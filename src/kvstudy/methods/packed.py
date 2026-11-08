"""Real packed storage (systems_real): K/V held as mx.quantize payloads, never dequantized in the cache.

A cache layer is a list of segments, each with its own key/value bits:
uniform uses [context @ bits, window]; selective uses [high, low, window].
Tokens appended after conversion go to the window segment at the window
precision; on overflow the oldest are demoted (dequantize, requantize at the
method's low bits) into the low segment, exactly as the simulation does.

Attention runs over the segments with mx.quantized_matmul and one softmax across
all of them. Keys are stored after RoPE, so attention does not depend on token
order: context tokens are visible to every query, and appended tokens are masked
causally by their absolute positions (identical across heads). Selective storage
keeps a uint16 position index per high-precision token and head; it is counted
in memory but not read by attention.

mlx-lm sends any cache with a `bits` attribute to
models.base.quantized_scaled_dot_product_attention; install() wraps that function
so it handles PackedSegments and delegates everything else unchanged.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import mlx.core as mx

Quantized = tuple[mx.array, mx.array, mx.array]


@dataclass
class Segment:
    key_bits: int
    value_bits: int
    keys: Quantized | None = None
    values: Quantized | None = None
    n_context: int = 0  # leading tokens from the conversion point: visible to every query
    positions: list[int] = field(default_factory=list)  # absolute positions of appended tokens, in order

    @property
    def length(self) -> int:
        return 0 if self.keys is None else self.keys[0].shape[-2]

    def append(self, k: Quantized, v: Quantized, positions: list[int]) -> None:
        if self.keys is None:
            self.keys, self.values = k, v
        else:
            self.keys = tuple(mx.concatenate([a, b], axis=-2) for a, b in zip(self.keys, k))
            self.values = tuple(mx.concatenate([a, b], axis=-2) for a, b in zip(self.values, v))
        self.positions += positions

    def arrays(self) -> list[mx.array]:
        return [] if self.keys is None else [*self.keys, *self.values]


class PackedSegments(list):
    """What PackedCache.update_and_fetch returns: the segments attention must read."""


def quantize(x: mx.array, bits: int, group_size: int) -> Quantized:
    return tuple(mx.quantize(x, group_size=group_size, bits=bits))


def dequantize(q: Quantized, bits: int, group_size: int) -> mx.array:
    return mx.dequantize(*q, group_size=group_size, bits=bits)


class PackedCache:
    """One layer of packed segments; the last segment is the protected window."""

    def __init__(
        self,
        segments: list[Segment],
        low_segment: int,
        group_size: int,
        window_tokens: int,
        window_bits: int,
        offset: int,
        index: mx.array | None = None,
    ) -> None:
        self.segments = segments + [Segment(window_bits, window_bits)]
        self.low = low_segment
        self.group_size = group_size
        self.bits = window_bits  # only its presence matters: it routes mlx-lm to quantized attention
        self.window_tokens = window_tokens
        self.offset = offset
        self.index = index  # selective: uint16 positions of high-precision tokens, (1, kv_heads, n_high)

    @property
    def window(self) -> Segment:
        return self.segments[-1]

    def update_and_fetch(self, keys: mx.array, values: mx.array) -> tuple[PackedSegments, PackedSegments]:
        g, w = self.group_size, self.window
        n = keys.shape[2]
        w.append(quantize(keys, w.key_bits, g), quantize(values, w.value_bits, g), list(range(self.offset, self.offset + n)))
        self.offset += n
        overflow = w.length - self.window_tokens
        if overflow > 0:
            low = self.segments[self.low]
            old_k = dequantize(tuple(a[..., :overflow, :] for a in w.keys), w.key_bits, g)
            old_v = dequantize(tuple(a[..., :overflow, :] for a in w.values), w.value_bits, g)
            low.append(quantize(old_k, low.key_bits, g), quantize(old_v, low.value_bits, g), w.positions[:overflow])
            w.keys = tuple(a[..., overflow:, :] for a in w.keys)
            w.values = tuple(a[..., overflow:, :] for a in w.values)
            w.positions = w.positions[overflow:]
        segments = PackedSegments(s for s in self.segments if s.length)
        return segments, segments  # model code unpacks (keys, values); attention reads both from segments

    def make_mask(self, n: int, **kwargs: Any) -> None:
        return None  # segmented attention builds its own mask from positions

    @property
    def state(self) -> list[mx.array]:
        arrays = [a for s in self.segments for a in s.arrays()]
        return arrays + ([self.index] if self.index is not None else [])

    def nbytes(self) -> int:
        return sum(a.nbytes for a in self.state)

    @property
    def n_window(self) -> int:
        return self.window.length


def segmented_attention(queries: mx.array, segments: PackedSegments, scale: float, group_size: int) -> mx.array:
    B, n_q_heads, L, D = queries.shape
    n_kv_heads = segments[0].keys[0].shape[-3]
    rep = n_q_heads // n_kv_heads
    q = queries * scale
    expand = (lambda t: tuple(mx.expand_dims(a, axis=-3) for a in t)) if rep > 1 else (lambda t: t)
    if rep > 1:
        q = mx.reshape(q, (B, n_kv_heads, rep, L, D))

    q_pos = mx.array(segments_query_positions(segments, L))
    scores = []
    for s in segments:
        sc = mx.quantized_matmul(q, *expand(s.keys), transpose=True, group_size=group_size, bits=s.key_bits)
        if s.positions and L > 1:
            k_pos = mx.array([-1] * s.n_context + s.positions)  # context tokens precede every query
            sc = mx.where(k_pos[None, :] <= q_pos[:, None], sc, mx.finfo(sc.dtype).min)
        scores.append(sc)
    probs = mx.softmax(mx.concatenate(scores, axis=-1), axis=-1, precise=True)

    out, start = 0, 0
    for s in segments:
        p = probs[..., start : start + s.length]
        out = out + mx.quantized_matmul(p, *expand(s.values), transpose=False, group_size=group_size, bits=s.value_bits)
        start += s.length
    if rep > 1:
        out = mx.reshape(out, (B, n_q_heads, L, D))
    return out


def segments_query_positions(segments: PackedSegments, n_queries: int) -> list[int]:
    """Absolute positions of the current queries: the last n_queries appended tokens."""
    appended = sorted(p for s in segments for p in s.positions)
    return appended[-n_queries:]


_installed = False


def install() -> None:
    """Route PackedSegments through segmented_attention; other caches are unaffected."""
    global _installed
    if _installed:
        return
    from mlx_lm.models import base

    original = base.quantized_scaled_dot_product_attention

    def dispatch(queries, q_keys, q_values, scale, mask, group_size=64, bits=8):
        if isinstance(q_keys, PackedSegments):
            return segmented_attention(queries, q_keys, scale, group_size)
        return original(queries, q_keys, q_values, scale=scale, mask=mask, group_size=group_size, bits=bits)

    base.quantized_scaled_dot_product_attention = dispatch
    _installed = True
