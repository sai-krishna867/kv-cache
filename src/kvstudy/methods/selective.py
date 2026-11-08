"""Selective precision, simulated: selected context tokens at high_bits, the rest at low_bits.

Selection (methods/selectors.py) is per layer and KV head and shared by K and V.
Groups run along the head dimension, so each token is quantized independently
and mixing precisions per token is exact for the format.
"""

from __future__ import annotations

from typing import Any

import mlx.core as mx
import numpy as np

from kvstudy.backends.mlx import Arch
from kvstudy.methods.selectors import CacheView, select_tokens
from kvstudy.methods.uniform import (
    ProtectedWindowCache,
    bytes_per_token,
    fake_quantize,
    verify_fake_quantized,
)


def index_bytes(n_context: int) -> int:
    """Bytes per stored position index: uint16 while positions fit, else uint32."""
    return 2 if n_context <= 2**16 else 4


def context_bytes(arch: Arch, n_context: int, n_high: int, high_bits: int, low_bits: int, group_size: int) -> int:
    """Context cache bytes with n_high tokens per layer and KV head at high_bits.

    Counts packed payload, per-group scales and biases at each precision, and one
    position index per high-precision token per layer and head (shared by K and V).
    Every head holds the same number of high tokens, so there is no padding.
    """
    per_head = arch.num_kv_heads * arch.num_layers
    return (
        n_high * bytes_per_token(arch, high_bits, high_bits, group_size)
        + (n_context - n_high) * bytes_per_token(arch, low_bits, low_bits, group_size)
        + n_high * index_bytes(n_context) * per_head
    )


def _verify(original: mx.array, converted: mx.array, mask: np.ndarray, cfg: Any) -> None:
    """Head 0 of a layer: selected tokens hold high_bits, the others low_bits."""
    for selected, bits in ((True, cfg.high_bits), (False, cfg.low_bits)):
        idx = mx.array(np.flatnonzero(mask[0] == selected).astype(np.int32))
        if idx.size:
            verify_fake_quantized(original[:, :1, idx, :], converted[:, :1, idx, :], bits, cfg.group_size)


class SelectiveMethod:
    def __init__(self, method_cfg: Any, protocol: Any) -> None:
        self.cfg = method_cfg
        self.window = protocol.protected_window

    def apply(self, cache: list[Any], arch: Arch):
        from kvstudy.methods.base import MethodReport

        cfg = self.cfg
        n = cache[0].offset
        view = CacheView(
            keys=[c.keys[..., : c.offset, :] for c in cache], values=[c.values[..., : c.offset, :] for c in cache]
        )
        masks = select_tokens(view, cfg)
        n_high = int(masks[0][0].sum())
        layer_bytes = context_bytes(arch, n, n_high, cfg.high_bits, cfg.low_bits, cfg.group_size) // arch.num_layers
        for i, (k, v, mask) in enumerate(zip(view.keys, view.values, masks)):
            sel = mx.array(mask)[None, :, :, None]
            out = []
            for x in (k, v):
                high = fake_quantize(x, cfg.high_bits, cfg.group_size)
                out.append(mx.where(sel, high, fake_quantize(x, cfg.low_bits, cfg.group_size)))
            if i == 0:
                _verify(k, out[0], mask, cfg)
                _verify(v, out[1], mask, cfg)
            pc = ProtectedWindowCache(
                out[0], out[1], cfg.low_bits, cfg.low_bits, cfg.group_size, self.window.tokens, self.window.precision_bits
            )
            pc.context_bytes = layer_bytes
            cache[i] = pc
        mx.eval([c.state for c in cache])
        report_bytes = self.estimate_bytes(cache, arch)
        elements = n * arch.num_layers * arch.num_kv_heads * arch.head_dim * 2
        return MethodReport(
            cache_bytes_estimated=report_bytes,
            label="analytic_affine_simulated",
            metadata={
                "selector": cfg.selector,
                "high_fraction_configured": cfg.high_fraction,
                "high_fraction_realized": n_high / n,
                "n_high_tokens_per_head": n_high,
                "effective_bits_context": report_bytes * 8 / elements,
            },
        )

    def estimate_bytes(self, cache: list[Any], arch: Arch) -> int:
        """Context bytes as selected, plus demoted tokens at low_bits and the protected window."""
        c = cache[0]
        if any((x.window_start, x.offset) != (c.window_start, c.offset) for x in cache):
            raise AssertionError("layers disagree on cache layout")
        g = self.cfg.group_size
        low = bytes_per_token(arch, self.cfg.low_bits, self.cfg.low_bits, g)
        win = bytes_per_token(arch, self.window.precision_bits, self.window.precision_bits, g)
        return sum(x.context_bytes for x in cache) + (c.window_start - c.n_context) * low + c.n_window * win
