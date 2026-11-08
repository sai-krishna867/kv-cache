"""Uniform per-token affine quantization, simulated by quantize-then-dequantize.

The cache keeps dequantized BF16 values, so quality is exact for the format while
memory is an analytic estimate. Groups run along the head dimension (per token),
using mx.quantize so the rounding matches MLX's real quantized storage.
"""

from __future__ import annotations

from typing import Any

import mlx.core as mx
import numpy as np
from mlx_lm.models.cache import KVCache

from kvstudy.backends.mlx import Arch

SCALE_BIAS_BYTES = 4  # one bf16 scale + one bf16 bias per group
VERIFY_SAMPLE_TOKENS = 64


def fake_quantize(x: mx.array, bits: int, group_size: int) -> mx.array:
    wq, scales, biases = mx.quantize(x, group_size=group_size, bits=bits)
    return mx.dequantize(wq, scales, biases, group_size=group_size, bits=bits)


def bytes_per_token(arch: Arch, key_bits: int, value_bits: int, group_size: int) -> int:
    """Packed K and V payload plus per-group scales and biases, over all layers and KV heads."""
    groups = arch.head_dim // group_size
    per_head = arch.head_dim * (key_bits + value_bits) // 8 + 2 * groups * SCALE_BIAS_BYTES
    return per_head * arch.num_kv_heads * arch.num_layers


def verify_fake_quantized(original: mx.array, converted: mx.array, bits: int, group_size: int) -> None:
    """Check a sample of converted values against the format they claim to hold.

    Conversion must have changed values, the error must stay within two nominal
    quantization steps per group (MLX shifts the grid so zero is exact, and BF16
    output rounding adds error at 8 bits), and when a group is larger than the
    number of levels it must take at most 2**bits distinct values.
    """
    n = original.shape[2]
    idx = mx.array(np.linspace(0, n - 1, num=min(n, VERIFY_SAMPLE_TOKENS)).astype(np.int32))
    x = np.array(original[:, :, idx, :].astype(mx.float32)).reshape(-1, group_size)
    y = np.array(converted[:, :, idx, :].astype(mx.float32)).reshape(-1, group_size)
    if np.array_equal(x, y):
        raise AssertionError(f"{bits}-bit conversion left the sampled values unchanged")
    step = (x.max(axis=1) - x.min(axis=1)) / (2**bits - 1)
    err = np.abs(x - y).max(axis=1)
    if np.any(err > 2 * step + 1e-6):
        raise AssertionError(f"rounding error exceeds the {bits}-bit bound")
    if 2**bits < group_size:
        levels = max(len(np.unique(row)) for row in y)
        if levels > 2**bits:
            raise AssertionError(f"group holds {levels} distinct values; {bits}-bit allows {2**bits}")


class ProtectedWindowCache(KVCache):
    """KV cache for a converted branch, with the protocol's protected window.

    Positions [0, window_start) are held at the method's key/value bits. Tokens
    appended after conversion (question, then generated tokens) are fake-quantized
    at window_bits; once more than window_tokens are held, the oldest are demoted
    to the method's bits. Tokens are never promoted.

    Deliberately has no `bits` attribute: mlx-lm would route attention to the
    quantized kernels, but this cache holds dequantized values.
    """

    def __init__(
        self,
        keys: mx.array,
        values: mx.array,
        key_bits: int,
        value_bits: int,
        group_size: int,
        window_tokens: int,
        window_bits: int,
    ) -> None:
        super().__init__()
        self.keys, self.values, self.offset = keys, values, keys.shape[2]
        self.key_bits, self.value_bits, self.group_size_ = key_bits, value_bits, group_size
        self.window_tokens, self.window_bits = window_tokens, window_bits
        self.n_context = self.window_start = self.offset

    def update_and_fetch(self, keys, values):
        g = self.group_size_
        super().update_and_fetch(fake_quantize(keys, self.window_bits, g), fake_quantize(values, self.window_bits, g))
        overflow = self.offset - self.window_start - self.window_tokens
        if overflow > 0:
            a, b = self.window_start, self.window_start + overflow
            self.keys[..., a:b, :] = fake_quantize(self.keys[..., a:b, :], self.key_bits, g)
            self.values[..., a:b, :] = fake_quantize(self.values[..., a:b, :], self.value_bits, g)
            self.window_start = b
        return self.keys_and_values()

    @property
    def n_window(self) -> int:
        return self.offset - self.window_start


class UniformMethod:
    def __init__(self, method_cfg: Any, protocol: Any) -> None:
        self.cfg = method_cfg
        self.window = protocol.protected_window

    def apply(self, cache: list[Any], arch: Arch):
        from kvstudy.methods.base import MethodReport

        cfg = self.cfg
        for i, c in enumerate(cache):
            k, v = c.keys[..., : c.offset, :], c.values[..., : c.offset, :]
            qk = fake_quantize(k, cfg.key_bits, cfg.group_size)
            qv = fake_quantize(v, cfg.value_bits, cfg.group_size)
            if i == 0:
                verify_fake_quantized(k, qk, cfg.key_bits, cfg.group_size)
                verify_fake_quantized(v, qv, cfg.value_bits, cfg.group_size)
            cache[i] = ProtectedWindowCache(
                qk, qv, cfg.key_bits, cfg.value_bits, cfg.group_size, self.window.tokens, self.window.precision_bits
            )
        mx.eval([c.state for c in cache])
        return MethodReport(cache_bytes_estimated=self.estimate_bytes(cache, arch), label="analytic_affine_simulated")

    def estimate_bytes(self, cache: list[Any], arch: Arch) -> int:
        """Estimated bytes of the cache as it stands, protected window included."""
        c = cache[0]
        if any((x.window_start, x.offset) != (c.window_start, c.offset) for x in cache):
            raise AssertionError("layers disagree on cache layout")
        g = self.cfg.group_size
        low = bytes_per_token(arch, self.cfg.key_bits, self.cfg.value_bits, g)
        win = bytes_per_token(arch, self.window.precision_bits, self.window.precision_bits, g)
        return c.window_start * low + c.n_window * win
