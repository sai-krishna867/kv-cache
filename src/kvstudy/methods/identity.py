"""BF16 identity: the cache stays unquantized."""

from __future__ import annotations

from typing import Any

from kvstudy.backends.mlx import Arch, cache_nbytes

BF16_BYTES = 2


class IdentityMethod:
    """The protected window does not apply: the baseline stays BF16 throughout."""

    def __init__(self, method_cfg: Any, protocol: Any) -> None:
        self.cfg = method_cfg

    def apply(self, cache: list[Any], arch: Arch):
        from kvstudy.methods.base import MethodReport

        return MethodReport(cache_bytes_estimated=self.estimate_bytes(cache, arch), label="analytic_bf16")

    def estimate_bytes(self, cache: list[Any], arch: Arch) -> int:
        tokens = cache[0].offset
        analytic = tokens * arch.num_layers * arch.num_kv_heads * arch.head_dim * 2 * BF16_BYTES
        measured = cache_nbytes(cache)
        if measured != analytic:
            raise AssertionError(f"bf16 cache holds {measured} bytes; analytic prediction is {analytic}")
        return analytic
