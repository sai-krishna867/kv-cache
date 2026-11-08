"""Cache method interface and registry.

A method converts a cache branch in place under its precision policy. Methods
never know which benchmark is running.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Protocol

from kvstudy.backends.mlx import Arch


@dataclass(frozen=True)
class MethodReport:
    cache_bytes_estimated: int
    label: str  # how cache_bytes_estimated was obtained
    metadata: dict[str, Any] = field(default_factory=dict)  # method-specific facts saved with the record


class CacheMethod(Protocol):
    def apply(self, cache: list[Any], arch: Arch) -> MethodReport:
        """Convert the branch in place and report its estimated cache bytes."""

    def estimate_bytes(self, cache: list[Any], arch: Arch) -> int:
        """Estimated bytes of the cache as it currently stands."""


from kvstudy.methods.identity import IdentityMethod  # noqa: E402
from kvstudy.methods.selective import SelectiveMethod  # noqa: E402
from kvstudy.methods.uniform import UniformMethod  # noqa: E402

METHODS: dict[str, type] = {"identity": IdentityMethod, "uniform": UniformMethod, "selective": SelectiveMethod}


def build_method(method_cfg: Any, protocol: Any) -> CacheMethod:
    if method_cfg.kind not in METHODS:
        raise ValueError(f"unknown method kind {method_cfg.kind!r}")
    return METHODS[method_cfg.kind](method_cfg, protocol)
