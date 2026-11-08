"""Token selection for selective precision, kept separate from storage.

A selector sees the context K/V of every layer and returns, per layer, a boolean
mask of shape (kv_heads, tokens): True marks a token held at high precision.
Every selector picks exactly round(high_fraction * tokens) tokens per layer and
KV head, so heads need no padding. Add a policy by writing a function and
registering it in SELECTORS (and naming it in config.SELECTORS).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable

import mlx.core as mx
import numpy as np


@dataclass(frozen=True)
class CacheView:
    keys: list[mx.array]  # per layer, (1, kv_heads, tokens, head_dim), context positions only
    values: list[mx.array]

    @property
    def n_tokens(self) -> int:
        return self.keys[0].shape[2]

    @property
    def n_heads(self) -> int:
        return self.keys[0].shape[1]


def n_selected(n_tokens: int, high_fraction: float) -> int:
    return int(round(high_fraction * n_tokens))


def _top_k_mask(scores: np.ndarray, k: int) -> np.ndarray:
    """Per row, True at the k highest scores (ties broken by lower position)."""
    order = np.argsort(-scores, axis=-1, kind="stable")[:, :k]
    mask = np.zeros(scores.shape, dtype=bool)
    np.put_along_axis(mask, order, True, axis=-1)
    return mask


def select_random(view: CacheView, cfg: Any) -> list[np.ndarray]:
    """Uniformly random tokens, independently per layer and head, seeded by (seed, layer)."""
    k = n_selected(view.n_tokens, cfg.high_fraction)
    masks = []
    for layer in range(len(view.keys)):
        rng = np.random.default_rng([cfg.seed, layer])
        masks.append(_top_k_mask(rng.random((view.n_heads, view.n_tokens)), k))
    return masks


def select_sinks_recency(view: CacheView, cfg: Any) -> list[np.ndarray]:
    """The first sink_tokens context tokens, then the most recent ones; the same for every head."""
    n, k = view.n_tokens, n_selected(view.n_tokens, cfg.high_fraction)
    sinks = min(cfg.sink_tokens, k)
    row = np.zeros(n, dtype=bool)
    row[:sinks] = True
    row[n - (k - sinks) :] = True
    return [np.broadcast_to(row, (view.n_heads, n)).copy() for _ in view.keys]


def residual_scores(k: mx.array, v: mx.array, bits: int, group_size: int) -> np.ndarray:
    """Relative squared quantization error of each token at the low precision.

    score_t = ||K_t - Q(K_t)||^2 / mean_s ||K_s||^2 + ||V_t - Q(V_t)||^2 / mean_s ||V_s||^2,
    per KV head. Normalising each term by its head's mean squared norm keeps K and V
    on a comparable scale, so neither dominates because of its magnitude alone.
    """
    from kvstudy.methods.uniform import fake_quantize

    total = 0
    for x in (k, v):
        x32 = x.astype(mx.float32)
        err = ((x32 - fake_quantize(x, bits, group_size).astype(mx.float32)) ** 2).sum(axis=-1)
        norm = (x32**2).sum(axis=-1).mean(axis=-1, keepdims=True)
        total = total + err / mx.maximum(norm, 1e-12)
    return np.array(total[0])


def select_quantization_residual(view: CacheView, cfg: Any) -> list[np.ndarray]:
    """Tokens that low-bit quantization distorts most (see residual_scores)."""
    k = n_selected(view.n_tokens, cfg.high_fraction)
    return [
        _top_k_mask(residual_scores(kl, vl, cfg.low_bits, cfg.group_size), k) for kl, vl in zip(view.keys, view.values)
    ]


SELECTORS: dict[str, Callable[[CacheView, Any], list[np.ndarray]]] = {
    "random": select_random,
    "sinks_recency": select_sinks_recency,
    "quantization_residual": select_quantization_residual,
}


def select_tokens(view: CacheView, cfg: Any) -> list[np.ndarray]:
    if cfg.selector not in SELECTORS:
        raise ValueError(f"unknown selector {cfg.selector!r}")
    masks = SELECTORS[cfg.selector](view, cfg)
    k = n_selected(view.n_tokens, cfg.high_fraction)
    for m in masks:
        if m.shape != (view.n_heads, view.n_tokens) or np.any(m.sum(axis=-1) != k):
            raise AssertionError(f"selector {cfg.selector} must select exactly {k} tokens per head")
    return masks
