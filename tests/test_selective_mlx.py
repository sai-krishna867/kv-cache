"""Selective precision: selectors, mixed-precision storage and memory accounting, on a tiny random model."""

from types import SimpleNamespace

import numpy as np
import pytest

mx = pytest.importorskip("mlx.core")

from kvstudy.backends import mlx as backend  # noqa: E402
from kvstudy.methods.selective import SelectiveMethod, context_bytes, index_bytes  # noqa: E402
from kvstudy.methods.selectors import CacheView, residual_scores, select_tokens  # noqa: E402
from kvstudy.methods.uniform import ProtectedWindowCache, bytes_per_token  # noqa: E402

pytestmark = pytest.mark.mlx

ARCH = backend.Arch(num_layers=2, num_kv_heads=2, head_dim=64)
QWEN = backend.Arch(num_layers=36, num_kv_heads=8, head_dim=128)
STUB_TOKENIZER = SimpleNamespace(eos_token_ids=[], decode=lambda ids: " ".join(map(str, ids)))


@pytest.fixture(scope="module")
def model():
    from mlx_lm.models import qwen3

    mx.random.seed(0)
    args = qwen3.ModelArgs(
        model_type="qwen3", hidden_size=128, num_hidden_layers=ARCH.num_layers, intermediate_size=256,
        num_attention_heads=4, rms_norm_eps=1e-6, vocab_size=97, num_key_value_heads=ARCH.num_kv_heads,
        max_position_embeddings=4096, head_dim=ARCH.head_dim, tie_word_embeddings=True, rope_theta=10000.0,
    )  # fmt: skip
    m = qwen3.Model(args)
    mx.eval(m.parameters())
    return m


def _ids(n, seed=1):
    return np.random.default_rng(seed).integers(0, 97, size=n).tolist()


def _cfg(selector="random", high_bits=8, low_bits=2, high_fraction=0.25, **kw):
    defaults = {"random": {"seed": 0}, "sinks_recency": {"sink_tokens": 4}}.get(selector, {})
    fields = {"seed": None, "sink_tokens": None, **defaults, **kw}
    return SimpleNamespace(
        selector=selector, high_bits=high_bits, low_bits=low_bits, high_fraction=high_fraction, group_size=64, **fields
    )


def _method(**kw):
    protocol = SimpleNamespace(protected_window=SimpleNamespace(tokens=512, precision_bits=8))
    return SelectiveMethod(_cfg(**kw), protocol)


def _view(n=100, layers=2, heads=2, seed=0):
    keys = [mx.random.normal((1, heads, n, 64), key=mx.random.key(seed + i)).astype(mx.bfloat16) for i in range(layers)]
    values = [mx.random.normal((1, heads, n, 64), key=mx.random.key(seed + 10 + i)).astype(mx.bfloat16) for i in range(layers)]
    return CacheView(keys=keys, values=values)


def _levels(x):
    return max(len(np.unique(row)) for row in np.array(x.astype(mx.float32)).reshape(-1, 64))


# --- accounting ------------------------------------------------------------------


def test_context_bytes_counts_payload_scales_and_indices():
    n, n_high = 4000, 660
    expected = (
        n_high * bytes_per_token(QWEN, 8, 8, 64)
        + (n - n_high) * bytes_per_token(QWEN, 2, 2, 64)
        + n_high * 2 * QWEN.num_layers * QWEN.num_kv_heads  # one uint16 index per high token per layer-head
    )
    assert context_bytes(QWEN, n, n_high, 8, 2, 64) == expected
    # effective bits = 2.5 + f * (8.5 - 2.5) + f * 16 / (2 * head_dim)
    elements = n * QWEN.num_layers * QWEN.num_kv_heads * QWEN.head_dim * 2
    f = n_high / n
    assert context_bytes(QWEN, n, n_high, 8, 2, 64) * 8 / elements == pytest.approx(2.5 + 6 * f + 16 * f / 256)


def test_index_width_grows_past_uint16():
    assert index_bytes(2**16) == 2 and index_bytes(2**16 + 1) == 4


# --- selectors ---------------------------------------------------------------------


@pytest.mark.parametrize("selector", ["random", "sinks_recency", "quantization_residual"])
def test_selectors_pick_exact_count_per_head(selector):
    view = _view(n=101)
    masks = select_tokens(view, _cfg(selector, high_fraction=0.3))
    assert len(masks) == 2
    for m in masks:
        assert m.shape == (2, 101) and np.all(m.sum(axis=-1) == round(0.3 * 101))


def test_random_selection_is_seeded_and_varies_by_layer_and_head():
    view = _view()
    a, b = select_tokens(view, _cfg("random")), select_tokens(view, _cfg("random"))
    assert all(np.array_equal(x, y) for x, y in zip(a, b))
    assert not np.array_equal(a[0], a[1]) and not np.array_equal(a[0][0], a[0][1])
    c = select_tokens(view, _cfg("random", seed=1))
    assert not np.array_equal(a[0], c[0])


def test_sinks_recency_selects_first_and_last_tokens():
    [m, _] = select_tokens(_view(n=40), _cfg("sinks_recency", high_fraction=0.25, sink_tokens=3))
    expected = np.zeros(40, dtype=bool)
    expected[:3] = True
    expected[-7:] = True
    assert np.array_equal(m[0], expected) and np.array_equal(m[1], expected)


def test_residual_selects_tokens_with_largest_low_bit_error():
    view = _view(n=64, layers=1, heads=1)
    k = np.array(view.keys[0].astype(mx.float32))
    k[0, 0, [5, 17, 40], :32] *= 30  # outliers: large in-group range, large 2-bit error
    view = CacheView(keys=[mx.array(k).astype(mx.bfloat16)], values=view.values)
    scores = residual_scores(view.keys[0], view.values[0], 2, 64)
    assert set(np.argsort(-scores[0])[:3]) == {5, 17, 40}
    [m] = select_tokens(view, _cfg("quantization_residual", high_fraction=3 / 64))
    assert set(np.flatnonzero(m[0])) == {5, 17, 40}


def test_select_tokens_rejects_wrong_count(monkeypatch):
    from kvstudy.methods import selectors

    monkeypatch.setitem(selectors.SELECTORS, "random", lambda view, cfg: [np.ones((view.n_heads, view.n_tokens), bool)] * 2)
    with pytest.raises(AssertionError, match="exactly"):
        select_tokens(_view(), _cfg("random"))


# --- storage on a branch -------------------------------------------------------------


@pytest.mark.parametrize("selector", ["random", "sinks_recency", "quantization_residual"])
def test_selected_tokens_hold_high_bits_and_reference_is_untouched(model, selector):
    reference = backend.prefill(model, _ids(128))
    before = [np.array(c.keys[..., : c.offset, :].astype(mx.float32)) for c in reference]
    branch = backend.branch_cache(reference)
    method = _method(selector=selector, high_bits=8, low_bits=2, high_fraction=0.25)
    view = CacheView(keys=[c.keys[..., : c.offset, :] for c in branch], values=[c.values[..., : c.offset, :] for c in branch])
    masks = select_tokens(view, method.cfg)
    report = method.apply(branch, ARCH)

    for c, mask in zip(branch, masks):
        assert isinstance(c, ProtectedWindowCache) and not hasattr(c, "bits")
        for h in range(ARCH.num_kv_heads):
            low = np.flatnonzero(~mask[h])
            assert _levels(c.keys[:, h : h + 1, mx.array(low), :]) <= 4
            assert _levels(c.values[:, h : h + 1, mx.array(low), :]) <= 4
            high = np.flatnonzero(mask[h])
            assert _levels(c.keys[:, h : h + 1, mx.array(high), :]) > 4  # kept at high precision
    backend.generate(model, STUB_TOKENIZER, _ids(10, seed=3), branch, max_new_tokens=4)
    for c, k in zip(reference, before):
        np.testing.assert_array_equal(np.array(c.keys[..., : c.offset, :].astype(mx.float32)), k)

    assert report.cache_bytes_estimated == context_bytes(ARCH, 128, 32, 8, 2, 64)
    assert report.metadata["high_fraction_realized"] == 0.25
    assert report.metadata["n_high_tokens_per_head"] == 32


def test_final_bytes_add_window_tokens(model):
    branch = backend.branch_cache(backend.prefill(model, _ids(100)))
    method = _method(high_fraction=0.2)
    method.apply(branch, ARCH)
    model(mx.array(_ids(12, seed=5))[None], cache=branch)
    expected = context_bytes(ARCH, 100, 20, 8, 2, 64) + 12 * bytes_per_token(ARCH, 8, 8, 64)
    assert method.estimate_bytes(branch, ARCH) == expected
