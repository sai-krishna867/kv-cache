"""Backend behaviour on a tiny random Qwen3-architecture model (no downloads)."""

from types import SimpleNamespace

import numpy as np
import pytest

mx = pytest.importorskip("mlx.core")

from kvstudy.backends import mlx as backend  # noqa: E402
from kvstudy.methods.identity import IdentityMethod  # noqa: E402

pytestmark = pytest.mark.mlx

ARCH = backend.Arch(num_layers=2, num_kv_heads=2, head_dim=16)
STUB_TOKENIZER = SimpleNamespace(eos_token_ids=[], decode=lambda ids: " ".join(map(str, ids)))


@pytest.fixture(scope="module")
def model():
    from mlx.utils import tree_map
    from mlx_lm.models import qwen3

    mx.random.seed(0)
    args = qwen3.ModelArgs(
        model_type="qwen3", hidden_size=64, num_hidden_layers=ARCH.num_layers, intermediate_size=128,
        num_attention_heads=4, rms_norm_eps=1e-6, vocab_size=97, num_key_value_heads=ARCH.num_kv_heads,
        max_position_embeddings=4096, head_dim=ARCH.head_dim, tie_word_embeddings=True, rope_theta=10000.0,
    )  # fmt: skip
    m = qwen3.Model(args)
    m.update(tree_map(lambda p: p.astype(mx.float32), m.parameters()))
    mx.eval(m.parameters())
    return m


def _ids(n, seed=1):
    return np.random.default_rng(seed).integers(0, 97, size=n).tolist()


def _snapshot(cache):
    return [(c.offset, np.array(c.keys[..., : c.offset, :]), np.array(c.values[..., : c.offset, :])) for c in cache]


def _assert_same(cache, snap):
    for c, (offset, k, v) in zip(cache, snap):
        assert c.offset == offset
        np.testing.assert_array_equal(np.array(c.keys[..., : c.offset, :]), k)
        np.testing.assert_array_equal(np.array(c.values[..., : c.offset, :]), v)


@pytest.mark.parametrize("step", [7, 2048])
def test_split_prefill_matches_single_pass(model, step):
    ids = _ids(301)
    boundary = 263
    full = model(mx.array(ids)[None])[:, -1, :]
    cache = backend.prefill(model, ids[:boundary], step=step)
    split = model(mx.array(ids[boundary:])[None], cache=cache)[:, -1, :]
    np.testing.assert_allclose(np.array(split), np.array(full), atol=1e-4, rtol=1e-4)


def test_branch_does_not_mutate_reference(model):
    ids = _ids(300)
    reference = backend.prefill(model, ids[:256])  # offset on a 256-step boundary
    before = _snapshot(reference)
    branch = backend.branch_cache(reference)
    backend.generate(model, STUB_TOKENIZER, ids[256:], branch, max_new_tokens=8)
    assert branch[0].offset > reference[0].offset
    _assert_same(reference, before)


def test_branches_are_independent(model):
    ids = _ids(250, seed=2)
    reference = backend.prefill(model, ids[:237])  # offset off the 256-step boundary
    before = _snapshot(reference)
    b1 = backend.branch_cache(reference)
    out1 = backend.generate(model, STUB_TOKENIZER, ids[237:], b1, max_new_tokens=8)
    b1_state = _snapshot(b1)
    b2 = backend.branch_cache(reference)
    out2 = backend.generate(model, STUB_TOKENIZER, ids[237:], b2, max_new_tokens=8)
    assert out1 == out2
    _assert_same(b1, b1_state)
    _assert_same(reference, before)


def test_identity_bytes_match_analytic_bf16(model):
    cache = backend.prefill(model, _ids(100))
    for c in cache:
        c.keys, c.values = c.keys.astype(mx.bfloat16), c.values.astype(mx.bfloat16)
    report = IdentityMethod(None, None).apply(cache, ARCH)
    assert report.cache_bytes_estimated == 100 * ARCH.num_layers * ARCH.num_kv_heads * ARCH.head_dim * 2 * 2
    assert backend.cache_nbytes(cache) == report.cache_bytes_estimated


def test_identity_rejects_non_bf16_cache(model):
    cache = backend.prefill(model, _ids(100))  # float32 test model
    with pytest.raises(AssertionError, match="analytic prediction"):
        IdentityMethod(None, None).apply(cache, ARCH)


def test_read_arch_uses_explicit_head_dim():
    cfg = {"num_attention_heads": 32, "num_hidden_layers": 36, "num_key_value_heads": 8, "head_dim": 128, "hidden_size": 2560}
    assert backend.read_arch(cfg) == backend.Arch(36, 8, 128)
