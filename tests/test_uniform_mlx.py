"""Uniform simulated quantization and the protected window, on a tiny random model."""

from types import SimpleNamespace

import numpy as np
import pytest

mx = pytest.importorskip("mlx.core")

from kvstudy.backends import mlx as backend  # noqa: E402
from kvstudy.methods.uniform import (  # noqa: E402
    ProtectedWindowCache,
    UniformMethod,
    bytes_per_token,
    fake_quantize,
    verify_fake_quantized,
)

pytestmark = pytest.mark.mlx

ARCH = backend.Arch(num_layers=2, num_kv_heads=2, head_dim=64)
STUB_TOKENIZER = SimpleNamespace(eos_token_ids=[], decode=lambda ids: " ".join(map(str, ids)))
QWEN = backend.Arch(num_layers=36, num_kv_heads=8, head_dim=128)


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
    mx.eval(m.parameters())  # bf16 weights, as in the experiment
    return m


def _ids(n, seed=1):
    return np.random.default_rng(seed).integers(0, 97, size=n).tolist()


def _method(key_bits=4, value_bits=4, group_size=64, window_tokens=512, window_bits=8):
    cfg = SimpleNamespace(key_bits=key_bits, value_bits=value_bits, group_size=group_size)
    protocol = SimpleNamespace(protected_window=SimpleNamespace(tokens=window_tokens, precision_bits=window_bits))
    return UniformMethod(cfg, protocol)


def _np(x):
    return np.array(x.astype(mx.float32))


def _max_levels(x, group_size):
    return max(len(np.unique(row)) for row in _np(x).reshape(-1, group_size))


# --- accounting ----------------------------------------------------------------


@pytest.mark.parametrize("bits", [2, 3, 4, 5, 6, 8])
@pytest.mark.parametrize("group_size", [32, 64, 128])
def test_bytes_per_token_matches_effective_bits_formula(bits, group_size):
    # Experiment design: b_eff = b + 32 / g (one bf16 scale and bias per group)
    elements = QWEN.num_layers * QWEN.num_kv_heads * QWEN.head_dim * 2
    assert bytes_per_token(QWEN, bits, bits, group_size) * 8 / elements == bits + 32 / group_size


def test_bytes_per_token_matches_mlx_packed_buffers():
    x = mx.zeros((1, QWEN.num_kv_heads, 1, QWEN.head_dim), dtype=mx.bfloat16)
    for bits in (2, 3, 4, 5, 6, 8):
        wq, scales, biases = mx.quantize(x, group_size=64, bits=bits)
        per_layer_one_tensor = wq.nbytes + scales.nbytes + biases.nbytes
        assert bytes_per_token(QWEN, bits, bits, 64) == 2 * per_layer_one_tensor * QWEN.num_layers


def test_asymmetric_bits_accounting():
    assert bytes_per_token(QWEN, 8, 4, 64) * 8 / (36 * 8 * 128 * 2) == 6.5


# --- format verification ---------------------------------------------------------


def test_verification_accepts_real_conversion():
    x = mx.random.normal((1, 2, 100, 64)).astype(mx.bfloat16)
    for bits in (2, 3, 4, 5, 6, 8):
        verify_fake_quantized(x, fake_quantize(x, bits, 32), bits, 32)


def test_verification_rejects_unconverted_values():
    x = mx.random.normal((1, 2, 100, 64)).astype(mx.bfloat16)
    with pytest.raises(AssertionError, match="unchanged"):
        verify_fake_quantized(x, x, 4, 64)


def test_verification_rejects_wrong_claimed_bits():
    x = mx.random.normal((1, 2, 100, 64)).astype(mx.bfloat16)
    with pytest.raises(AssertionError):
        verify_fake_quantized(x, fake_quantize(x, 8, 64), 4, 64)  # too many levels for 4-bit
    with pytest.raises(AssertionError):
        verify_fake_quantized(x, fake_quantize(x, 2, 64), 8, 64)  # error too large for 8-bit


# --- conversion on a branch ------------------------------------------------------


def test_uniform_conversion_leaves_reference_untouched(model):
    reference = backend.prefill(model, _ids(200))
    before = [(_np(c.keys[..., : c.offset, :]), _np(c.values[..., : c.offset, :])) for c in reference]
    branch = backend.branch_cache(reference)
    method = _method(key_bits=4, value_bits=2)
    report = method.apply(branch, ARCH)
    backend.generate(model, STUB_TOKENIZER, _ids(20, seed=3), branch, max_new_tokens=5)

    for c, (k, v) in zip(reference, before):
        np.testing.assert_array_equal(_np(c.keys[..., : c.offset, :]), k)
        np.testing.assert_array_equal(_np(c.values[..., : c.offset, :]), v)
    assert report.cache_bytes_estimated == 200 * bytes_per_token(ARCH, 4, 2, 64)
    assert report.label == "analytic_affine_simulated"


def test_converted_context_holds_configured_bits(model):
    branch = backend.branch_cache(backend.prefill(model, _ids(150)))
    _method(key_bits=3, value_bits=2, group_size=64).apply(branch, ARCH)
    for c in branch:
        assert isinstance(c, ProtectedWindowCache)
        assert not hasattr(c, "bits")  # would route mlx-lm attention to quantized kernels
        assert _max_levels(c.keys[..., : c.offset, :], 64) <= 2**3
        assert _max_levels(c.values[..., : c.offset, :], 64) <= 2**2


def test_conversion_is_deterministic_across_branches(model):
    reference = backend.prefill(model, _ids(120))
    b1, b2 = backend.branch_cache(reference), backend.branch_cache(reference)
    _method().apply(b1, ARCH)
    _method().apply(b2, ARCH)
    out1 = backend.generate(model, STUB_TOKENIZER, _ids(10, seed=4), b1, max_new_tokens=6)
    out2 = backend.generate(model, STUB_TOKENIZER, _ids(10, seed=4), b2, max_new_tokens=6)
    assert out1 == out2


# --- protected window ------------------------------------------------------------


def test_appended_tokens_enter_window_at_window_bits(model):
    n_ctx = 100
    branch = backend.branch_cache(backend.prefill(model, _ids(n_ctx)))
    method = _method(key_bits=2, value_bits=2, window_tokens=512, window_bits=4)
    method.apply(branch, ARCH)
    model(mx.array(_ids(30, seed=5))[None], cache=branch)
    for c in branch:
        assert (c.window_start, c.n_window) == (n_ctx, 30)
        assert _max_levels(c.keys[..., n_ctx : c.offset, :], 64) <= 2**4
        assert _max_levels(c.keys[..., n_ctx : c.offset, :], 64) > 2**2  # not demoted
    assert method.estimate_bytes(branch, ARCH) == n_ctx * bytes_per_token(ARCH, 2, 2, 64) + 30 * bytes_per_token(
        ARCH, 4, 4, 64
    )


def test_window_overflow_demotes_oldest(model):
    n_ctx, window = 100, 8
    branch = backend.branch_cache(backend.prefill(model, _ids(n_ctx)))
    method = _method(key_bits=2, value_bits=2, window_tokens=window, window_bits=8)
    method.apply(branch, ARCH)
    model(mx.array(_ids(5, seed=6))[None], cache=branch)  # chunk: 5 in window
    for t in _ids(7, seed=7):  # then one token at a time: 12 appended, 4 demoted
        model(mx.array([t])[None], cache=branch)
    for c in branch:
        assert c.offset == n_ctx + 12
        assert (c.window_start, c.n_window) == (n_ctx + 4, window)
        assert _max_levels(c.keys[..., n_ctx : n_ctx + 4, :], 64) <= 2**2  # demoted to the method's bits
        assert _max_levels(c.values[..., n_ctx : n_ctx + 4, :], 64) <= 2**2
    expected = (n_ctx + 4) * bytes_per_token(ARCH, 2, 2, 64) + window * bytes_per_token(ARCH, 8, 8, 64)
    assert method.estimate_bytes(branch, ARCH) == expected
