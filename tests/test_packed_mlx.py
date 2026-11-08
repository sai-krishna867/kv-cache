"""Real packed storage validated against the simulation, on a tiny random model."""

from types import SimpleNamespace

import numpy as np
import pytest

mx = pytest.importorskip("mlx.core")

from kvstudy.backends import mlx as backend  # noqa: E402
from kvstudy.methods.packed import PackedCache  # noqa: E402
from kvstudy.methods.selective import SelectiveMethod  # noqa: E402
from kvstudy.methods.uniform import UniformMethod  # noqa: E402

pytestmark = pytest.mark.mlx

ARCH = backend.Arch(num_layers=2, num_kv_heads=2, head_dim=64)
STUB_TOKENIZER = SimpleNamespace(eos_token_ids=[], decode=lambda ids: " ".join(map(str, ids)))


@pytest.fixture(scope="module")
def model():
    from mlx.utils import tree_map
    from mlx_lm.models import qwen3

    mx.random.seed(0)
    args = qwen3.ModelArgs(
        model_type="qwen3", hidden_size=128, num_hidden_layers=ARCH.num_layers, intermediate_size=256,
        num_attention_heads=4, rms_norm_eps=1e-6, vocab_size=97, num_key_value_heads=ARCH.num_kv_heads,
        max_position_embeddings=4096, head_dim=ARCH.head_dim, tie_word_embeddings=True, rope_theta=10000.0,
    )  # fmt: skip
    m = qwen3.Model(args)
    m.update(tree_map(lambda p: p.astype(mx.bfloat16), m.parameters()))  # bf16 cache, as with the real model
    mx.eval(m.parameters())
    return m


def _ids(n, seed=1):
    return np.random.default_rng(seed).integers(0, 97, size=n).tolist()


def _protocol(window_tokens=512, window_bits=8):
    return SimpleNamespace(protected_window=SimpleNamespace(tokens=window_tokens, precision_bits=window_bits))


METHODS = {
    "uniform_4": lambda p: UniformMethod(SimpleNamespace(key_bits=4, value_bits=4, group_size=64), p),
    "uniform_8_4": lambda p: UniformMethod(SimpleNamespace(key_bits=8, value_bits=4, group_size=64), p),
    "selective_residual": lambda p: SelectiveMethod(
        SimpleNamespace(selector="quantization_residual", high_bits=8, low_bits=2, high_fraction=0.2, group_size=64, seed=None, sink_tokens=None), p
    ),
    "selective_random": lambda p: SelectiveMethod(
        SimpleNamespace(selector="random", high_bits=6, low_bits=3, high_fraction=0.3, group_size=64, seed=7, sink_tokens=None), p
    ),
}


def _logits_after(model, cache, question):
    return model(mx.array(question)[None], cache=cache)[0].astype(mx.float32)


@pytest.mark.parametrize("name", list(METHODS))
@pytest.mark.parametrize("window_tokens", [512, 6])  # 6: the question and decode overflow the window
def test_real_storage_matches_simulation(model, name, window_tokens):
    protocol = _protocol(window_tokens=window_tokens)
    reference = backend.prefill(model, _ids(150))
    sim = backend.branch_cache(reference)
    method = METHODS[name](protocol)
    report = method.apply(sim, ARCH)
    real = METHODS[name](protocol).convert_real(backend.branch_cache(reference), ARCH)
    assert all(isinstance(c, PackedCache) for c in real)

    # measured packed bytes at the conversion point equal the analytic estimate
    assert sum(c.nbytes() for c in real) == report.cache_bytes_estimated

    question = _ids(9, seed=4)
    a, b = _logits_after(model, sim, question), _logits_after(model, real, question)
    diff = float(mx.abs(a - b).max())
    assert diff < 0.05 * float(mx.abs(a).max())
    # same prediction wherever the top-1 margin exceeds the numerical difference (bf16 near-ties excepted)
    top2 = np.sort(np.array(a), axis=-1)[:, -2:]
    decisive = (top2[:, 1] - top2[:, 0]) > 2 * diff
    assert decisive.any()
    assert np.array_equal(np.array(a.argmax(-1))[decisive], np.array(b.argmax(-1))[decisive])
    out_sim = backend.generate(model, STUB_TOKENIZER, [3], sim, max_new_tokens=12)
    out_real = backend.generate(model, STUB_TOKENIZER, [3], real, max_new_tokens=12)
    assert out_sim == out_real
    # after decode (window overflowed when window_tokens=6) bytes still match the estimate
    assert sum(c.nbytes() for c in real) == method.estimate_bytes(sim, ARCH)
    assert real[0].n_window == sim[0].n_window


def test_selective_real_stores_uint16_indices(model):
    real = METHODS["selective_residual"](_protocol()).convert_real(backend.branch_cache(backend.prefill(model, _ids(150))), ARCH)
    assert real[0].index.dtype == mx.uint16 and real[0].index.shape == (1, 2, 30)
    assert [s.key_bits for s in real[0].segments] == [8, 2, 8]


def test_ordinary_quantized_cache_still_uses_mlx_attention(model):
    from mlx_lm.models.cache import make_prompt_cache

    from kvstudy.methods import packed

    packed.install()
    ids = mx.array(_ids(40))[None]
    plain = make_prompt_cache(model)
    model(ids, cache=plain)
    q = [c.to_quantized(group_size=64, bits=8) for c in plain]
    out = model(mx.array([[5]]), cache=q)  # goes through the delegated original function
    assert out.shape == (1, 1, 97)
