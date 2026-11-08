"""Checks against the downloaded Qwen tokenizer and the pinned RULER clone."""

import ast
import importlib.util
import inspect
import random
import textwrap

import pytest

from conftest import QWEN_ID, QWEN_REVISION, REPO
from kvstudy.benchmarks._vendor import ruler_eval

pytestmark = pytest.mark.integration

RULER_DIR = REPO / "data" / "upstream" / "RULER"


@pytest.fixture(scope="module")
def qwen_tokenizer():
    from types import SimpleNamespace

    from kvstudy.backends.mlx import local_snapshot

    path = local_snapshot(SimpleNamespace(id=QWEN_ID, revision=QWEN_REVISION))
    if path is None:
        pytest.skip("Qwen3-4B snapshot not in the local HF cache")
    from mlx_lm.tokenizer_utils import load

    return load(path)


@pytest.mark.parametrize(
    "context,question",
    [
        ("Intro.\nThe grass is green. There and back again.", "\nWhat is the special magic number for foo?"),
        ("Memorize.\n\nVAR AB = 12\nHere we go.", "\nQuestion: Find all variables that are assigned the value 12."),
        ("No trailing punctuation here", " and then a question?"),
    ],
)
def test_boundary_split_reproduces_full_tokenization(qwen_tokenizer, context, question):
    from kvstudy.backends.mlx import SYSTEM_PROMPT, render_prompt

    prompt = render_prompt(qwen_tokenizer, SYSTEM_PROMPT, context + question, len(context), " Answer:")
    hf = qwen_tokenizer._tokenizer
    assert prompt.token_ids == hf(prompt.text, add_special_tokens=False)["input_ids"]
    head = hf.decode(prompt.token_ids[: prompt.boundary])
    tail = hf.decode(prompt.token_ids[prompt.boundary :])
    assert head + tail == prompt.text
    assert tail.endswith(" Answer:")
    # The question side starts at the question, or one straddling token before it.
    assert context in head + tail[: tail.index(question)]
    assert tail.index(question) <= 16
    assert "<think>\n\n</think>" in prompt.text  # thinking disabled


def _upstream_module(relpath, name):
    path = RULER_DIR / relpath
    if not path.exists():
        pytest.skip(f"{path} not present")
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_vendored_scorer_matches_upstream():
    upstream = _upstream_module("scripts/eval/synthetic/constants.py", "_ruler_eval_constants")
    rng = random.Random(0)
    words = ["alpha", "Beta", "GAMMA", "delta", "42", "7"]
    for _ in range(500):
        refs = [[rng.choice(words) for _ in range(rng.randint(1, 4))] for _ in range(rng.randint(1, 5))]
        preds = [" ".join(rng.choice(words) for _ in range(rng.randint(0, 5))) for _ in refs]
        assert ruler_eval.string_match_all(preds, refs) == upstream.string_match_all(preds, refs)
        assert ruler_eval.string_match_part(preds, refs) == upstream.string_match_part(preds, refs)
    assert {k: v["metric_fn"].__name__ for k, v in ruler_eval.TASKS.items()} == {
        k: v["metric_fn"].__name__ for k, v in upstream.TASKS.items()
    }


def test_vendored_postprocess_matches_upstream_source():
    path = RULER_DIR / "scripts" / "eval" / "evaluate.py"
    if not path.exists():
        pytest.skip(f"{path} not present")
    source = path.read_text()
    node = next(n for n in ast.parse(source).body if isinstance(n, ast.FunctionDef) and n.name == "postprocess_pred")
    upstream_src = ast.get_source_segment(source, node)
    assert textwrap.dedent(inspect.getsource(ruler_eval.postprocess_pred)).strip() == upstream_src.strip()
