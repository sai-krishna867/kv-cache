import ast
from pathlib import Path

import pytest
from _pytest.mark.expression import Expression

REPO = Path(__file__).resolve().parents[1]
QWEN_ID = "Qwen/Qwen3-4B"
QWEN_REVISION = "1cfa9a7208912126459214e8b04321603b3df60c"


def base_config() -> dict:
    return {
        "schema_version": 1,
        "name": "test",
        "model": {"backend": "mlx", "id": QWEN_ID, "revision": QWEN_REVISION, "weight_dtype": "bfloat16"},
        "execution": {"mode": "quality_simulated", "seed": 42},
        "protocol": {
            "prefill_dtype": "bfloat16",
            "conversion_point": "after_context_before_question",
            "key_position": "post_rope",
        },
        "benchmarks": [
            {
                "name": "ruler",
                "split_manifest": "manifest.jsonl",
                "tasks": ["niah_single_1"],
                "context_lengths": [4096],
                "max_new_tokens": 16,
            }
        ],
        "methods": [{"id": "bf16", "kind": "identity"}],
        "output": {"root": "results"},
    }


@pytest.fixture
def config_dict() -> dict:
    return base_config()


def _module_marks(path: Path) -> set[str]:
    """Marker names assigned to `pytestmark` at module level, read without importing the module."""
    marks = set()
    for node in ast.parse(path.read_text(encoding="utf-8")).body:
        if isinstance(node, ast.Assign) and any(isinstance(t, ast.Name) and t.id == "pytestmark" for t in node.targets):
            for n in ast.walk(node.value):
                if isinstance(n, ast.Attribute) and isinstance(n.value, ast.Attribute) and n.value.attr == "mark":
                    marks.add(n.attr)
    return marks


def pytest_ignore_collect(collection_path, config):
    # `-m` deselects only after a module is imported, and importing the MLX test
    # modules loads MLX and initialises Metal. Skip importing a module whose
    # module-level marks already exclude it, e.g. under -m "not mlx and not integration".
    markexpr = config.getoption("markexpr")
    if not markexpr or collection_path.suffix != ".py" or not collection_path.name.startswith("test_"):
        return None
    marks = _module_marks(collection_path)
    if marks and not Expression.compile(markexpr).evaluate(lambda name, **kwargs: name in marks):
        return True
    return None
