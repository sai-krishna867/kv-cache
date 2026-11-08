"""Resume correctness end to end with the real model (2 examples x 2 methods)."""

import json
from pathlib import Path

import pytest

from conftest import REPO, base_config
from kvstudy.config import ExperimentConfig
from kvstudy.records import ResultRecord, append_jsonl, read_jsonl
from kvstudy.runner import ResumeError, method_results_path, resume, run

pytestmark = pytest.mark.integration

MANIFEST = REPO / "data" / "manifests" / "ruler_qwen3-4b_s42" / "ruler_dev.jsonl"


@pytest.fixture(scope="module")
def completed_run(tmp_path_factory):
    from types import SimpleNamespace

    from kvstudy.backends.mlx import local_snapshot

    cfg_dict = base_config()
    if not MANIFEST.exists() or local_snapshot(SimpleNamespace(**cfg_dict["model"])) is None:
        pytest.skip("needs the Qwen3-4B snapshot and the prepared RULER manifest")
    cfg_dict["output"]["root"] = str(tmp_path_factory.mktemp("results"))
    cfg_dict["benchmarks"][0].update(split_manifest=str(MANIFEST), max_examples_per_cell=2)
    cfg_dict["protocol"]["protected_window"] = {"tokens": 512, "precision_bits": 8, "overflow": "demote_oldest"}
    cfg_dict["methods"].append(
        {"id": "u2", "kind": "uniform", "key_bits": 2, "value_bits": 2, "group_size": 64, "axis": "per_token"}
    )
    run_dir = run(ExperimentConfig.model_validate(cfg_dict))
    return run_dir, {m: read_jsonl(run_dir / method_results_path(m)) for m in ("bf16", "u2")}


def _events(run_dir):
    return [e["event"] for e in read_jsonl(run_dir / "events.jsonl")]


def _rewrite(path, rows, partial=None):
    path.unlink()
    for r in rows:
        append_jsonl(path, r)
    if partial is not None:
        with open(path, "a") as f:
            f.write(partial)


def test_resume_after_interrupted_write(completed_run):
    run_dir, original = completed_run
    path = run_dir / method_results_path("u2")
    full_line = json.dumps(original["u2"][1], sort_keys=True)
    _rewrite(path, original["u2"][:1], partial=full_line[: len(full_line) // 2])

    resume(run_dir)

    rows = read_jsonl(path)
    assert [r["example_id"] for r in rows] == [r["example_id"] for r in original["u2"]]
    assert rows[1]["prediction"] == original["u2"][1]["prediction"]  # deterministic rerun
    assert rows[1]["run_id"] == original["u2"][1]["run_id"]
    assert read_jsonl(run_dir / method_results_path("bf16")) == original["bf16"]  # completed: skipped
    assert "interrupted_write_removed" in _events(run_dir)


def test_failed_records_retried_only_when_asked(completed_run):
    run_dir, original = completed_run
    path = run_dir / method_results_path("bf16")
    failed = {**original["bf16"][0], "status": "oom", "score": None, "prediction": None, "error": "simulated"}
    _rewrite(path, [failed, original["bf16"][1]])

    resume(run_dir)
    assert len(read_jsonl(path)) == 2  # failure kept, not silently rerun

    resume(run_dir, retry_failed=True)
    rows = read_jsonl(path)
    assert len(rows) == 3  # appended, never overwritten
    assert (rows[2]["example_id"], rows[2]["attempt"], rows[2]["status"]) == (failed["example_id"], 2, "ok")
    assert rows[2]["prediction"] == original["bf16"][0]["prediction"]
    ResultRecord.from_dict(rows[2])

    summary = json.loads((run_dir / "methods/bf16/summary.json").read_text())
    assert summary["n_superseded_attempts"] == 1
    assert sum(c["n_ok"] for c in summary["cells"]) == 2


def test_resume_refuses_changed_identity(completed_run):
    run_dir, _ = completed_run
    config_path = run_dir / "resolved_config.yaml"
    original = config_path.read_text()
    config_path.write_text(original.replace("max_new_tokens: 16", "max_new_tokens: 17"))
    try:
        with pytest.raises(ResumeError, match="config_hash"):
            resume(run_dir)
    finally:
        config_path.write_text(original)


def test_resume_refuses_changed_source(completed_run, tmp_path, monkeypatch):
    import shutil

    import kvstudy
    from kvstudy import runner

    run_dir, _ = completed_run
    root = tmp_path / "kvstudy"
    shutil.copytree(Path(kvstudy.__file__).parent, root, ignore=shutil.ignore_patterns("__pycache__"))
    (root / "methods" / "uniform.py").write_text((root / "methods" / "uniform.py").read_text() + "\n# uncommitted edit\n")
    real = runner.code_fingerprint
    monkeypatch.setattr(runner, "code_fingerprint", lambda: real(root))

    with pytest.raises(ResumeError, match=r"code_fingerprint; changed source files: methods/uniform\.py"):
        resume(run_dir)
