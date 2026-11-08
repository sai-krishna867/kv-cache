import json
import re
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

from kvstudy.analysis.compare import summarize_run
from kvstudy.config import ExperimentConfig
from kvstudy.records import ResultRecord, append_jsonl
from kvstudy.runner import classify_failure, create_run_dir, method_results_path, new_run_id


def test_run_id_format():
    now = datetime(2026, 11, 7, 18, 0, 0, tzinfo=timezone.utc)
    assert re.fullmatch(r"2026-11-07T180000Z_[0-9a-f]{8}", new_run_id(now))


def test_same_config_gets_new_run_dir(tmp_path, config_dict):
    config_dict["output"]["root"] = str(tmp_path)
    cfg = ExperimentConfig.model_validate(config_dict)
    _, a, _ = create_run_dir(cfg)
    _, b, _ = create_run_dir(cfg)
    assert a != b and a.parent == b.parent == tmp_path / "test"


def test_classify_failure():
    assert classify_failure(RuntimeError("[metal::malloc] Attempting to allocate 1 bytes")) == "oom"
    assert classify_failure(ValueError("bad")) == "failed"


def _rec(method, example, status, score=None, length=4096):
    return ResultRecord(
        run_id="r1", example_id=example, source_id=example, method_id=method, model_id="m",
        model_revision="v", protocol_id="p", benchmark="ruler", task="niah_single_1", split="dev",
        context_length=length, status=status, score=score, n_input_tokens=10, n_context_tokens=8,
        n_generated_tokens=2, cache_bytes_estimated=100, cache_bytes_label="analytic_bf16",
    )  # fmt: skip


def test_summarize_run_uses_manifest_and_excludes_failures(tmp_path):
    manifest = {"run_id": "r1", "config_hash": "h", "methods": {"bf16": method_results_path("bf16")}}
    (tmp_path / "manifest.json").write_text(json.dumps(manifest))
    (tmp_path / "resolved_config.yaml").write_text("methods: [{id: bf16, kind: identity}]\n")
    path = tmp_path / method_results_path("bf16")
    append_jsonl(path, _rec("bf16", "e1", "ok", 100.0))
    append_jsonl(path, _rec("bf16", "e2", "ok", 50.0))
    append_jsonl(path, _rec("bf16", "e3", "failed"))
    append_jsonl(path, _rec("bf16", "e4", "oom"))

    summarize_run(tmp_path)

    summary = json.loads((tmp_path / "methods/bf16/summary.json").read_text())
    [cell] = summary["cells"]
    assert cell["mean_score"] == 75.0
    assert (cell["n"], cell["n_ok"], cell["n_failed"], cell["n_oom"]) == (4, 2, 1, 1)
    comparison = json.loads((tmp_path / "comparisons/summary.json").read_text())
    assert list(comparison["methods"]) == ["bf16"]


def _write_run(tmp_path, methods):
    import yaml

    manifest = {"run_id": "r1", "config_hash": "h", "methods": {m: method_results_path(m) for m in methods}}
    (tmp_path / "manifest.json").write_text(json.dumps(manifest))
    kinds = [{"id": m, "kind": "identity" if m == "bf16" else "uniform"} for m in methods]
    (tmp_path / "resolved_config.yaml").write_text(yaml.safe_dump({"methods": kinds}))


def test_summary_uses_latest_attempt(tmp_path):
    _write_run(tmp_path, ["bf16"])
    path = tmp_path / method_results_path("bf16")
    append_jsonl(path, _rec("bf16", "e1", "oom"))
    retry = _rec("bf16", "e1", "ok", 100.0)
    retry.attempt = 2
    append_jsonl(path, retry)
    summarize_run(tmp_path)
    summary = json.loads((tmp_path / "methods/bf16/summary.json").read_text())
    assert summary["n_superseded_attempts"] == 1
    assert (summary["cells"][0]["n_ok"], summary["cells"][0]["n_oom"]) == (1, 0)


def test_latest_records_picks_highest_attempt(tmp_path):
    from kvstudy.runner import latest_records

    path = tmp_path / method_results_path("u4")
    append_jsonl(path, _rec("u4", "e1", "failed"))
    second = _rec("u4", "e1", "ok", 10.0)
    second.attempt = 2
    append_jsonl(path, second)
    append_jsonl(path, _rec("u4", "e2", "failed"))
    latest = latest_records(tmp_path, ["u4", "missing"])
    assert latest[("u4", "e1")]["status"] == "ok" and latest[("u4", "e1")]["attempt"] == 2
    assert latest[("u4", "e2")]["status"] == "failed"
    assert len(latest) == 2


def _pred(method, example, status, score=None, prediction="x", length=4096):
    r = _rec(method, example, status, score, length)
    r.prediction = prediction if status == "ok" else None
    return r


def test_paired_comparison_pairs_by_example_and_excludes_failures(tmp_path):
    _write_run(tmp_path, ["bf16", "u4"])
    ref, meth = tmp_path / method_results_path("bf16"), tmp_path / method_results_path("u4")
    for r in [
        _pred("bf16", "e1", "ok", 100.0, "a"),
        _pred("bf16", "e2", "ok", 100.0, "b"),
        _pred("bf16", "e3", "ok", 50.0, "c"),
        _pred("bf16", "e4", "failed"),
    ]:
        append_jsonl(ref, r)
    # written in a different order than the reference
    for r in [
        _pred("u4", "e3", "ok", 50.0, "c"),
        _pred("u4", "e2", "ok", 0.0, "wrong"),
        _pred("u4", "e1", "oom"),
        _pred("u4", "e4", "ok", 100.0, "d"),
    ]:
        append_jsonl(meth, r)

    comparison = summarize_run(tmp_path)

    paired = comparison["paired_vs_reference"]
    assert paired["reference"] == "bf16"
    [pooled] = paired["methods"]["u4"]["pooled"]
    assert pooled["n_pairs"] == 2  # e2, e3; e1 (oom) and e4 (reference failed) excluded
    assert pooled["n_excluded"] == 2
    assert pooled["mean_diff"] == -50.0  # (0-100 + 50-50) / 2
    assert pooled["flip_rate"] == 0.5
    [cell] = paired["methods"]["u4"]["cells"]
    assert cell["n_pairs"] == 2 and cell["context_length"] == 4096


def _package_copy(tmp_path):
    import shutil

    import kvstudy

    root = tmp_path / "kvstudy"
    shutil.copytree(Path(kvstudy.__file__).parent, root, ignore=shutil.ignore_patterns("__pycache__"))
    return root


def test_code_fingerprint_is_deterministic_and_tracks_uncommitted_edits(tmp_path):
    from kvstudy.runner import code_fingerprint

    root = _package_copy(tmp_path)
    before = code_fingerprint(root)
    assert before == code_fingerprint(root)
    assert "methods/uniform.py" in before["files"] and "runner.py" in before["files"]

    path = root / "methods" / "uniform.py"
    path.write_text(path.read_text() + "\n# local edit\n")
    edited = code_fingerprint(root)
    assert edited["sha256"] != before["sha256"]
    assert [f for f in edited["files"] if edited["files"][f] != before["files"][f]] == ["methods/uniform.py"]

    (root / "methods" / "untracked.py").write_text("X = 1\n")
    assert code_fingerprint(root)["sha256"] != edited["sha256"]


def test_code_fingerprint_ignores_analysis_and_cli(tmp_path):
    from kvstudy.runner import code_fingerprint

    root = _package_copy(tmp_path)
    before = code_fingerprint(root)
    for rel in ("analysis/compare.py", "cli.py"):
        (root / rel).write_text((root / rel).read_text() + "\n# edit\n")
        assert rel not in before["files"]
    assert code_fingerprint(root) == before


def test_bootstrap_keeps_sources_together():
    from kvstudy.analysis.compare import bootstrap_means

    # source a: two related examples at -100; source b: one example at 0
    means = bootstrap_means([-100.0, -100.0, 0.0], ["a", "a", "b"], n_resamples=2000, seed=0)
    # only aa, bb, and ab/ba (-200 / 3) are possible when whole sources are drawn
    assert set(np.round(means, 6)) == {-100.0, 0.0, round(-200 / 3, 6)}
    assert np.array_equal(means, bootstrap_means([-100.0, -100.0, 0.0], ["a", "a", "b"], 2000, 0))


def test_paired_bootstrap_cells_pooled_and_counts(tmp_path):
    _write_run(tmp_path, ["bf16", "u4"])
    ref, meth = tmp_path / method_results_path("bf16"), tmp_path / method_results_path("u4")
    rows = [("e1", 4096, 100.0, 100.0), ("e2", 4096, 100.0, 0.0), ("e3", 8192, 100.0, 100.0), ("e4", 8192, 50.0, 50.0)]
    for example, length, ref_score, meth_score in rows:
        append_jsonl(ref, _pred("bf16", example, "ok", ref_score, "a", length))
        append_jsonl(meth, _pred("u4", example, "ok", meth_score, "a" if meth_score == ref_score else "b", length))
    append_jsonl(ref, _pred("bf16", "e5", "ok", 100.0, "a", 8192))
    append_jsonl(meth, _pred("u4", "e5", "failed", length=8192))

    paired = summarize_run(tmp_path)["paired_vs_reference"]

    assert paired["bootstrap"]["unit"] == "source_id"
    result = paired["methods"]["u4"]
    [pooled] = result["pooled"]
    assert pooled["scope"] == "pooled"
    assert (pooled["n_pairs"], pooled["n_sources"], pooled["n_excluded"]) == (4, 4, 1)
    assert pooled["mean_diff"] == -25.0
    lo, hi = pooled["mean_diff_ci"]
    assert -100.0 <= lo <= -25.0 <= hi <= 0.0
    cells = {c["context_length"]: c for c in result["cells"]}
    assert all(c["scope"] == "cell" for c in cells.values())
    assert (cells[8192]["n_pairs"], cells[8192]["n_excluded"]) == (2, 1)
    assert cells[8192]["mean_diff_ci"] == [0.0, 0.0]  # no differences in this cell
    assert cells[4096]["flip_rate"] == 0.5


def test_bootstrap_interval_needs_two_sources(tmp_path):
    _write_run(tmp_path, ["bf16", "u4"])
    append_jsonl(tmp_path / method_results_path("bf16"), _pred("bf16", "e1", "ok", 100.0, "a"))
    append_jsonl(tmp_path / method_results_path("u4"), _pred("u4", "e1", "ok", 0.0, "b"))
    [cell] = summarize_run(tmp_path)["paired_vs_reference"]["methods"]["u4"]["cells"]
    assert cell["n_sources"] == 1 and cell["mean_diff_ci"] is None
