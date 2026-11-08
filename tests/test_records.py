import pytest

from kvstudy.records import ResultRecord, append_jsonl, read_jsonl


def _record(**kw):
    base = dict(
        run_id="r", example_id="e", source_id="s", method_id="m", model_id="x", model_revision="v",
        protocol_id="p", benchmark="ruler", task="t", split="dev", context_length=4096,
    )  # fmt: skip
    return ResultRecord(**{**base, **kw})


def test_jsonl_roundtrip(tmp_path):
    path = tmp_path / "a" / "results.jsonl"
    recs = [_record(status="ok", score=50.0, prediction="x"), _record(status="failed", error="boom")]
    for r in recs:
        append_jsonl(path, r)
    rows = read_jsonl(path)
    assert [ResultRecord.from_dict(r) for r in rows] == recs


def test_failure_records_keep_score_none(tmp_path):
    path = tmp_path / "results.jsonl"
    append_jsonl(path, _record(status="oom", error="[metal::malloc]"))
    assert read_jsonl(path)[0]["score"] is None


def test_failure_record_cannot_carry_score():
    with pytest.raises(ValueError):
        _record(status="failed", score=0.0)


def test_malformed_line_raises(tmp_path):
    path = tmp_path / "results.jsonl"
    append_jsonl(path, {"a": 1})
    with open(path, "a") as f:
        f.write('{"a": ')
    with pytest.raises(ValueError, match="malformed"):
        read_jsonl(path)


def test_repair_interrupted_final_write(tmp_path):
    from kvstudy.records import repair_interrupted_jsonl

    path = tmp_path / "results.jsonl"
    append_jsonl(path, _record(status="ok", score=1.0))
    with open(path, "a") as f:
        f.write('{"run_id": "r", "exam')
    assert repair_interrupted_jsonl(path) == '{"run_id": "r", "exam'
    assert len(read_jsonl(path)) == 1
    assert repair_interrupted_jsonl(path) is None


def test_repair_leaves_intact_file_alone(tmp_path):
    from kvstudy.records import repair_interrupted_jsonl

    path = tmp_path / "results.jsonl"
    append_jsonl(path, _record(status="ok", score=1.0))
    before = path.read_bytes()
    assert repair_interrupted_jsonl(path) is None
    assert path.read_bytes() == before
