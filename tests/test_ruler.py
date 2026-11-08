from types import SimpleNamespace

import pytest

from kvstudy.benchmarks._vendor import ruler_eval
from kvstudy.benchmarks.ruler import RulerAdapter, assign_split, convert_rows, question_marker, split_input
from kvstudy.records import Example, append_jsonl

# Templates as in RULER scripts/data/synthetic/constants.py at the pinned commit.
NIAH_TEMPLATE = (
    "Some special magic {type_needle_v} are hidden within the following text. Make sure to memorize it. "
    "I will quiz you about the {type_needle_v} afterwards.\n{context}\n"
    "What are all the special magic {type_needle_v} for {query} mentioned in the provided text?"
)
VT_TEMPLATE = (
    "Memorize and track the chain(s) of variable assignment hidden in the following text.\n\n{context}\n"
    "Question: Find all variables that are assigned the value {query} in the text above."
)
NOISE = "The grass is green. The sky is blue. The sun is yellow. Here we go. There and back again."


def test_string_match_all_partial_credit():
    assert ruler_eval.string_match_all(["alpha beta"], [["ALPHA", "beta", "gamma"]]) == 66.67


def test_string_match_part_any_reference():
    assert ruler_eval.string_match_part(["the answer is Paris"], [["london", "paris"]]) == 100.0
    assert ruler_eval.string_match_part(["no"], [["london", "paris"]]) == 0.0


def test_postprocess_strips_control_characters():
    assert ruler_eval.postprocess_pred("  12\x0034\t ", {}) == "12\n34"


def test_adapter_score_uses_task_family_metric():
    ex = _example("vt", ["AAA", "BBB"], task_family="variable_tracking")
    assert RulerAdapter().score(ex, " AAA and ccc") == 50.0


def test_niah_single_marker_uses_singular_rewrite():
    args = {"num_needle_k": 1, "num_needle_v": 1, "num_needle_q": 1}
    assert question_marker(NIAH_TEMPLATE, "niah", args) == "\nWhat is the special magic "


def test_niah_multi_marker_keeps_plural():
    args = {"num_needle_k": 1, "num_needle_v": 4, "num_needle_q": 1}
    assert question_marker(NIAH_TEMPLATE, "niah", args) == "\nWhat are all the special magic "


def test_vt_marker():
    assert question_marker(VT_TEMPLATE, "variable_tracking", {}) == "\nQuestion: Find all variables that are assigned the value "


def test_split_input_uses_last_marker():
    marker = "\nQuestion: Find all variables that are assigned the value "
    icl = f"VAR A = 1{marker}1 in the text above. Answer: A\n"
    text = f"Memorize...\n\n{icl}{NOISE}\nVAR B = 2{marker}2 in the text above."
    context, question = split_input(text, marker)
    assert context + question == text
    assert question == f"{marker}2 in the text above."


def test_split_input_missing_marker_fails():
    with pytest.raises(ValueError, match="not found"):
        split_input("no question here", "\nWhat is")


def test_split_input_marker_too_early_fails():
    with pytest.raises(ValueError, match="too early"):
        split_input("\nWhat is " + "x" * 2000, "\nWhat is")


def _upstream_row(index, needle_value="1234567"):
    body = f"{NOISE} One of the special magic numbers for foo-bar is: {needle_value}. {NOISE}"
    text = (
        "A special magic number is hidden within the following text. Make sure to memorize it. "
        f"I will quiz you about the number afterwards.\n{body}\n"
        "What is the special magic number for foo-bar mentioned in the provided text?"
    )
    prefix = " The special magic number for foo-bar mentioned in the provided text is"
    return {"index": index, "input": text, "outputs": [needle_value], "length": 90, "answer_prefix": prefix}


def test_convert_rows():
    marker = "\nWhat is the special magic "
    [ex] = convert_rows([_upstream_row(3)], "niah_single_1", 4096, "niah", marker, {"ruler_commit": "c"})
    assert ex.example_id == "ruler/niah_single_1/4096/3"
    assert ex.source_id == ex.example_id
    assert ex.question.startswith(marker)
    assert ex.context.endswith(NOISE)
    assert ex.references == ["1234567"]
    assert ex.metadata["context_length"] == 4096
    assert ex.metadata["answer_prefix"].startswith(" The special magic")
    assert ex.split in ("dev", "test")


def test_convert_rows_rejects_reference_outside_context():
    row = _upstream_row(0)
    row["outputs"] = ["9999999"]
    with pytest.raises(ValueError, match="not in context"):
        convert_rows([row], "niah_single_1", 4096, "niah", "\nWhat is the special magic ", {})


def test_assign_split_deterministic_and_roughly_20_percent():
    ids = [f"ruler/vt/4096/{i}" for i in range(2000)]
    splits = [assign_split(i) for i in ids]
    assert splits == [assign_split(i) for i in ids]
    assert 0.17 < splits.count("dev") / len(splits) < 0.23


def _example(task, refs, task_family="niah", length=4096, idx=0):
    eid = f"ruler/{task}/{length}/{idx}"
    return Example(
        example_id=eid, source_id=eid, benchmark="ruler", task=task, split="dev",
        context="ctx", question="q", reference_continuation=None, references=refs,
        metadata={"context_length": length, "task_family": task_family, "answer_prefix": " A:"},
    )  # fmt: skip


def _bench(path, **kw):
    return SimpleNamespace(
        split_manifest=path,
        tasks=kw.get("tasks", ["niah_single_1"]),
        context_lengths=kw.get("context_lengths", [4096]),
        max_examples_per_cell=kw.get("limit"),
    )


def test_load_examples_filters_and_limits(tmp_path):
    path = tmp_path / "m.jsonl"
    for i in range(4):
        append_jsonl(path, _example("niah_single_1", ["x"], idx=i))
    append_jsonl(path, _example("niah_single_1", ["x"], length=8192))
    append_jsonl(path, _example("vt", ["x"], task_family="variable_tracking"))
    got = RulerAdapter().load_examples(_bench(path, limit=2))
    assert [e.example_id for e in got] == ["ruler/niah_single_1/4096/0", "ruler/niah_single_1/4096/1"]


def test_load_examples_missing_cell_fails(tmp_path):
    path = tmp_path / "m.jsonl"
    append_jsonl(path, _example("niah_single_1", ["x"]))
    with pytest.raises(ValueError, match="niah_single_1@8192"):
        RulerAdapter().load_examples(_bench(path, context_lengths=[4096, 8192]))


def test_user_text_and_question_offset():
    ex = _example("niah_single_1", ["x"])
    adapter = RulerAdapter()
    text = adapter.user_text(ex)
    assert text[adapter.question_char_offset(ex) :] == ex.question


def _variant(ex, **changes):
    from dataclasses import replace

    return replace(ex, **changes)


def test_check_disjoint_accepts_reproduced_and_new_examples():
    from kvstudy.benchmarks.ruler import check_disjoint

    old = _example("vt", ["A"], idx=0)
    new = _variant(_example("vt", ["B"], idx=1), context="other ctx", split="test")
    check_disjoint([old, new], existing=[old])


def test_check_disjoint_rejects_content_moving_split():
    from kvstudy.benchmarks.ruler import check_disjoint

    old = _variant(_example("vt", ["A"], idx=0), split="test")
    moved = _variant(_example("vt", ["A"], idx=7), split="dev")  # same context and question, new id
    with pytest.raises(ValueError, match="same content already assigned to split 'test'"):
        check_disjoint([moved], existing=[old])
    with pytest.raises(ValueError, match="same content"):
        check_disjoint([old, moved], existing=[])


def test_check_disjoint_rejects_reused_id_with_new_content():
    from kvstudy.benchmarks.ruler import check_disjoint

    old = _example("vt", ["A"], idx=0)
    with pytest.raises(ValueError, match="id already used for different content"):
        check_disjoint([_variant(old, context="regenerated")], existing=[old])


def test_check_disjoint_rejects_source_in_two_splits():
    from kvstudy.benchmarks.ruler import check_disjoint

    a = _variant(_example("vt", ["A"], idx=0), source_id="doc1")
    b = _variant(_example("vt", ["A"], idx=1), source_id="doc1", context="other", split="test")
    with pytest.raises(ValueError, match="source doc1 already in split 'dev'"):
        check_disjoint([a, b], existing=[])


def test_qa_examples_share_a_source_across_lengths():
    marker = "\n\nQuestion: "
    rows = lambda length: [{"index": 3, "input": f"docs{marker}Is it?", "outputs": ["yes"], "answer_prefix": " Answer:", "length": length}]
    a = convert_rows(rows(4096), "qa_2", 4096, "qa", marker, {})
    b = convert_rows(rows(8192), "qa_2", 8192, "qa", marker, {})
    assert a[0].example_id != b[0].example_id
    assert a[0].source_id == b[0].source_id == "ruler/qa_2/q3"
    assert a[0].split == b[0].split == assign_split("ruler/qa_2/q3")
