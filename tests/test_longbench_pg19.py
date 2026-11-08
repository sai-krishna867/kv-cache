"""LongBench and PG-19 adapters: prompt split, length buckets, scoring, windows."""

import json
from types import SimpleNamespace

import pytest

from kvstudy.benchmarks._vendor import longbench_eval
from kvstudy.benchmarks.longbench import LongBenchAdapter, bucket_for, split_prompt
from kvstudy.benchmarks.pg19 import PG19Adapter, window_start
from kvstudy.records import Example, append_jsonl

HOTPOT = (
    "Answer the question based on the given passages. Only give me the answer and do not output any other words."
    "\n\nThe following are given passages.\n{context}\n\nAnswer the question based on the given passages. "
    "Only give me the answer and do not output any other words.\n\nQuestion: {input}\nAnswer:"
)


def test_split_prompt_reproduces_official_format():
    context, question = split_prompt(HOTPOT, "Passage 1: {braces} kept", "Who?")
    assert context + question == HOTPOT.replace("{context}", "Passage 1: {braces} kept").replace("{input}", "Who?")
    assert question.startswith("\n\nAnswer the question") and question.endswith("Question: Who?\nAnswer:")


def test_split_prompt_rejects_question_first():
    with pytest.raises(ValueError):
        split_prompt("Q: {input}\n{context}", "c", "q")


def test_bucket_uses_natural_length_and_generation_budget():
    assert bucket_for(4000, 32, [4096, 8192, 16384]) == 4096
    assert bucket_for(4070, 32, [4096, 8192, 16384]) == 8192  # input + budget must fit
    assert bucket_for(16353, 32, [4096, 8192, 16384]) is None  # excluded, never truncated


def _lb_example(task="hotpotqa", refs=("Paris",)):
    return Example(
        example_id=f"longbench/{task}/x", source_id="longbench/doc", benchmark="longbench", task=task, split="dev",
        context="ctx", question="q", reference_continuation=None, references=list(refs),
        metadata={"context_length": 4096, "max_new_tokens": 32, "all_classes": None},
    )  # fmt: skip


@pytest.mark.parametrize(
    "prediction,refs",
    [("Paris", ["paris"]), ("The city of Paris.", ["Paris", "Lyon"]), ("no idea", ["Paris"]), ("Eiffel Tower in Paris", ["the Eiffel Tower"])],
)
def test_longbench_score_matches_official_scorer(prediction, refs):
    ours = LongBenchAdapter().score(_lb_example(refs=refs), prediction)
    assert round(ours, 2) == longbench_eval.scorer("hotpotqa", [prediction], [refs], None)


def test_longbench_budget_is_official_per_dataset():
    assert LongBenchAdapter().max_new_tokens(_lb_example(), SimpleNamespace(max_new_tokens=999)) == 32


def test_pg19_window_is_deterministic_and_fits():
    a = window_start(100_000, 8192, 512, 42, "10146")
    assert a == window_start(100_000, 8192, 512, 42, "10146")
    assert 0 <= a <= 100_000 - 8192 - 512
    assert a != window_start(100_000, 8192, 512, 42, "10321")
    assert window_start(8000, 8192, 512, 42, "1") is None


def _pg_example(n_cont):
    return Example(
        example_id="pg19/1/4096", source_id="pg19/1", benchmark="pg19", task="continuation", split="dev",
        context="c", question=None, reference_continuation="r", references=[],
        metadata={"context_length": 4096, "context_token_ids": [1, 2, 3], "continuation_token_ids": list(range(n_cont))},
    )  # fmt: skip


def test_pg19_load_checks_continuation_length(tmp_path):
    path = tmp_path / "pg19.jsonl"
    append_jsonl(path, _pg_example(512))
    cfg = SimpleNamespace(split_manifest=path, context_lengths=[4096], continuation_tokens=512, max_examples_per_cell=None)
    [ex] = PG19Adapter().load_examples(cfg)
    assert PG19Adapter().token_ids(ex) == ([1, 2, 3], list(range(512)))
    with pytest.raises(ValueError, match="continuation tokens"):
        PG19Adapter().load_examples(SimpleNamespace(**{**vars(cfg), "continuation_tokens": 256}))


def test_benchmark_configs_validate(config_dict):
    from kvstudy.config import ExperimentConfig

    config_dict["benchmarks"] += [
        {"name": "longbench", "split_manifest": "lb.jsonl", "tasks": ["hotpotqa", "qasper"], "context_lengths": [8192]},
        {"name": "pg19", "split_manifest": "pg.jsonl", "context_lengths": [4096], "continuation_tokens": 512},
    ]
    cfg = ExperimentConfig.model_validate(config_dict)
    assert [b.name for b in cfg.benchmarks] == ["ruler", "longbench", "pg19"]
    config_dict["benchmarks"][1]["tasks"] = ["lcc"]  # needs metrics this harness does not vendor
    with pytest.raises(Exception):
        ExperimentConfig.model_validate(config_dict)
