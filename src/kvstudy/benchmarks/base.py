"""Benchmark adapter interface and registry.

Adapters load examples, describe how each example's input is built, and
invoke official scoring. They never touch the KV cache.
"""

from __future__ import annotations

from typing import Any, Protocol

from kvstudy.records import Example


class BenchmarkAdapter(Protocol):
    """`kind` selects the execution path: "generate" adapters supply a prompt and a
    scorer; "continuation" adapters supply context and reference token ids, which the
    runner teacher-forces and scores by NLL.
    """

    name: str
    kind: str  # "generate" | "continuation"

    def load_examples(self, bench_cfg: Any) -> list[Example]:
        """Examples for the configured tasks and lengths, in manifest order."""

    def context_length(self, example: Example) -> int:
        """Nominal context-length bucket the example belongs to."""

    def evaluator_fingerprint(self) -> dict[str, str]:
        """Evaluator revision and file hashes for the run manifest."""

    # generate adapters

    def user_text(self, example: Example) -> str:
        """The full user-turn text (context followed by question)."""

    def question_char_offset(self, example: Example) -> int:
        """Character offset in user_text() where the question begins."""

    def answer_prefix(self, example: Example) -> str:
        """Text appended after the assistant generation prompt."""

    def max_new_tokens(self, example: Example, bench_cfg: Any) -> int:
        """Generation budget for the example."""

    def score(self, example: Example, prediction: str) -> float:
        """Official per-example score."""

    # continuation adapters

    def token_ids(self, example: Example) -> tuple[list[int], list[int]]:
        """(context ids, reference continuation ids)."""


from kvstudy.benchmarks.longbench import LongBenchAdapter  # noqa: E402
from kvstudy.benchmarks.pg19 import PG19Adapter  # noqa: E402
from kvstudy.benchmarks.ruler import RulerAdapter  # noqa: E402

BENCHMARKS: dict[str, type] = {"ruler": RulerAdapter, "longbench": LongBenchAdapter, "pg19": PG19Adapter}


def get_adapter(name: str) -> BenchmarkAdapter:
    if name not in BENCHMARKS:
        raise ValueError(f"unknown benchmark {name!r}")
    return BENCHMARKS[name]()
