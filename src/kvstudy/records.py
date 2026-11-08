"""Common example format, per-example result records, and JSONL io."""

from __future__ import annotations

import json
import os
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Literal

# Post-prefill cache conversion: context prefilled in BF16, cache converted once,
# then question and answer processed through the converted cache. Quantizing
# during prefill is a different protocol and must get a different identifier.
PROTOCOL_ID = "post_prefill_conversion/v1"

Status = Literal["ok", "failed", "oom"]


@dataclass
class Example:
    example_id: str
    source_id: str
    benchmark: str
    task: str
    split: str
    context: str
    question: str | None
    reference_continuation: str | None
    references: list[str]
    metadata: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> Example:
        return cls(**d)


@dataclass
class ResultRecord:
    run_id: str
    example_id: str
    source_id: str
    method_id: str
    model_id: str
    model_revision: str
    protocol_id: str
    benchmark: str
    task: str
    split: str
    context_length: int
    status: Status
    attempt: int = 1  # explicit retries append a new row with attempt + 1
    prediction: str | None = None
    references: list[str] | None = None
    score: float | None = None  # continuation benchmarks: mean NLL in nats per scored token (lower is better)
    nll_sum: float | None = None  # continuation benchmarks: summed NLL over scored tokens
    n_scored_tokens: int | None = None
    error: str | None = None
    n_input_tokens: int | None = None
    n_context_tokens: int | None = None
    n_question_tokens: int | None = None
    n_generated_tokens: int | None = None
    cache_bytes_estimated: int | None = None  # at the conversion point (context only)
    cache_bytes_estimated_final: int | None = None  # after decode, protected window included
    cache_bytes_label: str | None = None
    memory_bytes: dict[str, int] = field(default_factory=dict)  # systems_real: measured buffers and MLX memory
    timings_s: dict[str, float] = field(default_factory=dict)
    timings_label: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.status != "ok" and self.score is not None:
            raise ValueError("non-ok records must not carry a score")

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> ResultRecord:
        return cls(**d)


def append_jsonl(path: Path, record: Any) -> None:
    """Append one record as a JSON line and force it to disk."""
    obj = asdict(record) if hasattr(record, "__dataclass_fields__") else record
    line = json.dumps(obj, ensure_ascii=False, sort_keys=True)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "a", encoding="utf-8") as f:
        f.write(line + "\n")
        f.flush()
        os.fsync(f.fileno())


def repair_interrupted_jsonl(path: Path) -> str | None:
    """Cut an interrupted final write (a last line without newline) from a JSONL file.

    Returns the discarded partial text, or None if the file was intact.
    """
    data = path.read_bytes()
    if not data or data.endswith(b"\n"):
        return None
    cut = data.rfind(b"\n") + 1
    with open(path, "r+b") as f:
        f.truncate(cut)
        f.flush()
        os.fsync(f.fileno())
    return data[cut:].decode("utf-8", errors="replace")


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    """Read a JSONL file, raising on any malformed line."""
    rows = []
    with open(path, encoding="utf-8") as f:
        for lineno, line in enumerate(f, 1):
            if not line.strip():
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError as e:
                raise ValueError(f"{path}:{lineno}: malformed JSONL line: {e}") from e
    return rows
