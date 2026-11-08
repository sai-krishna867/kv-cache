"""Experiment configuration schema and validation.

Unknown fields and unsupported values are rejected. Nothing is silently
substituted: a value the harness cannot honour is a validation error.
"""

from __future__ import annotations

import hashlib
import json
from collections import Counter
from pathlib import Path
from typing import Annotated, Literal

import yaml
from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

SAFE_ID = r"^[A-Za-z0-9][A-Za-z0-9_.-]*$"

# Bit widths and group sizes supported by mx.quantize (affine mode).
QuantBits = Literal[2, 3, 4, 5, 6, 8]
GroupSize = Literal[32, 64, 128]


class Strict(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)


class ModelConfig(Strict):
    backend: Literal["mlx"]
    id: str
    revision: str = Field(pattern=r"^[0-9a-f]{40}$")
    weight_dtype: Literal["bfloat16"]


class SystemsConfig(Strict):
    """systems_real measurement settings: every (example, method) is run warmup_runs
    + repeats times from a fresh prefill, decoding exactly decode_tokens tokens."""

    warmup_runs: int = Field(ge=0)
    repeats: int = Field(ge=1)
    decode_tokens: int = Field(gt=0)


class ExecutionConfig(Strict):
    mode: str
    seed: int
    systems: SystemsConfig | None = None

    @field_validator("mode")
    @classmethod
    def _mode(cls, v: str) -> str:
        if v not in ("quality_simulated", "systems_real"):
            raise ValueError(f"unknown execution.mode {v!r}; supported: 'quality_simulated', 'systems_real'")
        return v

    @model_validator(mode="after")
    def _systems(self) -> ExecutionConfig:
        if self.mode == "systems_real" and self.systems is None:
            raise ValueError("execution.mode 'systems_real' requires execution.systems (warmup_runs, repeats, decode_tokens)")
        if self.mode != "systems_real" and self.systems is not None:
            raise ValueError("execution.systems applies only to execution.mode 'systems_real'")
        return self


class ProtectedWindowConfig(Strict):
    """Tokens appended after conversion (question, generated) are held at precision_bits.

    When more than `tokens` are held, the oldest are demoted to the method's own
    precision. Applies to every compressed method; the identity baseline stays BF16.
    """

    tokens: int = Field(gt=0)
    precision_bits: QuantBits
    overflow: Literal["demote_oldest"]


class ProtocolConfig(Strict):
    prefill_dtype: Literal["bfloat16"]
    conversion_point: Literal["after_context_before_question"]
    key_position: Literal["post_rope"]
    protected_window: ProtectedWindowConfig | None = None


class RulerBenchmarkConfig(Strict):
    name: Literal["ruler"]
    split_manifest: Path
    tasks: list[str] = Field(min_length=1)
    context_lengths: list[int] = Field(min_length=1)
    max_new_tokens: int = Field(gt=0)
    max_examples_per_cell: int | None = Field(default=None, gt=0)


LongBenchTask = Literal["narrativeqa", "qasper", "multifieldqa_en", "hotpotqa", "2wikimqa", "musique"]


class LongBenchBenchmarkConfig(Strict):
    """English QA datasets (official qa_f1_score). context_lengths are natural-length
    buckets fixed at preparation; generation budgets are the official dataset2maxlen."""

    name: Literal["longbench"]
    split_manifest: Path
    tasks: list[LongBenchTask] = Field(min_length=1)
    context_lengths: list[int] = Field(min_length=1)
    max_examples_per_cell: int | None = Field(default=None, gt=0)


class PG19BenchmarkConfig(Strict):
    name: Literal["pg19"]
    split_manifest: Path
    context_lengths: list[int] = Field(min_length=1)
    continuation_tokens: int = Field(gt=1)
    max_examples_per_cell: int | None = Field(default=None, gt=0)


BenchmarkConfig = Annotated[
    RulerBenchmarkConfig | LongBenchBenchmarkConfig | PG19BenchmarkConfig, Field(discriminator="name")
]


class IdentityMethodConfig(Strict):
    id: str = Field(pattern=SAFE_ID)
    kind: Literal["identity"]


class UniformMethodConfig(Strict):
    id: str = Field(pattern=SAFE_ID)
    kind: Literal["uniform"]
    key_bits: QuantBits
    value_bits: QuantBits
    group_size: GroupSize
    axis: str

    @field_validator("axis")
    @classmethod
    def _axis(cls, v: str) -> str:
        if v == "per_channel":
            raise ValueError("axis 'per_channel' (KIVI-style key groups) is not implemented yet")
        if v != "per_token":
            raise ValueError(f"unknown axis {v!r}; supported: 'per_token'")
        return v


SELECTORS = ("random", "sinks_recency", "quantization_residual")


class SelectiveMethodConfig(Strict):
    """Per layer and KV head, a fraction of context tokens is kept at high_bits, the rest at low_bits.

    K and V of a token share one selection. Selector-specific fields: `seed`
    (random) and `sink_tokens` (sinks_recency); other selectors reject them.
    """

    id: str = Field(pattern=SAFE_ID)
    kind: Literal["selective"]
    selector: Literal[SELECTORS]
    high_bits: QuantBits
    low_bits: QuantBits
    high_fraction: float = Field(gt=0, lt=1)
    group_size: GroupSize
    axis: str
    seed: int | None = None
    sink_tokens: int | None = Field(default=None, ge=0)

    _axis = field_validator("axis")(UniformMethodConfig._axis.__func__)

    @model_validator(mode="after")
    def _selector_fields(self) -> SelectiveMethodConfig:
        if self.high_bits <= self.low_bits:
            raise ValueError(f"method {self.id}: high_bits must exceed low_bits")
        needs = {"random": "seed", "sinks_recency": "sink_tokens"}.get(self.selector)
        for name in ("seed", "sink_tokens"):
            given = getattr(self, name) is not None
            if name == needs and not given:
                raise ValueError(f"method {self.id}: selector {self.selector!r} requires {name}")
            if name != needs and given:
                raise ValueError(f"method {self.id}: {name} does not apply to selector {self.selector!r}")
        return self


def max_bits(method: IdentityMethodConfig | UniformMethodConfig | SelectiveMethodConfig) -> int:
    """Highest precision a compressed method stores context tokens at."""
    if method.kind == "selective":
        return method.high_bits
    return max(method.key_bits, method.value_bits)


MethodConfig = Annotated[
    IdentityMethodConfig | UniformMethodConfig | SelectiveMethodConfig, Field(discriminator="kind")
]


class OutputConfig(Strict):
    root: Path


class ExperimentConfig(Strict):
    schema_version: Literal[1]
    name: str = Field(pattern=SAFE_ID)
    model: ModelConfig
    execution: ExecutionConfig
    protocol: ProtocolConfig
    benchmarks: list[BenchmarkConfig] = Field(min_length=1)
    methods: list[MethodConfig] = Field(min_length=1)
    output: OutputConfig

    @model_validator(mode="after")
    def _unique_ids(self) -> ExperimentConfig:
        dup = [k for k, n in Counter(m.id for m in self.methods).items() if n > 1]
        if dup:
            raise ValueError(f"duplicate method ids: {dup}")
        dup = [k for k, n in Counter(b.name for b in self.benchmarks).items() if n > 1]
        if dup:
            raise ValueError(f"duplicate benchmarks: {dup}")
        return self

    @model_validator(mode="after")
    def _systems_benchmarks(self) -> ExperimentConfig:
        if self.execution.mode == "systems_real":
            continuation = [b.name for b in self.benchmarks if b.name == "pg19"]
            if continuation:
                raise ValueError("systems_real times generation; continuation benchmarks (pg19) are not supported")
        return self

    @model_validator(mode="after")
    def _protected_window(self) -> ExperimentConfig:
        compressed = [m for m in self.methods if m.kind != "identity"]
        window = self.protocol.protected_window
        if compressed and window is None:
            raise ValueError("compressed methods require protocol.protected_window")
        for m in compressed:
            # Tokens are only ever demoted, never promoted.
            if window.precision_bits < max_bits(m):
                raise ValueError(
                    f"method {m.id}: protected_window.precision_bits ({window.precision_bits}) "
                    f"is below the method's own precision"
                )
        return self


class ComparisonPair(Strict):
    method: str = Field(pattern=SAFE_ID)
    against: str = Field(pattern=SAFE_ID)


class ComparisonConfig(Strict):
    """Declared comparisons on one saved run (`kvstudy compare`).

    Methods are named explicitly; nothing is chosen from results. With `holm`,
    the listed comparisons form one family, corrected per benchmark at pooled scope.
    """

    schema_version: Literal[1]
    name: str = Field(pattern=SAFE_ID)
    run_dir: Path
    comparisons: list[ComparisonPair] = Field(min_length=1)
    holm: bool


def load_comparison_config(path: Path) -> ComparisonConfig:
    with open(path, encoding="utf-8") as f:
        return ComparisonConfig.model_validate(yaml.safe_load(f))


def load_config(path: Path) -> ExperimentConfig:
    with open(path, encoding="utf-8") as f:
        raw = yaml.safe_load(f)
    return ExperimentConfig.model_validate(raw)


def config_to_dict(cfg: ExperimentConfig) -> dict:
    return cfg.model_dump(mode="json")


def config_hash(cfg: ExperimentConfig) -> str:
    canonical = json.dumps(config_to_dict(cfg), sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(canonical.encode()).hexdigest()


def validate_config(cfg: ExperimentConfig) -> list[str]:
    """Check local prerequisites without downloading. Returns problems found.

    Prints the expanded run count per benchmark.
    """
    from kvstudy.backends import mlx as backend
    from kvstudy.benchmarks.base import get_adapter

    problems: list[str] = []
    snapshot = backend.local_snapshot(cfg.model)
    if snapshot is None:
        problems.append(
            f"model {cfg.model.id}@{cfg.model.revision} is not in the local HF cache; "
            f"download it with: hf download {cfg.model.id} --revision {cfg.model.revision}"
        )
    else:
        with open(snapshot / "config.json", encoding="utf-8") as f:
            arch = backend.read_arch(json.load(f))
        for m in cfg.methods:
            if m.kind != "identity" and arch.head_dim % m.group_size != 0:
                problems.append(f"method {m.id}: group_size {m.group_size} does not divide head_dim {arch.head_dim}")

    total = 0
    for bench in cfg.benchmarks:
        adapter = get_adapter(bench.name)
        if not bench.split_manifest.exists():
            problems.append(f"{bench.name}: manifest {bench.split_manifest} does not exist")
            continue
        try:
            examples = adapter.load_examples(bench)
        except ValueError as e:
            problems.append(f"{bench.name}: {e}")
            continue
        runs = len(examples) * len(cfg.methods)
        total += runs
        print(f"{bench.name}: {len(examples)} examples x {len(cfg.methods)} methods = {runs} runs")
    print(f"total: {total} runs")
    return problems
