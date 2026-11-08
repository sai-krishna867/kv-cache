"""RULER adapter: dataset preparation via the upstream generator, official scoring."""

from __future__ import annotations

import hashlib
import importlib.util
import os
import shutil
import subprocess
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any

import yaml

from kvstudy.benchmarks._vendor import ruler_eval
from kvstudy.records import Example, append_jsonl, read_jsonl

RULER_COMMIT = ruler_eval.RULER_COMMIT
DEV_FRACTION_PERCENT = 20
_VENDOR_FILE = Path(ruler_eval.__file__)


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


class RulerAdapter:
    name = "ruler"
    kind = "generate"

    def load_examples(self, bench_cfg: Any) -> list[Example]:
        rows = [Example.from_dict(r) for r in read_jsonl(bench_cfg.split_manifest)]
        cells: dict[tuple[str, int], list[Example]] = defaultdict(list)
        for ex in rows:
            if ex.benchmark != self.name:
                raise ValueError(f"{ex.example_id}: benchmark {ex.benchmark!r} in a RULER manifest")
            cells[(ex.task, self.context_length(ex))].append(ex)

        selected: list[Example] = []
        missing = []
        for task in bench_cfg.tasks:
            for length in bench_cfg.context_lengths:
                cell = cells.get((task, length), [])
                if not cell:
                    missing.append(f"{task}@{length}")
                    continue
                if bench_cfg.max_examples_per_cell is not None:
                    cell = cell[: bench_cfg.max_examples_per_cell]
                selected.extend(cell)
        if missing:
            raise ValueError(f"manifest {bench_cfg.split_manifest} has no examples for: {', '.join(missing)}")
        return selected

    def user_text(self, example: Example) -> str:
        return example.context + example.question

    def question_char_offset(self, example: Example) -> int:
        return len(example.context)

    def answer_prefix(self, example: Example) -> str:
        return example.metadata["answer_prefix"]

    def context_length(self, example: Example) -> int:
        return int(example.metadata["context_length"])

    def max_new_tokens(self, example: Example, bench_cfg: Any) -> int:
        return bench_cfg.max_new_tokens

    def score(self, example: Example, prediction: str) -> float:
        metric_fn = ruler_eval.TASKS[example.metadata["task_family"]]["metric_fn"]
        return metric_fn([ruler_eval.postprocess_pred(prediction, {})], [example.references])

    def evaluator_fingerprint(self) -> dict[str, str]:
        return {
            "evaluator": "NVIDIA/RULER scripts/eval (vendored)",
            "revision": RULER_COMMIT,
            "vendored_file_sha256": sha256_file(_VENDOR_FILE),
        }


# --- dataset preparation -------------------------------------------------------


def assign_split(source_id: str) -> str:
    bucket = int(hashlib.sha256(source_id.encode()).hexdigest()[:8], 16) % 100
    return "dev" if bucket < DEV_FRACTION_PERCENT else "test"


def question_marker(task_template: str, task_family: str, task_args: dict[str, Any]) -> str:
    """Literal text that starts the question part of a RULER input.

    It is the template text between ``{context}`` and the next placeholder,
    with the same singular rewrite niah.py applies when one value is asked for.
    """
    marker = task_template.split("{context}", 1)[1].split("{", 1)[0]
    if task_family == "niah" and task_args.get("num_needle_q", 1) * task_args.get("num_needle_v", 1) == 1:
        marker = marker.replace("Some", "A").replace("are all", "is").replace("are", "is").replace("answers", "answer")
    if not marker.strip():
        raise ValueError(f"cannot derive a question marker from template {task_template!r}")
    return marker


def split_input(input_text: str, marker: str, max_question_chars: int = 1000) -> tuple[str, str]:
    """Split an upstream RULER input into (context, question) at the last marker."""
    idx = input_text.rfind(marker)
    if idx < 0:
        raise ValueError(f"question marker {marker!r} not found in RULER input")
    question = input_text[idx:]
    if len(question) > max_question_chars:
        raise ValueError(f"question part is {len(question)} chars; marker {marker!r} matched too early")
    return input_text[:idx], question


def content_key(example: Example) -> str:
    return hashlib.sha256(f"{example.context}\0{example.question}".encode()).hexdigest()


def check_disjoint(examples: list[Example], existing: list[Example]) -> None:
    """Refuse examples that would put one source in two splits.

    Checked within `examples` and against previously written manifests: an id
    must keep its content, and identical content must keep its split.
    """
    by_id = {ex.example_id: ex for ex in existing}
    split_of: dict[str, str] = {content_key(ex): ex.split for ex in existing}
    sources: dict[str, str] = {ex.source_id: ex.split for ex in existing}
    for ex in examples:
        key = content_key(ex)
        prev = by_id.get(ex.example_id)
        if prev is not None and content_key(prev) != key:
            raise ValueError(f"{ex.example_id}: id already used for different content in an existing manifest")
        if split_of.setdefault(key, ex.split) != ex.split:
            raise ValueError(f"{ex.example_id}: same content already assigned to split {split_of[key]!r}")
        if sources.setdefault(ex.source_id, ex.split) != ex.split:
            raise ValueError(f"{ex.example_id}: source {ex.source_id} already in split {sources[ex.source_id]!r}")


def _load_module(path: Path, name: str):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def check_ruler_clone(ruler_dir: Path) -> None:
    head = subprocess.run(
        ["git", "-C", str(ruler_dir), "rev-parse", "HEAD"], capture_output=True, text=True, check=True
    ).stdout.strip()
    if head != RULER_COMMIT:
        raise ValueError(f"{ruler_dir} is at {head}; expected pinned RULER commit {RULER_COMMIT}")


def convert_rows(
    rows: list[dict[str, Any]],
    task: str,
    length: int,
    task_family: str,
    marker: str,
    provenance: dict[str, Any],
) -> list[Example]:
    examples = []
    for row in rows:
        context, question = split_input(row["input"], marker)
        refs = list(row["outputs"])
        # QA answers are not always extractive (e.g. hotpotqa yes/no), so only synthetic tasks are checked.
        for ref in refs if task_family != "qa" else []:
            if ref not in context:
                raise ValueError(f"{task}@{length} index {row['index']}: reference {ref!r} not in context")
        example_id = f"ruler/{task}/{length}/{row['index']}"
        # Synthetic examples are independent. QA index i asks the same question at every
        # length (upstream qa.py), so the question is the source and keeps one split.
        source_id = f"ruler/{task}/q{row['index']}" if task_family == "qa" else example_id
        examples.append(
            Example(
                example_id=example_id,
                source_id=source_id,
                benchmark="ruler",
                task=task,
                split=assign_split(source_id),
                context=context,
                question=question,
                reference_continuation=None,
                references=refs,
                metadata={
                    "context_length": length,
                    "task_family": task_family,
                    "answer_prefix": row["answer_prefix"],
                    "ruler_length": row["length"],
                    "token_position_answer": row.get("token_position_answer"),
                    "upstream_index": row["index"],
                    **provenance,
                },
            )
        )
    return examples


def prepare_ruler(
    ruler_dir: Path,
    tokenizer_path: Path,
    tokenizer_id: str,
    tasks: list[str],
    lengths: list[int],
    num_samples: int,
    seed: int,
    out_dir: Path,
    raw_dir: Path,
    disjoint_from: list[Path] | None = None,
) -> dict[str, Path]:
    """Generate RULER examples with the upstream generator and write split manifests.

    Manifests in `disjoint_from` (earlier versions) are read, never modified, and
    the new examples are checked against them with `check_disjoint`.
    """
    ruler_dir = ruler_dir.resolve()
    check_ruler_clone(ruler_dir)
    scripts = ruler_dir / "scripts"
    constants = _load_module(scripts / "data" / "synthetic" / "constants.py", "_ruler_constants")
    with open(scripts / "synthetic.yaml", encoding="utf-8") as f:
        task_configs = yaml.safe_load(f)

    targets = {split: out_dir / f"ruler_{split}.jsonl" for split in ("dev", "test")}
    for path in targets.values():
        if path.exists():
            raise FileExistsError(f"{path} exists; manifests are versioned, write to a new --out-dir")

    existing = [Example.from_dict(r) for path in disjoint_from or [] for r in read_jsonl(path)]
    essay = scripts / "data" / "synthetic" / "json" / "PaulGrahamEssays.json"
    if any(task_configs[t]["args"].get("type_haystack") == "essay" for t in tasks if t in task_configs) and not essay.exists():
        raise FileNotFoundError(f"{essay} is missing; run upstream download_paulgraham_essay.py in its directory first")

    # prepare.py shells out to `python`, so put this interpreter first on PATH.
    env = dict(os.environ)
    env["PATH"] = str(Path(sys.executable).parent) + os.pathsep + env.get("PATH", "")

    examples: list[Example] = []
    for task in tasks:
        if task not in task_configs:
            raise ValueError(f"unknown RULER task {task!r}")
        task_family = task_configs[task]["task"]
        marker = question_marker(constants.TASKS[task_family]["template"], task_family, task_configs[task]["args"])
        for length in lengths:
            save_dir = (raw_dir / f"seed{seed}" / str(length)).resolve()
            raw_file = save_dir / task / "validation.jsonl"
            if raw_file.exists():
                raw_file.unlink()  # upstream skips existing files; always regenerate
            cmd = [
                sys.executable, "prepare.py",
                "--save_dir", str(save_dir),
                "--benchmark", "synthetic",
                "--task", task,
                "--tokenizer_path", str(tokenizer_path),
                "--tokenizer_type", "hf",
                "--max_seq_length", str(length),
                "--model_template_type", "base",
                "--num_samples", str(num_samples),
                "--random_seed", str(seed),
            ]  # fmt: skip
            proc = subprocess.run(cmd, cwd=scripts / "data", env=env, capture_output=True, text=True)
            # prepare.py swallows generator failures, so check the output itself.
            rows = read_jsonl(raw_file) if raw_file.exists() else []
            if proc.returncode != 0 or len(rows) != num_samples:
                raise RuntimeError(
                    f"RULER generation failed for {task}@{length} "
                    f"(exit {proc.returncode}, {len(rows)}/{num_samples} rows)\n{proc.stdout[-4000:]}\n{proc.stderr[-4000:]}"
                )
            provenance = {"ruler_commit": RULER_COMMIT, "tokenizer": tokenizer_id, "seed": seed}
            if task_configs[task]["args"].get("type_haystack") == "essay":
                provenance["haystack_sha256"] = sha256_file(essay)  # downloaded text, not pinned upstream
            examples.extend(convert_rows(rows, task, length, task_family, marker, provenance))
            print(f"{task}@{length}: {len(rows)} examples")

    check_disjoint(examples, existing)
    out_dir.mkdir(parents=True, exist_ok=True)
    for ex in examples:
        append_jsonl(targets[ex.split], ex)
    for split, path in targets.items():
        if not path.exists():
            path.touch()
        Path(f"{path}.sha256").write_text(f"{sha256_file(path)}  {path.name}\n")
        n = sum(1 for ex in examples if ex.split == split)
        print(f"wrote {path} ({n} examples)")
    shutil.rmtree(raw_dir / f"seed{seed}", ignore_errors=True)
    return targets
