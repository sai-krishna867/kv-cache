"""LongBench v1 adapter: English QA datasets, official prompts and F1 scoring.

Length policy: examples are bucketed by their natural rendered length (chat
template, system prompt and official generation budget included) into the
smallest configured bucket that holds them. Nothing is truncated or padded;
examples longer than the largest bucket are excluded and counted at preparation.
"""

from __future__ import annotations

import hashlib
import json
import subprocess
import zipfile
from collections import Counter
from pathlib import Path
from typing import Any

from kvstudy.benchmarks._vendor import longbench_eval
from kvstudy.benchmarks.ruler import RulerAdapter, assign_split, check_disjoint, sha256_file
from kvstudy.records import Example, append_jsonl

LONGBENCH_COMMIT = longbench_eval.LONGBENCH_COMMIT
DATASET_REPO = "THUDM/LongBench"
DATASET_REVISION = "5e628be450b7e67fb7ae6e201bd6d8f7056f7672"
SUPPORTED_TASKS = tuple(longbench_eval.dataset2metric)
_VENDOR_FILE = Path(longbench_eval.__file__)


class LongBenchAdapter:
    name = "longbench"
    kind = "generate"

    load_examples = RulerAdapter.load_examples  # same task x length cell selection

    def user_text(self, example: Example) -> str:
        return example.context + example.question

    def question_char_offset(self, example: Example) -> int:
        return len(example.context)

    def answer_prefix(self, example: Example) -> str:
        return ""  # the official prompt already ends with "Answer:"

    def context_length(self, example: Example) -> int:
        return int(example.metadata["context_length"])

    def max_new_tokens(self, example: Example, bench_cfg: Any) -> int:
        return int(example.metadata["max_new_tokens"])  # official dataset2maxlen

    def score(self, example: Example, prediction: str) -> float:
        # eval.py scorer for one prediction: max over references, scaled to 0-100.
        metric = longbench_eval.dataset2metric[example.task]
        best = max(metric(prediction, ref, all_classes=example.metadata.get("all_classes")) for ref in example.references)
        return 100.0 * best

    def evaluator_fingerprint(self) -> dict[str, str]:
        return {
            "evaluator": "THUDM/LongBench LongBench/metrics.py qa_f1_score (vendored)",
            "revision": LONGBENCH_COMMIT,
            "vendored_file_sha256": sha256_file(_VENDOR_FILE),
        }


# --- dataset preparation -------------------------------------------------------


def split_prompt(template: str, context: str, question: str) -> tuple[str, str]:
    """(context part, question part) of the official prompt; their concatenation is template.format(...)."""
    if template.count("{context}") != 1 or template.count("{input}") != 1:
        raise ValueError(f"unexpected prompt template {template!r}")
    head, tail = template.split("{context}")
    if "{input}" not in tail:
        raise ValueError("prompt places the question before the context")
    return head + context, tail.replace("{input}", question)


def bucket_for(n_tokens: int, max_new_tokens: int, lengths: list[int]) -> int | None:
    """Smallest bucket holding the rendered input plus the generation budget."""
    for length in sorted(lengths):
        if n_tokens + max_new_tokens <= length:
            return length
    return None


def check_longbench_clone(longbench_dir: Path) -> None:
    head = subprocess.run(
        ["git", "-C", str(longbench_dir), "rev-parse", "HEAD"], capture_output=True, text=True, check=True
    ).stdout.strip()
    if head != LONGBENCH_COMMIT:
        raise ValueError(f"{longbench_dir} is at {head}; expected pinned LongBench commit {LONGBENCH_COMMIT}")


def prepare_longbench(
    longbench_dir: Path,
    data_zip: Path,
    tokenizer: Any,
    tokenizer_id: str,
    tasks: list[str],
    lengths: list[int],
    out_dir: Path,
    disjoint_from: list[Path] | None = None,
) -> dict[str, Path]:
    """Write dev/test manifests of LongBench examples bucketed by rendered length.

    `tokenizer` is the model's HF tokenizer; token counts use the same chat
    template and system prompt as the runner.
    """
    from types import SimpleNamespace

    from kvstudy.backends.mlx import SYSTEM_PROMPT, render_prompt
    from kvstudy.records import read_jsonl

    check_longbench_clone(longbench_dir)
    config = longbench_dir / "LongBench" / "config"
    prompts = json.loads((config / "dataset2prompt.json").read_text())
    maxlen = json.loads((config / "dataset2maxlen.json").read_text())
    unsupported = [t for t in tasks if t not in SUPPORTED_TASKS]
    if unsupported:
        raise ValueError(f"unsupported LongBench tasks {unsupported}; supported: {list(SUPPORTED_TASKS)}")

    targets = {split: out_dir / f"longbench_{split}.jsonl" for split in ("dev", "test")}
    for path in targets.values():
        if path.exists():
            raise FileExistsError(f"{path} exists; manifests are versioned, write to a new --out-dir")

    wrapper = SimpleNamespace(_tokenizer=tokenizer)
    provenance = {
        "longbench_commit": LONGBENCH_COMMIT,
        "dataset": f"{DATASET_REPO}@{DATASET_REVISION}",
        "data_zip_sha256": sha256_file(data_zip),
        "tokenizer": tokenizer_id,
    }
    examples: list[Example] = []
    with zipfile.ZipFile(data_zip) as zf:
        for task in tasks:
            rows = [json.loads(line) for line in zf.read(f"data/{task}.jsonl").decode("utf-8").splitlines() if line]
            counts: Counter = Counter()
            for row in rows:
                context, question = split_prompt(prompts[task], row["context"], row["input"])
                prompt = render_prompt(wrapper, SYSTEM_PROMPT, context + question, len(context), "")
                n_tokens = len(prompt.token_ids)
                length = bucket_for(n_tokens, maxlen[task], lengths)
                if length is None:
                    counts["excluded_too_long"] += 1
                    continue
                counts[length] += 1
                # Questions about one document share it: the document is the source.
                source_id = f"longbench/{task}/doc-{hashlib.sha256(row['context'].encode()).hexdigest()[:16]}"
                examples.append(
                    Example(
                        example_id=f"longbench/{task}/{row['_id']}",
                        source_id=source_id,
                        benchmark="longbench",
                        task=task,
                        split=assign_split(source_id),
                        context=context,
                        question=question,
                        reference_continuation=None,
                        references=list(row["answers"]),
                        metadata={
                            "context_length": length,
                            "n_tokens_rendered": n_tokens,
                            "longbench_length": row["length"],
                            "max_new_tokens": maxlen[task],
                            "all_classes": row["all_classes"],
                            **provenance,
                        },
                    )
                )
            print(f"{task}: {dict(sorted(counts.items(), key=str))}")

    existing = [Example.from_dict(r) for path in disjoint_from or [] for r in read_jsonl(path)]
    check_disjoint(examples, existing)
    out_dir.mkdir(parents=True, exist_ok=True)
    for ex in examples:
        append_jsonl(targets[ex.split], ex)
    for split, path in targets.items():
        if not path.exists():
            path.touch()
        Path(f"{path}.sha256").write_text(f"{sha256_file(path)}  {path.name}\n")
        print(f"wrote {path} ({sum(ex.split == split for ex in examples)} examples)")
    return targets
