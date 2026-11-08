"""PG-19 adapter: cache-aware continuation NLL on fixed token windows of books.

Each example is one window of a book: `context_length` tokens of context, then
`continuation_tokens` reference tokens that are teacher-forced. Token ids are
stored in the manifest so every run scores exactly the same tokens.

The prediction of continuation token 0 comes from the logits of the last context
position, which are computed during the BF16 prefill, before cache conversion.
It is therefore excluded: tokens 1..N-1 are scored, each predicted through the
converted cache, identically for every method.
"""

from __future__ import annotations

import base64
import hashlib
import json
import urllib.request
from pathlib import Path
from typing import Any

import numpy as np

from kvstudy.benchmarks.ruler import RulerAdapter, assign_split, check_disjoint, sha256_file
from kvstudy.records import Example, append_jsonl

BUCKET_URL = "https://storage.googleapis.com/storage/v1/b/deepmind-gutenberg/o"
TASK = "continuation"
_THIS_FILE = Path(__file__)


class PG19Adapter:
    name = "pg19"
    kind = "continuation"

    def load_examples(self, bench_cfg: Any) -> list[Example]:
        from types import SimpleNamespace

        cells = SimpleNamespace(
            split_manifest=bench_cfg.split_manifest,
            tasks=[TASK],
            context_lengths=bench_cfg.context_lengths,
            max_examples_per_cell=bench_cfg.max_examples_per_cell,
        )
        examples = RulerAdapter.load_examples(self, cells)
        for ex in examples:
            n = len(ex.metadata["continuation_token_ids"])
            if n != bench_cfg.continuation_tokens:
                raise ValueError(f"{ex.example_id}: {n} continuation tokens; config asks {bench_cfg.continuation_tokens}")
        return examples

    def context_length(self, example: Example) -> int:
        return int(example.metadata["context_length"])

    def token_ids(self, example: Example) -> tuple[list[int], list[int]]:
        return example.metadata["context_token_ids"], example.metadata["continuation_token_ids"]

    def evaluator_fingerprint(self) -> dict[str, str]:
        return {
            "evaluator": "kvstudy cache-aware continuation NLL (first continuation token excluded)",
            "adapter_file_sha256": sha256_file(_THIS_FILE),
        }


# --- dataset preparation -------------------------------------------------------


def _list_split(split: str) -> list[dict[str, Any]]:
    items, token = [], None
    while True:
        url = f"{BUCKET_URL}?prefix={split}/" + (f"&pageToken={token}" if token else "")
        with urllib.request.urlopen(url) as resp:
            page = json.load(resp)
        items += page.get("items", [])
        token = page.get("nextPageToken")
        if not token:
            return [i for i in items if i["name"].endswith(".txt")]


def download_pg19(split: str, books_dir: Path) -> list[Path]:
    """Fetch a PG-19 split from DeepMind's public bucket, verifying each file's md5."""
    books_dir.mkdir(parents=True, exist_ok=True)
    paths = []
    for item in sorted(_list_split(split), key=lambda i: i["name"]):
        path = books_dir / Path(item["name"]).name
        expected = base64.b64decode(item["md5Hash"]).hex()
        if not path.exists() or hashlib.md5(path.read_bytes()).hexdigest() != expected:
            with urllib.request.urlopen(item["mediaLink"]) as resp:
                path.write_bytes(resp.read())
            if hashlib.md5(path.read_bytes()).hexdigest() != expected:
                raise ValueError(f"{path}: md5 mismatch after download")
        paths.append(path)
    return paths


def window_start(n_tokens: int, length: int, continuation: int, seed: int, book_id: str) -> int | None:
    """Deterministic window start for one book and length, or None if the book is too short."""
    span = length + continuation
    if n_tokens < span:
        return None
    rng = np.random.default_rng([seed, int(book_id), length])
    return int(rng.integers(0, n_tokens - span + 1))


def prepare_pg19(
    split: str,
    books_dir: Path,
    tokenizer: Any,
    tokenizer_id: str,
    lengths: list[int],
    continuation_tokens: int,
    num_books: int,
    seed: int,
    out_dir: Path,
    disjoint_from: list[Path] | None = None,
) -> dict[str, Path]:
    """One window per book and length from the first `num_books` books (sorted by id) of a PG-19 split.

    The book is the source: all its windows share one dev/test split.
    """
    from kvstudy.records import read_jsonl

    targets = {s: out_dir / f"pg19_{s}.jsonl" for s in ("dev", "test")}
    for path in targets.values():
        if path.exists():
            raise FileExistsError(f"{path} exists; manifests are versioned, write to a new --out-dir")

    books = download_pg19(split, books_dir)[:num_books]
    examples: list[Example] = []
    skipped = 0
    for path in books:
        book_id = path.stem
        ids = tokenizer(path.read_text(encoding="utf-8"), add_special_tokens=False)["input_ids"]
        source_id = f"pg19/{book_id}"
        for length in lengths:
            start = window_start(len(ids), length, continuation_tokens, seed, book_id)
            if start is None:
                skipped += 1
                continue
            ctx = ids[start : start + length]
            cont = ids[start + length : start + length + continuation_tokens]
            examples.append(
                Example(
                    example_id=f"pg19/{book_id}/{length}",
                    source_id=source_id,
                    benchmark="pg19",
                    task=TASK,
                    split=assign_split(source_id),
                    context=tokenizer.decode(ctx),
                    question=None,
                    reference_continuation=tokenizer.decode(cont),
                    references=[],
                    metadata={
                        "context_length": length,
                        "window_start": start,
                        "book_tokens": len(ids),
                        "book_sha256": sha256_file(path),
                        "context_token_ids": ctx,
                        "continuation_token_ids": cont,
                        "pg19_split": split,
                        "tokenizer": tokenizer_id,
                        "seed": seed,
                    },
                )
            )
    print(f"{len(books)} books, {len(examples)} windows, {skipped} (book, length) pairs too short")

    existing = [Example.from_dict(r) for path in disjoint_from or [] for r in read_jsonl(path)]
    check_disjoint(examples, existing)
    out_dir.mkdir(parents=True, exist_ok=True)
    for ex in examples:
        append_jsonl(targets[ex.split], ex)
    for s, path in targets.items():
        if not path.exists():
            path.touch()
        Path(f"{path}.sha256").write_text(f"{sha256_file(path)}  {path.name}\n")
        print(f"wrote {path} ({sum(ex.split == s for ex in examples)} examples)")
    return targets
