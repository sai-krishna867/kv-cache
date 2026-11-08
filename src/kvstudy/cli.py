"""Command-line interface."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from pydantic import ValidationError


def _load(path: Path):
    from kvstudy.config import load_config

    try:
        return load_config(path)
    except ValidationError as e:
        sys.exit(f"invalid config {path}:\n{e}")


def cmd_validate(args: argparse.Namespace) -> int:
    from kvstudy.config import validate_config

    problems = validate_config(_load(args.config))
    for p in problems:
        print(f"ERROR: {p}", file=sys.stderr)
    if not problems:
        print("config OK")
    return 1 if problems else 0


def cmd_run(args: argparse.Namespace) -> int:
    from kvstudy.config import validate_config
    from kvstudy.runner import run

    cfg = _load(args.config)
    problems = validate_config(cfg)
    if problems:
        for p in problems:
            print(f"ERROR: {p}", file=sys.stderr)
        return 1
    run_dir = run(cfg)
    print(f"done: {run_dir}")
    return 0


def cmd_resume(args: argparse.Namespace) -> int:
    from kvstudy.runner import ResumeError, resume

    try:
        run_dir = resume(args.run_dir, retry_failed=args.retry_failed)
    except ResumeError as e:
        print(f"ERROR: {e}", file=sys.stderr)
        return 1
    print(f"done: {run_dir}")
    return 0


def cmd_analyze(args: argparse.Namespace) -> int:
    from kvstudy.analysis.compare import summarize_run

    comparison = summarize_run(args.run_dir)
    for method_id, cells in comparison["methods"].items():
        for c in cells:
            score = "n/a" if c["mean_score"] is None else f"{c['mean_score']:.2f}"
            print(
                f"{method_id} {c['benchmark']}/{c['task']}@{c['context_length']}: "
                f"score={score} ok={c['n_ok']}/{c['n']} failed={c['n_failed']} oom={c['n_oom']}"
            )
    paired = comparison.get("paired_vs_reference")
    if paired and paired["methods"]:
        boot = paired["bootstrap"]
        print(f"paired vs {paired['reference']} ({boot['confidence']:.0%} {boot['unit']} bootstrap intervals):")
        for method_id, result in paired["methods"].items():
            for c in result["pooled"] + result["cells"]:
                where = "/".join(str(c[k]) for k in ("benchmark", "task", "context_length") if k in c)
                diff = "n/a" if c["mean_diff"] is None else f"{c['mean_diff']:+.2f}"
                if c["mean_diff_ci"] is not None:
                    diff += f" [{c['mean_diff_ci'][0]:+.2f}, {c['mean_diff_ci'][1]:+.2f}]"
                flips = "n/a" if c["flip_rate"] is None else f"{c['flip_rate']:.2f}"
                print(
                    f"  {method_id} {c['scope']} {where}: diff={diff} flip_rate={flips} "
                    f"pairs={c['n_pairs']} sources={c['n_sources']} excluded={c['n_excluded']}"
                )
    return 0


def cmd_prepare_ruler(args: argparse.Namespace) -> int:
    from kvstudy.backends.mlx import local_snapshot
    from kvstudy.benchmarks.ruler import prepare_ruler
    from kvstudy.config import ModelConfig

    model_cfg = ModelConfig(backend="mlx", id=args.model_id, revision=args.revision, weight_dtype="bfloat16")
    snapshot = local_snapshot(model_cfg)
    if snapshot is None:
        sys.exit(f"{args.model_id}@{args.revision} is not in the local HF cache; download it first")
    prepare_ruler(
        ruler_dir=args.ruler_dir,
        tokenizer_path=snapshot,
        tokenizer_id=f"{args.model_id}@{args.revision}",
        tasks=args.tasks,
        lengths=args.lengths,
        num_samples=args.num_samples,
        seed=args.seed,
        out_dir=args.out_dir,
        raw_dir=args.raw_dir,
        disjoint_from=args.disjoint_from,
    )
    return 0


def _snapshot_tokenizer(args: argparse.Namespace):
    from transformers import AutoTokenizer

    from kvstudy.backends.mlx import local_snapshot
    from kvstudy.config import ModelConfig

    model_cfg = ModelConfig(backend="mlx", id=args.model_id, revision=args.revision, weight_dtype="bfloat16")
    snapshot = local_snapshot(model_cfg)
    if snapshot is None:
        sys.exit(f"{args.model_id}@{args.revision} is not in the local HF cache; download it first")
    return AutoTokenizer.from_pretrained(snapshot), f"{args.model_id}@{args.revision}"


def cmd_prepare_longbench(args: argparse.Namespace) -> int:
    from huggingface_hub import hf_hub_download

    from kvstudy.benchmarks.longbench import DATASET_REPO, DATASET_REVISION, prepare_longbench

    try:
        data_zip = hf_hub_download(
            DATASET_REPO, "data.zip", repo_type="dataset", revision=DATASET_REVISION, local_files_only=True
        )
    except Exception:  # noqa: BLE001
        sys.exit(
            f"LongBench data is not in the local HF cache; download it with: "
            f"hf download {DATASET_REPO} data.zip --repo-type dataset --revision {DATASET_REVISION}"
        )
    tokenizer, tokenizer_id = _snapshot_tokenizer(args)
    prepare_longbench(
        longbench_dir=args.longbench_dir,
        data_zip=Path(data_zip),
        tokenizer=tokenizer,
        tokenizer_id=tokenizer_id,
        tasks=args.tasks,
        lengths=args.lengths,
        out_dir=args.out_dir,
        disjoint_from=args.disjoint_from,
    )
    return 0


def cmd_prepare_pg19(args: argparse.Namespace) -> int:
    from kvstudy.benchmarks.pg19 import prepare_pg19

    tokenizer, tokenizer_id = _snapshot_tokenizer(args)
    prepare_pg19(
        split=args.pg19_split,
        books_dir=args.books_dir,
        tokenizer=tokenizer,
        tokenizer_id=tokenizer_id,
        lengths=args.lengths,
        continuation_tokens=args.continuation_tokens,
        num_books=args.num_books,
        seed=args.seed,
        out_dir=args.out_dir,
        disjoint_from=args.disjoint_from,
    )
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="kvstudy")
    sub = parser.add_subparsers(dest="command", required=True)

    p = sub.add_parser("validate", help="validate a config and summarize the run count")
    p.add_argument("config", type=Path)
    p.set_defaults(func=cmd_validate)

    p = sub.add_parser("run", help="run an experiment into a new run directory")
    p.add_argument("config", type=Path)
    p.set_defaults(func=cmd_run)

    p = sub.add_parser("resume", help="continue an interrupted run in its original directory")
    p.add_argument("run_dir", type=Path)
    p.add_argument("--retry-failed", action="store_true", help="rerun failed and oom records as new attempts")
    p.set_defaults(func=cmd_resume)

    p = sub.add_parser("analyze", help="summarize saved results of a run")
    p.add_argument("run_dir", type=Path)
    p.set_defaults(func=cmd_analyze)

    p = sub.add_parser("prepare", help="generate versioned dataset split manifests")
    prep = p.add_subparsers(dest="dataset", required=True)
    r = prep.add_parser("ruler", help="generate RULER examples with the pinned upstream generator")
    r.add_argument("--ruler-dir", type=Path, default=Path("data/upstream/RULER"))
    r.add_argument("--model-id", required=True)
    r.add_argument("--revision", required=True)
    r.add_argument("--tasks", nargs="+", required=True)
    r.add_argument("--lengths", nargs="+", type=int, required=True)
    r.add_argument("--num-samples", type=int, required=True)
    r.add_argument("--seed", type=int, default=42)
    r.add_argument("--out-dir", type=Path, required=True)
    r.add_argument("--raw-dir", type=Path, default=Path("data/ruler_raw"))
    r.add_argument(
        "--disjoint-from", type=Path, nargs="+", default=[],
        help="existing manifests (read only); fail if a source would land in a different split",
    )  # fmt: skip
    r.set_defaults(func=cmd_prepare_ruler)

    lb = prep.add_parser("longbench", help="LongBench English QA, bucketed by natural rendered length")
    lb.add_argument("--longbench-dir", type=Path, default=Path("data/upstream/LongBench"))
    lb.add_argument("--model-id", required=True)
    lb.add_argument("--revision", required=True)
    lb.add_argument("--tasks", nargs="+", required=True)
    lb.add_argument("--lengths", nargs="+", type=int, required=True, help="length bucket upper bounds")
    lb.add_argument("--out-dir", type=Path, required=True)
    lb.add_argument("--disjoint-from", type=Path, nargs="+", default=[])
    lb.set_defaults(func=cmd_prepare_longbench)

    pg = prep.add_parser("pg19", help="PG-19 continuation windows (downloads the split from DeepMind's bucket)")
    pg.add_argument("--pg19-split", choices=["test", "validation"], default="test")
    pg.add_argument("--books-dir", type=Path, default=Path("data/upstream/pg19"))
    pg.add_argument("--model-id", required=True)
    pg.add_argument("--revision", required=True)
    pg.add_argument("--lengths", nargs="+", type=int, required=True)
    pg.add_argument("--continuation-tokens", type=int, default=512)
    pg.add_argument("--num-books", type=int, required=True)
    pg.add_argument("--seed", type=int, default=42)
    pg.add_argument("--out-dir", type=Path, required=True)
    pg.add_argument("--disjoint-from", type=Path, nargs="+", default=[])
    pg.set_defaults(func=cmd_prepare_pg19)

    args = parser.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
