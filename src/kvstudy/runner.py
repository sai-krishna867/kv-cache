"""Controlled execution procedure, run directories, and run manifests."""

from __future__ import annotations

import hashlib
import json
import platform
import random
import secrets
import subprocess
import time
import traceback
from datetime import datetime, timezone
from importlib import metadata
from pathlib import Path
from typing import Any

import yaml

import kvstudy
from kvstudy.config import ExperimentConfig, config_hash, config_to_dict
from kvstudy.records import PROTOCOL_ID, Example, ResultRecord, append_jsonl

TIMINGS_LABEL = "quality_simulated wall-clock; not a systems measurement"
SYSTEMS_TIMINGS_LABEL = "systems_real: median over repeats after warm-ups; computation forced at every timer boundary"
PACKAGES = ["kvstudy", "mlx", "mlx-metal", "mlx-lm", "transformers", "tokenizers", "huggingface-hub", "pydantic", "numpy", "pyyaml"]


def new_run_id(now: datetime | None = None) -> str:
    now = now or datetime.now(timezone.utc)
    return f"{now.strftime('%Y-%m-%dT%H%M%SZ')}_{secrets.token_hex(4)}"


def create_run_dir(cfg: ExperimentConfig) -> tuple[str, Path, datetime]:
    started = datetime.now(timezone.utc)
    run_id = new_run_id(started)
    run_dir = cfg.output.root / cfg.name / run_id
    run_dir.mkdir(parents=True, exist_ok=False)
    return run_id, run_dir, started


def method_results_path(method_id: str) -> str:
    return f"methods/{method_id}/results.jsonl"


def _cmd(args: list[str]) -> str | None:
    try:
        return subprocess.run(args, capture_output=True, text=True, check=True).stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        return None


def git_state() -> dict[str, Any]:
    status = _cmd(["git", "status", "--porcelain"])
    return {"commit": _cmd(["git", "rev-parse", "HEAD"]), "dirty": None if status is None else bool(status)}


# Source files that cannot change what a run produces: analysis re-reads saved
# results, and the CLI only dispatches.
CODE_FINGERPRINT_EXCLUDE = ("analysis/", "cli.py")


def code_fingerprint(root: Path | None = None) -> dict[str, Any]:
    """sha256 over the experiment-relevant source files as they are on disk.

    Uncommitted edits and untracked files under the package change it, unlike a
    commit id. Per-file digests make a mismatch traceable to the files involved.
    """
    root = root or Path(kvstudy.__file__).parent
    files = {}
    for path in sorted(root.rglob("*.py")):
        rel = path.relative_to(root).as_posix()
        if not rel.startswith(CODE_FINGERPRINT_EXCLUDE):
            files[rel] = hashlib.sha256(path.read_bytes()).hexdigest()
    combined = hashlib.sha256("".join(f"{rel}\0{digest}\n" for rel, digest in files.items()).encode())
    return {"sha256": combined.hexdigest(), "files": files}


def package_versions() -> dict[str, str | None]:
    versions = {}
    for name in PACKAGES:
        try:
            versions[name] = metadata.version(name)
        except metadata.PackageNotFoundError:
            versions[name] = None
    return versions


def hardware_info() -> dict[str, Any]:
    mem = _cmd(["sysctl", "-n", "hw.memsize"])
    return {
        "cpu": _cmd(["sysctl", "-n", "machdep.cpu.brand_string"]),
        "memory_bytes": int(mem) if mem else None,
        "machine": platform.machine(),
        "os": platform.platform(),
        "macos_version": platform.mac_ver()[0] or None,
        "python": platform.python_version(),
    }


def build_manifest(
    cfg: ExperimentConfig,
    run_id: str,
    started: datetime,
    loaded: Any,
    datasets: dict[str, Any],
    evaluators: dict[str, Any],
    system_prompt: str,
) -> dict[str, Any]:
    from kvstudy.benchmarks.ruler import sha256_file

    tokenizer_files = {
        p.name: sha256_file(p)
        for p in sorted(loaded.snapshot_path.iterdir())
        if p.name.startswith("tokenizer") or p.name in ("vocab.json", "merges.txt")
    }
    return {
        "run_id": run_id,
        "experiment": cfg.name,
        "started_utc": started.isoformat(),
        "schema_version": cfg.schema_version,
        "config_hash": config_hash(cfg),
        "code": git_state(),
        "code_fingerprint": code_fingerprint(),
        "model": {
            "id": cfg.model.id,
            "revision": loaded.revision,
            "tokenizer_revision": loaded.revision,
            "tokenizer_files_sha256": tokenizer_files,
            "weight_dtype": cfg.model.weight_dtype,
            "arch": vars(loaded.arch),
        },
        "packages": package_versions(),
        "hardware": hardware_info(),
        "datasets": datasets,
        "evaluators": evaluators,
        "execution": {"mode": cfg.execution.mode, "seed": cfg.execution.seed},
        "protocol": {"id": PROTOCOL_ID, **cfg.protocol.model_dump(mode="json")},
        "system_prompt": system_prompt,
        "methods": {m.id: method_results_path(m.id) for m in cfg.methods},
    }


def classify_failure(exc: BaseException) -> str:
    msg = str(exc).lower()
    return "oom" if "metal::malloc" in msg or "out of memory" in msg else "failed"


def _seed_everything(seed: int) -> None:
    import mlx.core as mx
    import numpy as np

    random.seed(seed)
    np.random.seed(seed)
    mx.random.seed(seed)


# Manifest fields that define the experiment identity; resume requires all to match.
IDENTITY_KEYS = [
    "schema_version", "config_hash", "code_fingerprint", "model", "packages", "hardware",
    "datasets", "evaluators", "execution", "protocol", "system_prompt", "methods",
]  # fmt: skip


class ResumeError(RuntimeError):
    pass


def _prepare(cfg: ExperimentConfig) -> tuple[list[tuple[Any, Any, list[Example]]], dict, dict, list[tuple[Any, Any]]]:
    """Resolve examples, fingerprints and methods; everything that can fail before the model loads."""
    from kvstudy.benchmarks.base import get_adapter
    from kvstudy.benchmarks.ruler import sha256_file
    from kvstudy.methods.base import build_method

    plan: list[tuple[Any, Any, list[Example]]] = []
    datasets, evaluators = {}, {}
    for bench in cfg.benchmarks:
        adapter = get_adapter(bench.name)
        examples = adapter.load_examples(bench)
        plan.append((bench, adapter, examples))
        datasets[bench.name] = {
            "manifest": str(bench.split_manifest),
            "manifest_sha256": sha256_file(bench.split_manifest),
            "splits": sorted({ex.split for ex in examples}),
            "n_examples": len(examples),
        }
        evaluators[bench.name] = adapter.evaluator_fingerprint()
    methods = [(m, build_method(m, cfg.protocol)) for m in cfg.methods]
    return plan, datasets, evaluators, methods


def run(cfg: ExperimentConfig) -> Path:
    from kvstudy.backends import mlx as backend

    _seed_everything(cfg.execution.seed)
    plan, datasets, evaluators, methods = _prepare(cfg)
    loaded = backend.load_model(cfg.model)

    run_id, run_dir, started = create_run_dir(cfg)
    with open(run_dir / "resolved_config.yaml", "w", encoding="utf-8") as f:
        yaml.safe_dump(config_to_dict(cfg), f, sort_keys=False)
    manifest = build_manifest(cfg, run_id, started, loaded, datasets, evaluators, backend.SYSTEM_PROMPT)
    with open(run_dir / "manifest.json", "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)
    append_jsonl(run_dir / "events.jsonl", {"event": "run_started", "time_utc": started.isoformat(), "run_id": run_id})
    print(f"run directory: {run_dir}")

    todo = [(bench, adapter, ex, [(m_cfg, m, 1) for m_cfg, m in methods]) for bench, adapter, examples in plan for ex in examples]
    _dispatch(cfg, run_dir, run_id, loaded, todo)
    return run_dir


def _dispatch(cfg: ExperimentConfig, run_dir: Path, run_id: str, loaded: Any, todo: list) -> None:
    if cfg.execution.mode == "systems_real":
        _execute_systems(cfg, run_dir, run_id, loaded, todo)
    else:
        _execute(cfg, run_dir, run_id, loaded, todo)


def latest_records(run_dir: Path, method_ids: list[str]) -> dict[tuple[str, str], dict[str, Any]]:
    """Last saved attempt per (method_id, example_id)."""
    from kvstudy.records import read_jsonl

    latest = {}
    for method_id in method_ids:
        path = run_dir / method_results_path(method_id)
        if not path.exists():
            continue
        for r in read_jsonl(path):
            key = (method_id, r["example_id"])
            if key not in latest or r["attempt"] >= latest[key]["attempt"]:
                latest[key] = r
    return latest


def resume(run_dir: Path, retry_failed: bool = False) -> Path:
    """Continue an interrupted run in its original directory.

    Skips (example, method) pairs that already have a record. Failed and oom
    records are kept unless retry_failed is set, in which case they are rerun and
    the new attempt is appended.
    """
    from kvstudy.backends import mlx as backend
    from kvstudy.config import load_config
    from kvstudy.records import repair_interrupted_jsonl

    with open(run_dir / "manifest.json", encoding="utf-8") as f:
        manifest = json.load(f)
    cfg = load_config(run_dir / "resolved_config.yaml")
    run_id = manifest["run_id"]
    _seed_everything(cfg.execution.seed)
    plan, datasets, evaluators, methods = _prepare(cfg)
    loaded = backend.load_model(cfg.model)

    current = build_manifest(
        cfg, run_id, datetime.now(timezone.utc), loaded, datasets, evaluators, backend.SYSTEM_PROMPT
    )
    # Runs that predate a field (e.g. code_fingerprint) cannot be verified, so they count as mismatched.
    mismatched = [k for k in IDENTITY_KEYS if current[k] != manifest.get(k)]
    if mismatched:
        detail = ""
        if "code_fingerprint" in mismatched and manifest.get("code_fingerprint"):
            old, new = manifest["code_fingerprint"]["files"], current["code_fingerprint"]["files"]
            changed = sorted(f for f in old.keys() | new.keys() if old.get(f) != new.get(f))
            detail = f"; changed source files: {', '.join(changed)}"
        raise ResumeError(f"experiment identity differs from {run_dir} in: {', '.join(mismatched)}{detail}")

    events = run_dir / "events.jsonl"
    for method_cfg, _ in methods:
        path = run_dir / method_results_path(method_cfg.id)
        if path.exists():
            partial = repair_interrupted_jsonl(path)
            if partial is not None:
                append_jsonl(events, {"event": "interrupted_write_removed", "path": str(path), "partial": partial})

    latest = latest_records(run_dir, [m.id for m, _ in methods])
    todo = []
    for bench, adapter, examples in plan:
        for ex in examples:
            pending = []
            for m_cfg, m in methods:
                prev = latest.get((m_cfg.id, ex.example_id))
                if prev is None:
                    pending.append((m_cfg, m, 1))
                elif prev["status"] != "ok" and retry_failed:
                    pending.append((m_cfg, m, prev["attempt"] + 1))
            if pending:
                todo.append((bench, adapter, ex, pending))

    append_jsonl(
        events,
        {
            "event": "resume_started",
            "time_utc": datetime.now(timezone.utc).isoformat(),
            "retry_failed": retry_failed,
            "pending_runs": sum(len(p) for *_, p in todo),
            "code": git_state(),
        },
    )
    print(f"resuming {run_dir}: {sum(len(p) for *_, p in todo)} pending runs")
    _dispatch(cfg, run_dir, run_id, loaded, todo)
    return run_dir


def _execute(cfg: ExperimentConfig, run_dir: Path, run_id: str, loaded: Any, todo: list) -> None:
    """For each example: prefill once, then branch, convert, generate and score per method."""
    import mlx.core as mx

    from kvstudy.analysis.compare import summarize_run
    from kvstudy.backends import mlx as backend

    events = run_dir / "events.jsonl"
    counts = {"ok": 0, "failed": 0, "oom": 0}
    try:
        for i, (bench, adapter, ex, pending) in enumerate(todo, 1):
            base = dict(
                run_id=run_id,
                example_id=ex.example_id,
                source_id=ex.source_id,
                model_id=cfg.model.id,
                model_revision=loaded.revision,
                protocol_id=PROTOCOL_ID,
                benchmark=ex.benchmark,
                task=ex.task,
                split=ex.split,
                context_length=adapter.context_length(ex),
                references=ex.references,
            )
            reference = None
            try:
                if adapter.kind == "continuation":
                    context_ids, continuation_ids = adapter.token_ids(ex)
                    ids, boundary, extra = context_ids + continuation_ids, len(context_ids), {}
                else:
                    prompt = backend.render_prompt(
                        loaded.tokenizer,
                        backend.SYSTEM_PROMPT,
                        adapter.user_text(ex),
                        adapter.question_char_offset(ex),
                        adapter.answer_prefix(ex),
                    )
                    ids, boundary = prompt.token_ids, prompt.boundary
                    extra = {"boundary_straddled": prompt.boundary_straddled}
                t0 = time.perf_counter()
                reference = backend.prefill(loaded.model, ids[:boundary])
                prefill_s = time.perf_counter() - t0
            except Exception as exc:  # noqa: BLE001 - recorded, never scored as zero
                status = classify_failure(exc)
                append_jsonl(events, {"event": "prefill_failed", "example_id": ex.example_id, "error": repr(exc)})
                for method_cfg, _, attempt in pending:
                    append_jsonl(
                        run_dir / method_results_path(method_cfg.id),
                        ResultRecord(
                            **base, method_id=method_cfg.id, status=status, attempt=attempt, error=traceback.format_exc()
                        ),
                    )
                    counts[status] += 1
                mx.clear_cache()
                continue

            tokens = dict(
                n_input_tokens=len(ids),
                n_context_tokens=boundary,
                n_question_tokens=len(ids) - boundary,
            )
            for method_cfg, method, attempt in pending:
                branch = None
                try:
                    t0 = time.perf_counter()
                    branch = backend.branch_cache(reference)
                    report = method.apply(branch, loaded.arch)
                    branch_s = time.perf_counter() - t0
                    t0 = time.perf_counter()
                    if adapter.kind == "continuation":
                        nll_sum, n_scored = backend.score_continuation(loaded.model, branch, ids[boundary:])
                        outcome = dict(score=nll_sum / n_scored, nll_sum=nll_sum, n_scored_tokens=n_scored)
                    else:
                        text, n_gen = backend.generate(
                            loaded.model,
                            loaded.tokenizer,
                            ids[boundary:],
                            branch,
                            adapter.max_new_tokens(ex, bench),
                        )
                        outcome = dict(prediction=text, score=adapter.score(ex, text), n_generated_tokens=n_gen)
                    decode_s = time.perf_counter() - t0
                    record = ResultRecord(
                        **base,
                        **tokens,
                        **outcome,
                        method_id=method_cfg.id,
                        status="ok",
                        attempt=attempt,
                        cache_bytes_estimated=report.cache_bytes_estimated,
                        cache_bytes_estimated_final=method.estimate_bytes(branch, loaded.arch),
                        cache_bytes_label=report.label,
                        timings_s={
                            "prefill_context": prefill_s,
                            "branch_and_convert": branch_s,
                            "question_and_decode": decode_s,
                        },
                        timings_label=TIMINGS_LABEL,
                        metadata={**extra, **report.metadata},
                    )
                except Exception as exc:  # noqa: BLE001
                    record = ResultRecord(
                        **base,
                        **tokens,
                        method_id=method_cfg.id,
                        status=classify_failure(exc),
                        attempt=attempt,
                        error=traceback.format_exc(),
                    )
                finally:
                    del branch
                append_jsonl(run_dir / method_results_path(method_cfg.id), record)
                counts[record.status] += 1
                shown = f"score={record.score}" if record.status == "ok" else record.status
                print(f"[{i}/{len(todo)}] {ex.example_id} {method_cfg.id}: {shown}")
            del reference
            mx.clear_cache()
    except KeyboardInterrupt:
        append_jsonl(
            events, {"event": "run_interrupted", "time_utc": datetime.now(timezone.utc).isoformat(), "counts": counts}
        )
        raise

    append_jsonl(
        events,
        {"event": "run_finished", "time_utc": datetime.now(timezone.utc).isoformat(), "counts": counts},
    )
    summarize_run(run_dir)


def _execute_systems(cfg: ExperimentConfig, run_dir: Path, run_id: str, loaded: Any, todo: list) -> None:
    """systems_real: per (example, method), fresh BF16 prefill, real conversion, fixed-length decode.

    No BF16 reference is retained, so memory reflects deployment. Each run repeats
    warmup_runs + repeats times; timings are medians over the repeats.
    """
    import statistics

    import mlx.core as mx

    from kvstudy.analysis.compare import summarize_run
    from kvstudy.backends import mlx as backend

    sysc = cfg.execution.systems
    events = run_dir / "events.jsonl"
    counts = {"ok": 0, "failed": 0, "oom": 0}
    try:
        for i, (bench, adapter, ex, pending) in enumerate(todo, 1):
            prompt = backend.render_prompt(
                loaded.tokenizer,
                backend.SYSTEM_PROMPT,
                adapter.user_text(ex),
                adapter.question_char_offset(ex),
                adapter.answer_prefix(ex),
            )
            ids, boundary = prompt.token_ids, prompt.boundary
            base = dict(
                run_id=run_id,
                example_id=ex.example_id,
                source_id=ex.source_id,
                model_id=cfg.model.id,
                model_revision=loaded.revision,
                protocol_id=PROTOCOL_ID,
                benchmark=ex.benchmark,
                task=ex.task,
                split=ex.split,
                context_length=adapter.context_length(ex),
                references=ex.references,
                n_input_tokens=len(ids),
                n_context_tokens=boundary,
                n_question_tokens=len(ids) - boundary,
            )
            for method_cfg, method, attempt in pending:
                try:
                    reps = []
                    for rep in range(sysc.warmup_runs + sysc.repeats):
                        cache = None
                        mx.clear_cache()
                        mx.reset_peak_memory()
                        t0 = time.perf_counter()
                        cache = backend.prefill(loaded.model, ids[:boundary])
                        prefill_s = time.perf_counter() - t0
                        bf16_bytes = backend.cache_buffer_bytes(cache)
                        t0 = time.perf_counter()
                        cache = method.convert_real(cache, loaded.arch)
                        mx.eval([c.state for c in cache])
                        convert_s = time.perf_counter() - t0
                        mx.clear_cache()
                        converted_bytes = backend.cache_buffer_bytes(cache)
                        active_bytes = mx.get_active_memory()
                        tokens, question_s, decode_s = backend.generate_fixed(
                            loaded.model, ids[boundary:], cache, sysc.decode_tokens
                        )
                        reps.append(
                            {
                                "prefill_context": prefill_s,
                                "convert": convert_s,
                                "question_prefill": question_s,
                                "decode": decode_s,
                                "bf16_cache_bytes": bf16_bytes,
                                "cache_bytes": converted_bytes,
                                "cache_bytes_final": backend.cache_buffer_bytes(cache),
                                "active_bytes_after_convert": active_bytes,
                                "peak_bytes": mx.get_peak_memory(),
                                "tokens": tokens,
                            }
                        )
                        del cache
                    measured = reps[sysc.warmup_runs :]
                    med = {k: statistics.median(r[k] for r in measured) for k in ("prefill_context", "convert", "question_prefill", "decode")}
                    record = ResultRecord(
                        **base,
                        method_id=method_cfg.id,
                        status="ok",
                        attempt=attempt,
                        prediction=loaded.tokenizer.decode(measured[-1]["tokens"]),
                        n_generated_tokens=sysc.decode_tokens,
                        cache_bytes_estimated=measured[-1]["cache_bytes"],
                        cache_bytes_label="measured_cache_buffers",
                        memory_bytes={
                            "bf16_cache_before_conversion": measured[-1]["bf16_cache_bytes"],
                            "cache_after_conversion": measured[-1]["cache_bytes"],
                            "cache_after_decode": measured[-1]["cache_bytes_final"],
                            "mlx_active_after_conversion": max(r["active_bytes_after_convert"] for r in measured),
                            "mlx_peak": max(r["peak_bytes"] for r in measured),
                        },
                        timings_s={**med, "decode_tokens_per_s": (sysc.decode_tokens - 1) / med["decode"]},
                        timings_label=SYSTEMS_TIMINGS_LABEL,
                        metadata={
                            "boundary_straddled": prompt.boundary_straddled,
                            "warmup_runs": sysc.warmup_runs,
                            "repeats": [{k: v for k, v in r.items() if k != "tokens"} for r in measured],
                            "outputs_identical_across_repeats": all(r["tokens"] == measured[0]["tokens"] for r in measured),
                        },
                    )
                except Exception as exc:  # noqa: BLE001
                    record = ResultRecord(
                        **base, method_id=method_cfg.id, status=classify_failure(exc), attempt=attempt, error=traceback.format_exc()
                    )
                append_jsonl(run_dir / method_results_path(method_cfg.id), record)
                counts[record.status] += 1
                shown = f"{record.timings_s.get('decode_tokens_per_s', 0):.1f} tok/s" if record.status == "ok" else record.status
                print(f"[{i}/{len(todo)}] {ex.example_id} {method_cfg.id}: {shown}")
                mx.clear_cache()
    except KeyboardInterrupt:
        append_jsonl(
            events, {"event": "run_interrupted", "time_utc": datetime.now(timezone.utc).isoformat(), "counts": counts}
        )
        raise

    append_jsonl(
        events,
        {"event": "run_finished", "time_utc": datetime.now(timezone.utc).isoformat(), "counts": counts},
    )
    summarize_run(run_dir)
