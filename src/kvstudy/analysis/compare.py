"""Summaries computed from saved per-example results; never loads a model."""

from __future__ import annotations

import json
import math
from collections import defaultdict
from pathlib import Path
from statistics import mean
from typing import Any

import numpy as np
import yaml

from kvstudy.records import read_jsonl

# Paired bootstrap: fixed so that re-running analysis reproduces the intervals.
BOOTSTRAP_RESAMPLES = 10_000
BOOTSTRAP_SEED = 0
BOOTSTRAP_CONFIDENCE = 0.95

# What `score` means per benchmark. Metrics are never combined across benchmarks.
METRICS = {
    "ruler": {"metric": "RULER string match (0-100)", "higher_is_better": True},
    "longbench": {"metric": "LongBench qa_f1_score (0-100)", "higher_is_better": True},
    "pg19": {"metric": "mean NLL, nats per scored token", "higher_is_better": False},
}


def _perplexity(rows: list[dict[str, Any]]) -> float | None:
    """exp(total NLL / total scored tokens) over continuation records."""
    tokens = sum(r.get("n_scored_tokens") or 0 for r in rows)
    return math.exp(sum(r["nll_sum"] for r in rows) / tokens) if tokens else None


def _mean(values: list[float]) -> float | None:
    return mean(values) if values else None


def latest_attempts(records: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Keep the last attempt per example; earlier failed attempts are superseded by explicit retries."""
    latest: dict[str, dict[str, Any]] = {}
    for r in records:
        prev = latest.get(r["example_id"])
        if prev is None or r.get("attempt", 1) >= prev.get("attempt", 1):  # milestone A rows lack attempt
            latest[r["example_id"]] = r
    return list(latest.values())


def summarize_records(records: list[dict[str, Any]], elements_per_token: int | None = None) -> dict[str, Any]:
    """Per benchmark x task x context_length aggregates.

    Only ok records enter means; failed and oom records are counted, never scored.
    `elements_per_token` (layers x KV heads x head_dim x 2) turns estimated context
    bytes into effective bits per cached element.
    """
    current = latest_attempts(records)
    cells: dict[tuple[str, str, int], list[dict[str, Any]]] = defaultdict(list)
    for r in current:
        cells[(r["benchmark"], r["task"], r["context_length"])].append(r)

    out = []
    for (benchmark, task, length), rows in sorted(cells.items()):
        ok = [r for r in rows if r["status"] == "ok"]
        out.append(
            {
                "benchmark": benchmark,
                "task": task,
                "context_length": length,
                "n": len(rows),
                "n_ok": len(ok),
                "n_failed": sum(r["status"] == "failed" for r in rows),
                "n_oom": sum(r["status"] == "oom" for r in rows),
                "mean_score": _mean([r["score"] for r in ok if r["score"] is not None]),  # systems_real: unscored
                "mean_n_input_tokens": _mean([r["n_input_tokens"] for r in ok]),
                "mean_n_context_tokens": _mean([r["n_context_tokens"] for r in ok]),
                "mean_n_generated_tokens": _mean([r["n_generated_tokens"] for r in ok if r["n_generated_tokens"] is not None]),
                "mean_cache_bytes_estimated": _mean([r["cache_bytes_estimated"] for r in ok]),
                "mean_cache_bytes_estimated_final": _mean([r["cache_bytes_estimated_final"] for r in ok if r.get("cache_bytes_estimated_final") is not None]),
                "cache_bytes_labels": sorted({r["cache_bytes_label"] for r in ok}),
                "mean_effective_bits": _mean(
                    [r["cache_bytes_estimated"] * 8 / (r["n_context_tokens"] * elements_per_token) for r in ok]
                )
                if elements_per_token
                else None,
                "metric": METRICS.get(benchmark, {}).get("metric"),
                "perplexity": _perplexity(ok) if ok and all(r.get("nll_sum") is not None for r in ok) else None,
                **(_systems_cell(ok) if ok and all(r.get("memory_bytes") for r in ok) else {}),
            }
        )
    return {"n_superseded_attempts": len(records) - len(current), "cells": out}


def _systems_cell(rows: list[dict[str, Any]]) -> dict[str, Any]:
    """systems_real aggregates: medians of per-example median timings, measured memory."""
    from statistics import median

    return {
        "median_prefill_context_s": median(r["timings_s"]["prefill_context"] for r in rows),
        "median_convert_s": median(r["timings_s"]["convert"] for r in rows),
        "median_question_prefill_s": median(r["timings_s"]["question_prefill"] for r in rows),
        "median_decode_tokens_per_s": median(r["timings_s"]["decode_tokens_per_s"] for r in rows),
        "mean_cache_bytes_measured": _mean([r["memory_bytes"]["cache_after_conversion"] for r in rows]),
        "max_mlx_peak_bytes": max(r["memory_bytes"]["mlx_peak"] for r in rows),
    }


def bootstrap_means(values: list[float], clusters: list[str], n_resamples: int, seed: int) -> np.ndarray:
    """Means of `values` over cluster bootstrap resamples.

    Whole clusters (sources) are drawn with replacement, keeping every value of a
    drawn cluster together, so related examples are never split apart. Each
    resample's mean is over its values, matching the example-weighted point estimate.
    """
    ids = sorted(set(clusters))
    index = {c: i for i, c in enumerate(ids)}
    sums, counts = np.zeros(len(ids)), np.zeros(len(ids))
    for v, c in zip(values, clusters):
        sums[index[c]] += v
        counts[index[c]] += 1
    draws = np.random.default_rng(seed).integers(0, len(ids), size=(n_resamples, len(ids)))
    return sums[draws].sum(axis=1) / counts[draws].sum(axis=1)


def _ci(values: list[float], clusters: list[str]) -> list[float] | None:
    if len(set(clusters)) < 2:
        return None  # a single source gives no sampling distribution
    means = bootstrap_means(values, clusters, BOOTSTRAP_RESAMPLES, BOOTSTRAP_SEED)
    tail = (1 - BOOTSTRAP_CONFIDENCE) / 2 * 100
    lo, hi = np.percentile(means, [tail, 100 - tail])
    return [float(lo), float(hi)]


def bootstrap_p_value(values: list[float], clusters: list[str]) -> float | None:
    """Two-sided paired bootstrap p-value for mean(values) = 0: 2 x the smaller tail mass at 0."""
    if len(set(clusters)) < 2:
        return None
    means = bootstrap_means(values, clusters, BOOTSTRAP_RESAMPLES, BOOTSTRAP_SEED)
    return float(min(1.0, 2 * min(np.mean(means <= 0), np.mean(means >= 0))))


def mcnemar_exact(pairs: list[tuple[dict[str, Any], dict[str, Any]]]) -> dict[str, Any] | None:
    """Exact McNemar test, only when every score on both sides is binary (0 or 100)."""
    if not pairs or any(x["score"] not in (0.0, 100.0) for pair in pairs for x in pair):
        return None
    b = sum(m["score"] == 100.0 and r["score"] == 0.0 for m, r in pairs)  # method right, reference wrong
    c = sum(m["score"] == 0.0 and r["score"] == 100.0 for m, r in pairs)
    n = b + c
    p = 1.0 if n == 0 else min(1.0, 2 * sum(math.comb(n, i) for i in range(min(b, c) + 1)) / 2**n)
    return {"method_only_correct": b, "reference_only_correct": c, "p_value": p}


def holm(p_values: list[float | None]) -> list[float | None]:
    """Holm step-down adjusted p-values over one declared family (None entries are skipped)."""
    present = sorted((p, i) for i, p in enumerate(p_values) if p is not None)
    adjusted: list[float | None] = [None] * len(p_values)
    running = 0.0
    for rank, (p, i) in enumerate(present):
        running = max(running, min(1.0, (len(present) - rank) * p))
        adjusted[i] = running
    return adjusted


def _paired_cell(pairs: list[tuple[dict[str, Any], dict[str, Any]]], n_unpaired: int) -> dict[str, Any]:
    diffs = [m["score"] - r["score"] for m, r in pairs]
    has_text = all(m.get("prediction") is not None and r.get("prediction") is not None for m, r in pairs)
    flips = [float(m["prediction"] != r["prediction"]) for m, r in pairs] if has_text else []
    sources = [m["source_id"] for m, _ in pairs]
    stats = {
        "n_pairs": len(pairs),
        "n_sources": len(set(sources)),
        "n_excluded": n_unpaired,  # one or both sides not ok or unscored, or example missing
        "mean_score": _mean([m["score"] for m, _ in pairs]),
        "mean_score_reference": _mean([r["score"] for _, r in pairs]),
        "mean_diff": _mean(diffs),
        "mean_diff_ci": _ci(diffs, sources),
        "p_value_bootstrap": bootstrap_p_value(diffs, sources),
        "flip_rate": _mean(flips),
        "flip_rate_ci": _ci(flips, sources) if flips else None,
        "mcnemar": mcnemar_exact(pairs),
    }
    if pairs and all(m.get("nll_sum") is not None for m, _ in pairs):
        ppl_m, ppl_r = _perplexity([m for m, _ in pairs]), _perplexity([r for _, r in pairs])
        stats.update(perplexity=ppl_m, perplexity_reference=ppl_r, perplexity_ratio=ppl_m / ppl_r)
    return stats


def paired_vs_reference(method: list[dict[str, Any]], reference: list[dict[str, Any]]) -> dict[str, Any]:
    """Paired differences (method - reference) on the same examples.

    Scopes, all per benchmark and weighted by example (never a mean of cell means):
    `cells` task x context_length; `by_task` all lengths of a task; `by_length` all
    tasks at a length; `pooled` every pair of the benchmark. Intervals resample source_id.
    """
    ref_by_id = {r["example_id"]: r for r in latest_attempts(reference)}
    groups: dict[tuple, list] = defaultdict(list)
    excluded: dict[tuple, int] = defaultdict(int)
    for m in latest_attempts(method):
        b, t, length = m["benchmark"], m["task"], m["context_length"]
        r = ref_by_id.get(m["example_id"])
        for key in (("cell", b, t, length), ("task", b, t), ("length", b, length), ("pooled", b)):
            if r is not None and m["status"] == r["status"] == "ok" and None not in (m["score"], r["score"]):
                groups[key].append((m, r))
            else:
                excluded[key] += 1

    out: dict[str, list] = {"cells": [], "by_task": [], "by_length": [], "pooled": []}
    for key in sorted(set(groups) | set(excluded), key=str):
        stats = _paired_cell(groups.get(key, []), excluded.get(key, 0))
        scope, b = key[0], key[1]
        if scope == "cell":
            out["cells"].append({"scope": "cell", "benchmark": b, "task": key[2], "context_length": key[3], **stats})
        elif scope == "task":
            out["by_task"].append({"scope": "task", "benchmark": b, "task": key[2], **stats})
        elif scope == "length":
            out["by_length"].append({"scope": "length", "benchmark": b, "context_length": key[2], **stats})
        else:
            out["pooled"].append({"scope": "pooled", "benchmark": b, **stats})
    for rows in out.values():
        rows.sort(key=lambda c: (c["benchmark"], c.get("task", ""), c.get("context_length", 0)))
    return out


def _elements_per_token(manifest: dict[str, Any]) -> int | None:
    arch = manifest.get("model", {}).get("arch")
    return arch["num_layers"] * arch["num_kv_heads"] * arch["head_dim"] * 2 if arch else None


def summarize_run(run_dir: Path) -> dict[str, Any]:
    """Write methods/<id>/summary.json and comparisons/summary.json for a run."""
    with open(run_dir / "manifest.json", encoding="utf-8") as f:
        manifest = json.load(f)
    with open(run_dir / "resolved_config.yaml", encoding="utf-8") as f:
        kinds = {m["id"]: m["kind"] for m in yaml.safe_load(f)["methods"]}

    records, per_method = {}, {}
    for method_id, rel_path in manifest["methods"].items():
        path = run_dir / rel_path
        records[method_id] = read_jsonl(path) if path.exists() else []
        summary = {
            "run_id": manifest["run_id"],
            "method_id": method_id,
            **summarize_records(records[method_id], _elements_per_token(manifest)),
        }
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path.parent / "summary.json", "w", encoding="utf-8") as f:
            json.dump(summary, f, indent=2)
        per_method[method_id] = summary["cells"]

    comparison: dict[str, Any] = {
        "run_id": manifest["run_id"],
        "config_hash": manifest["config_hash"],
        "metrics": METRICS,
        "methods": per_method,
    }
    identity = [m for m, k in kinds.items() if k == "identity"]
    if len(identity) == 1:
        ref = identity[0]
        comparison["paired_vs_reference"] = {
            "reference": ref,
            "bootstrap": {
                "unit": "source_id",
                "interval": "percentile",
                "confidence": BOOTSTRAP_CONFIDENCE,
                "n_resamples": BOOTSTRAP_RESAMPLES,
                "seed": BOOTSTRAP_SEED,
                "pooled_weighting": "per example",
            },
            "methods": {m: paired_vs_reference(records[m], records[ref]) for m in records if m != ref},
        }
    (run_dir / "comparisons").mkdir(exist_ok=True)
    with open(run_dir / "comparisons" / "summary.json", "w", encoding="utf-8") as f:
        json.dump(comparison, f, indent=2)
    return comparison


def compare_declared(cfg: Any) -> dict[str, Any]:
    """Run the comparisons a ComparisonConfig names and write comparisons/<name>.json."""
    run_dir = Path(cfg.run_dir)
    with open(run_dir / "manifest.json", encoding="utf-8") as f:
        manifest = json.load(f)
    missing = {x for c in cfg.comparisons for x in (c.method, c.against)} - set(manifest["methods"])
    if missing:
        raise ValueError(f"methods not in run {run_dir}: {sorted(missing)}")
    records = {m: read_jsonl(run_dir / rel) for m, rel in manifest["methods"].items() if (run_dir / rel).exists()}

    results = []
    for c in cfg.comparisons:
        results.append({"method": c.method, "against": c.against, **paired_vs_reference(records[c.method], records[c.against])})
    if cfg.holm:
        benchmarks = sorted({row["benchmark"] for r in results for row in r["pooled"]})
        for b in benchmarks:
            rows = [row for r in results for row in r["pooled"] if row["benchmark"] == b]
            for row, adj in zip(rows, holm([row["p_value_bootstrap"] for row in rows])):
                row["p_value_holm"] = adj
    out = {
        "name": cfg.name,
        "run_id": manifest["run_id"],
        "config_hash": manifest["config_hash"],
        "metrics": METRICS,
        "bootstrap": {"unit": "source_id", "n_resamples": BOOTSTRAP_RESAMPLES, "seed": BOOTSTRAP_SEED},
        "holm": {"applied": cfg.holm, "family": "listed comparisons, pooled scope, per benchmark"},
        "comparisons": results,
    }
    (run_dir / "comparisons").mkdir(exist_ok=True)
    with open(run_dir / "comparisons" / f"{cfg.name}.json", "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2)
    return out
