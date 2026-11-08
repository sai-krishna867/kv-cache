"""Summaries computed from saved per-example results; never loads a model."""

from __future__ import annotations

import json
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


def summarize_records(records: list[dict[str, Any]]) -> dict[str, Any]:
    """Per benchmark x task x context_length aggregates.

    Only ok records enter means; failed and oom records are counted, never scored.
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
                "mean_score": _mean([r["score"] for r in ok]),
                "mean_n_input_tokens": _mean([r["n_input_tokens"] for r in ok]),
                "mean_n_context_tokens": _mean([r["n_context_tokens"] for r in ok]),
                "mean_n_generated_tokens": _mean([r["n_generated_tokens"] for r in ok]),
                "mean_cache_bytes_estimated": _mean([r["cache_bytes_estimated"] for r in ok]),
                "mean_cache_bytes_estimated_final": _mean([r["cache_bytes_estimated_final"] for r in ok if r.get("cache_bytes_estimated_final") is not None]),
                "cache_bytes_labels": sorted({r["cache_bytes_label"] for r in ok}),
            }
        )
    return {"n_superseded_attempts": len(records) - len(current), "cells": out}


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


def _paired_cell(pairs: list[tuple[dict[str, Any], dict[str, Any]]], n_unpaired: int) -> dict[str, Any]:
    diffs = [m["score"] - r["score"] for m, r in pairs]
    flips = [float(m["prediction"] != r["prediction"]) for m, r in pairs]
    sources = [m["source_id"] for m, _ in pairs]
    return {
        "n_pairs": len(pairs),
        "n_sources": len(set(sources)),
        "n_excluded": n_unpaired,  # one or both sides not ok, or example missing
        "mean_score": _mean([m["score"] for m, _ in pairs]),
        "mean_score_reference": _mean([r["score"] for _, r in pairs]),
        "mean_diff": _mean(diffs),
        "mean_diff_ci": _ci(diffs, sources),
        "flip_rate": _mean(flips),
        "flip_rate_ci": _ci(flips, sources),
    }


def paired_vs_reference(method: list[dict[str, Any]], reference: list[dict[str, Any]]) -> dict[str, Any]:
    """Paired differences (method - reference) on the same examples, per cell and per benchmark.

    `cells` are task x context_length; `pooled` is every pair of a benchmark,
    weighted by example (not a mean of cell means). Intervals resample source_id.
    """
    ref_by_id = {r["example_id"]: r for r in latest_attempts(reference)}
    groups: dict[tuple, list] = defaultdict(list)
    excluded: dict[tuple, int] = defaultdict(int)
    for m in latest_attempts(method):
        cell = (m["benchmark"], m["task"], m["context_length"])
        r = ref_by_id.get(m["example_id"])
        for key in (cell, (m["benchmark"],)):
            if r is not None and m["status"] == "ok" and r["status"] == "ok":
                groups[key].append((m, r))
            else:
                excluded[key] += 1

    cells, pooled = [], []
    for key in sorted(set(groups) | set(excluded), key=lambda k: (len(k), k)):
        stats = _paired_cell(groups.get(key, []), excluded.get(key, 0))
        if len(key) == 3:
            cells.append({"scope": "cell", "benchmark": key[0], "task": key[1], "context_length": key[2], **stats})
        else:
            pooled.append({"scope": "pooled", "benchmark": key[0], **stats})
    return {"cells": cells, "pooled": pooled}


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
        summary = {"run_id": manifest["run_id"], "method_id": method_id, **summarize_records(records[method_id])}
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path.parent / "summary.json", "w", encoding="utf-8") as f:
            json.dump(summary, f, indent=2)
        per_method[method_id] = summary["cells"]

    comparison: dict[str, Any] = {"run_id": manifest["run_id"], "config_hash": manifest["config_hash"], "methods": per_method}
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
