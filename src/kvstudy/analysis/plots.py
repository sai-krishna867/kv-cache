"""Figures from saved summaries; never loads a model. Needs the `plots` dependency group.

One quality-memory figure per benchmark (metrics are never combined) and, for
systems_real runs, a decode-speed and measured-memory figure. Written to
<run>/comparisons/figures/.
"""

from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path
from typing import Any


def _load(run_dir: Path) -> tuple[dict[str, Any], dict[str, Any]]:
    comparison = json.loads((run_dir / "comparisons" / "summary.json").read_text())
    summaries = {}
    for method_id in comparison["methods"]:
        path = run_dir / "methods" / method_id / "summary.json"
        summaries[method_id] = json.loads(path.read_text())
    return comparison, summaries


def _pooled_bits(cells: list[dict[str, Any]], benchmark: str) -> float | None:
    """Example-weighted mean effective bits of a method on one benchmark."""
    rows = [c for c in cells if c["benchmark"] == benchmark and c.get("mean_effective_bits") is not None]
    n = sum(c["n_ok"] for c in rows)
    return sum(c["mean_effective_bits"] * c["n_ok"] for c in rows) / n if n else None


def quality_memory(run_dir: Path) -> list[Path]:
    """Per benchmark: paired difference vs the BF16 reference (95% CI) against effective bits."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    comparison, _ = _load(run_dir)
    paired = comparison.get("paired_vs_reference")
    if not paired or not paired["methods"]:
        return []
    out_dir = run_dir / "comparisons" / "figures"
    out_dir.mkdir(parents=True, exist_ok=True)
    written = []
    benchmarks = sorted({row["benchmark"] for r in paired["methods"].values() for row in r["pooled"]})
    for b in benchmarks:
        metric = comparison["metrics"][b]
        fig, ax = plt.subplots(figsize=(7, 4.5))
        plotted = 0
        for method_id, result in paired["methods"].items():
            [row] = [r for r in result["pooled"] if r["benchmark"] == b]
            bits = _pooled_bits(comparison["methods"][method_id], b)
            if row["mean_diff"] is None or bits is None:
                continue
            ci = row["mean_diff_ci"] or [row["mean_diff"], row["mean_diff"]]
            kind = "o" if method_id.startswith("U") else "s"
            ax.errorbar(bits, row["mean_diff"], yerr=[[row["mean_diff"] - ci[0]], [ci[1] - row["mean_diff"]]], fmt=kind, capsize=3)
            ax.annotate(method_id, (bits, row["mean_diff"]), textcoords="offset points", xytext=(5, 4), fontsize=8)
            plotted += 1
        if not plotted:  # e.g. systems_real runs carry no scores
            plt.close(fig)
            continue
        ax.axhline(0, color="grey", lw=0.8)
        ax.set_xlabel("effective bits per cached element (estimated, context)")
        direction = "higher is better" if metric["higher_is_better"] else "lower is better"
        ax.set_ylabel(f"{metric['metric']}\nmethod - {paired['reference']} ({direction})")
        ax.set_title(f"{b}: paired difference vs {paired['reference']}, 95% {paired['bootstrap']['unit']} bootstrap CI")
        fig.tight_layout()
        path = out_dir / f"{b}_quality_memory.png"
        fig.savefig(path, dpi=150)
        plt.close(fig)
        written.append(path)
    return written


def systems(run_dir: Path) -> list[Path]:
    """systems_real: decode tokens/s and measured cache MiB per method and context length."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    _, summaries = _load(run_dir)
    by_method: dict[str, dict[int, dict[str, Any]]] = defaultdict(dict)
    for method_id, summary in summaries.items():
        for c in summary["cells"]:
            if c.get("median_decode_tokens_per_s") is not None:
                by_method[method_id][c["context_length"]] = c
    if not by_method:
        return []
    out_dir = run_dir / "comparisons" / "figures"
    out_dir.mkdir(parents=True, exist_ok=True)
    fig, (speed, memory) = plt.subplots(1, 2, figsize=(11, 4.5))
    for method_id, cells in by_method.items():
        lengths = sorted(cells)
        speed.plot(lengths, [cells[n]["median_decode_tokens_per_s"] for n in lengths], marker="o", label=method_id)
        memory.plot(lengths, [cells[n]["mean_cache_bytes_measured"] / 2**20 for n in lengths], marker="o", label=method_id)
    speed.set(xlabel="context length (tokens)", ylabel="decode tokens/s (median)", title="Decode speed, systems_real")
    memory.set(xlabel="context length (tokens)", ylabel="cache MiB (measured buffers)", title="KV cache after conversion")
    for ax in (speed, memory):
        ax.set_xscale("log", base=2)
        ax.legend(fontsize=8)
    fig.tight_layout()
    path = out_dir / "systems_latency_memory.png"
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return [path]


def plot_run(run_dir: Path) -> list[Path]:
    return quality_memory(run_dir) + systems(run_dir)
