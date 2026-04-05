from __future__ import annotations

from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Sequence

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def _ensure_parent(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)


def _safe_values(items: Sequence[object]) -> np.ndarray:
    values: List[float] = []
    for item in items:
        if isinstance(item, (int, float)):
            val = float(item)
            if np.isfinite(val):
                values.append(val)
    return np.asarray(values, dtype=np.float64)


def plot_live_run_progress(progress_events: Sequence[Dict[str, object]], output_png: Path, title: str) -> None:
    _ensure_parent(output_png)

    generation_x: List[int] = []
    generation_best: List[float] = []
    generation_mean: List[float] = []

    evaluated = 0
    compiled = 0
    compiled_curve_x: List[int] = []
    compiled_curve_y: List[float] = []

    for event in progress_events:
        event_name = str(event.get("event", ""))

        if event_name == "generation_completed":
            generation = event.get("generation")
            best = event.get("best_fitness")
            mean = event.get("population_mean_fitness")
            if isinstance(generation, int) and isinstance(best, (int, float)) and isinstance(mean, (int, float)):
                generation_x.append(generation)
                generation_best.append(float(best))
                generation_mean.append(float(mean))

        if event_name in {"bootstrap_candidate_evaluated", "offspring_evaluated"}:
            evaluated += 1
            if bool(event.get("compiled", False)):
                compiled += 1
            compiled_curve_x.append(evaluated)
            compiled_curve_y.append(compiled / max(1, evaluated))

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    if generation_x:
        axes[0].plot(generation_x, generation_best, marker="o", label="Best fitness")
        axes[0].plot(generation_x, generation_mean, marker="x", label="Population mean fitness")
    axes[0].set_title("Generation fitness")
    axes[0].set_xlabel("Generation")
    axes[0].set_ylabel("Fitness")
    axes[0].grid(True, alpha=0.3)
    axes[0].legend(loc="best")

    if compiled_curve_x:
        axes[1].plot(compiled_curve_x, compiled_curve_y, color="tab:green", marker=".")
    axes[1].set_title("Compile success over evaluated candidates")
    axes[1].set_xlabel("Evaluated candidate count")
    axes[1].set_ylabel("Compile success rate")
    axes[1].set_ylim(0.0, 1.0)
    axes[1].grid(True, alpha=0.3)

    fig.suptitle(title)
    fig.tight_layout()
    fig.savefig(str(output_png), dpi=160)
    plt.close(fig)


def plot_overall_progress(run_records: Sequence[Dict[str, object]], output_png: Path) -> None:
    _ensure_parent(output_png)
    sorted_records = sorted(
        run_records,
        key=lambda row: str(row.get("started_at", "")),
    )

    algorithms = sorted({str(row.get("algorithm", "unknown")) for row in sorted_records})
    completed_counts = {alg: [] for alg in algorithms}
    x = list(range(1, len(sorted_records) + 1))

    cumulative = {alg: 0 for alg in algorithms}
    for record in sorted_records:
        algorithm = str(record.get("algorithm", "unknown"))
        if str(record.get("status", "")) == "success":
            cumulative[algorithm] = cumulative.get(algorithm, 0) + 1
        for alg in algorithms:
            completed_counts[alg].append(cumulative.get(alg, 0))

    fig, ax = plt.subplots(figsize=(10, 5))
    for alg in algorithms:
        ax.step(x, completed_counts[alg], where="post", label=alg)

    ax.set_title("Experiment progress: successful runs over time")
    ax.set_xlabel("Run launch order")
    ax.set_ylabel("Cumulative successful runs")
    ax.grid(True, alpha=0.3)
    ax.legend(loc="best")
    fig.tight_layout()
    fig.savefig(str(output_png), dpi=160)
    plt.close(fig)


def plot_metric_distributions(
    run_metrics: Sequence[Dict[str, object]],
    metric: str,
    output_png: Path,
    ylabel: str,
) -> None:
    _ensure_parent(output_png)
    grouped: Dict[str, List[float]] = defaultdict(list)

    for record in run_metrics:
        if str(record.get("status", "")) != "success":
            continue
        algorithm = str(record.get("algorithm", "unknown"))
        value = record.get(metric)
        if isinstance(value, (int, float)) and np.isfinite(float(value)):
            grouped[algorithm].append(float(value))

    algorithms = sorted(grouped.keys())
    if not algorithms:
        return

    data = [_safe_values(grouped[algorithm]) for algorithm in algorithms]

    fig, ax = plt.subplots(figsize=(10, 5))
    box = ax.boxplot(data, patch_artist=True, labels=algorithms, showmeans=True)
    colors = ["#4C72B0", "#55A868", "#C44E52", "#8172B2"]
    for patch, color in zip(box["boxes"], colors * 5):
        set_facecolor = getattr(patch, "set_facecolor", None)
        if callable(set_facecolor):
            set_facecolor(color)
        set_alpha = getattr(patch, "set_alpha", None)
        if callable(set_alpha):
            set_alpha(0.45)

    for idx, algorithm in enumerate(algorithms, start=1):
        jitter = np.random.uniform(-0.08, 0.08, size=len(grouped[algorithm]))
        ax.scatter(np.full(len(grouped[algorithm]), idx) + jitter, grouped[algorithm], s=24, alpha=0.8)

    ax.set_title(f"Distribution of {metric}")
    ax.set_ylabel(ylabel)
    ax.grid(True, axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(str(output_png), dpi=160)
    plt.close(fig)


def plot_convergence_by_algorithm(run_metrics: Sequence[Dict[str, object]], output_png: Path) -> None:
    _ensure_parent(output_png)

    grouped_curves: Dict[str, List[np.ndarray]] = defaultdict(list)
    for record in run_metrics:
        if str(record.get("status", "")) != "success":
            continue
        algorithm = str(record.get("algorithm", "unknown"))
        history_raw = record.get("history_best_fitness", [])
        history_values = history_raw if isinstance(history_raw, list) else []
        curve = _safe_values(history_values)
        if len(curve) > 0:
            grouped_curves[algorithm].append(curve)

    fig, ax = plt.subplots(figsize=(10, 5))
    for algorithm, curves in sorted(grouped_curves.items()):
        max_len = max(len(curve) for curve in curves)
        padded = np.full((len(curves), max_len), np.nan, dtype=np.float64)
        for idx, curve in enumerate(curves):
            padded[idx, : len(curve)] = curve

        mean_curve = np.nanmean(padded, axis=0)
        std_curve = np.nanstd(padded, axis=0)
        x = np.arange(len(mean_curve))
        ax.plot(x, mean_curve, label=f"{algorithm} mean")
        ax.fill_between(x, mean_curve - std_curve, mean_curve + std_curve, alpha=0.2)

    ax.set_title("Best fitness convergence (mean ± std across seeds)")
    ax.set_xlabel("Generation")
    ax.set_ylabel("Best fitness")
    ax.grid(True, alpha=0.3)
    ax.legend(loc="best")
    fig.tight_layout()
    fig.savefig(str(output_png), dpi=160)
    plt.close(fig)


def plot_run_comparison_scatter(run_metrics: Sequence[Dict[str, object]], output_png: Path) -> None:
    _ensure_parent(output_png)

    fig, ax = plt.subplots(figsize=(10, 5))
    for record in run_metrics:
        if str(record.get("status", "")) != "success":
            continue
        algorithm = str(record.get("algorithm", "unknown"))
        x = record.get("compile_success_rate")
        y = record.get("best_quant_acc1")
        if isinstance(x, (int, float)) and isinstance(y, (int, float)):
            ax.scatter(float(x), float(y), label=algorithm, alpha=0.8)

    handles, labels = ax.get_legend_handles_labels()
    unique = {}
    for handle, label in zip(handles, labels):
        if label not in unique:
            unique[label] = handle

    ax.legend(unique.values(), unique.keys(), loc="best")
    ax.set_title("Run-level tradeoff: compile success vs best quantized accuracy")
    ax.set_xlabel("Compile success rate")
    ax.set_ylabel("Best quantized accuracy (acc1)")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(str(output_png), dpi=160)
    plt.close(fig)


def plot_statistical_pvalues(statistics_payload: Dict[str, object], output_png: Path) -> None:
    _ensure_parent(output_png)

    tests: List[Dict[str, object]] = []
    pairwise = statistics_payload.get("pairwise_results", [])
    if isinstance(pairwise, list) and pairwise:
        first = pairwise[0]
        if isinstance(first, dict) and isinstance(first.get("tests", []), list):
            tests = first["tests"]

    if not tests:
        return

    metrics = [str(test.get("metric", "")) for test in tests]
    pvals = []
    for test in tests:
        value = test.get("p_value_holm_bonferroni", test.get("p_value", np.nan))
        pvals.append(float(value) if isinstance(value, (int, float)) else np.nan)

    fig, ax = plt.subplots(figsize=(12, 5))
    ax.bar(metrics, pvals, color="#4C72B0", alpha=0.8)
    ax.axhline(0.05, color="red", linestyle="--", linewidth=1.0, label="alpha=0.05")
    ax.set_yscale("log")
    ax.set_ylabel("Adjusted p-value (log scale)")
    ax.set_title("Statistical significance summary")
    ax.tick_params(axis="x", rotation=30)
    ax.legend(loc="best")
    ax.grid(True, axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(str(output_png), dpi=160)
    plt.close(fig)


def plot_effect_sizes(statistics_payload: Dict[str, object], output_png: Path) -> None:
    _ensure_parent(output_png)

    tests: List[Dict[str, object]] = []
    pairwise = statistics_payload.get("pairwise_results", [])
    if isinstance(pairwise, list) and pairwise:
        first = pairwise[0]
        if isinstance(first, dict) and isinstance(first.get("tests", []), list):
            tests = first["tests"]

    if not tests:
        return

    metrics = [str(test.get("metric", "")) for test in tests]
    effects = []
    for test in tests:
        value = test.get("effect_size", np.nan)
        effects.append(float(value) if isinstance(value, (int, float)) else np.nan)

    y = np.arange(len(metrics))
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.axvline(0.0, color="black", linewidth=1.0)
    ax.barh(y, effects, color="#55A868", alpha=0.8)
    ax.set_yticks(y)
    ax.set_yticklabels(metrics)
    ax.set_xlabel("Effect size")
    ax.set_title("Effect sizes by metric")
    ax.grid(True, axis="x", alpha=0.3)
    fig.tight_layout()
    fig.savefig(str(output_png), dpi=160)
    plt.close(fig)
