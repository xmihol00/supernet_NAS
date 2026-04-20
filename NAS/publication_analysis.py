#!/usr/bin/env python3
"""
Comprehensive publication analysis for NAS multi-run parallel experiment.
Generates statistical analyses and plots suitable for academic publication.

Usage:
    python publication_analysis.py --sga-dir <path> --reg-evo-dir <path> --output-dir <path>
"""

import argparse
import json
import os
import glob
import math
import sys
from collections import defaultdict
from datetime import datetime

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.lines import Line2D
from scipy import stats

# ─── Colour palette (colorblind-friendly) ────────────────────────────────────
COLORS = {
    "baseline_sga": "#0072B2",       # blue
    "regularized_evolution": "#D55E00",  # orange-red
}
LABELS = {
    "baseline_sga": "Baseline SGA",
    "regularized_evolution": "Regularized Evolution",
}

ALG_NAMES = ["regularized_evolution", "baseline_sga"]

# ─── Data loading helpers ─────────────────────────────────────────────────────

def load_run_records(exp_dir: str) -> list[dict]:
    """Load run_records.json and flatten summary fields to top level."""
    path = os.path.join(exp_dir, "run_records.json")
    with open(path) as f:
        records = json.load(f)
    # Flatten: merge summary dict into top-level record
    flat = []
    for r in records:
        entry = {k: v for k, v in r.items() if k not in ("summary", "history", "progress_events_count", "command")}
        summary = r.get("summary") or {}
        for k, v in summary.items():
            if k not in entry:
                entry[k] = v
        flat.append(entry)
    return flat


def load_all_histories(exp_dir: str, alg_name: str) -> list[list[dict]]:
    """Return list of per-run histories from run_records.json (history field)."""
    path = os.path.join(exp_dir, "run_records.json")
    with open(path) as f:
        records = json.load(f)
    histories = []
    for r in records:
        if r.get("status") == "success" and r.get("history"):
            histories.append(r["history"])
    return histories


def load_all_summaries(exp_dir: str) -> list[dict]:
    """Load summaries from run_records.json summary fields."""
    path = os.path.join(exp_dir, "run_records.json")
    with open(path) as f:
        records = json.load(f)
    summaries = []
    for r in records:
        if r.get("status") == "success" and r.get("summary"):
            summaries.append(r["summary"])
    return summaries


def load_population_snapshot(exp_dir: str, run_glob: str, gen: int) -> list[dict]:
    """Load population_gen_XXX.json for a specific run and generation."""
    pattern = os.path.join(exp_dir, "raw_runs", run_glob, "*",
                           f"population_gen_{gen:03d}.json")
    results = []
    for pf in sorted(glob.glob(pattern)):
        with open(pf) as f:
            results.append(json.load(f))
    return results


# ─── Statistical helpers ──────────────────────────────────────────────────────

def cliffs_delta(a: list[float], b: list[float]) -> float:
    """Cliff's delta effect size (a > b positive)."""
    n, m = len(a), len(b)
    count = sum(1 if ai > bj else (-1 if ai < bj else 0)
                for ai in a for bj in b)
    return count / (n * m)


def cohens_d(a: list[float], b: list[float]) -> float:
    na, nb = len(a), len(b)
    pooled = math.sqrt(((na - 1) * np.std(a, ddof=1) ** 2 +
                        (nb - 1) * np.std(b, ddof=1) ** 2) / (na + nb - 2))
    if pooled == 0:
        return 0.0
    return (np.mean(a) - np.mean(b)) / pooled


def bootstrap_ci(a: list[float], b: list[float], n_boot=10000,
                 confidence=0.95, rng=None) -> tuple[float, float]:
    """Bootstrap CI for mean(a) - mean(b)."""
    if rng is None:
        rng = np.random.default_rng(42)
    diffs = []
    for _ in range(n_boot):
        sa = rng.choice(a, size=len(a), replace=True)
        sb = rng.choice(b, size=len(b), replace=True)
        diffs.append(np.mean(sa) - np.mean(sb))
    alpha = 1 - confidence
    return np.percentile(diffs, 100 * alpha / 2), np.percentile(diffs, 100 * (1 - alpha / 2))


def shapiro_wilk_p(x: list[float]) -> float:
    if len(x) < 3:
        return float("nan")
    _, p = stats.shapiro(x)
    return p


def effect_magnitude(d: float, kind: str = "cohens_d") -> str:
    """Interpret effect size magnitude."""
    if kind == "cohens_d":
        if abs(d) < 0.2:
            return "negligible"
        if abs(d) < 0.5:
            return "small"
        if abs(d) < 0.8:
            return "medium"
        return "large"
    else:  # cliffs_delta
        if abs(d) < 0.147:
            return "negligible"
        if abs(d) < 0.33:
            return "small"
        if abs(d) < 0.474:
            return "medium"
        return "large"


# ─── Analysis functions ───────────────────────────────────────────────────────

def compute_convergence_stats(histories: list[list[dict]]) -> dict:
    """Per-generation mean, std, min, max of best_fitness (ignoring NaN/failed)."""
    n_gens = max(len(h) for h in histories)
    gen_vals = defaultdict(list)
    for hist in histories:
        for g in hist:
            val = g["best_fitness"]
            if val is not None and val > -1e8:
                gen_vals[g["generation"]].append(val)

    result = {}
    for gen in range(n_gens):
        vals = gen_vals.get(gen, [])
        if vals:
            result[gen] = {
                "mean": np.mean(vals),
                "std": np.std(vals, ddof=1) if len(vals) > 1 else 0.0,
                "median": np.median(vals),
                "min": np.min(vals),
                "max": np.max(vals),
                "n": len(vals),
            }
    return result


def compute_pop_mean_stats(histories: list[list[dict]]) -> dict:
    """Per-generation mean of population_mean_fitness, skipping non-compiled penalty."""
    n_gens = max(len(h) for h in histories)
    gen_vals = defaultdict(list)
    for hist in histories:
        for g in hist:
            val = g.get("population_mean_fitness")
            if val is not None and val > -1e6:
                gen_vals[g["generation"]].append(val)
    result = {}
    for gen in range(n_gens):
        vals = gen_vals.get(gen, [])
        if vals:
            result[gen] = {
                "mean": np.mean(vals),
                "std": np.std(vals, ddof=1) if len(vals) > 1 else 0.0,
                "n": len(vals),
            }
    return result


def compute_auc(best_vals: list[float]) -> float:
    """Trapezoidal AUC of the best-fitness-per-generation curve."""
    valid = [v for v in best_vals if v is not None and v > -1e8]
    if not valid:
        return float("nan")
    return float(np.trapezoid(valid) / len(valid))


def compute_generations_to_threshold(histories: list[list[dict]],
                                      threshold: float) -> list[int | None]:
    """For each run, the first generation where best_fitness >= threshold."""
    results = []
    for hist in histories:
        found = None
        for g in hist:
            val = g["best_fitness"]
            if val is not None and val >= threshold:
                found = g["generation"]
                break
        results.append(found)
    return results


def compute_search_efficiency(records: list[dict]) -> dict:
    """Accuracy-per-candidate-evaluated ratio."""
    effs = {}
    for rec in records:
        if rec.get("status") == "success" and rec.get("best_quant_acc1") is not None:
            alg = rec["algorithm"]
            eff = rec["best_quant_acc1"] / rec["total_candidates_evaluated"]
            effs.setdefault(alg, []).append(eff)
    return effs


def compute_architecture_stats(summaries: list[dict]) -> dict:
    """Distribution of architecture parameters across best architectures."""
    resolutions, stem_widths = [], []
    stage_depths = [[], [], [], []]
    stage_widths = [[], [], [], []]
    for s in summaries:
        bc = s.get("best_config", {})
        if not bc:
            continue
        resolutions.append(bc.get("resolution"))
        stem_widths.append(bc.get("stem_width"))
        for i, v in enumerate(bc.get("stage_depths", [])):
            stage_depths[i].append(v)
        for i, v in enumerate(bc.get("stage_widths", [])):
            stage_widths[i].append(v)
    return {
        "resolutions": resolutions,
        "stem_widths": stem_widths,
        "stage_depths": stage_depths,
        "stage_widths": stage_widths,
    }


# ─── Plotting functions ───────────────────────────────────────────────────────

def savefig(fig, path: str):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {path}")


def plot_combined_convergence(conv_stats: dict[str, dict], output_dir: str):
    """Mean best fitness ± 1σ per generation for both algorithms."""
    fig, ax = plt.subplots(figsize=(8, 5))
    for alg in ALG_NAMES:
        cs = conv_stats[alg]
        gens = sorted(cs.keys())
        means = [cs[g]["mean"] for g in gens]
        stds = [cs[g]["std"] for g in gens]
        lo = [m - s for m, s in zip(means, stds)]
        hi = [m + s for m, s in zip(means, stds)]
        color = COLORS[alg]
        ax.plot(gens, means, color=color, label=LABELS[alg], lw=2, marker="o", ms=4)
        ax.fill_between(gens, lo, hi, alpha=0.18, color=color)

    ax.set_xlabel("Generation", fontsize=12)
    ax.set_ylabel("Best Quantized Accuracy (%)", fontsize=12)
    ax.set_title("Convergence: Best Fitness per Generation\n(Mean ± 1σ across independent runs)", fontsize=12)
    ax.legend(fontsize=11)
    ax.set_xlim(-0.5, 24.5)
    ax.yaxis.grid(True, alpha=0.3)
    ax.set_axisbelow(True)
    savefig(fig, os.path.join(output_dir, "combined_convergence.png"))


def plot_all_run_trajectories(data_by_alg: dict[str, list[list[dict]]], output_dir: str):
    """Individual run trajectories for each algorithm."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 5), sharey=True)
    for ax, alg in zip(axes, ALG_NAMES):
        color = COLORS[alg]
        histories = data_by_alg[alg]
        for idx, hist in enumerate(histories):
            gens = [g["generation"] for g in hist]
            vals = [g["best_fitness"] for g in hist]
            # Replace penalty values with NaN
            vals = [v if v is not None and v > -1e8 else float("nan") for v in vals]
            ax.plot(gens, vals, color=color, alpha=0.45, lw=1.5, label=f"Run {idx}")
        # Mean
        cs = compute_convergence_stats(histories)
        gens_s = sorted(cs.keys())
        means = [cs[g]["mean"] for g in gens_s]
        ax.plot(gens_s, means, color="black", lw=2.5, linestyle="--", label="Mean")
        ax.set_title(LABELS[alg], fontsize=12)
        ax.set_xlabel("Generation", fontsize=11)
        ax.yaxis.grid(True, alpha=0.3)
        ax.set_axisbelow(True)
    axes[0].set_ylabel("Best Quantized Accuracy (%)", fontsize=11)
    fig.suptitle("Individual Run Convergence Trajectories", fontsize=13, y=1.02)
    handles = [
        Line2D([0], [0], color=COLORS[ALG_NAMES[0]], lw=1.5, alpha=0.7, label="Reg-Evo runs"),
        Line2D([0], [0], color=COLORS[ALG_NAMES[1]], lw=1.5, alpha=0.7, label="SGA runs"),
        Line2D([0], [0], color="black", lw=2.5, linestyle="--", label="Mean"),
    ]
    fig.legend(handles=handles, loc="upper right", fontsize=10)
    savefig(fig, os.path.join(output_dir, "individual_run_trajectories.png"))


def plot_pop_mean_evolution(pop_stats: dict[str, dict], conv_stats: dict[str, dict], output_dir: str):
    """Population mean fitness evolution (SGA only; RegEvo omitted since age-based selection includes uncompiled)."""
    fig, ax = plt.subplots(figsize=(8, 5))
    alg = "baseline_sga"
    ps = pop_stats[alg]
    gens_s = sorted(ps.keys())
    means = [ps[g]["mean"] for g in gens_s]
    stds = [ps[g]["std"] for g in gens_s]
    color = COLORS[alg]
    ax.plot(gens_s, means, color=color, lw=2, label="Pop. Mean (SGA)", marker="s", ms=4)
    ax.fill_between(gens_s,
                    [m - s for m, s in zip(means, stds)],
                    [m + s for m, s in zip(means, stds)],
                    alpha=0.2, color=color)
    # overlay best fitness
    cs = conv_stats[alg]
    gens_b = sorted(cs.keys())
    best_means = [cs[g]["mean"] for g in gens_b]
    ax.plot(gens_b, best_means, color=color, lw=2, linestyle="--", label="Best Fitness (SGA)", marker="o", ms=4)

    ax.set_xlabel("Generation", fontsize=12)
    ax.set_ylabel("Quantized Accuracy (%)", fontsize=12)
    ax.set_title("SGA Population Mean vs. Best Fitness\n(Mean ± 1σ, compiled candidates only)", fontsize=12)
    ax.legend(fontsize=11)
    ax.yaxis.grid(True, alpha=0.3)
    ax.set_axisbelow(True)
    ax.set_xlim(-0.5, 24.5)
    savefig(fig, os.path.join(output_dir, "sga_population_evolution.png"))


def plot_violin_distributions(records_by_alg: dict[str, list[float]],
                               metric: str, ylabel: str, title: str, output_dir: str):
    """Violin + box + scatter for a given metric."""
    fig, ax = plt.subplots(figsize=(7, 5))
    positions = list(range(1, len(ALG_NAMES) + 1))
    parts = ax.violinplot([records_by_alg[a] for a in ALG_NAMES],
                          positions=positions, showmedians=True, showextrema=True)
    for body, alg in zip(parts["bodies"], ALG_NAMES):
        body.set_facecolor(COLORS[alg])
        body.set_alpha(0.5)
    for comp in ["cmedians", "cmins", "cmaxes", "cbars"]:
        if comp in parts:
            parts[comp].set_color("black")

    for pos, alg in zip(positions, ALG_NAMES):
        vals = records_by_alg[alg]
        jitter = np.random.default_rng(42).uniform(-0.08, 0.08, len(vals))
        ax.scatter([pos + j for j in jitter], vals,
                   color=COLORS[alg], s=40, zorder=3, edgecolors="white", lw=0.5)

    ax.set_xticks(positions)
    ax.set_xticklabels([LABELS[a] for a in ALG_NAMES], fontsize=11)
    ax.set_ylabel(ylabel, fontsize=12)
    ax.set_title(title, fontsize=12)
    ax.yaxis.grid(True, alpha=0.3)
    ax.set_axisbelow(True)
    savefig(fig, os.path.join(output_dir, f"violin_{metric}.png"))


def plot_generations_to_threshold(data: dict[str, dict[float, list]], output_dir: str):
    """For several thresholds, show fraction of runs reaching them by generation."""
    thresholds = [87.33, 88.0, 89.33, 90.0, 90.67, 91.33]
    fig, axes = plt.subplots(1, len(thresholds), figsize=(3.2 * len(thresholds), 5), sharey=True)
    for ax, thr in zip(axes, thresholds):
        for alg in ALG_NAMES:
            gen_hits = data[alg][thr]
            n_total = len(gen_hits)
            hit_counts = [sum(1 for g in gen_hits if g is not None and g <= gen) for gen in range(25)]
            fractions = [c / n_total for c in hit_counts]
            ax.plot(range(25), fractions, color=COLORS[alg], lw=2, label=LABELS[alg])
        ax.set_title(f"≥{thr:.1f}%", fontsize=10)
        ax.set_xlabel("Gen.", fontsize=9)
        ax.set_ylim(-0.05, 1.05)
        ax.yaxis.grid(True, alpha=0.3)
        ax.set_axisbelow(True)

    axes[0].set_ylabel("Fraction of Runs Reached (%)", fontsize=11)
    fig.suptitle("Cumulative Fraction of Runs Reaching Accuracy Thresholds", fontsize=12, y=1.02)
    handles = [mpatches.Patch(color=COLORS[a], label=LABELS[a]) for a in ALG_NAMES]
    fig.legend(handles=handles, loc="upper right", fontsize=10)
    savefig(fig, os.path.join(output_dir, "generations_to_threshold.png"))


def plot_search_efficiency(eff: dict[str, list[float]], output_dir: str):
    """Scatter of accuracy vs total_candidates_evaluated."""
    fig, ax = plt.subplots(figsize=(7, 5))
    for alg in ALG_NAMES:
        accs = [r["best_quant_acc1"] for r in eff if r["algorithm"] == alg and r.get("best_quant_acc1") is not None]
        cands = [r["total_candidates_evaluated"] for r in eff if r["algorithm"] == alg and r.get("best_quant_acc1") is not None]
        ax.scatter(cands, accs, color=COLORS[alg], label=LABELS[alg], s=80, zorder=3,
                   edgecolors="white", lw=0.8)

    ax.set_xlabel("Total Candidates Evaluated", fontsize=12)
    ax.set_ylabel("Best Quantized Accuracy (%)", fontsize=12)
    ax.set_title("Search Efficiency:\nAccuracy vs. Candidates Evaluated per Run", fontsize=12)
    ax.legend(fontsize=11)
    ax.yaxis.grid(True, alpha=0.3)
    ax.xaxis.grid(True, alpha=0.3)
    ax.set_axisbelow(True)
    savefig(fig, os.path.join(output_dir, "search_efficiency.png"))


def plot_architecture_distributions(arch_by_alg: dict[str, dict], output_dir: str):
    """Distribution of key architecture parameters for best architectures per algorithm."""
    param_names = ["resolutions", "stem_widths"]
    param_labels = ["Input Resolution (px)", "Stem Width (channels)"]
    stage_params = [
        ("stage_depths", "Stage Depth (layers)", 4),
        ("stage_widths", "Stage Width (channels)", 4),
    ]

    # Resolution and stem_width side-by-side
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    for ax, pname, plabel in zip(axes, param_names, param_labels):
        all_vals = sorted(set(
            v for alg in ALG_NAMES
            for v in arch_by_alg[alg][pname]
            if v is not None
        ))
        positions = {v: i for i, v in enumerate(all_vals)}
        width = 0.35
        for offset, alg in zip([-width / 2, width / 2], ALG_NAMES):
            counts = defaultdict(int)
            for v in arch_by_alg[alg][pname]:
                if v is not None:
                    counts[v] += 1
            ax.bar([positions[v] + offset for v in all_vals],
                   [counts[v] for v in all_vals],
                   width=width, color=COLORS[alg], alpha=0.85, label=LABELS[alg])
        ax.set_xticks(range(len(all_vals)))
        ax.set_xticklabels(all_vals, fontsize=10)
        ax.set_xlabel(plabel, fontsize=11)
        ax.set_ylabel("Count", fontsize=11)
        ax.yaxis.grid(True, alpha=0.3)
        ax.set_axisbelow(True)
        ax.legend(fontsize=10)
    fig.suptitle("Best Architecture Parameter Distributions", fontsize=13)
    savefig(fig, os.path.join(output_dir, "architecture_resolution_stem.png"))

    # Stage depths and widths
    for pname, plabel, n_stages in stage_params:
        fig, axes = plt.subplots(1, n_stages, figsize=(3.5 * n_stages, 5), sharey=True)
        for stage_idx, ax in enumerate(axes):
            for alg in ALG_NAMES:
                vals = arch_by_alg[alg][pname][stage_idx]
                if not vals:
                    continue
                all_options = sorted(set(vals))
                counts = [vals.count(v) for v in all_options]
                width = 0.35
                offset = -width / 2 if alg == ALG_NAMES[0] else width / 2
                pos_vals = list(range(len(all_options)))
                ax.bar([p + offset for p in pos_vals], counts,
                       width=width, color=COLORS[alg], alpha=0.85, label=LABELS[alg])
            ax.set_xticks(range(len(all_options)))
            ax.set_xticklabels(all_options, fontsize=9)
            ax.set_title(f"Stage {stage_idx + 1}", fontsize=11)
            ax.set_xlabel(plabel, fontsize=10)
            ax.yaxis.grid(True, alpha=0.3)
            ax.set_axisbelow(True)
            if stage_idx == 0:
                ax.legend(fontsize=9)
        axes[0].set_ylabel("Count", fontsize=11)
        fig.suptitle(f"Best Architecture {plabel} per Stage", fontsize=13)
        savefig(fig, os.path.join(output_dir, f"architecture_{pname}.png"))


def plot_auc_comparison(auc_by_alg: dict[str, list[float]], output_dir: str):
    """Box + violin comparison of AUC of convergence curves."""
    fig, ax = plt.subplots(figsize=(6, 5))
    positions = list(range(1, len(ALG_NAMES) + 1))
    data = [auc_by_alg[a] for a in ALG_NAMES]
    parts = ax.violinplot(data, positions=positions, showmedians=True)
    for body, alg in zip(parts["bodies"], ALG_NAMES):
        body.set_facecolor(COLORS[alg])
        body.set_alpha(0.5)
    for comp in ["cmedians", "cmins", "cmaxes", "cbars"]:
        if comp in parts:
            parts[comp].set_color("black")
    for pos, alg, vals in zip(positions, ALG_NAMES, data):
        jitter = np.random.default_rng(42).uniform(-0.07, 0.07, len(vals))
        ax.scatter([pos + j for j in jitter], vals, color=COLORS[alg], s=50, zorder=3, edgecolors="white", lw=0.5)
    ax.set_xticks(positions)
    ax.set_xticklabels([LABELS[a] for a in ALG_NAMES], fontsize=11)
    ax.set_ylabel("Mean AUC of Best-Fitness Curve (%)", fontsize=12)
    ax.set_title("Area Under Convergence Curve\n(Higher = faster convergence to high accuracy)", fontsize=12)
    ax.yaxis.grid(True, alpha=0.3)
    ax.set_axisbelow(True)
    savefig(fig, os.path.join(output_dir, "auc_comparison.png"))


def plot_compile_success_vs_accuracy(records: list[dict], output_dir: str):
    """Scatter: compile success rate vs best accuracy, colored by algorithm, with run labels."""
    fig, ax = plt.subplots(figsize=(8, 6))
    for alg in ALG_NAMES:
        recs = [r for r in records if r["algorithm"] == alg and r.get("best_quant_acc1") is not None]
        xs = [r["compile_success_rate"] * 100 for r in recs]
        ys = [r["best_quant_acc1"] for r in recs]
        ax.scatter(xs, ys, color=COLORS[alg], s=70, label=LABELS[alg], zorder=3, edgecolors="white", lw=0.8)
        for r in recs:
            ax.annotate(f'S{r["seed"]}',
                        (r["compile_success_rate"] * 100, r["best_quant_acc1"]),
                        textcoords="offset points", xytext=(4, 3), fontsize=7, color=COLORS[alg], alpha=0.8)

    ax.set_xlabel("Compile Success Rate (%)", fontsize=12)
    ax.set_ylabel("Best Quantized Accuracy (%)", fontsize=12)
    ax.set_title("Tradeoff: Compile Success Rate vs. Best Accuracy\n(each point = one independent run)", fontsize=12)
    ax.legend(fontsize=11)
    ax.yaxis.grid(True, alpha=0.3)
    ax.xaxis.grid(True, alpha=0.3)
    ax.set_axisbelow(True)
    savefig(fig, os.path.join(output_dir, "compile_vs_accuracy_annotated.png"))


def plot_statistical_summary(stat_results: dict, output_dir: str):
    """Combined bar chart of p-values and effect sizes."""
    metrics = [m for m in stat_results.keys()]
    p_vals = [stat_results[m]["p_holm"] for m in metrics]
    effect_vals = [stat_results[m]["effect_size"] for m in metrics]
    effect_types = [stat_results[m]["effect_type"] for m in metrics]

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 9))

    # p-values
    colors_bar = ["#c0392b" if p < 0.05 else "#7f8c8d" for p in p_vals]
    bars1 = ax1.bar(range(len(metrics)), p_vals, color=colors_bar, alpha=0.85, edgecolor="white")
    ax1.axhline(y=0.05, color="red", linestyle="--", lw=1.5, label="α = 0.05")
    ax1.set_yscale("log")
    ax1.set_xticks(range(len(metrics)))
    ax1.set_xticklabels([m.replace("_", "\n") for m in metrics], fontsize=9)
    ax1.set_ylabel("p-value (Holm-Bonferroni, log scale)", fontsize=11)
    ax1.set_title("Statistical Significance: Holm-Bonferroni Adjusted p-values\n(Regularized Evolution vs. Baseline SGA)", fontsize=12)
    ax1.legend(fontsize=10)
    ax1.yaxis.grid(True, alpha=0.3)
    for bar, p in zip(bars1, p_vals):
        ax1.text(bar.get_x() + bar.get_width() / 2, bar.get_height() * 1.5,
                 f"p={p:.4f}", ha="center", va="bottom", fontsize=8)

    # effect sizes
    colors_eff = [COLORS["regularized_evolution"] if v < 0 else COLORS["baseline_sga"] for v in effect_vals]
    bars2 = ax2.barh(range(len(metrics)), effect_vals, color=colors_eff, alpha=0.85, edgecolor="white")
    ax2.axvline(x=0, color="black", lw=1)
    ax2.set_yticks(range(len(metrics)))
    ax2.set_yticklabels([f"{m}\n({et})" for m, et in zip(metrics, effect_types)], fontsize=9)
    ax2.set_xlabel("Effect Size (negative = Reg-Evo higher, positive = SGA higher)", fontsize=11)
    ax2.set_title("Effect Sizes (Cliff's δ or Cohen's d)", fontsize=12)
    ax2.xaxis.grid(True, alpha=0.3)
    for bar, v, m in zip(bars2, effect_vals, metrics):
        mag = effect_magnitude(v, kind=stat_results[m]["effect_type"])
        ax2.text(v + (0.02 if v >= 0 else -0.02), bar.get_y() + bar.get_height() / 2,
                 mag, ha="left" if v >= 0 else "right", va="center", fontsize=8, style="italic")

    fig.tight_layout()
    savefig(fig, os.path.join(output_dir, "statistical_summary.png"))


def plot_search_cost_breakdown(records: list[dict], output_dir: str):
    """Box + violin of elapsed_hours by algorithm."""
    hours_by_alg = {}
    for alg in ALG_NAMES:
        hours_by_alg[alg] = [
            r["elapsed_seconds"] / 3600
            for r in records
            if r["algorithm"] == alg and r.get("elapsed_seconds") is not None and r.get("status") == "success"
        ]

    fig, ax = plt.subplots(figsize=(7, 5))
    positions = list(range(1, len(ALG_NAMES) + 1))
    parts = ax.violinplot([hours_by_alg[a] for a in ALG_NAMES], positions=positions, showmedians=True)
    for body, alg in zip(parts["bodies"], ALG_NAMES):
        body.set_facecolor(COLORS[alg])
        body.set_alpha(0.5)
    for comp in ["cmedians", "cmins", "cmaxes", "cbars"]:
        if comp in parts:
            parts[comp].set_color("black")
    for pos, alg in zip(positions, ALG_NAMES):
        vals = hours_by_alg[alg]
        jitter = np.random.default_rng(42).uniform(-0.07, 0.07, len(vals))
        ax.scatter([pos + j for j in jitter], vals, color=COLORS[alg], s=55, zorder=3, edgecolors="white", lw=0.5)

    ax.set_xticks(positions)
    ax.set_xticklabels([LABELS[a] for a in ALG_NAMES], fontsize=11)
    ax.set_ylabel("Search Duration (GPU-hours per run)", fontsize=12)
    ax.set_title("Search Cost per Run\n(1 GPU, single-threaded evaluation)", fontsize=12)
    ax.yaxis.grid(True, alpha=0.3)
    ax.set_axisbelow(True)
    savefig(fig, os.path.join(output_dir, "search_cost.png"))


# ─── Statistics computation & reporting ──────────────────────────────────────

def compute_full_statistics(records_by_alg: dict) -> dict:
    """Full pairwise stats for key metrics."""
    metrics = [
        ("best_quant_acc1", "Best Quantized Accuracy (%)"),
        ("compile_success_rate", "Compile Success Rate"),
        ("elapsed_seconds", "Elapsed Time (s)"),
        ("total_candidates_evaluated", "Total Candidates"),
        ("compiled_candidates", "Compiled Candidates"),
    ]
    results = {}
    alg_a, alg_b = ALG_NAMES[0], ALG_NAMES[1]  # reg_evo, sga
    vals_a_raw = {m: [r[m] for r in records_by_alg[alg_a] if r.get(m) is not None] for m, _ in metrics}
    vals_b_raw = {m: [r[m] for r in records_by_alg[alg_b] if r.get(m) is not None] for m, _ in metrics}

    raw_p_values = []
    for metric, _ in metrics:
        va = vals_a_raw[metric]
        vb = vals_b_raw[metric]
        sw_p_a = shapiro_wilk_p(va)
        sw_p_b = shapiro_wilk_p(vb)
        both_normal = sw_p_a > 0.05 and sw_p_b > 0.05

        if both_normal:
            stat, p = stats.ttest_ind(va, vb, equal_var=False)
            test_type = "welch_t"
            effect = cohens_d(va, vb)
            etype = "cohens_d"
        else:
            stat, p = stats.mannwhitneyu(va, vb, alternative="two-sided")
            test_type = "mann_whitney_u"
            effect = cliffs_delta(va, vb)
            etype = "cliffs_delta"

        ci_lo, ci_hi = bootstrap_ci(va, vb)
        raw_p_values.append(p)
        results[metric] = {
            "n_a": len(va), "n_b": len(vb),
            "mean_a": np.mean(va), "std_a": np.std(va, ddof=1),
            "median_a": np.median(va), "iqr_a": float(np.subtract(*np.percentile(va, [75, 25]))),
            "mean_b": np.mean(vb), "std_b": np.std(vb, ddof=1),
            "median_b": np.median(vb), "iqr_b": float(np.subtract(*np.percentile(vb, [75, 25]))),
            "sw_p_a": sw_p_a, "sw_p_b": sw_p_b,
            "test_type": test_type, "statistic": float(stat), "p_raw": float(p),
            "effect_size": float(effect), "effect_type": etype,
            "mean_diff_a_minus_b": float(np.mean(va) - np.mean(vb)),
            "bootstrap_ci_lo": float(ci_lo), "bootstrap_ci_hi": float(ci_hi),
            "significant_alpha_0_05": p < 0.05,
        }

    # Holm-Bonferroni correction
    m = len(raw_p_values)
    sorted_pairs = sorted(enumerate(raw_p_values), key=lambda x: x[1])
    holm_p = [None] * m
    prev_corrected = 0.0
    for rank, (idx, p) in enumerate(sorted_pairs):
        corrected = min(1.0, max(prev_corrected, p * (m - rank)))
        holm_p[idx] = corrected
        prev_corrected = corrected

    for i, (metric, _) in enumerate(metrics):
        results[metric]["p_holm"] = holm_p[i]
        results[metric]["significant_alpha_0_05_holm"] = holm_p[i] < 0.05

    return results


def write_stats_report(stat_results: dict, auc_by_alg: dict,
                        gtt_data: dict, records: list[dict], output_path: str):
    """Write a machine-readable statistics summary JSON."""
    report = {
        "generated_at": datetime.utcnow().isoformat(),
        "algorithms": ALG_NAMES,
        "algorithm_labels": LABELS,
        "pairwise_statistics": stat_results,
        "auc_by_algorithm": {
            alg: {
                "mean": float(np.mean(vals)),
                "std": float(np.std(vals, ddof=1)),
                "median": float(np.median(vals)),
                "min": float(np.min(vals)),
                "max": float(np.max(vals)),
                "values": [float(v) for v in vals],
            }
            for alg, vals in auc_by_alg.items()
        },
        "generations_to_threshold": {
            alg: {
                str(thr): [g for g in gtt_data[alg][thr]]
                for thr in gtt_data[alg]
            }
            for alg in ALG_NAMES
        },
        "success_rate_by_algorithm": {
            alg: {
                "n_success": sum(1 for r in records if r["algorithm"] == alg and r.get("status") == "success"),
                "n_total": sum(1 for r in records if r["algorithm"] == alg),
            }
            for alg in ALG_NAMES
        },
    }
    class _Enc(json.JSONEncoder):
        def default(self, o):
            if isinstance(o, (np.integer,)): return int(o)
            if isinstance(o, (np.floating,)): return float(o)
            if isinstance(o, (np.bool_,)): return bool(o)
            return super().default(o)

    with open(output_path, "w") as f:
        json.dump(report, f, indent=2, cls=_Enc)
    print(f"  Saved: {output_path}")


# ─── Main ─────────────────────────────────────────────────────────────────────

def main():
    ap = argparse.ArgumentParser(description="Publication-quality NAS analysis")
    ap.add_argument("--sga-dir", required=True, help="Path to baseline_sga experiment dir")
    ap.add_argument("--reg-evo-dir", required=True, help="Path to regularized_evolution experiment dir")
    ap.add_argument("--output-dir", required=True, help="Where to write output plots & reports")
    args = ap.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    plot_dir = os.path.join(args.output_dir, "plots")
    os.makedirs(plot_dir, exist_ok=True)

    print("Loading data...")
    dir_map = {
        "regularized_evolution": args.reg_evo_dir,
        "baseline_sga": args.sga_dir,
    }

    records_all = []
    records_by_alg = {}
    histories_by_alg = {}
    summaries_by_alg = {}

    for alg, d in dir_map.items():
        recs = load_run_records(d)
        records_all.extend(recs)
        records_by_alg[alg] = [r for r in recs if r["algorithm"] == alg]
        histories_by_alg[alg] = load_all_histories(d, alg)
        summaries_by_alg[alg] = load_all_summaries(d)
        print(f"  {alg}: {len(records_by_alg[alg])} runs, "
              f"{len(histories_by_alg[alg])} histories, "
              f"{len(summaries_by_alg[alg])} summaries")

    # Filter successful records for metric analysis
    successful_by_alg = {alg: [r for r in recs if r.get("status") == "success"]
                         for alg, recs in records_by_alg.items()}

    print("\nComputing statistics...")
    conv_stats = {alg: compute_convergence_stats(histories_by_alg[alg]) for alg in ALG_NAMES}
    pop_stats = {alg: compute_pop_mean_stats(histories_by_alg[alg]) for alg in ALG_NAMES}

    auc_by_alg = {}
    for alg in ALG_NAMES:
        aucs = []
        for hist in histories_by_alg[alg]:
            best_vals = [g["best_fitness"] for g in hist]
            aucs.append(compute_auc(best_vals))
        auc_by_alg[alg] = [v for v in aucs if not math.isnan(v)]

    thresholds = [87.33, 88.0, 89.33, 90.0, 90.67, 91.33]
    gtt_data = {alg: {thr: compute_generations_to_threshold(histories_by_alg[alg], thr)
                      for thr in thresholds}
                for alg in ALG_NAMES}

    arch_by_alg = {alg: compute_architecture_stats(summaries_by_alg[alg]) for alg in ALG_NAMES}
    stat_results = compute_full_statistics(successful_by_alg)

    print("\nGenerating plots...")
    plot_combined_convergence(conv_stats, plot_dir)
    plot_all_run_trajectories(histories_by_alg, plot_dir)
    plot_pop_mean_evolution(pop_stats, conv_stats, plot_dir)

    for metric, label, title in [
        ("best_quant_acc1", "Best Quantized Accuracy (%)", "Final Accuracy Distribution"),
        ("compile_success_rate", "Compile Success Rate", "Compile Success Rate Distribution"),
        ("elapsed_seconds", "Search Time (s)", "Search Time Distribution"),
        ("total_candidates_evaluated", "Total Candidates Evaluated", "Candidates Evaluated Distribution"),
    ]:
        data = {alg: [r[metric] for r in successful_by_alg[alg] if r.get(metric) is not None]
                for alg in ALG_NAMES}
        plot_violin_distributions(data, metric, label, title, plot_dir)

    plot_generations_to_threshold(gtt_data, plot_dir)
    plot_search_efficiency(records_all, plot_dir)
    plot_auc_comparison(auc_by_alg, plot_dir)
    plot_compile_success_vs_accuracy(records_all, plot_dir)
    plot_statistical_summary(stat_results, plot_dir)
    plot_search_cost_breakdown(records_all, plot_dir)
    plot_architecture_distributions(arch_by_alg, plot_dir)

    print("\nWriting reports...")
    report_path = os.path.join(args.output_dir, "publication_statistics.json")
    write_stats_report(stat_results, auc_by_alg, gtt_data, records_all, report_path)

    # Print summary table to stdout
    print("\n" + "=" * 70)
    print("STATISTICAL SUMMARY")
    print("=" * 70)
    print(f"{'Metric':<35} {'Test':<16} {'p-raw':>8} {'p-Holm':>8} {'Effect':>8} {'Sig?':>5}")
    print("-" * 70)
    for metric, (test_type, p_raw, p_holm, effect, etype) in {
        m: (v["test_type"], v["p_raw"], v["p_holm"], v["effect_size"], v["effect_type"])
        for m, v in stat_results.items()
    }.items():
        sig = "yes" if p_holm < 0.05 else "no"
        print(f"{metric:<35} {test_type:<16} {p_raw:>8.4f} {p_holm:>8.4f} {effect:>8.3f} {sig:>5}")
    print("=" * 70)
    print(f"\nRegularized Evolution: n={len(successful_by_alg['regularized_evolution'])}/10 successful, "
          f"mean acc = {np.mean([r['best_quant_acc1'] for r in successful_by_alg['regularized_evolution']]):.2f}%")
    print(f"Baseline SGA:          n={len(successful_by_alg['baseline_sga'])}/10 successful, "
          f"mean acc = {np.mean([r['best_quant_acc1'] for r in successful_by_alg['baseline_sga']]):.2f}%")
    print("\nDone.")


if __name__ == "__main__":
    main()
