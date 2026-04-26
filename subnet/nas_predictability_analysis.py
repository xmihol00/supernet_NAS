#!/usr/bin/env python3
"""
NAS Proxy Predictability Analysis.

Reads results from full_dataset_experiment/ and generates plots and a
statistical summary examining how well the NAS proxy metric
(6-class IMX500 quantised accuracy) predicts full-dataset floating-point
validation accuracy after stand-alone training.

Usage:
    python subnet/nas_predictability_analysis.py \
        --experiment-dir ./full_dataset_experiment \
        --output-dir ./full_dataset_experiment/nas_predictability_analysis
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Dict, List, Tuple

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
from scipy import stats as scipy_stats


# ── colours ──────────────────────────────────────────────────────────────────
_CMAP = matplotlib.colormaps.get_cmap("tab20").resampled(20)

def _arch_color(i: int):
    return _CMAP(i % 20)


# ── helpers ───────────────────────────────────────────────────────────────────

def _sig_label(p: float) -> str:
    if p < 0.001: return "p<0.001 ✓✓✓"
    if p < 0.01:  return f"p={p:.4f} ✓✓"
    if p < 0.05:  return f"p={p:.4f} ✓"
    return f"p={p:.4f} ✗"


def _bootstrap_spearman_ci(x: np.ndarray, y: np.ndarray,
                            n_boot: int = 5000, seed: int = 42) -> Tuple[float, float]:
    rng = np.random.default_rng(seed)
    n = len(x)
    boot_r = []
    for _ in range(n_boot):
        idx = rng.integers(0, n, size=n)
        if len(set(idx.tolist())) < 3:
            continue
        r_b, _ = scipy_stats.spearmanr(x[idx], y[idx])
        boot_r.append(float(r_b))
    if boot_r:
        return float(np.percentile(boot_r, 2.5)), float(np.percentile(boot_r, 97.5))
    return float("nan"), float("nan")


def _top_k_recall(nas_accs: np.ndarray, val_accs: np.ndarray, k: int) -> float:
    """Fraction of the true top-k (by val_acc) captured by the top-k NAS selection."""
    n = len(nas_accs)
    k = min(k, n)
    top_k_nas = set(np.argsort(-nas_accs)[:k].tolist())
    top_k_val = set(np.argsort(-val_accs)[:k].tolist())
    return len(top_k_nas & top_k_val) / k


# ── data loading ──────────────────────────────────────────────────────────────

def load_data(exp_dir: Path) -> Tuple[List[dict], List[dict], List[dict]]:
    with (exp_dir / "selected_architectures.json").open() as f:
        arch_list: List[dict] = json.load(f)
    with (exp_dir / "best_acc_summary.json").open() as f:
        best_summary: List[dict] = json.load(f)
    with (exp_dir / "stats_history.json").open() as f:
        stats_history: List[dict] = json.load(f)
    return arch_list, best_summary, stats_history


# ── plot 1: NAS score vs. best full-dataset accuracy ─────────────────────────

def plot_nas_vs_full_acc(best_summary: List[dict], out_dir: Path) -> None:
    nas  = np.array([r["nas_quant_acc1"] for r in best_summary])
    val  = np.array([r["best_val_acc1"]  for r in best_summary])
    ema  = np.array([r["best_ema_acc1"]  for r in best_summary])
    idxs = [r["arch_index"]              for r in best_summary]
    n    = len(best_summary)

    sr_v, sp_v = scipy_stats.spearmanr(nas, val)
    pr_v, pp_v = scipy_stats.pearsonr(nas, val)
    sr_e, sp_e = scipy_stats.spearmanr(nas, ema)
    pr_e, pp_e = scipy_stats.pearsonr(nas, ema)
    ci_v = _bootstrap_spearman_ci(nas, val)
    ci_e = _bootstrap_spearman_ci(nas, ema, seed=43)

    fig, axes = plt.subplots(1, 2, figsize=(16, 7))

    for ax, acc, sr, sp, pr, ci, title in [
        (axes[0], val, sr_v, sp_v, pr_v, ci_v, "Best val accuracy (non-EMA)"),
        (axes[1], ema, sr_e, sp_e, pr_e, ci_e, "Best EMA val accuracy"),
    ]:
        for i, (x, y, ai) in enumerate(zip(nas, acc, idxs)):
            ax.scatter(x, y, color=_arch_color(i), s=100, zorder=3,
                       edgecolors="white", lw=0.8)
            ax.annotate(f"A{ai}", (x, y), textcoords="offset points",
                        xytext=(6, 3), fontsize=8, color=_arch_color(i))

        # OLS regression
        if n >= 2:
            m, b, *_ = scipy_stats.linregress(nas, acc)
            xs = np.linspace(nas.min(), nas.max(), 100)
            ax.plot(xs, m * xs + b, "k--", lw=1.4, alpha=0.55, label="OLS fit")

        ax.set_xlabel("NAS proxy score — 6-class IMX500 quantised top-1 (%)", fontsize=11)
        ax.set_ylabel("Full ImageNet val top-1 accuracy (%)", fontsize=11)
        ax.set_title(
            f"{title}\n"
            f"Spearman ρ={sr:.3f} ({_sig_label(sp)}) | Pearson r={pr:.3f} ({_sig_label(pp_v if ax is axes[0] else pp_e)})\n"
            f"Bootstrap 95% CI: [{ci[0]:.3f}, {ci[1]:.3f}]  (N={n})",
            fontsize=9.5,
        )
        ax.xaxis.grid(True, alpha=0.3); ax.yaxis.grid(True, alpha=0.3)
        ax.set_axisbelow(True)
        ax.legend(fontsize=9)

    fig.suptitle("NAS Proxy Accuracy vs. Full-Dataset Stand-Alone Training Accuracy\n"
                 "(after 5 training epochs per architecture)", fontsize=12, fontweight="bold")
    fig.tight_layout()
    fig.savefig(out_dir / "nas_vs_full_acc.png", dpi=140, bbox_inches="tight")
    plt.close(fig)


# ── plot 2: rank comparison ───────────────────────────────────────────────────

def plot_rank_comparison(best_summary: List[dict], out_dir: Path) -> None:
    n = len(best_summary)
    nas  = np.array([r["nas_quant_acc1"] for r in best_summary])
    val  = np.array([r["best_val_acc1"]  for r in best_summary])
    ema  = np.array([r["best_ema_acc1"]  for r in best_summary])
    idxs = [r["arch_index"]              for r in best_summary]

    nas_ranks  = np.argsort(np.argsort(-nas))  # 0 = best
    val_ranks  = np.argsort(np.argsort(-val))
    ema_ranks  = np.argsort(np.argsort(-ema))

    order = np.argsort(nas_ranks)  # sort left-to-right by NAS rank (best→worst)
    ai_labels = [f"A{idxs[i]}" for i in order]

    fig, axes = plt.subplots(1, 2, figsize=(16, 6))

    x = np.arange(n)
    w = 0.25
    b0 = axes[0].bar(x - w,    nas_ranks[order], w, label="NAS rank",       color="#0072B2", alpha=0.85)
    b1 = axes[0].bar(x,        val_ranks[order], w, label="Val rank",        color="#D55E00", alpha=0.85)
    b2 = axes[0].bar(x + w,    ema_ranks[order], w, label="Best EMA rank",   color="#009E73", alpha=0.85)
    axes[0].set_xticks(x); axes[0].set_xticklabels(ai_labels, fontsize=9)
    axes[0].set_ylabel("Rank  (0 = best)", fontsize=11)
    axes[0].set_title("NAS rank vs. full-dataset ranks\n(architectures sorted by NAS rank, best → worst)",
                      fontsize=11)
    axes[0].legend(fontsize=9)
    axes[0].yaxis.grid(True, alpha=0.3)
    axes[0].set_axisbelow(True)
    axes[0].invert_yaxis()  # lower rank bar = taller

    diff_val = (val_ranks - nas_ranks)[order]
    diff_ema = (ema_ranks - nas_ranks)[order]
    axes[1].bar(x - 0.18, diff_val,
                0.35, color=["#009E73" if d <= 0 else "#CC79A7" for d in diff_val],
                alpha=0.85, edgecolor="white", label="Val rank − NAS rank")
    axes[1].bar(x + 0.18, diff_ema,
                0.35, color=["#56B4E9" if d <= 0 else "#E69F00" for d in diff_ema],
                alpha=0.85, edgecolor="white", label="EMA rank − NAS rank")
    axes[1].axhline(0, color="black", lw=0.8, linestyle="--")
    axes[1].set_xticks(x); axes[1].set_xticklabels(ai_labels, fontsize=9)
    axes[1].set_ylabel("Rank shift  (negative = architecture moved up)", fontsize=11)
    axes[1].set_title("Rank shift: NAS → full-dataset\n(negative = better than NAS predicted)",
                      fontsize=11)
    axes[1].yaxis.grid(True, alpha=0.3)
    axes[1].set_axisbelow(True)
    axes[1].legend(fontsize=9)

    fig.suptitle("Architecture Ranking: NAS Proxy vs. Full-Dataset Training", fontsize=12,
                 fontweight="bold")
    fig.tight_layout()
    fig.savefig(out_dir / "rank_comparison.png", dpi=140, bbox_inches="tight")
    plt.close(fig)


# ── plot 3: correlation evolution over cycles ─────────────────────────────────

def plot_correlation_evolution(stats_history: List[dict], out_dir: Path) -> None:
    if len(stats_history) < 2:
        return

    cycles = [s["cycle"] for s in stats_history]
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    # Row 0: Spearman & Pearson over cycles for val / EMA / best
    ax = axes[0][0]
    for key, label, color in [
        ("spearman_r",      "Current val",  "#0072B2"),
        ("spearman_r_ema",  "EMA val",      "#D55E00"),
        ("spearman_r_best", "Best val",     "#009E73"),
    ]:
        sr = [s.get(key, float("nan")) for s in stats_history]
        sp = [s.get(key.replace("spearman_r", "spearman_p"), 1.0) for s in stats_history]
        ax.plot(cycles, sr, "o-", color=color, lw=2, label=f"Spearman ρ ({label})")
        for c, s_val, s_p in zip(cycles, sr, sp):
            if s_p < 0.05:
                ax.scatter(c, s_val, s=120, marker="*", color="gold",
                           zorder=4, edgecolors="black", lw=0.5)
    ax.axhline(0, color="black", lw=0.7, linestyle="--")
    ax.set_ylim(-1.05, 1.05)
    ax.set_xlabel("Cycle (epochs per architecture)", fontsize=10)
    ax.set_ylabel("Spearman ρ", fontsize=10)
    ax.set_title("Spearman rank correlation over training  (★=p<0.05)", fontsize=10)
    ax.legend(fontsize=8); ax.yaxis.grid(True, alpha=0.3)

    ax = axes[0][1]
    for key, label, color in [
        ("pearson_r",      "Current val", "#0072B2"),
        ("pearson_r_ema",  "EMA val",     "#D55E00"),
        ("pearson_r_best", "Best val",    "#009E73"),
    ]:
        pr = [s.get(key, float("nan")) for s in stats_history]
        ax.plot(cycles, pr, "s--", color=color, lw=2, label=f"Pearson r ({label})")
    ax.axhline(0, color="black", lw=0.7, linestyle="--")
    ax.set_ylim(-1.05, 1.05)
    ax.set_xlabel("Cycle (epochs per architecture)", fontsize=10)
    ax.set_ylabel("Pearson r", fontsize=10)
    ax.set_title("Pearson linear correlation over training", fontsize=10)
    ax.legend(fontsize=8); ax.yaxis.grid(True, alpha=0.3)

    # Row 1: p-values and bootstrap CI width
    ax = axes[1][0]
    for key, label, color in [
        ("spearman_p",      "Current val", "#0072B2"),
        ("spearman_p_ema",  "EMA val",     "#D55E00"),
        ("spearman_p_best", "Best val",    "#009E73"),
    ]:
        pv = [s.get(key, 1.0) for s in stats_history]
        ax.semilogy(cycles, pv, "o-", color=color, lw=2, label=f"{label}")
    ax.axhline(0.05, color="red", lw=1.0, linestyle="--", label="α=0.05")
    ax.axhline(0.01, color="darkred", lw=1.0, linestyle=":", label="α=0.01")
    ax.set_xlabel("Cycle (epochs per architecture)", fontsize=10)
    ax.set_ylabel("p-value (log scale)", fontsize=10)
    ax.set_title("Spearman p-value over training", fontsize=10)
    ax.legend(fontsize=8); ax.yaxis.grid(True, alpha=0.3)

    ax = axes[1][1]
    for key_lo, key_hi, label, color in [
        ("bootstrap_ci_spearman",
         "bootstrap_ci_spearman",      "Val CI",     "#0072B2"),
        ("bootstrap_ci_spearman_best",
         "bootstrap_ci_spearman_best", "Best val CI", "#009E73"),
    ]:
        lo = [s.get(key_lo,  [float("nan"), float("nan")])[0] for s in stats_history]
        hi = [s.get(key_hi,  [float("nan"), float("nan")])[1] for s in stats_history]
        sr = [s.get("spearman_r", float("nan")) for s in stats_history]
        ax.fill_between(cycles, lo, hi, alpha=0.25, color=color)
        ax.plot(cycles, sr, "o-", color=color, lw=2, label=label)
    ax.axhline(0, color="black", lw=0.8, linestyle="--")
    ax.set_ylim(-1.05, 1.05)
    ax.set_xlabel("Cycle (epochs per architecture)", fontsize=10)
    ax.set_ylabel("Spearman ρ  (shaded = bootstrap 95% CI)", fontsize=10)
    ax.set_title("Spearman ρ with bootstrap confidence intervals", fontsize=10)
    ax.legend(fontsize=8); ax.yaxis.grid(True, alpha=0.3)

    fig.suptitle("NAS Proxy–Full-Dataset Correlation over Training\n"
                 "Cycle = number of full training epochs each architecture has completed",
                 fontsize=12, fontweight="bold")
    fig.tight_layout()
    fig.savefig(out_dir / "correlation_evolution.png", dpi=140, bbox_inches="tight")
    plt.close(fig)


# ── plot 4: top-K recall ──────────────────────────────────────────────────────

def plot_topk_recall(best_summary: List[dict], out_dir: Path) -> None:
    nas  = np.array([r["nas_quant_acc1"] for r in best_summary])
    val  = np.array([r["best_val_acc1"]  for r in best_summary])
    ema  = np.array([r["best_ema_acc1"]  for r in best_summary])
    n    = len(best_summary)

    ks = list(range(1, n + 1))
    recall_val = [_top_k_recall(nas, val, k) for k in ks]
    recall_ema = [_top_k_recall(nas, ema, k) for k in ks]
    random_bl  = [k / n for k in ks]

    fig, ax = plt.subplots(figsize=(9, 6))
    ax.plot(ks, recall_val, "o-", color="#0072B2", lw=2.0, label="Recall (val)")
    ax.plot(ks, recall_ema, "s--", color="#D55E00", lw=2.0, label="Recall (EMA)")
    ax.plot(ks, random_bl,  "k:",  lw=1.4, label="Random baseline (k/N)")
    ax.fill_between(ks, random_bl, recall_val, alpha=0.1, color="#0072B2")

    ax.set_xlabel("k  (top-k architectures selected)", fontsize=12)
    ax.set_ylabel("Recall  (fraction of true top-k recovered)", fontsize=12)
    ax.set_title("Top-K Recall: fraction of the true top-k (by full-dataset accuracy)\n"
                 "that the NAS proxy ranking correctly identifies", fontsize=11)
    ax.set_xticks(ks)
    ax.set_ylim(-0.05, 1.10)
    ax.legend(fontsize=10)
    ax.yaxis.grid(True, alpha=0.3)
    ax.set_axisbelow(True)

    # Annotate k=1 and k=3 explicitly
    for k, rv, re in zip(ks[:5], recall_val[:5], recall_ema[:5]):
        ax.annotate(f"{rv:.0%}", (k, rv), textcoords="offset points", xytext=(0, 8),
                    ha="center", fontsize=9, color="#0072B2")

    fig.tight_layout()
    fig.savefig(out_dir / "topk_recall.png", dpi=140, bbox_inches="tight")
    plt.close(fig)


# ── plot 5: identical NAS score, divergent performance ───────────────────────

def plot_tied_nas_case(best_summary: List[dict], arch_list: List[dict],
                       out_dir: Path) -> None:
    """Highlight architectures with identical NAS scores but different full accuracy."""
    nas  = np.array([r["nas_quant_acc1"] for r in best_summary])
    val  = np.array([r["best_val_acc1"]  for r in best_summary])
    idxs = [r["arch_index"]              for r in best_summary]

    # Find NAS ties
    from collections import defaultdict
    by_nas: Dict[float, List[int]] = defaultdict(list)
    for i, (nv, ai) in enumerate(zip(nas, idxs)):
        by_nas[round(nv, 4)].append(i)
    ties = {k: v for k, v in by_nas.items() if len(v) > 1}

    if not ties:
        return

    fig, axes = plt.subplots(1, len(ties), figsize=(7 * len(ties), 6), squeeze=False)

    for col, (nas_val, indices) in enumerate(ties.items()):
        ax = axes[0][col]
        arch_info = {a["arch_index"]: a for a in arch_list}

        for rank, idx in enumerate(sorted(indices, key=lambda i: -val[i])):
            ai = idxs[idx]
            cfg = arch_info[ai]["config"]
            label = (
                f"A{ai}  (val={val[idx]:.2f}%)\n"
                f"res={cfg['resolution']}, stem={cfg['stem_width']}\n"
                f"depths={cfg['stage_depths']}\n"
                f"widths={cfg['stage_widths']}"
            )
            ax.bar(rank, val[idx], color=_arch_color(idx), alpha=0.85,
                   edgecolor="white", label=label)
            ax.text(rank, val[idx] + 0.3, f"{val[idx]:.2f}%", ha="center",
                    fontsize=10, fontweight="bold")

        ax.set_xticks(range(len(indices)))
        ax.set_xticklabels([f"A{idxs[i]}" for i in sorted(indices, key=lambda i: -val[i])],
                           fontsize=11)
        ax.set_ylabel("Best full-dataset val top-1 (%)", fontsize=11)
        ax.set_title(f"Architectures with identical NAS proxy score: {nas_val:.2f}%\n"
                     f"Full-dataset accuracy spread: "
                     f"{max(val[i] for i in indices):.2f}% vs {min(val[i] for i in indices):.2f}%",
                     fontsize=10)
        ax.legend(fontsize=8, loc="lower right")
        ax.set_ylim(0, max(val[i] for i in indices) * 1.15)
        ax.yaxis.grid(True, alpha=0.3)
        ax.set_axisbelow(True)

    fig.suptitle("Identical NAS Proxy Score, Divergent Full-Dataset Performance\n"
                 "Illustrates unreliability of the proxy metric for fine-grained ranking",
                 fontsize=11, fontweight="bold")
    fig.tight_layout()
    fig.savefig(out_dir / "tied_nas_score_comparison.png", dpi=140, bbox_inches="tight")
    plt.close(fig)


# ── plot 6: per-arch NAS score vs best val — annotated with rank positions ────

def plot_rank_matrix(best_summary: List[dict], out_dir: Path) -> None:
    """Heatmap-style rank displacement matrix."""
    n    = len(best_summary)
    nas  = np.array([r["nas_quant_acc1"] for r in best_summary])
    val  = np.array([r["best_val_acc1"]  for r in best_summary])
    idxs = [r["arch_index"]              for r in best_summary]

    nas_order = np.argsort(-nas)   # index 0 = highest NAS
    val_order = np.argsort(-val)   # index 0 = highest val

    nas_rank = np.argsort(nas_order)
    val_rank = np.argsort(val_order)

    fig, ax = plt.subplots(figsize=(10, 7))

    xs = nas_rank
    ys = val_rank
    for i, (x, y, ai) in enumerate(zip(xs, ys, idxs)):
        col = _arch_color(i)
        ax.scatter(x, y, s=180, color=col, zorder=3, edgecolors="white", lw=0.8)
        ax.annotate(f"A{ai}", (x, y), textcoords="offset points",
                    xytext=(7, 3), fontsize=8.5, color=col, fontweight="bold")

    # diagonal = perfect correlation
    ax.plot([0, n - 1], [0, n - 1], "k--", lw=1.2, alpha=0.4, label="Perfect correlation")

    sr, sp = scipy_stats.spearmanr(xs, ys)
    ax.set_xlabel("NAS rank  (0 = highest NAS proxy score)", fontsize=12)
    ax.set_ylabel("Full-dataset rank  (0 = highest full val accuracy)", fontsize=12)
    ax.set_title(
        f"NAS rank vs. full-dataset rank per architecture\n"
        f"Spearman ρ={sr:.3f} ({_sig_label(sp)})  — points on diagonal = perfect prediction",
        fontsize=11,
    )
    ax.xaxis.grid(True, alpha=0.3); ax.yaxis.grid(True, alpha=0.3)
    ax.set_axisbelow(True)
    ax.legend(fontsize=10)
    ax.set_xticks(range(n))
    ax.set_yticks(range(n))

    fig.tight_layout()
    fig.savefig(out_dir / "rank_matrix.png", dpi=140, bbox_inches="tight")
    plt.close(fig)


# ── text summary ──────────────────────────────────────────────────────────────

def write_text_summary(best_summary: List[dict], arch_list: List[dict],
                       stats_history: List[dict], out_dir: Path) -> None:
    nas  = np.array([r["nas_quant_acc1"] for r in best_summary])
    val  = np.array([r["best_val_acc1"]  for r in best_summary])
    ema  = np.array([r["best_ema_acc1"]  for r in best_summary])
    idxs = [r["arch_index"]              for r in best_summary]
    n    = len(best_summary)

    sr_v, sp_v = scipy_stats.spearmanr(nas, val)
    pr_v, pp_v = scipy_stats.pearsonr(nas, val)
    kt_v, kp_v = scipy_stats.kendalltau(nas, val)
    ci_v = _bootstrap_spearman_ci(nas, val)

    sr_e, sp_e = scipy_stats.spearmanr(nas, ema)
    pr_e, pp_e = scipy_stats.pearsonr(nas, ema)
    ci_e = _bootstrap_spearman_ci(nas, ema, seed=43)

    nas_order = np.argsort(-nas)
    val_order = np.argsort(-val)
    nas_ranks = np.argsort(nas_order)
    val_ranks = np.argsort(val_order)

    best_nas_ai  = idxs[int(np.argmax(nas))]
    best_nas_val = float(val[np.argmax(nas)])
    best_val_ai  = idxs[int(np.argmax(val))]
    best_val_nas = float(nas[np.argmax(val)])

    top1_recall = _top_k_recall(nas, val, 1)
    top3_recall = _top_k_recall(nas, val, 3)
    top5_recall = _top_k_recall(nas, val, 5)

    last_cycle = stats_history[-1] if stats_history else {}
    n_cycles = len(stats_history)

    lines = [
        "NAS Proxy Predictability Analysis — Summary",
        "=" * 60,
        "",
        f"Architectures evaluated : {n}",
        f"Training cycles completed: {n_cycles}  ({n_cycles} epochs per architecture)",
        "",
        "── Final correlation (best val accuracy after all cycles) ──",
        f"  Spearman ρ  = {sr_v:.4f}  (p={sp_v:.4f}, {_sig_label(sp_v)})",
        f"  Pearson  r  = {pr_v:.4f}  (p={pp_v:.4f})",
        f"  Kendall  τ  = {kt_v:.4f}  (p={kp_v:.4f})",
        f"  Bootstrap 95% CI (Spearman): [{ci_v[0]:.4f}, {ci_v[1]:.4f}]",
        "",
        "── Final correlation (best EMA accuracy after all cycles) ──",
        f"  Spearman ρ  = {sr_e:.4f}  (p={sp_e:.4f}, {_sig_label(sp_e)})",
        f"  Pearson  r  = {pr_e:.4f}  (p={pp_e:.4f})",
        f"  Bootstrap 95% CI (Spearman): [{ci_e[0]:.4f}, {ci_e[1]:.4f}]",
        "",
        "── Top-K selection recall ──────────────────────────────────",
        f"  Top-1 recall : {top1_recall:.0%}  (NAS picks the best arch: {'Yes' if top1_recall==1 else 'No'})",
        f"  Top-3 recall : {top3_recall:.0%}  ({int(round(top3_recall*3))}/3 true top-3 captured)",
        f"  Top-5 recall : {top5_recall:.0%}  ({int(round(top5_recall*5))}/5 true top-5 captured)",
        "",
        "── Best-architecture comparison ────────────────────────────",
        f"  Best by NAS : A{best_nas_ai}  (NAS={nas.max():.2f}%)  →  full val={best_nas_val:.2f}%  "
        f"(val rank {int(val_ranks[np.argmax(nas)])+1}/{n})",
        f"  Best by val : A{best_val_ai}  (full val={val.max():.2f}%)  →  NAS={best_val_nas:.2f}%  "
        f"(NAS rank {int(nas_ranks[np.argmax(val)])+1}/{n})",
        "",
        "── Per-architecture summary (sorted by NAS score) ──────────",
        f"  {'Arch':>5}  {'NAS%':>7}  {'NAS rank':>9}  {'Val%':>7}  {'Val rank':>9}  {'Rank shift':>10}",
        "  " + "-" * 60,
    ]

    sorted_by_nas = sorted(range(n), key=lambda i: -nas[i])
    for pos, i in enumerate(sorted_by_nas):
        shift = int(val_ranks[i]) - int(nas_ranks[i])
        sign  = "+" if shift > 0 else ""
        lines.append(
            f"  A{idxs[i]:>2}    {nas[i]:>7.2f}  "
            f"{int(nas_ranks[i])+1:>9}  "
            f"{val[i]:>7.2f}  "
            f"{int(val_ranks[i])+1:>9}  "
            f"{sign}{shift:>+10}"
        )

    lines += [
        "",
        "── Correlation trend over cycles ───────────────────────────",
    ]
    for s in stats_history:
        lines.append(
            f"  Cycle {s['cycle']:>2}: Spearman ρ={s.get('spearman_r', float('nan')):.4f} "
            f"(p={s.get('spearman_p', 1.0):.4f})  "
            f"Best-val ρ={s.get('spearman_r_best', float('nan')):.4f} "
            f"(p={s.get('spearman_p_best', 1.0):.4f})"
        )

    summary_text = "\n".join(lines)
    (out_dir / "summary.txt").write_text(summary_text)
    print(summary_text)


# ── main ──────────────────────────────────────────────────────────────────────

def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="NAS proxy predictability analysis")
    ap.add_argument("--experiment-dir", default="./full_dataset_experiment",
                    help="Path to full_dataset_experiment directory")
    ap.add_argument("--output-dir", default=None,
                    help="Output directory for plots/summary (default: <experiment-dir>/nas_predictability_analysis)")
    return ap.parse_args()


def main() -> None:
    args = parse_args()
    exp_dir = Path(args.experiment_dir)
    out_dir = Path(args.output_dir) if args.output_dir else exp_dir / "nas_predictability_analysis"
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"Loading data from {exp_dir}")
    arch_list, best_summary, stats_history = load_data(exp_dir)
    print(f"  {len(best_summary)} architectures, {len(stats_history)} cycles of stats")

    print("Generating plots…")
    plot_nas_vs_full_acc(best_summary, out_dir)
    print("  ✓ nas_vs_full_acc.png")
    plot_rank_comparison(best_summary, out_dir)
    print("  ✓ rank_comparison.png")
    plot_correlation_evolution(stats_history, out_dir)
    print("  ✓ correlation_evolution.png")
    plot_topk_recall(best_summary, out_dir)
    print("  ✓ topk_recall.png")
    plot_tied_nas_case(best_summary, arch_list, out_dir)
    print("  ✓ tied_nas_score_comparison.png")
    plot_rank_matrix(best_summary, out_dir)
    print("  ✓ rank_matrix.png")

    print("\nWriting text summary…")
    write_text_summary(best_summary, arch_list, stats_history, out_dir)
    print(f"\nAll outputs saved to {out_dir}")


if __name__ == "__main__":
    main()
