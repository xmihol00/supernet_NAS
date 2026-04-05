from __future__ import annotations

import math
from dataclasses import asdict, dataclass
from typing import Dict, Iterable, List, Sequence, Tuple

import numpy as np

try:
    from scipy import stats

    SCIPY_AVAILABLE = True
except Exception:
    stats = None
    SCIPY_AVAILABLE = False


@dataclass
class RunMetrics:
    algorithm: str
    run_index: int
    seed: int
    status: str
    run_dir: str
    best_quant_acc1: float
    best_fitness: float
    compile_success_rate: float
    total_candidates_evaluated: int
    compiled_candidates: int
    elapsed_seconds: float
    generations_completed: int
    history_best_fitness: List[float]
    history_population_mean_fitness: List[float]

    def to_dict(self) -> Dict[str, object]:
        return asdict(self)


def _to_float(value: object, default: float = float("nan")) -> float:
    if isinstance(value, (int, float)):
        return float(value)
    return default


def _to_int(value: object, default: int = 0) -> int:
    if isinstance(value, bool):
        return int(value)
    if isinstance(value, int):
        return value
    if isinstance(value, float):
        return int(value)
    return default


def build_run_metrics(run_record: Dict[str, object]) -> RunMetrics:
    raw_summary = run_record.get("summary")
    summary: Dict[str, object] = raw_summary if isinstance(raw_summary, dict) else {}

    raw_history = run_record.get("history")
    history: List[object] = raw_history if isinstance(raw_history, list) else []

    history_best_fitness: List[float] = []
    history_population_mean_fitness: List[float] = []
    for item in history:
        if not isinstance(item, dict):
            continue
        history_best_fitness.append(_to_float(item.get("best_fitness"), default=float("nan")))
        history_population_mean_fitness.append(
            _to_float(item.get("population_mean_fitness"), default=float("nan"))
        )

    return RunMetrics(
        algorithm=str(run_record.get("algorithm", "unknown")),
        run_index=_to_int(run_record.get("run_index"), default=-1),
        seed=_to_int(run_record.get("seed"), default=-1),
        status=str(run_record.get("status", "unknown")),
        run_dir=str(run_record.get("run_dir", "")),
        best_quant_acc1=_to_float(summary.get("best_quant_acc1"), default=float("nan")),
        best_fitness=_to_float(summary.get("best_fitness"), default=float("nan")),
        compile_success_rate=_to_float(summary.get("compile_success_rate"), default=float("nan")),
        total_candidates_evaluated=_to_int(summary.get("total_candidates_evaluated"), default=0),
        compiled_candidates=_to_int(summary.get("compiled_candidates"), default=0),
        elapsed_seconds=_to_float(summary.get("elapsed_seconds"), default=float("nan")),
        generations_completed=len(history_best_fitness),
        history_best_fitness=history_best_fitness,
        history_population_mean_fitness=history_population_mean_fitness,
    )


def _clean(values: Sequence[float]) -> np.ndarray:
    arr = np.asarray(values, dtype=np.float64)
    return arr[np.isfinite(arr)]


def _bootstrap_difference_ci(
    values_a: Sequence[float],
    values_b: Sequence[float],
    rng: np.random.Generator,
    confidence: float = 0.95,
    num_bootstrap: int = 10000,
) -> Tuple[float, float]:
    a = _clean(values_a)
    b = _clean(values_b)
    if len(a) == 0 or len(b) == 0:
        return float("nan"), float("nan")

    diffs = np.empty(num_bootstrap, dtype=np.float64)
    for idx in range(num_bootstrap):
        sample_a = rng.choice(a, size=len(a), replace=True)
        sample_b = rng.choice(b, size=len(b), replace=True)
        diffs[idx] = float(np.mean(sample_a) - np.mean(sample_b))

    alpha = 1.0 - confidence
    lo = float(np.quantile(diffs, alpha / 2.0))
    hi = float(np.quantile(diffs, 1.0 - alpha / 2.0))
    return lo, hi


def _cohen_d(values_a: Sequence[float], values_b: Sequence[float]) -> float:
    a = _clean(values_a)
    b = _clean(values_b)
    if len(a) < 2 or len(b) < 2:
        return float("nan")

    mean_diff = float(np.mean(a) - np.mean(b))
    var_a = float(np.var(a, ddof=1))
    var_b = float(np.var(b, ddof=1))
    pooled = math.sqrt(((len(a) - 1) * var_a + (len(b) - 1) * var_b) / (len(a) + len(b) - 2))
    if pooled == 0.0:
        return 0.0
    return mean_diff / pooled


def _cliffs_delta(values_a: Sequence[float], values_b: Sequence[float]) -> float:
    a = _clean(values_a)
    b = _clean(values_b)
    if len(a) == 0 or len(b) == 0:
        return float("nan")

    greater = 0
    lower = 0
    for a_val in a:
        greater += int(np.sum(a_val > b))
        lower += int(np.sum(a_val < b))
    total = len(a) * len(b)
    return float((greater - lower) / max(1, total))


def _normality_pvalue(values: Sequence[float]) -> float:
    clean = _clean(values)
    if len(clean) < 3:
        return float("nan")
    if not SCIPY_AVAILABLE:
        return float("nan")
    assert stats is not None
    if len(clean) > 5000:
        clean = clean[:5000]
    return float(stats.shapiro(clean).pvalue)


def _welch_ttest(values_a: Sequence[float], values_b: Sequence[float]) -> Tuple[float, float]:
    if not SCIPY_AVAILABLE:
        return float("nan"), float("nan")
    assert stats is not None
    result = stats.ttest_ind(values_a, values_b, equal_var=False, nan_policy="omit")
    return float(result.statistic), float(result.pvalue)


def _mann_whitney(values_a: Sequence[float], values_b: Sequence[float]) -> Tuple[float, float]:
    if not SCIPY_AVAILABLE:
        return float("nan"), float("nan")
    assert stats is not None
    result = stats.mannwhitneyu(values_a, values_b, alternative="two-sided")
    return float(result.statistic), float(result.pvalue)


def _holm_bonferroni(pvalues: Sequence[float]) -> List[float]:
    indexed = [(idx, p) for idx, p in enumerate(pvalues)]
    indexed.sort(key=lambda item: item[1])
    m = len(indexed)

    adjusted_sorted = [0.0] * m
    running_max = 0.0
    for rank, (_, pvalue) in enumerate(indexed):
        adjusted = (m - rank) * pvalue
        running_max = max(running_max, adjusted)
        adjusted_sorted[rank] = min(1.0, running_max)

    adjusted = [1.0] * m
    for rank, (original_idx, _) in enumerate(indexed):
        adjusted[original_idx] = adjusted_sorted[rank]
    return adjusted


def summarize_by_algorithm(
    run_metrics: Sequence[RunMetrics],
    metrics: Sequence[str],
) -> Dict[str, Dict[str, Dict[str, float]]]:
    summary: Dict[str, Dict[str, Dict[str, float]]] = {}
    algorithms = sorted(set(item.algorithm for item in run_metrics))

    for algorithm in algorithms:
        summary[algorithm] = {}
        group = [item for item in run_metrics if item.algorithm == algorithm and item.status == "success"]
        for metric in metrics:
            values = _clean([_to_float(getattr(item, metric, float("nan"))) for item in group])
            if len(values) == 0:
                summary[algorithm][metric] = {
                    "count": 0.0,
                    "mean": float("nan"),
                    "std": float("nan"),
                    "median": float("nan"),
                    "iqr": float("nan"),
                    "min": float("nan"),
                    "max": float("nan"),
                }
                continue

            q1 = float(np.quantile(values, 0.25))
            q3 = float(np.quantile(values, 0.75))
            summary[algorithm][metric] = {
                "count": float(len(values)),
                "mean": float(np.mean(values)),
                "std": float(np.std(values, ddof=1)) if len(values) > 1 else 0.0,
                "median": float(np.median(values)),
                "iqr": q3 - q1,
                "min": float(np.min(values)),
                "max": float(np.max(values)),
            }

    return summary


def pairwise_compare_algorithms(
    run_metrics: Sequence[RunMetrics],
    algorithm_a: str,
    algorithm_b: str,
    metrics: Sequence[str],
    bootstrap_samples: int = 10000,
    confidence: float = 0.95,
    random_seed: int = 12345,
) -> Dict[str, object]:
    rng = np.random.default_rng(random_seed)
    group_a = [item for item in run_metrics if item.algorithm == algorithm_a and item.status == "success"]
    group_b = [item for item in run_metrics if item.algorithm == algorithm_b and item.status == "success"]

    tests: List[Dict[str, object]] = []
    raw_pvalues: List[float] = []

    for metric in metrics:
        values_a = _clean([_to_float(getattr(item, metric, float("nan"))) for item in group_a])
        values_b = _clean([_to_float(getattr(item, metric, float("nan"))) for item in group_b])

        if len(values_a) == 0 or len(values_b) == 0:
            tests.append(
                {
                    "metric": metric,
                    "n_a": int(len(values_a)),
                    "n_b": int(len(values_b)),
                    "test_type": "insufficient_data",
                    "statistic": float("nan"),
                    "p_value": float("nan"),
                    "normality_p_a": float("nan"),
                    "normality_p_b": float("nan"),
                    "effect_size_type": "none",
                    "effect_size": float("nan"),
                    "mean_diff_a_minus_b": float("nan"),
                    "bootstrap_ci_low": float("nan"),
                    "bootstrap_ci_high": float("nan"),
                }
            )
            raw_pvalues.append(1.0)
            continue

        normality_a = _normality_pvalue(values_a)
        normality_b = _normality_pvalue(values_b)
        both_normal = (
            np.isfinite(normality_a)
            and np.isfinite(normality_b)
            and normality_a > 0.05
            and normality_b > 0.05
            and len(values_a) >= 3
            and len(values_b) >= 3
        )

        if both_normal:
            statistic, pvalue = _welch_ttest(values_a, values_b)
            effect_size_type = "cohen_d"
            effect_size = _cohen_d(values_a, values_b)
            test_type = "welch_t_test"
        else:
            statistic, pvalue = _mann_whitney(values_a, values_b)
            effect_size_type = "cliffs_delta"
            effect_size = _cliffs_delta(values_a, values_b)
            test_type = "mann_whitney_u"

        mean_diff = float(np.mean(values_a) - np.mean(values_b))
        ci_low, ci_high = _bootstrap_difference_ci(
            values_a,
            values_b,
            rng=rng,
            confidence=confidence,
            num_bootstrap=bootstrap_samples,
        )

        tests.append(
            {
                "metric": metric,
                "n_a": int(len(values_a)),
                "n_b": int(len(values_b)),
                "test_type": test_type,
                "statistic": statistic,
                "p_value": pvalue,
                "normality_p_a": normality_a,
                "normality_p_b": normality_b,
                "effect_size_type": effect_size_type,
                "effect_size": effect_size,
                "mean_diff_a_minus_b": mean_diff,
                "bootstrap_ci_low": ci_low,
                "bootstrap_ci_high": ci_high,
            }
        )
        raw_pvalues.append(pvalue if np.isfinite(pvalue) else 1.0)

    adjusted = _holm_bonferroni(raw_pvalues)
    for idx, adj in enumerate(adjusted):
        tests[idx]["p_value_holm_bonferroni"] = adj
        tests[idx]["significant_alpha_0_05"] = bool(adj < 0.05)

    return {
        "algorithm_a": algorithm_a,
        "algorithm_b": algorithm_b,
        "metrics_tested": list(metrics),
        "scipy_available": SCIPY_AVAILABLE,
        "tests": tests,
    }


def build_full_statistics(
    run_records: Sequence[Dict[str, object]],
    metric_names: Sequence[str],
    algorithms: Sequence[str],
    bootstrap_samples: int = 10000,
    confidence: float = 0.95,
    random_seed: int = 12345,
) -> Dict[str, object]:
    metrics = [build_run_metrics(record) for record in run_records]

    per_algorithm_summary = summarize_by_algorithm(metrics, metrics=metric_names)

    pairwise_results: List[Dict[str, object]] = []
    for idx in range(len(algorithms)):
        for jdx in range(idx + 1, len(algorithms)):
            pairwise_results.append(
                pairwise_compare_algorithms(
                    run_metrics=metrics,
                    algorithm_a=algorithms[idx],
                    algorithm_b=algorithms[jdx],
                    metrics=metric_names,
                    bootstrap_samples=bootstrap_samples,
                    confidence=confidence,
                    random_seed=random_seed,
                )
            )

    return {
        "metric_names": list(metric_names),
        "run_metrics": [item.to_dict() for item in metrics],
        "per_algorithm_summary": per_algorithm_summary,
        "pairwise_results": pairwise_results,
        "scipy_available": SCIPY_AVAILABLE,
    }
