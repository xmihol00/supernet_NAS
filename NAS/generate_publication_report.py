from __future__ import annotations

import argparse
import csv
import json
import math
from datetime import datetime
from pathlib import Path
from typing import Dict, Iterable, List, Sequence, Tuple


def utc_now() -> str:
    return datetime.utcnow().isoformat(timespec="seconds") + "Z"


def read_json(path: Path, default: object) -> object:
    if not path.exists():
        return default
    try:
        with path.open("r", encoding="utf-8") as handle:
            return json.load(handle)
    except Exception:
        return default


def read_csv(path: Path) -> List[Dict[str, str]]:
    if not path.exists():
        return []
    with path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        return list(reader)


def to_float(value: object, default: float = float("nan")) -> float:
    if isinstance(value, bool):
        return float(value)
    if isinstance(value, (int, float)):
        return float(value)
    if isinstance(value, str):
        try:
            return float(value)
        except Exception:
            return default
    return default


def to_int(value: object, default: int = 0) -> int:
    if isinstance(value, bool):
        return int(value)
    if isinstance(value, int):
        return value
    if isinstance(value, float):
        return int(value)
    if isinstance(value, str):
        try:
            return int(float(value))
        except Exception:
            return default
    return default


def fmt(value: object, digits: int = 4, nan: str = "NA") -> str:
    number = to_float(value)
    if math.isnan(number):
        return nan
    return f"{number:.{digits}f}"


def rel_path(target: Path, base: Path) -> str:
    try:
        return str(target.resolve().relative_to(base.resolve()))
    except Exception:
        return str(target)


def markdown_table(headers: Sequence[str], rows: Sequence[Sequence[object]]) -> str:
    line_header = "| " + " | ".join(headers) + " |"
    line_sep = "| " + " | ".join(["---"] * len(headers)) + " |"
    body = ["| " + " | ".join(str(col) for col in row) + " |" for row in rows]
    return "\n".join([line_header, line_sep, *body])


def collect_run_records(experiment_dir: Path) -> List[Dict[str, object]]:
    run_records_json = experiment_dir / "run_records.json"
    run_records_csv = experiment_dir / "run_records.csv"

    records_json = read_json(run_records_json, default=[])
    if isinstance(records_json, list):
        valid = [item for item in records_json if isinstance(item, dict)]
        if valid:
            return valid

    csv_rows = read_csv(run_records_csv)
    fallback: List[Dict[str, object]] = []
    for row in csv_rows:
        fallback.append(
            {
                "algorithm": row.get("algorithm", ""),
                "run_index": to_int(row.get("run_index", "-1"), default=-1),
                "seed": to_int(row.get("seed", "-1"), default=-1),
                "status": row.get("status", "unknown"),
                "return_code": to_int(row.get("return_code", "0"), default=0),
                "run_dir": row.get("run_dir", ""),
                "summary": {
                    "best_quant_acc1": to_float(row.get("best_quant_acc1", "nan")),
                    "best_fitness": to_float(row.get("best_fitness", "nan")),
                    "compile_success_rate": to_float(row.get("compile_success_rate", "nan")),
                    "total_candidates_evaluated": to_int(row.get("total_candidates_evaluated", "0")),
                    "compiled_candidates": to_int(row.get("compiled_candidates", "0")),
                    "elapsed_seconds": to_float(row.get("elapsed_seconds", "nan")),
                },
            }
        )
    return fallback


def build_algorithm_table(per_algorithm_summary: Dict[str, object]) -> Tuple[List[str], List[List[object]]]:
    headers = [
        "Algorithm",
        "Metric",
        "N",
        "Mean",
        "Std",
        "Median",
        "IQR",
        "Min",
        "Max",
    ]

    rows: List[List[object]] = []
    for algorithm in sorted(per_algorithm_summary.keys()):
        metric_payload = per_algorithm_summary.get(algorithm)
        if not isinstance(metric_payload, dict):
            continue

        for metric_name in sorted(metric_payload.keys()):
            metric_stats = metric_payload.get(metric_name)
            if not isinstance(metric_stats, dict):
                continue
            rows.append(
                [
                    algorithm,
                    metric_name,
                    to_int(metric_stats.get("count", 0)),
                    fmt(metric_stats.get("mean", float("nan"))),
                    fmt(metric_stats.get("std", float("nan"))),
                    fmt(metric_stats.get("median", float("nan"))),
                    fmt(metric_stats.get("iqr", float("nan"))),
                    fmt(metric_stats.get("min", float("nan"))),
                    fmt(metric_stats.get("max", float("nan"))),
                ]
            )
    return headers, rows


def build_stat_tests_table(pairwise_results: Sequence[object]) -> Tuple[List[str], List[List[object]]]:
    headers = [
        "Algorithm A",
        "Algorithm B",
        "Metric",
        "Test",
        "Statistic",
        "p",
        "p (Holm)",
        "Effect",
        "Effect Type",
        "Mean Diff (A-B)",
        "CI Low",
        "CI High",
        "Significant (0.05)",
    ]
    rows: List[List[object]] = []

    for pair in pairwise_results:
        if not isinstance(pair, dict):
            continue
        algorithm_a = str(pair.get("algorithm_a", ""))
        algorithm_b = str(pair.get("algorithm_b", ""))
        tests = pair.get("tests", [])
        if not isinstance(tests, list):
            continue

        for test in tests:
            if not isinstance(test, dict):
                continue
            rows.append(
                [
                    algorithm_a,
                    algorithm_b,
                    str(test.get("metric", "")),
                    str(test.get("test_type", "")),
                    fmt(test.get("statistic", float("nan"))),
                    fmt(test.get("p_value", float("nan"))),
                    fmt(test.get("p_value_holm_bonferroni", float("nan"))),
                    fmt(test.get("effect_size", float("nan"))),
                    str(test.get("effect_size_type", "")),
                    fmt(test.get("mean_diff_a_minus_b", float("nan"))),
                    fmt(test.get("bootstrap_ci_low", float("nan"))),
                    fmt(test.get("bootstrap_ci_high", float("nan"))),
                    str(bool(test.get("significant_alpha_0_05", False))),
                ]
            )

    return headers, rows


def build_run_table(run_records: Sequence[Dict[str, object]]) -> Tuple[List[str], List[List[object]]]:
    headers = [
        "Algorithm",
        "Run",
        "Seed",
        "Status",
        "Best Quant Acc1",
        "Best Fitness",
        "Compile Success Rate",
        "Candidates",
        "Compiled",
        "Elapsed Seconds",
    ]
    rows: List[List[object]] = []

    sorted_records = sorted(
        run_records,
        key=lambda row: (str(row.get("algorithm", "")), to_int(row.get("run_index", -1))),
    )
    for record in sorted_records:
        summary_raw = record.get("summary")
        summary = summary_raw if isinstance(summary_raw, dict) else {}
        rows.append(
            [
                str(record.get("algorithm", "")),
                to_int(record.get("run_index", -1)),
                to_int(record.get("seed", -1)),
                str(record.get("status", "")),
                fmt(summary.get("best_quant_acc1", float("nan"))),
                fmt(summary.get("best_fitness", float("nan"))),
                fmt(summary.get("compile_success_rate", float("nan"))),
                to_int(summary.get("total_candidates_evaluated", 0)),
                to_int(summary.get("compiled_candidates", 0)),
                fmt(summary.get("elapsed_seconds", float("nan"))),
            ]
        )

    return headers, rows


def build_best_model_table(run_records: Sequence[Dict[str, object]]) -> Tuple[List[str], List[List[object]]]:
    headers = [
        "Algorithm",
        "Run",
        "Seed",
        "Best Quant Acc1",
        "Resolution",
        "Stem Width",
        "Stage Depths",
        "Stage Widths",
        "Run Dir",
    ]
    rows: List[List[object]] = []

    best_by_algorithm: Dict[str, Dict[str, object]] = {}
    for record in run_records:
        if str(record.get("status", "")) != "success":
            continue
        algorithm = str(record.get("algorithm", "unknown"))

        summary_raw = record.get("summary")
        summary = summary_raw if isinstance(summary_raw, dict) else {}
        score = to_float(summary.get("best_quant_acc1", float("nan")))
        if math.isnan(score):
            continue

        current = best_by_algorithm.get(algorithm)
        if current is None:
            best_by_algorithm[algorithm] = record
            continue

        current_summary_raw = current.get("summary")
        current_summary = current_summary_raw if isinstance(current_summary_raw, dict) else {}
        current_score = to_float(current_summary.get("best_quant_acc1", float("nan")))
        if math.isnan(current_score) or score > current_score:
            best_by_algorithm[algorithm] = record

    for algorithm in sorted(best_by_algorithm.keys()):
        record = best_by_algorithm[algorithm]
        summary_raw = record.get("summary")
        summary = summary_raw if isinstance(summary_raw, dict) else {}
        config_raw = summary.get("best_config")
        config = config_raw if isinstance(config_raw, dict) else {}

        rows.append(
            [
                algorithm,
                to_int(record.get("run_index", -1)),
                to_int(record.get("seed", -1)),
                fmt(summary.get("best_quant_acc1", float("nan"))),
                to_int(config.get("resolution", -1)),
                to_int(config.get("stem_width", -1)),
                str(config.get("stage_depths", "NA")),
                str(config.get("stage_widths", "NA")),
                str(record.get("run_dir", "")),
            ]
        )

    return headers, rows


def save_csv(path: Path, headers: Sequence[str], rows: Sequence[Sequence[object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(headers)
        writer.writerows(rows)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        "Generate publication-ready report from multi-run NAS experiment outputs."
    )
    parser.add_argument(
        "--experiment-dir",
        type=Path,
        required=True,
        help="Path to experiment output directory created by `multi_run_nas_experiment.py`.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Output directory for report artifacts. Defaults to <experiment-dir>/publication_report.",
    )
    parser.add_argument(
        "--title",
        type=str,
        default="Neural Architecture Search Comparison Report",
        help="Report title.",
    )
    parser.add_argument(
        "--author",
        type=str,
        default="Auto-generated",
        help="Author/owner string included in report header.",
    )
    parser.add_argument(
        "--notes",
        type=str,
        default="",
        help="Optional free-text note included in the report.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    experiment_dir = args.experiment_dir.resolve()
    if not experiment_dir.exists() or not experiment_dir.is_dir():
        raise FileNotFoundError(f"Experiment directory does not exist: {experiment_dir}")

    output_dir = (args.output_dir.resolve() if args.output_dir else (experiment_dir / "publication_report"))
    output_dir.mkdir(parents=True, exist_ok=True)

    statistics_path = experiment_dir / "statistics.json"
    config_path = experiment_dir / "experiment_config.json"
    summary_path = experiment_dir / "experiment_summary.json"

    statistics = read_json(statistics_path, default={})
    if not isinstance(statistics, dict):
        statistics = {}

    config = read_json(config_path, default={})
    if not isinstance(config, dict):
        config = {}

    experiment_summary = read_json(summary_path, default={})
    if not isinstance(experiment_summary, dict):
        experiment_summary = {}

    run_records = collect_run_records(experiment_dir)

    algo_headers, algo_rows = build_algorithm_table(
        statistics.get("per_algorithm_summary", {})
        if isinstance(statistics.get("per_algorithm_summary", {}), dict)
        else {}
    )
    stat_headers, stat_rows = build_stat_tests_table(
        statistics.get("pairwise_results", []) if isinstance(statistics.get("pairwise_results", []), list) else []
    )
    run_headers, run_rows = build_run_table(run_records)
    best_headers, best_rows = build_best_model_table(run_records)

    tables_dir = output_dir / "tables"
    save_csv(tables_dir / "algorithm_summary.csv", algo_headers, algo_rows)
    save_csv(tables_dir / "statistical_tests.csv", stat_headers, stat_rows)
    save_csv(tables_dir / "runs_overview.csv", run_headers, run_rows)
    save_csv(tables_dir / "best_models.csv", best_headers, best_rows)

    visualizations_dir = experiment_dir / "visualizations"
    figures = [
        visualizations_dir / "overall_progress.png",
        visualizations_dir / "convergence_best_fitness.png",
        visualizations_dir / "distribution_best_quant_acc1.png",
        visualizations_dir / "distribution_compile_success_rate.png",
        visualizations_dir / "distribution_elapsed_seconds.png",
        visualizations_dir / "run_tradeoff_scatter.png",
        visualizations_dir / "statistical_pvalues.png",
        visualizations_dir / "effect_sizes.png",
    ]

    existing_figures = [path for path in figures if path.exists()]

    config_args = config.get("args", {}) if isinstance(config.get("args", {}), dict) else {}
    key_args = [
        "algorithms",
        "runs_per_algorithm",
        "base_seed",
        "seed_stride",
        "generations",
        "population_size",
        "offspring_per_generation",
        "epochs_per_candidate",
        "train_dataset",
        "eval_dataset",
        "checkpoint",
        "device",
    ]

    setup_rows: List[List[object]] = []
    for key in key_args:
        setup_rows.append([key, config_args.get(key, "NA")])

    successful_runs = to_int(experiment_summary.get("successful_runs", 0), default=0)
    failed_runs = to_int(experiment_summary.get("failed_runs", 0), default=0)
    total_finished = to_int(experiment_summary.get("total_runs_finished", len(run_records)), default=len(run_records))

    report_lines: List[str] = []
    report_lines.append(f"# {args.title}")
    report_lines.append("")
    report_lines.append(f"- Generated at: `{utc_now()}`")
    report_lines.append(f"- Author: `{args.author}`")
    report_lines.append(f"- Experiment directory: `{experiment_dir}`")
    report_lines.append(f"- Successful runs: `{successful_runs}` / `{max(1, total_finished)}`")
    report_lines.append(f"- Failed runs: `{failed_runs}`")
    if args.notes:
        report_lines.append(f"- Notes: {args.notes}")
    report_lines.append("")

    report_lines.append("## Experimental Setup")
    report_lines.append("")
    report_lines.append(markdown_table(["Parameter", "Value"], setup_rows))
    report_lines.append("")

    report_lines.append("## Run-Level Overview")
    report_lines.append("")
    report_lines.append(markdown_table(run_headers, run_rows))
    report_lines.append("")

    report_lines.append("## Algorithm Summary Statistics")
    report_lines.append("")
    report_lines.append(markdown_table(algo_headers, algo_rows))
    report_lines.append("")

    report_lines.append("## Pairwise Statistical Tests")
    report_lines.append("")
    report_lines.append(markdown_table(stat_headers, stat_rows))
    report_lines.append("")

    report_lines.append("## Best Models per Algorithm")
    report_lines.append("")
    report_lines.append(markdown_table(best_headers, best_rows))
    report_lines.append("")

    report_lines.append("## Figures")
    report_lines.append("")
    if existing_figures:
        for figure_path in existing_figures:
            figure_rel = rel_path(figure_path, output_dir)
            caption = figure_path.stem.replace("_", " ").title()
            report_lines.append(f"### {caption}")
            report_lines.append("")
            report_lines.append(f"![{caption}]({figure_rel})")
            report_lines.append("")
    else:
        report_lines.append("No figure files were found in the experiment visualization directory.")
        report_lines.append("")

    report_lines.append("## Reproducibility Artifacts")
    report_lines.append("")
    report_lines.append("- `tables/algorithm_summary.csv`")
    report_lines.append("- `tables/statistical_tests.csv`")
    report_lines.append("- `tables/runs_overview.csv`")
    report_lines.append("- `tables/best_models.csv`")
    report_lines.append(f"- Original statistics: `{statistics_path}`")
    report_lines.append(f"- Original experiment summary: `{summary_path}`")
    report_lines.append("")

    report_path = output_dir / "publication_report.md"
    with report_path.open("w", encoding="utf-8") as handle:
        handle.write("\n".join(report_lines).rstrip() + "\n")

    machine_summary = {
        "generated_at": utc_now(),
        "experiment_dir": str(experiment_dir),
        "report_markdown": str(report_path),
        "output_dir": str(output_dir),
        "tables": {
            "algorithm_summary": str(tables_dir / "algorithm_summary.csv"),
            "statistical_tests": str(tables_dir / "statistical_tests.csv"),
            "runs_overview": str(tables_dir / "runs_overview.csv"),
            "best_models": str(tables_dir / "best_models.csv"),
        },
        "figures_included": [str(path) for path in existing_figures],
    }
    with (output_dir / "publication_report_summary.json").open("w", encoding="utf-8") as handle:
        json.dump(machine_summary, handle, indent=2)

    print(json.dumps(machine_summary, indent=2), flush=True)


if __name__ == "__main__":
    main()
