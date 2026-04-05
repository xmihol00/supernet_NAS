from __future__ import annotations

import argparse
import csv
import json
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Sequence

from experiment_stats import build_full_statistics
from experiment_viz import (
    plot_convergence_by_algorithm,
    plot_effect_sizes,
    plot_metric_distributions,
    plot_overall_progress,
    plot_run_comparison_scatter,
    plot_statistical_pvalues,
)


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


def write_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2)


def write_run_records_csv(path: Path, records: Sequence[Dict[str, object]]) -> None:
    fields = [
        "algorithm",
        "run_index",
        "seed",
        "status",
        "return_code",
        "run_dir",
        "started_at",
        "finished_at",
        "best_quant_acc1",
        "best_fitness",
        "compile_success_rate",
        "total_candidates_evaluated",
        "compiled_candidates",
        "elapsed_seconds",
    ]

    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for record in records:
            summary_raw = record.get("summary")
            summary = summary_raw if isinstance(summary_raw, dict) else {}
            writer.writerow(
                {
                    "algorithm": record.get("algorithm", ""),
                    "run_index": record.get("run_index", ""),
                    "seed": record.get("seed", ""),
                    "status": record.get("status", ""),
                    "return_code": record.get("return_code", ""),
                    "run_dir": record.get("run_dir", ""),
                    "started_at": record.get("started_at", ""),
                    "finished_at": record.get("finished_at", ""),
                    "best_quant_acc1": summary.get("best_quant_acc1", ""),
                    "best_fitness": summary.get("best_fitness", ""),
                    "compile_success_rate": summary.get("compile_success_rate", ""),
                    "total_candidates_evaluated": summary.get("total_candidates_evaluated", ""),
                    "compiled_candidates": summary.get("compiled_candidates", ""),
                    "elapsed_seconds": summary.get("elapsed_seconds", ""),
                }
            )


def load_run_records(experiment_dir: Path) -> List[Dict[str, object]]:
    records_path = experiment_dir / "run_records.json"
    payload = read_json(records_path, default=[])
    if not isinstance(payload, list):
        return []
    return [item for item in payload if isinstance(item, dict)]


def render_visualizations(
    output_root: Path,
    run_records: Sequence[Dict[str, object]],
    statistics_payload: Dict[str, object],
) -> None:
    viz_dir = output_root / "visualizations"
    viz_dir.mkdir(parents=True, exist_ok=True)

    plot_overall_progress(run_records, viz_dir / "overall_progress.png")

    run_metrics = statistics_payload.get("run_metrics", [])
    if not isinstance(run_metrics, list):
        return

    plot_metric_distributions(
        run_metrics,
        metric="best_quant_acc1",
        output_png=viz_dir / "distribution_best_quant_acc1.png",
        ylabel="Best quantized accuracy (acc1)",
    )
    plot_metric_distributions(
        run_metrics,
        metric="compile_success_rate",
        output_png=viz_dir / "distribution_compile_success_rate.png",
        ylabel="Compile success rate",
    )
    plot_metric_distributions(
        run_metrics,
        metric="elapsed_seconds",
        output_png=viz_dir / "distribution_elapsed_seconds.png",
        ylabel="Elapsed seconds",
    )
    plot_convergence_by_algorithm(run_metrics, viz_dir / "convergence_best_fitness.png")
    plot_run_comparison_scatter(run_metrics, viz_dir / "run_tradeoff_scatter.png")
    plot_statistical_pvalues(statistics_payload, viz_dir / "statistical_pvalues.png")
    plot_effect_sizes(statistics_payload, viz_dir / "effect_sizes.png")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        "Merge separate per-algorithm NAS experiments (e.g., SGA and regularized evolution) into one comparison bundle."
    )
    parser.add_argument("--sga-dir", type=Path, required=True, help="Output directory from baseline_sga run.")
    parser.add_argument("--reg-evo-dir", type=Path, required=True, help="Output directory from regularized_evolution run.")
    parser.add_argument("--output-dir", type=Path, required=True, help="Where merged comparison artifacts will be written.")
    parser.add_argument("--bootstrap-samples", type=int, default=10000)
    parser.add_argument("--confidence", type=float, default=0.95)
    parser.add_argument("--random-seed", type=int, default=1234)
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    sga_dir = args.sga_dir.resolve()
    reg_evo_dir = args.reg_evo_dir.resolve()
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    sga_records = load_run_records(sga_dir)
    reg_records = load_run_records(reg_evo_dir)

    merged_records = [*sga_records, *reg_records]
    merged_records.sort(key=lambda item: str(item.get("started_at", "")))

    metric_names = [
        "best_quant_acc1",
        "best_fitness",
        "compile_success_rate",
        "elapsed_seconds",
        "total_candidates_evaluated",
        "compiled_candidates",
    ]
    algorithms = sorted({str(item.get("algorithm", "")) for item in merged_records if item.get("algorithm")})

    statistics_payload = build_full_statistics(
        run_records=merged_records,
        metric_names=metric_names,
        algorithms=algorithms,
        bootstrap_samples=args.bootstrap_samples,
        confidence=args.confidence,
        random_seed=args.random_seed,
    )

    write_json(output_dir / "run_records.json", merged_records)
    write_run_records_csv(output_dir / "run_records.csv", merged_records)
    write_json(output_dir / "statistics.json", statistics_payload)

    source_manifest = {
        "generated_at": utc_now(),
        "source_dirs": {
            "baseline_sga": str(sga_dir),
            "regularized_evolution": str(reg_evo_dir),
        },
        "record_counts": {
            "baseline_sga": len(sga_records),
            "regularized_evolution": len(reg_records),
            "merged_total": len(merged_records),
        },
        "algorithms_detected": algorithms,
    }
    write_json(output_dir / "merge_sources.json", source_manifest)

    render_visualizations(output_dir, merged_records, statistics_payload)

    successful = sum(1 for record in merged_records if str(record.get("status", "")) == "success")
    summary = {
        "timestamp": utc_now(),
        "output_root": str(output_dir),
        "total_runs_finished": len(merged_records),
        "successful_runs": successful,
        "failed_runs": len(merged_records) - successful,
        "algorithms": algorithms,
        "scipy_available": bool(statistics_payload.get("scipy_available", False)),
    }
    write_json(output_dir / "experiment_summary.json", summary)

    print(json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    main()
