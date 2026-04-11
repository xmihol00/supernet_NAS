from __future__ import annotations

import argparse
import csv
import json
import logging
import os
import select
import shlex
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Sequence

from experiment_stats import build_full_statistics
from experiment_viz import (
    plot_convergence_by_algorithm,
    plot_effect_sizes,
    plot_live_run_progress,
    plot_metric_distributions,
    plot_overall_progress,
    plot_run_comparison_scatter,
    plot_statistical_pvalues,
)


def utc_now() -> str:
    return datetime.utcnow().isoformat(timespec="seconds") + "Z"


def configure_logging(log_file: Path) -> logging.Logger:
    log_file.parent.mkdir(parents=True, exist_ok=True)

    logger = logging.getLogger("multi_run_nas_experiment")
    logger.setLevel(logging.INFO)
    logger.handlers.clear()

    formatter = logging.Formatter("%(asctime)s | %(levelname)s | %(message)s")

    stream_handler = logging.StreamHandler(sys.stdout)
    stream_handler.setFormatter(formatter)
    logger.addHandler(stream_handler)

    file_handler = logging.FileHandler(log_file, encoding="utf-8")
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)

    return logger


def append_jsonl(path: Path, payload: Dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(payload, ensure_ascii=False) + "\n")


def read_json(path: Path, default: object) -> object:
    if not path.exists():
        return default
    try:
        with path.open("r", encoding="utf-8") as handle:
            return json.load(handle)
    except Exception:
        return default


def read_progress_events(run_dir: Path) -> List[Dict[str, object]]:
    events: List[Dict[str, object]] = []
    progress_path = run_dir / "progress.jsonl"
    if not progress_path.exists():
        return events

    with progress_path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            try:
                payload = json.loads(line)
            except json.JSONDecodeError:
                continue
            if isinstance(payload, dict):
                events.append(payload)
    return events


def parse_run_dir_from_output(line: str) -> Path | None:
    marker = "Run directory:"
    if marker not in line:
        return None
    raw = line.split(marker, maxsplit=1)[1].strip()
    if not raw:
        return None
    return Path(raw)


def detect_run_dir(run_workspace: Path) -> Path | None:
    if not run_workspace.exists():
        return None
    candidates = [path for path in run_workspace.iterdir() if path.is_dir()]
    if not candidates:
        return None
    candidates.sort(key=lambda item: item.stat().st_mtime, reverse=True)
    return candidates[0]


def build_runner_command(args: argparse.Namespace, algorithm: str, seed: int, run_workspace: Path) -> List[str]:
    command = [
        args.python_executable,
        str(args.runner_script),
        "--algorithm",
        algorithm,
        "--seed",
        str(seed),
        "--train-dataset",
        args.train_dataset,
        "--eval-dataset",
        args.eval_dataset,
        "--initial-population-json",
        args.initial_population_json,
        "--generations",
        str(args.generations),
        "--population-size",
        str(args.population_size),
        "--offspring-per-generation",
        str(args.offspring_per_generation),
        "--epochs-per-candidate",
        str(args.epochs_per_candidate),
        "--train-batch-size",
        str(args.train_batch_size),
        "--eval-batch-size",
        str(args.eval_batch_size),
        "--num-workers",
        str(args.num_workers),
        "--lr",
        str(args.lr),
        "--weight-decay",
        str(args.weight_decay),
        "--momentum",
        str(args.momentum),
        "--label-smoothing",
        str(args.label_smoothing),
        "--num-classes",
        str(args.num_classes),
        "--images-per-class-train",
        str(args.images_per_class_train),
        "--images-per-class-eval",
        str(args.images_per_class_eval),
        "--checkpoint",
        args.checkpoint,
        "--device",
        args.device,
        "--calibration-dir",
        args.calibration_dir,
        "--num-calibration-images",
        str(args.num_calibration_images),
        "--calibration-batch-size",
        str(args.calibration_batch_size),
        "--tpc-version",
        args.tpc_version,
        "--opset-version",
        str(args.opset_version),
        "--compile-timeout-sec",
        str(args.compile_timeout_sec),
        "--eval-log-every",
        str(args.eval_log_every),
        "--mutation-rate",
        str(args.mutation_rate),
        "--tournament-size",
        str(args.tournament_size),
        "--regularized-sample-size",
        str(args.regularized_sample_size),
        "--output-root",
        str(run_workspace),
    ]
    if args.runner_extra_args:
        command.extend(args.runner_extra_args)
    return command


def render_experiment_visualizations(
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


def write_run_records_json(path: Path, records: Sequence[Dict[str, object]]) -> None:
    with path.open("w", encoding="utf-8") as handle:
        json.dump(records, handle, indent=2)


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

    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for record in records:
            summary_raw = record.get("summary")
            summary: Dict[str, object] = summary_raw if isinstance(summary_raw, dict) else {}
            row = {
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
            writer.writerow(row)


def run_single_search(
    args: argparse.Namespace,
    logger: logging.Logger,
    output_root: Path,
    algorithm: str,
    run_index: int,
    seed: int,
    manifest_path: Path,
    event_log_path: Path,
) -> Dict[str, object]:
    run_workspace = output_root / "raw_runs" / f"run_{run_index:03d}_seed_{seed}"
    run_workspace.mkdir(parents=True, exist_ok=True)

    command = build_runner_command(args, algorithm=algorithm, seed=seed, run_workspace=run_workspace)
    command_render = " ".join(shlex.quote(part) for part in command)

    stdout_log = run_workspace / "runner_stdout.log"
    logger.info("Launching run | algorithm=%s run=%d seed=%d", algorithm, run_index, seed)
    logger.info("Command: %s", command_render)

    append_jsonl(
        event_log_path,
        {
            "timestamp": utc_now(),
            "event": "run_launch",
            "algorithm": algorithm,
            "run_index": run_index,
            "seed": seed,
            "command": command,
        },
    )

    started_at = utc_now()
    started_perf = time.perf_counter()

    process = subprocess.Popen(
        command,
        cwd=str(args.nas_dir),
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1,
    )

    parsed_run_dir: Path | None = None
    live_plot_path = output_root / "visualizations" / "live" / f"{algorithm}_run_{run_index:03d}_seed_{seed}.png"

    with stdout_log.open("w", encoding="utf-8") as stdout_handle:
        while True:
            if process.stdout is None:
                break

            ready, _, _ = select.select([process.stdout], [], [], max(1.0, args.poll_interval_sec))
            if ready:
                line = process.stdout.readline()
                if line == "":
                    if process.poll() is not None:
                        break
                    continue
                message = line.rstrip("\n")
                stdout_handle.write(message + "\n")
                stdout_handle.flush()
                logger.info("[%s][run=%d][seed=%d] %s", algorithm, run_index, seed, message)

                maybe_run_dir = parse_run_dir_from_output(message)
                if maybe_run_dir is not None:
                    parsed_run_dir = maybe_run_dir
            else:
                if parsed_run_dir is not None:
                    events = read_progress_events(parsed_run_dir)
                    if events:
                        plot_live_run_progress(
                            progress_events=events,
                            output_png=live_plot_path,
                            title=f"Live run status | {algorithm} run={run_index} seed={seed}",
                        )

            if process.poll() is not None and not ready:
                break

        if process.stdout is not None:
            for line in process.stdout:
                message = line.rstrip("\n")
                stdout_handle.write(message + "\n")
                logger.info("[%s][run=%d][seed=%d] %s", algorithm, run_index, seed, message)
                maybe_run_dir = parse_run_dir_from_output(message)
                if maybe_run_dir is not None:
                    parsed_run_dir = maybe_run_dir

    return_code = int(process.wait())
    elapsed = time.perf_counter() - started_perf
    finished_at = utc_now()

    run_dir = parsed_run_dir if parsed_run_dir is not None else detect_run_dir(run_workspace)
    if run_dir is None:
        run_dir = run_workspace

    summary = read_json(run_dir / "summary.json", default={})
    if not isinstance(summary, dict):
        summary = {}

    history = read_json(run_dir / "history.json", default=[])
    if not isinstance(history, list):
        history = []

    progress_events = read_progress_events(run_dir)

    if progress_events:
        plot_live_run_progress(
            progress_events=progress_events,
            output_png=live_plot_path,
            title=f"Final run status | {algorithm} run={run_index} seed={seed}",
        )

    status = "success" if return_code == 0 and bool(summary) else "failed"

    if "elapsed_seconds" not in summary:
        summary["elapsed_seconds"] = elapsed
    if "seed" not in summary:
        summary["seed"] = seed
    if "algorithm" not in summary:
        summary["algorithm"] = algorithm

    run_record: Dict[str, object] = {
        "algorithm": algorithm,
        "run_index": run_index,
        "seed": seed,
        "status": status,
        "return_code": return_code,
        "started_at": started_at,
        "finished_at": finished_at,
        "elapsed_wall_seconds": elapsed,
        "command": command,
        "run_workspace": str(run_workspace),
        "run_dir": str(run_dir),
        "stdout_log": str(stdout_log),
        "summary": summary,
        "history": history,
        "progress_events_count": len(progress_events),
    }

    append_jsonl(
        manifest_path,
        {
            "timestamp": utc_now(),
            "event": "run_finished",
            "algorithm": algorithm,
            "run_index": run_index,
            "seed": seed,
            "status": status,
            "return_code": return_code,
            "run_dir": str(run_dir),
        },
    )
    append_jsonl(
        event_log_path,
        {
            "timestamp": utc_now(),
            "event": "run_result",
            "algorithm": algorithm,
            "run_index": run_index,
            "seed": seed,
            "status": status,
            "return_code": return_code,
            "elapsed_wall_seconds": elapsed,
            "run_dir": str(run_dir),
            "summary": summary,
        },
    )

    return run_record


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        "Run repeated NAS experiments for baseline SGA vs regularized evolution, with stats and plots."
    )

    parser.add_argument("--nas-dir", type=Path, default=Path(__file__).resolve().parent)
    parser.add_argument("--runner-script", type=Path, default=Path(__file__).resolve().parent / "genetic_NAS_runner.py")
    parser.add_argument("--python-executable", type=str, default=sys.executable)

    parser.add_argument("--algorithms", nargs="+", default=["baseline_sga", "regularized_evolution"])
    parser.add_argument("--runs-per-algorithm", type=int, default=10)
    parser.add_argument("--base-seed", type=int, default=1234)
    parser.add_argument("--seed-stride", type=int, default=97)

    parser.add_argument("--output-root", type=Path, default=Path(__file__).resolve().parent / "multi_run_experiments")
    parser.add_argument("--poll-interval-sec", type=float, default=100.0)
    parser.add_argument("--continue-on-failure", action="store_true")

    parser.add_argument("--train-dataset", type=str, default="/mnt/matylda5/xmihol00/datasets/imagenet/subset/train")
    parser.add_argument("--eval-dataset", type=str, default="/mnt/matylda5/xmihol00/datasets/imagenet/subset/val")
    parser.add_argument(
        "--initial-population-json",
        type=str,
        default="/mnt/matylda5/xmihol00/EUD/NAS/space_sampling_runs/sampling_results.json",
    )

    parser.add_argument("--generations", type=int, default=25)
    parser.add_argument("--population-size", type=int, default=25)
    parser.add_argument("--offspring-per-generation", type=int, default=8)
    parser.add_argument("--epochs-per-candidate", type=int, default=3)

    parser.add_argument("--train-batch-size", type=int, default=64)
    parser.add_argument("--eval-batch-size", type=int, default=50)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--lr", type=float, default=0.01)
    parser.add_argument("--weight-decay", type=float, default=5e-5)
    parser.add_argument("--momentum", type=float, default=0.9)
    parser.add_argument("--label-smoothing", type=float, default=0.0)

    parser.add_argument("--num-classes", type=int, default=6)
    parser.add_argument("--images-per-class-train", type=int, default=0)
    parser.add_argument("--images-per-class-eval", type=int, default=100)

    parser.add_argument(
        "--checkpoint",
        type=str,
        default="/mnt/matylda5/xmihol00/EUD/supernet/runs_imx500_supernet/20260402_200233/best.pt",
    )
    parser.add_argument("--device", type=str, default="cuda")

    parser.add_argument("--calibration-dir", type=str, default=str(Path(__file__).resolve().parent / "images"))
    parser.add_argument("--num-calibration-images", type=int, default=72)
    parser.add_argument("--calibration-batch-size", type=int, default=12)
    parser.add_argument("--tpc-version", type=str, default="1.0")
    parser.add_argument("--opset-version", type=int, default=15)
    parser.add_argument("--compile-timeout-sec", type=int, default=1800)
    parser.add_argument("--eval-log-every", type=int, default=5)

    parser.add_argument("--mutation-rate", type=float, default=0.25)
    parser.add_argument("--tournament-size", type=int, default=3)
    parser.add_argument("--regularized-sample-size", type=int, default=8)

    parser.add_argument("--bootstrap-samples", type=int, default=10000)
    parser.add_argument("--confidence", type=float, default=0.95)

    parser.add_argument(
        "--runner-extra-args",
        nargs=argparse.REMAINDER,
        default=[],
        help="Additional arguments passed verbatim to genetic_NAS_runner.py",
    )

    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output_root = args.output_root.resolve()
    output_root.mkdir(parents=True, exist_ok=True)

    logger = configure_logging(output_root / "experiment.log")
    logger.info("Starting multi-run NAS experiment orchestrator.")

    if args.runs_per_algorithm <= 0:
        raise ValueError("--runs-per-algorithm must be > 0")
    if args.seed_stride <= 0:
        raise ValueError("--seed-stride must be > 0")

    config_payload = {
        "timestamp": utc_now(),
        "args": {k: (str(v) if isinstance(v, Path) else v) for k, v in vars(args).items()},
        "working_directory": os.getcwd(),
        "python_executable": sys.executable,
        "algorithm_output_layout": "<output-root>/<algorithm>/...",
    }
    with (output_root / "experiment_config.json").open("w", encoding="utf-8") as handle:
        json.dump(config_payload, handle, indent=2)

    metric_names = [
        "best_quant_acc1",
        "best_fitness",
        "compile_success_rate",
        "elapsed_seconds",
        "total_candidates_evaluated",
        "compiled_candidates",
    ]

    all_run_records: List[Dict[str, object]] = []
    algorithm_output_dirs: Dict[str, str] = {}
    launch_order = 0

    for algorithm_index, algorithm in enumerate(args.algorithms):
        algorithm_output_root = output_root 
        algorithm_output_root.mkdir(parents=True, exist_ok=True)
        algorithm_output_dirs[algorithm] = str(algorithm_output_root)

        manifest_path = algorithm_output_root / "runs_manifest.jsonl"
        event_log_path = algorithm_output_root / "experiment_events.jsonl"
        run_records: List[Dict[str, object]] = []
        latest_algorithm_statistics: Dict[str, object] = {"scipy_available": False}

        algorithm_config = {
            "timestamp": utc_now(),
            "algorithm": algorithm,
            "args": {k: (str(v) if isinstance(v, Path) else v) for k, v in vars(args).items()},
            "working_directory": os.getcwd(),
            "python_executable": sys.executable,
        }
        with (algorithm_output_root / "experiment_config.json").open("w", encoding="utf-8") as handle:
            json.dump(algorithm_config, handle, indent=2)

        logger.info("Starting algorithm block: %s | output=%s", algorithm, algorithm_output_root)

        for run_index in range(args.runs_per_algorithm):
            launch_order += 1
            seed = args.base_seed + algorithm_index * 100000 + run_index * args.seed_stride

            run_record = run_single_search(
                args=args,
                logger=logger,
                output_root=algorithm_output_root,
                algorithm=algorithm,
                run_index=run_index,
                seed=seed,
                manifest_path=manifest_path,
                event_log_path=event_log_path,
            )
            run_record["launch_order"] = launch_order
            run_records.append(run_record)
            all_run_records.append(run_record)

            statistics_payload = build_full_statistics(
                run_records=run_records,
                metric_names=metric_names,
                algorithms=[algorithm],
                bootstrap_samples=args.bootstrap_samples,
                confidence=args.confidence,
                random_seed=args.base_seed,
            )
            latest_algorithm_statistics = statistics_payload

            write_run_records_json(algorithm_output_root / "run_records.json", run_records)
            write_run_records_csv(algorithm_output_root / "run_records.csv", run_records)
            with (algorithm_output_root / "statistics.json").open("w", encoding="utf-8") as handle:
                json.dump(statistics_payload, handle, indent=2)

            render_experiment_visualizations(
                output_root=algorithm_output_root,
                run_records=run_records,
                statistics_payload=statistics_payload,
            )

            if run_record.get("status") != "success":
                logger.error(
                    "Run failed | algorithm=%s run=%d seed=%d return_code=%s",
                    algorithm,
                    run_index,
                    seed,
                    run_record.get("return_code"),
                )
                if not args.continue_on_failure:
                    logger.error("Stopping algorithm %s because --continue-on-failure was not set.", algorithm)
                    break

        algorithm_successful = sum(1 for item in run_records if str(item.get("status", "")) == "success")
        algorithm_summary = {
            "timestamp": utc_now(),
            "algorithm": algorithm,
            "total_runs_requested": args.runs_per_algorithm,
            "total_runs_finished": len(run_records),
            "successful_runs": algorithm_successful,
            "failed_runs": len(run_records) - algorithm_successful,
            "output_root": str(algorithm_output_root),
            "scipy_available": bool(latest_algorithm_statistics.get("scipy_available", False)),
        }
        with (algorithm_output_root / "experiment_summary.json").open("w", encoding="utf-8") as handle:
            json.dump(algorithm_summary, handle, indent=2)

    merged_algorithms = sorted({str(item.get("algorithm", "")) for item in all_run_records if item.get("algorithm")})
    final_statistics = build_full_statistics(
        run_records=all_run_records,
        metric_names=metric_names,
        algorithms=merged_algorithms,
        bootstrap_samples=args.bootstrap_samples,
        confidence=args.confidence,
        random_seed=args.base_seed,
    )
    with (output_root / "statistics.json").open("w", encoding="utf-8") as handle:
        json.dump(final_statistics, handle, indent=2)

    write_run_records_json(output_root / "run_records.json", all_run_records)
    write_run_records_csv(output_root / "run_records.csv", all_run_records)

    render_experiment_visualizations(
        output_root=output_root,
        run_records=all_run_records,
        statistics_payload=final_statistics,
    )

    successful = sum(1 for item in all_run_records if str(item.get("status", "")) == "success")
    summary = {
        "timestamp": utc_now(),
        "total_runs_requested": len(args.algorithms) * args.runs_per_algorithm,
        "total_runs_finished": len(all_run_records),
        "successful_runs": successful,
        "failed_runs": len(all_run_records) - successful,
        "algorithms": args.algorithms,
        "output_root": str(output_root),
        "algorithm_output_dirs": algorithm_output_dirs,
        "scipy_available": bool(final_statistics.get("scipy_available", False)),
    }
    with (output_root / "experiment_summary.json").open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2)

    logger.info("Experiment finished. Successful runs: %d/%d", successful, len(all_run_records))
    logger.info("Output directory: %s", output_root)


if __name__ == "__main__":
    main()
