#!/usr/bin/env python3
"""
Extract all evaluated candidates from both NAS experiment directories and
select 15 architectures uniformly distributed across the quantised accuracy range.

Usage:
    python select_architectures.py \
        --sga-dir    multi_run_parallel/sga_2026-04-11_21-26-21 \
        --reg-evo-dir multi_run_parallel/reg_evo_2026-04-11_21-26-13 \
        --output selected_architectures.json
"""

import argparse
import glob
import hashlib
import json
import os
import re
import sys
from collections import defaultdict
from typing import Any


# ─── helpers ──────────────────────────────────────────────────────────────────

def config_hash(config: dict) -> str:
    return hashlib.md5(json.dumps(config, sort_keys=True).encode()).hexdigest()


def parse_run_meta_from_path(path: str) -> dict[str, Any]:
    """Extract algorithm, run_index, seed from a candidate result.json path."""
    # .../sga_2026-04-11_21-26-21/raw_runs/run_003_seed_1491/.../candidates/gen002_child004_000085/result.json
    meta: dict[str, Any] = {"algorithm": "unknown", "run_index": -1, "seed": -1, "generation": -1}

    # algorithm
    if "sga_" in path or "/sga" in path or "baseline_sga" in path:
        meta["algorithm"] = "baseline_sga"
    elif "reg_evo" in path or "regularized_evolution" in path:
        meta["algorithm"] = "regularized_evolution"

    # run_index and seed from directory name  e.g. run_003_seed_1491
    m = re.search(r"run_(\d+)_seed_(\d+)", path)
    if m:
        meta["run_index"] = int(m.group(1))
        meta["seed"] = int(m.group(2))

    # generation from candidate_id directory name  e.g. gen002_child004_000085
    m = re.search(r"gen(\d+)_child(\d+)_(\d+)", path)
    if m:
        meta["generation"] = int(m.group(1))
        meta["birth_id"] = int(m.group(3))

    return meta


# ─── extraction ───────────────────────────────────────────────────────────────

def extract_all_candidates(exp_dirs: list[str]) -> list[dict]:
    """Read every result.json and return a flat list of candidate dicts."""
    candidates = []
    for exp_dir in exp_dirs:
        pattern = os.path.join(exp_dir, "raw_runs", "run_*", "*", "candidates", "*", "result.json")
        result_files = sorted(glob.glob(pattern))
        print(f"  {exp_dir}: found {len(result_files)} result.json files")

        for rf in result_files:
            try:
                with open(rf) as f:
                    r = json.load(f)
            except Exception as e:
                print(f"    WARN: could not read {rf}: {e}", file=sys.stderr)
                continue

            if not r.get("compiled", False):
                continue  # only use IMX500-compilable architectures

            config = r.get("config")
            if not isinstance(config, dict):
                continue

            quant_eval = r.get("quant_eval") or {}
            quant_acc1 = quant_eval.get("acc1")
            if quant_acc1 is None or quant_acc1 <= 0:
                continue

            meta = parse_run_meta_from_path(rf)
            cid = r.get("candidate_id", os.path.basename(os.path.dirname(rf)))

            candidates.append({
                "config": config,
                "nas_quant_acc1": float(quant_acc1),
                "candidate_id": cid,
                "algorithm": meta["algorithm"],
                "run_index": meta["run_index"],
                "seed": meta["seed"],
                "generation": meta["generation"],
                "birth_id": meta.get("birth_id", -1),
                "result_json_path": rf,
            })

    return candidates


def deduplicate(candidates: list[dict]) -> list[dict]:
    """Keep the entry with the highest nas_quant_acc1 per unique config."""
    best: dict[str, dict] = {}
    for c in candidates:
        h = config_hash(c["config"])
        if h not in best or c["nas_quant_acc1"] > best[h]["nas_quant_acc1"]:
            best[h] = c
    return list(best.values())


def select_uniform(candidates: list[dict], n: int = 15) -> list[dict]:
    """Select n candidates uniformly distributed across the accuracy range."""
    sorted_cands = sorted(candidates, key=lambda c: c["nas_quant_acc1"])
    if len(sorted_cands) <= n:
        return sorted_cands

    # Uniformly spaced indices
    step = (len(sorted_cands) - 1) / (n - 1)
    indices = [round(i * step) for i in range(n)]
    # Remove duplicates while preserving order
    seen = set()
    selected = []
    for idx in indices:
        if idx not in seen:
            seen.add(idx)
            selected.append(sorted_cands[idx])
    # Fill up if rounding caused duplicates
    for c in sorted_cands:
        if len(selected) >= n:
            break
        if id(c) not in {id(s) for s in selected}:
            selected.append(c)
    return sorted(selected[:n], key=lambda c: c["nas_quant_acc1"])


# ─── main ─────────────────────────────────────────────────────────────────────

def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--sga-dir", required=True)
    ap.add_argument("--reg-evo-dir", required=True)
    ap.add_argument("--output", default="selected_architectures.json")
    ap.add_argument("--n", type=int, default=15)
    ap.add_argument("--min-acc", type=float, default=0.0, help="Minimum quant_acc1 filter")
    args = ap.parse_args()

    print("Extracting candidates...")
    all_cands = extract_all_candidates([args.sga_dir, args.reg_evo_dir])
    print(f"Total compiled+evaluated candidates: {len(all_cands)}")

    if args.min_acc > 0:
        all_cands = [c for c in all_cands if c["nas_quant_acc1"] >= args.min_acc]
        print(f"After min_acc={args.min_acc} filter: {len(all_cands)}")

    unique_cands = deduplicate(all_cands)
    print(f"Unique configs: {len(unique_cands)}")

    # Print accuracy distribution
    accs = sorted(c["nas_quant_acc1"] for c in unique_cands)
    print(f"Accuracy range: {accs[0]:.2f}% – {accs[-1]:.2f}%")
    # Histogram in 5-pp buckets
    buckets: dict[int, int] = defaultdict(int)
    for a in accs:
        buckets[int(a // 5) * 5] += 1
    for lo in sorted(buckets):
        print(f"  [{lo:3d}-{lo+5:3d}%): {buckets[lo]:4d} configs")

    selected = select_uniform(unique_cands, n=args.n)
    print(f"\nSelected {len(selected)} architectures (uniformly distributed):")
    print(f"{'#':>3}  {'nas_quant_acc1':>16}  {'algorithm':>24}  {'seed':>6}  {'run':>3}  {'gen':>4}  config")
    for i, s in enumerate(selected):
        print(f"{i:3d}  {s['nas_quant_acc1']:16.4f}  {s['algorithm']:>24}  {s['seed']:6}  {s['run_index']:3}  {s['generation']:4}  {s['config']}")

    # Assign stable short IDs
    for i, s in enumerate(selected):
        s["arch_index"] = i

    # Remove internal path (not needed in output)
    output = []
    for s in selected:
        entry = {k: v for k, v in s.items() if k != "result_json_path"}
        output.append(entry)

    with open(args.output, "w") as f:
        json.dump(output, f, indent=2)
    print(f"\nSaved to {args.output}")


if __name__ == "__main__":
    main()
