from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import List

from imx500_supernet import IMX500ResNetSupernet, SubnetConfig, create_default_supernet


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser("Sample subnet candidates around IMX500 memory target")
    parser.add_argument("--num-samples", type=int, default=200)
    parser.add_argument("--target-total-bytes", type=int, default=8_388_480)
    parser.add_argument("--tolerance-ratio", type=float, default=0.25)
    parser.add_argument("--firmware-bytes", type=int, default=1_572_864)
    parser.add_argument("--working-memory-factor", type=float, default=2.0)
    parser.add_argument("--num-classes", type=int, default=1000)
    parser.add_argument("--output", type=str, default="./imx500_candidate_subnets.json")
    return parser.parse_args()


def sample_candidates(model: IMX500ResNetSupernet, args: argparse.Namespace) -> List[dict]:
    candidates: List[dict] = []
    seen = set()

    for _ in range(args.num_samples):
        cfg: SubnetConfig = model.sample_subnet(
            mode="random",
            target_total_bytes=args.target_total_bytes,
            tolerance_ratio=args.tolerance_ratio,
            firmware_bytes=args.firmware_bytes,
            working_memory_factor=args.working_memory_factor,
        )
        key = model.config_to_json(cfg)
        if key in seen:
            continue
        seen.add(key)

        resources = model.estimate_subnet_resources(
            cfg,
            firmware_bytes=args.firmware_bytes,
            working_memory_factor=args.working_memory_factor,
        )

        candidates.append(
            {
                "config": cfg.to_dict(),
                "resources": resources,
                "distance_to_target": abs(resources["total_estimated_bytes"] - args.target_total_bytes),
            }
        )

    candidates.sort(key=lambda item: item["distance_to_target"])
    return candidates


def main() -> None:
    args = parse_args()
    model = create_default_supernet(num_classes=args.num_classes)

    candidates = sample_candidates(model, args)
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    payload = {
        "target_total_bytes": args.target_total_bytes,
        "tolerance_ratio": args.tolerance_ratio,
        "firmware_bytes": args.firmware_bytes,
        "working_memory_factor": args.working_memory_factor,
        "count": len(candidates),
        "candidates": candidates,
    }

    with output_path.open("w", encoding="utf-8") as fp:
        json.dump(payload, fp, indent=2)

    print(f"Wrote {len(candidates)} candidates to {output_path}")


if __name__ == "__main__":
    main()
