from __future__ import annotations

import argparse
import json
import os
import random
import sys
import time
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

import numpy as np
import torch
import torch.nn as nn
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms

PROJECT_ROOT = Path(__file__).resolve().parents[1]
NAS_DIR = PROJECT_ROOT / "NAS"
SUPERNET_DIR = PROJECT_ROOT / "supernet"
for path in [NAS_DIR, SUPERNET_DIR]:
    if str(path) not in sys.path:
        sys.path.append(str(path))

from genetic_algorithms import RegularizedEvolution, SearchSpace, SimpleGeneticAlgorithm
from imx500_supernet import IMX500ResNetSupernet, SubnetConfig
from space_sampling import (
    RepresentativeDataGenerator,
    as_jsonable,
    build_static_subnet_model,
    evaluate_onnx,
    export_to_onnx,
    load_supernet,
    log,
    quantize_and_export_onnx,
    run_imx500_compile,
    set_seed,
)

import safe_gpu
while True:
    try:
        safe_gpu.claim_gpus(1)
        break
    except:
        print("Waiting for free GPU")
        time.sleep(5)
        pass

SUPPORTED_IMAGE_EXTENSIONS = (".jpg", ".jpeg", ".png", ".bmp", ".webp")


@dataclass
class CandidateResult:
    config: SubnetConfig
    fitness: float
    compiled: bool
    quant_acc1: float
    sample_dir: str
    details: Dict[str, object]


class FolderClassificationDataset(Dataset):
    def __init__(
        self,
        samples: Sequence[Tuple[str, int]],
        transform: transforms.Compose,
    ) -> None:
        self.samples = list(samples)
        self.transform = transform

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int) -> Tuple[torch.Tensor, torch.Tensor]:
        image_path, label = self.samples[index]
        image = Image.open(image_path).convert("RGB")
        image_tensor = self.transform(image)
        return image_tensor, torch.tensor(label, dtype=torch.long)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser("Genetic NAS runner for IMX500 supernet")

    parser.add_argument("--train-dataset", type=str, default="/mnt/matylda5/xmihol00/datasets/imagenet/subset/train")
    parser.add_argument("--eval-dataset", type=str, default="/mnt/matylda5/xmihol00/datasets/imagenet/subset/val")
    parser.add_argument("--initial-population-json", type=str, default="/mnt/matylda5/xmihol00/EUD/NAS/space_sampling_runs/sampling_results.json")

    parser.add_argument("--algorithm", type=str, choices=["baseline_sga", "regularized_evolution"], default="regularized_evolution")
    parser.add_argument("--generations", type=int, default=10)
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

    parser.add_argument("--checkpoint", type=str, default="/mnt/matylda5/xmihol00/EUD/supernet/runs_imx500_supernet/20260402_200233/best.pt")
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--seed", type=int, default=42)

    parser.add_argument("--calibration-dir", type=str, default=str(Path(__file__).resolve().parent / "calibration"))
    parser.add_argument("--num-calibration-images", type=int, default=6*12)
    parser.add_argument("--calibration-batch-size", type=int, default=12)
    parser.add_argument("--tpc-version", type=str, default="1.0")
    parser.add_argument("--opset-version", type=int, default=15)
    parser.add_argument("--compile-timeout-sec", type=int, default=1800)
    parser.add_argument("--eval-log-every", type=int, default=5)

    parser.add_argument("--mutation-rate", type=float, default=0.25)
    parser.add_argument("--tournament-size", type=int, default=3)
    parser.add_argument("--regularized-sample-size", type=int, default=8)

    parser.add_argument("--output-root", type=str, default=str(Path(__file__).resolve().parent / "genetic_runs"))
    return parser.parse_args()


def load_state_dict_for_args(checkpoint_path: str) -> Dict[str, torch.Tensor]:
    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    if isinstance(checkpoint, dict) and "model" in checkpoint:
        state_dict = checkpoint["model"]
    elif isinstance(checkpoint, dict) and "state_dict" in checkpoint:
        state_dict = checkpoint["state_dict"]
    else:
        state_dict = checkpoint
    if not isinstance(state_dict, dict):
        raise ValueError("Unsupported checkpoint format.")
    return state_dict


def select_classes(dataset_root: Path, max_classes: int) -> List[str]:
    class_names = sorted([item.name for item in dataset_root.iterdir() if item.is_dir()])
    if not class_names:
        raise ValueError(f"No classes found in dataset: {dataset_root}")
    return class_names[:max_classes]


def collect_samples(dataset_root: Path, class_names: Sequence[str], images_per_class: int) -> List[Tuple[str, int]]:
    samples: List[Tuple[str, int]] = []
    for class_index, class_name in enumerate(class_names):
        class_dir = dataset_root / class_name
        class_images = [
            str(path)
            for path in sorted(class_dir.iterdir())
            if path.suffix.lower() in SUPPORTED_IMAGE_EXTENSIONS
        ]
        if images_per_class > 0:
            class_images = class_images[:images_per_class]
        samples.extend((path, class_index) for path in class_images)
    if not samples:
        raise ValueError(f"No images collected under dataset: {dataset_root}")
    return samples


def create_train_loader(
    dataset_root: Path,
    class_names: Sequence[str],
    images_per_class: int,
    batch_size: int,
    num_workers: int,
    max_resolution: int,
) -> DataLoader:
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    transform = transforms.Compose(
        [
            transforms.RandomResizedCrop(max_resolution, scale=(0.2, 1.0), interpolation=transforms.InterpolationMode.BILINEAR),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            normalize,
        ]
    )

    samples = collect_samples(dataset_root, class_names, images_per_class)
    dataset = FolderClassificationDataset(samples=samples, transform=transform)

    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=False,
        persistent_workers=num_workers > 0,
    )


def create_eval_entries(
    dataset_root: Path,
    class_names: Sequence[str],
    images_per_class: int,
) -> List[object]:
    entries: List[object] = []
    from space_sampling import DatasetEntry

    for class_index, class_name in enumerate(class_names):
        class_dir = dataset_root / class_name
        class_images = [
            path
            for path in sorted(class_dir.iterdir())
            if path.suffix.lower() in SUPPORTED_IMAGE_EXTENSIONS
        ]
        if images_per_class > 0:
            class_images = class_images[:images_per_class]

        for image_path in class_images:
            entries.append(
                DatasetEntry(
                    image_path=str(image_path),
                    class_name=class_name,
                    class_index=class_index,
                )
            )
    if not entries:
        raise ValueError(f"No eval images collected under dataset: {dataset_root}")
    return entries


def freeze_backbone(model: nn.Module) -> None:
    for name, parameter in model.named_parameters():
        parameter.requires_grad = name.startswith("classifier")


def unfreeze_all(model: nn.Module) -> None:
    for parameter in model.parameters():
        parameter.requires_grad = True


def train_candidate(
    model: nn.Module,
    train_loader: DataLoader,
    epochs: int,
    freeze_backbone_first_epoch: bool,
    lr: float,
    momentum: float,
    weight_decay: float,
    label_smoothing: float,
    device: torch.device,
) -> List[Dict[str, float]]:
    criterion = nn.CrossEntropyLoss(label_smoothing=label_smoothing)
    history: List[Dict[str, float]] = []

    for epoch in range(epochs):
        if freeze_backbone_first_epoch and epoch == 0:
            freeze_backbone(model)
            log("Training epoch 1 with frozen backbone (classifier adaptation phase).")
        elif freeze_backbone_first_epoch and epoch == 1:
            unfreeze_all(model)
            log("Unfroze backbone after adaptation epoch.")

        optimizer = torch.optim.SGD(
            [parameter for parameter in model.parameters() if parameter.requires_grad],
            lr=lr,
            momentum=momentum,
            weight_decay=weight_decay,
            nesterov=True,
        )

        model.train()
        epoch_loss = 0.0
        total = 0
        correct = 0

        for images, labels in train_loader:
            images = images.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)

            optimizer.zero_grad(set_to_none=True)
            logits = model(images)
            loss = criterion(logits, labels)
            loss.backward()
            optimizer.step()

            epoch_loss += float(loss.item()) * labels.size(0)
            predictions = torch.argmax(logits, dim=1)
            total += int(labels.size(0))
            correct += int((predictions == labels).sum().item())

        avg_loss = epoch_loss / max(1, total)
        acc1 = 100.0 * correct / max(1, total)
        history.append({"epoch": float(epoch + 1), "loss": avg_loss, "acc1": acc1})
        log(f"Train epoch {epoch + 1}/{epochs} | loss={avg_loss:.4f} acc1={acc1:.2f}%")

    return history


def config_key(config: SubnetConfig) -> str:
    return json.dumps(config.to_dict(), sort_keys=True)


def build_search_space(supernet: IMX500ResNetSupernet) -> SearchSpace:
    return SearchSpace(
        resolution_candidates=tuple(int(v) for v in supernet.resolution_candidates),
        stem_width_candidates=tuple(int(v) for v in supernet.stem_width_candidates),
        stage_depth_candidates=tuple(tuple(int(v) for v in stage) for stage in supernet.stage_depth_candidates),
        stage_width_candidates=tuple(tuple(int(v) for v in stage) for stage in supernet.stage_width_candidates),
    )


def make_unique_run_dir(output_root: Path) -> Path:
    output_root.mkdir(parents=True, exist_ok=True)
    run_id = f"{datetime.now().strftime('%Y%m%d_%H%M%S')}_pid{os.getpid()}"
    run_dir = output_root / run_id
    run_dir.mkdir(parents=True, exist_ok=False)
    return run_dir


def append_progress_event(run_dir: Path, event: Dict[str, object]) -> None:
    payload = {
        "timestamp": datetime.now().isoformat(timespec="seconds"),
        **event,
    }
    with (run_dir / "progress.jsonl").open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(as_jsonable(payload), ensure_ascii=False) + "\n")


def load_initial_population(
    json_path: Path,
    population_size: int,
) -> List[Dict[str, object]]:
    with json_path.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)

    if isinstance(payload, dict) and "results" in payload:
        records = payload["results"]
    else:
        records = payload

    candidates: List[Dict[str, object]] = []
    for record in records:
        compiled = bool(record.get("compiled", False))
        quant_eval = record.get("quant_eval", {})
        quant_acc = float(quant_eval.get("acc1", 0.0))
        config_dict = record.get("config")
        if not compiled or not isinstance(config_dict, dict):
            continue

        config = SubnetConfig.from_dict(config_dict)
        candidates.append(
            {
                "config": config,
                "fitness": quant_acc,
                "compiled": compiled,
                "quant_acc1": quant_acc,
                "birth_id": int(record.get("attempt", 0)),
                "source": "sampling_results",
                "details": record,
            }
        )

    dedup: Dict[str, Dict[str, object]] = {}
    for candidate in candidates:
        key = config_key(candidate["config"])
        current = dedup.get(key)
        if current is None or float(candidate["fitness"]) > float(current["fitness"]):
            dedup[key] = candidate

    unique_candidates = list(dedup.values())
    unique_candidates.sort(key=lambda item: float(item["fitness"]), reverse=True)
    return unique_candidates[:population_size]


def evaluate_candidate(
    config: SubnetConfig,
    supernet: IMX500ResNetSupernet,
    model_num_classes: int,
    train_num_classes: int,
    train_loader: DataLoader,
    eval_entries: Sequence[object],
    args: argparse.Namespace,
    device: torch.device,
    run_dir: Path,
    candidate_id: str,
) -> CandidateResult:
    sample_dir = run_dir / "candidates" / candidate_id
    sample_dir.mkdir(parents=True, exist_ok=True)

    supernet.set_active_subnet(config)
    model = build_static_subnet_model(
        supernet=supernet,
        config=config,
        num_classes=model_num_classes,
        device=device,
    )

    freeze_backbone_first_epoch = train_num_classes != model_num_classes
    if freeze_backbone_first_epoch:
        in_features = int(model.classifier.in_features)
        model.classifier = nn.Linear(in_features, train_num_classes).to(device)
        log(
            f"[{candidate_id}] Replaced classifier ({model_num_classes} -> {train_num_classes}) "
            "and enabling frozen first epoch."
        )

    train_history = train_candidate(
        model=model,
        train_loader=train_loader,
        epochs=args.epochs_per_candidate,
        freeze_backbone_first_epoch=freeze_backbone_first_epoch,
        lr=args.lr,
        momentum=args.momentum,
        weight_decay=args.weight_decay,
        label_smoothing=args.label_smoothing,
        device=device,
    )

    input_shape = (3, config.resolution, config.resolution)
    float_onnx = sample_dir / "model_float.onnx"
    quant_onnx = sample_dir / "model_quant.onnx"

    export_to_onnx(
        model=model,
        output_path=float_onnx,
        input_shape=input_shape,
        opset_version=args.opset_version,
        device=device,
    )

    rep_data_gen = RepresentativeDataGenerator(
        image_folder_path=args.calibration_dir,
        input_shape=input_shape,
        batch_size=args.calibration_batch_size,
        num_images=args.num_calibration_images,
        device=device,
    )

    quant_info = quantize_and_export_onnx(
        model=model,
        representative_data_gen=rep_data_gen,
        output_path=quant_onnx,
        tpc_version=args.tpc_version,
    )

    compiled, compile_log = run_imx500_compile(
        quantized_onnx_path=quant_onnx,
        compile_output_dir=sample_dir / "imx500_output",
        timeout_sec=args.compile_timeout_sec,
    )

    quant_eval = {"acc1": 0.0, "correct": 0.0, "total": 0.0}
    fitness = -1e9
    if compiled:
        try:
            quant_eval = evaluate_onnx(
                onnx_path=quant_onnx,
                dataset_entries=eval_entries,
                input_resolution=config.resolution,
                batch_size=args.eval_batch_size,
                selected_num_classes=train_num_classes,
                eval_log_every=args.eval_log_every,
            )
            evaluated = bool(float(quant_eval.get("evaluated", 1.0)) > 0.0)
            if evaluated:
                fitness = float(quant_eval["acc1"])
            else:
                log(
                    f"[{candidate_id}] ONNX evaluation did not complete successfully; "
                    "treating candidate as not evaluated (compile-failure-like fitness)."
                )
        except Exception as exc:
            quant_eval = {"acc1": 0.0, "correct": 0.0, "total": 0.0, "evaluated": 0.0}
            log(
                f"[{candidate_id}] ONNX evaluation crashed unexpectedly: {exc}. "
                "Treating candidate as not evaluated and continuing run."
            )

    details = {
        "candidate_id": candidate_id,
        "config": config.to_dict(),
        "compiled": compiled,
        "compile_log": compile_log,
        "quantization_info": as_jsonable(quant_info),
        "quant_eval": quant_eval,
        "train_history": train_history,
        "float_onnx": str(float_onnx),
        "quant_onnx": str(quant_onnx),
    }

    with (sample_dir / "result.json").open("w", encoding="utf-8") as handle:
        json.dump(as_jsonable(details), handle, indent=2)

    return CandidateResult(
        config=config,
        fitness=fitness,
        compiled=compiled,
        quant_acc1=float(quant_eval["acc1"]),
        sample_dir=str(sample_dir),
        details=details,
    )


def print_population_table(population: Sequence[Dict[str, object]], title: str) -> None:
    log(title)
    log("rank | fitness | resolution stem | depths | widths")
    ranked = sorted(population, key=lambda item: float(item["fitness"]), reverse=True)
    for index, individual in enumerate(ranked[:10], start=1):
        cfg: SubnetConfig = individual["config"]
        log(
            f"{index:>4} | {float(individual['fitness']):>7.2f} | "
            f"{cfg.resolution:>3} {cfg.stem_width:>3} | {cfg.stage_depths} | {cfg.stage_widths}"
        )


def candidate_to_json_record(candidate: Dict[str, object]) -> Dict[str, object]:
    config = candidate.get("config")
    config_to_dict = getattr(config, "to_dict", None)
    config_dict = config_to_dict() if callable(config_to_dict) else {}
    details = candidate.get("details")
    details_dict = details if isinstance(details, dict) else {}

    fitness_raw = candidate.get("fitness", float("nan"))
    quant_raw = candidate.get("quant_acc1", float("nan"))
    birth_raw = candidate.get("birth_id", 0)

    fitness = float(fitness_raw) if isinstance(fitness_raw, (int, float)) else float("nan")
    quant_acc1 = float(quant_raw) if isinstance(quant_raw, (int, float)) else float("nan")
    birth_id = int(birth_raw) if isinstance(birth_raw, (int, float)) else 0

    return {
        "fitness": fitness,
        "quant_acc1": quant_acc1,
        "compiled": bool(candidate.get("compiled", False)),
        "birth_id": birth_id,
        "source": str(candidate.get("source", "")),
        "sample_dir": str(candidate.get("sample_dir", "")),
        "config": config_dict,
        "candidate_id": str(details_dict.get("candidate_id", "")),
    }


def select_top_candidates(all_records: Sequence[Dict[str, object]], top_k: int = 3) -> List[Dict[str, object]]:
    compiled_candidates = [item for item in all_records if bool(item.get("compiled", False))]
    pool = compiled_candidates if compiled_candidates else list(all_records)

    def fitness_key(item: Dict[str, object]) -> float:
        fitness_raw = item.get("fitness")
        return float(fitness_raw) if isinstance(fitness_raw, (int, float)) else float("-inf")

    ranked = sorted(
        pool,
        key=fitness_key,
        reverse=True,
    )
    return ranked[: max(1, top_k)]


def main() -> None:
    run_started_at = time.perf_counter()
    args = parse_args()
    set_seed(args.seed)
    rng = random.Random(args.seed)
    np.random.seed(args.seed)

    run_dir = make_unique_run_dir(Path(args.output_root))
    (run_dir / "candidates").mkdir(parents=True, exist_ok=True)

    log(f"Run directory: {run_dir}")
    with (run_dir / "args.json").open("w", encoding="utf-8") as handle:
        json.dump(vars(args), handle, indent=2)
    append_progress_event(
        run_dir,
        {
            "event": "run_started",
            "algorithm": args.algorithm,
            "generations": args.generations,
            "population_size": args.population_size,
            "offspring_per_generation": args.offspring_per_generation,
            "seed": args.seed,
        },
    )

    requested_device = torch.device(args.device)
    device = requested_device if (requested_device.type != "cuda" or torch.cuda.is_available()) else torch.device("cpu")
    if requested_device.type == "cuda" and device.type != "cuda":
        log("CUDA requested but unavailable; falling back to CPU.")

    if args.checkpoint:
        state_dict = load_state_dict_for_args(args.checkpoint)
        inferred_classes = int(state_dict["classifier.weight"].shape[0]) if "classifier.weight" in state_dict else args.num_classes
        if inferred_classes != args.num_classes:
            log(
                f"Checkpoint classes ({inferred_classes}) differ from --num-classes ({args.num_classes}); "
                "model will use checkpoint classes, evaluation classes still follow --num-classes datasets."
            )

    supernet, model_num_classes = load_supernet(args, device)

    train_root = Path(args.train_dataset)
    eval_root = Path(args.eval_dataset)
    train_class_names = select_classes(train_root, args.num_classes)
    eval_class_names = select_classes(eval_root, args.num_classes)

    eval_class_set = set(eval_class_names)
    common_class_names = [class_name for class_name in train_class_names if class_name in eval_class_set]
    if not common_class_names:
        raise ValueError("No overlapping class names between train and eval datasets.")

    log(
        f"Using {len(common_class_names)} classes (alphabetically, overlap of train/eval). "
        f"First classes: {common_class_names[:5]}"
    )

    train_loader = create_train_loader(
        dataset_root=train_root,
        class_names=common_class_names,
        images_per_class=args.images_per_class_train,
        batch_size=args.train_batch_size,
        num_workers=args.num_workers,
        max_resolution=max(supernet.resolution_candidates),
    )
    eval_entries = create_eval_entries(
        dataset_root=eval_root,
        class_names=common_class_names,
        images_per_class=args.images_per_class_eval,
    )

    log(f"Train samples: {len(train_loader.dataset)} | Eval images: {len(eval_entries)}")

    population = load_initial_population(Path(args.initial_population_json), args.population_size)
    log(f"Loaded {len(population)} initial candidates from sampling results.")
    append_progress_event(
        run_dir,
        {
            "event": "initial_population_loaded",
            "count": len(population),
        },
    )

    archive: Dict[str, Dict[str, object]] = {}
    for individual in population:
        archive[config_key(individual["config"])] = individual

    birth_counter = max([int(item.get("birth_id", 0)) for item in population] + [0])

    while len(population) < args.population_size:
        random_config = supernet.sample_subnet(mode="random")
        key = config_key(random_config)
        if key in archive:
            continue

        birth_counter += 1
        candidate_id = f"bootstrap_{birth_counter:06d}"
        log(f"Evaluating bootstrap candidate {candidate_id}")
        result = evaluate_candidate(
            config=random_config,
            supernet=supernet,
            model_num_classes=model_num_classes,
            train_num_classes=len(common_class_names),
            train_loader=train_loader,
            eval_entries=eval_entries,
            args=args,
            device=device,
            run_dir=run_dir,
            candidate_id=candidate_id,
        )
        individual = {
            "config": result.config,
            "fitness": result.fitness,
            "compiled": result.compiled,
            "quant_acc1": result.quant_acc1,
            "birth_id": birth_counter,
            "source": "bootstrap",
            "details": result.details,
            "sample_dir": result.sample_dir,
        }
        population.append(individual)
        archive[key] = individual
        append_progress_event(
            run_dir,
            {
                "event": "bootstrap_candidate_evaluated",
                "candidate_id": candidate_id,
                "compiled": result.compiled,
                "fitness": result.fitness,
                "quant_acc1": result.quant_acc1,
                "population_size": len(population),
            },
        )

    search_space = build_search_space(supernet)
    if args.algorithm == "baseline_sga":
        algorithm = SimpleGeneticAlgorithm(
            mutation_rate=args.mutation_rate,
            tournament_size=args.tournament_size,
        )
    else:
        algorithm = RegularizedEvolution(
            sample_size=args.regularized_sample_size,
            mutation_rate=args.mutation_rate,
        )

    history: List[Dict[str, object]] = []
    for generation in range(args.generations):
        population = sorted(population, key=lambda item: float(item["fitness"]), reverse=True)[: args.population_size]
        print_population_table(population, title=f"Generation {generation} population summary")

        offspring_configs = algorithm.propose(
            population=population,
            search_space=search_space,
            num_offspring=args.offspring_per_generation,
            rng=rng,
        )

        offspring_records: List[Dict[str, object]] = []
        for child_index, child_config in enumerate(offspring_configs):
            key = config_key(child_config)
            if key in archive:
                continue

            birth_counter += 1
            candidate_id = f"gen{generation:03d}_child{child_index:03d}_{birth_counter:06d}"
            log(f"Evaluating offspring {candidate_id}")
            start = time.perf_counter()
            result = evaluate_candidate(
                config=child_config,
                supernet=supernet,
                model_num_classes=model_num_classes,
                train_num_classes=len(common_class_names),
                train_loader=train_loader,
                eval_entries=eval_entries,
                args=args,
                device=device,
                run_dir=run_dir,
                candidate_id=candidate_id,
            )
            elapsed = time.perf_counter() - start
            log(
                f"Finished {candidate_id} | compiled={result.compiled} "
                f"quant_acc1={result.quant_acc1:.2f} fitness={result.fitness:.2f} time={elapsed:.1f}s"
            )

            individual = {
                "config": result.config,
                "fitness": result.fitness,
                "compiled": result.compiled,
                "quant_acc1": result.quant_acc1,
                "birth_id": birth_counter,
                "source": f"generation_{generation}",
                "details": result.details,
                "sample_dir": result.sample_dir,
            }
            offspring_records.append(individual)
            archive[key] = individual
            append_progress_event(
                run_dir,
                {
                    "event": "offspring_evaluated",
                    "generation": generation,
                    "candidate_id": candidate_id,
                    "compiled": result.compiled,
                    "fitness": result.fitness,
                    "quant_acc1": result.quant_acc1,
                    "elapsed_seconds": elapsed,
                },
            )

        population = algorithm.select_next_population(
            population=population,
            offspring=offspring_records,
            population_size=args.population_size,
        )

        best = max(population, key=lambda item: float(item["fitness"]))
        generation_record = {
            "generation": generation,
            "best_fitness": float(best["fitness"]),
            "best_config": best["config"].to_dict(),
            "population_mean_fitness": float(np.mean([float(item["fitness"]) for item in population])),
            "population_size": len(population),
            "offspring_evaluated": len(offspring_records),
        }
        history.append(generation_record)
        append_progress_event(
            run_dir,
            {
                "event": "generation_completed",
                "generation": generation,
                "best_fitness": generation_record["best_fitness"],
                "population_mean_fitness": generation_record["population_mean_fitness"],
                "offspring_evaluated": generation_record["offspring_evaluated"],
            },
        )

        with (run_dir / "history.json").open("w", encoding="utf-8") as handle:
            json.dump(as_jsonable(history), handle, indent=2)

        with (run_dir / f"population_gen_{generation:03d}.json").open("w", encoding="utf-8") as handle:
            json.dump(
                as_jsonable(
                    [
                        {
                            "fitness": float(item["fitness"]),
                            "quant_acc1": float(item["quant_acc1"]),
                            "compiled": bool(item["compiled"]),
                            "birth_id": int(item["birth_id"]),
                            "source": item["source"],
                            "config": item["config"].to_dict(),
                            "sample_dir": item.get("sample_dir", ""),
                        }
                        for item in sorted(population, key=lambda row: float(row["fitness"]), reverse=True)
                    ]
                ),
                handle,
                indent=2,
            )

    final_best = max(population, key=lambda item: float(item["fitness"]))
    elapsed_seconds = time.perf_counter() - run_started_at
    all_records = list(archive.values())
    compiled_count = sum(1 for item in all_records if bool(item.get("compiled", False)))
    total_evaluated = len(all_records)
    top_3_candidates = select_top_candidates(all_records, top_k=3)
    top_3_payload = [candidate_to_json_record(candidate) for candidate in top_3_candidates]

    with (run_dir / "top_3_architectures.json").open("w", encoding="utf-8") as handle:
        json.dump(
            as_jsonable(
                {
                    "algorithm": args.algorithm,
                    "seed": int(args.seed),
                    "selection_policy": (
                        "top fitness among compiled candidates, "
                        "fallback to all candidates if none compiled"
                    ),
                    "top_k": 3,
                    "architectures": top_3_payload,
                }
            ),
            handle,
            indent=2,
        )

    summary = {
        "algorithm": args.algorithm,
        "seed": int(args.seed),
        "generations": args.generations,
        "population_size": args.population_size,
        "offspring_per_generation": args.offspring_per_generation,
        "best_fitness": float(final_best["fitness"]),
        "best_quant_acc1": float(final_best["quant_acc1"]),
        "total_candidates_evaluated": total_evaluated,
        "compiled_candidates": compiled_count,
        "compile_success_rate": float(compiled_count / max(1, total_evaluated)),
        "elapsed_seconds": elapsed_seconds,
        "best_config": final_best["config"].to_dict(),
        "top_3_architectures_file": str(run_dir / "top_3_architectures.json"),
        "top_3_architectures": top_3_payload,
        "run_dir": str(run_dir),
    }

    with (run_dir / "summary.json").open("w", encoding="utf-8") as handle:
        json.dump(as_jsonable(summary), handle, indent=2)

    append_progress_event(
        run_dir,
        {
            "event": "run_completed",
            "best_fitness": summary["best_fitness"],
            "best_quant_acc1": summary["best_quant_acc1"],
            "elapsed_seconds": elapsed_seconds,
        },
    )

    log("Genetic NAS search finished.")
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    main()
