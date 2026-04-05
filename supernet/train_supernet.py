from __future__ import annotations

import argparse
import json
import logging
import math
import random
import time
from pathlib import Path
from typing import Dict, Tuple

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.utils.data import Subset
from torchvision import datasets, transforms
from tqdm import tqdm

from imx500_supernet import IMX500ResNetSupernet, create_default_supernet

import safe_gpu
while True:
    try:
        safe_gpu.claim_gpus(1)
        break
    except:
        print("Waiting for free GPU")
        time.sleep(5)
        pass


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser("Train IMX500-aware supernet (single GPU)")

    parser.add_argument("--dataset-path", type=str, default="/mnt/matylda5/xmihol00/datasets/imagenet/train")
    parser.add_argument("--val-split", type=float, default=0.15)
    parser.add_argument("--num-classes", type=int, default=1000)

    parser.add_argument("--epochs", type=int, default=120)
    parser.add_argument("--batch-size", type=int, default=96)
    parser.add_argument("--num-workers", type=int, default=8)
    parser.add_argument("--lr", type=float, default=0.1)
    parser.add_argument("--weight-decay", type=float, default=5e-5)
    parser.add_argument("--momentum", type=float, default=0.9)
    parser.add_argument("--label-smoothing", type=float, default=0.1)
    parser.add_argument("--grad-clip", type=float, default=1.0)
    parser.add_argument("--warmup-epochs", type=int, default=5)

    parser.add_argument("--num-arch-training", type=int, default=3)
    parser.add_argument("--sandwich-rule", action="store_true", default=True)
    parser.add_argument("--no-sandwich-rule", dest="sandwich_rule", action="store_false")
    parser.add_argument("--inplace-distill", action="store_true", default=True)
    parser.add_argument("--no-inplace-distill", dest="inplace_distill", action="store_false")

    parser.add_argument("--target-total-bytes", type=int, default=8_388_480)
    parser.add_argument("--target-tolerance-ratio", type=float, default=None)
    parser.add_argument("--target-tolerance-ratio-low", type=float, default=0.20)
    parser.add_argument("--target-tolerance-ratio-high", type=float, default=0.35)
    parser.add_argument("--firmware-bytes", type=int, default=1_572_864)
    parser.add_argument("--working-memory-factor", type=float, default=2.0)

    parser.add_argument("--output-dir", type=str, default="./runs_imx500_supernet")
    parser.add_argument("--save-every", type=int, default=25)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--amp", action="store_true", default=True)
    parser.add_argument("--no-amp", dest="amp", action="store_false")

    return parser.parse_args()


def setup_logging(output_dir: Path) -> logging.Logger:
    output_dir.mkdir(parents=True, exist_ok=True)
    logger = logging.getLogger("supernet_train")
    logger.setLevel(logging.INFO)
    logger.handlers = []

    formatter = logging.Formatter("%(asctime)s | %(levelname)s | %(message)s")

    fh = logging.FileHandler(output_dir / "train.log")
    fh.setLevel(logging.INFO)
    fh.setFormatter(formatter)

    sh = logging.StreamHandler()
    sh.setLevel(logging.INFO)
    sh.setFormatter(formatter)

    logger.addHandler(fh)
    logger.addHandler(sh)
    return logger


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


class SoftTargetKLLoss(nn.Module):
    def __init__(self, temperature: float = 1.0) -> None:
        super().__init__()
        self.temperature = temperature

    def forward(self, student_logits: torch.Tensor, teacher_logits: torch.Tensor) -> torch.Tensor:
        temperature = self.temperature
        log_probs = torch.log_softmax(student_logits / temperature, dim=1)
        probs = torch.softmax(teacher_logits / temperature, dim=1)
        loss = torch.sum(-probs * log_probs, dim=1).mean()
        return loss * (temperature ** 2)


def create_splits(dataset_path: Path, val_split: float, seed: int) -> Tuple[list[int], list[int]]:
    base_dataset = datasets.ImageFolder(root=str(dataset_path))
    num_samples = len(base_dataset)
    all_indices = np.arange(num_samples)

    rng = np.random.default_rng(seed)
    rng.shuffle(all_indices)

    val_count = int(math.floor(num_samples * val_split))
    val_indices = all_indices[:val_count]
    train_indices = all_indices[val_count:]

    return train_indices.tolist(), val_indices.tolist()


def create_loaders(args: argparse.Namespace, max_resolution: int) -> Tuple[DataLoader, DataLoader]:
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])

    train_transform = transforms.Compose(
        [
            transforms.RandomResizedCrop(max_resolution, scale=(0.2, 1.0), interpolation=transforms.InterpolationMode.BILINEAR),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            normalize,
        ]
    )
    val_transform = transforms.Compose(
        [
            transforms.Resize(int(max_resolution * 1.14), interpolation=transforms.InterpolationMode.BILINEAR),
            transforms.CenterCrop(max_resolution),
            transforms.ToTensor(),
            normalize,
        ]
    )

    train_indices, val_indices = create_splits(Path(args.dataset_path), args.val_split, args.seed)

    train_dataset = datasets.ImageFolder(root=str(args.dataset_path), transform=train_transform)
    val_dataset = datasets.ImageFolder(root=str(args.dataset_path), transform=val_transform)

    train_subset = Subset(train_dataset, train_indices)
    val_subset = Subset(val_dataset, val_indices)

    train_loader = DataLoader(
        train_subset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=True,
        drop_last=True,
        persistent_workers=args.num_workers > 0,
    )
    val_loader = DataLoader(
        val_subset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True,
        drop_last=False,
        persistent_workers=args.num_workers > 0,
    )
    return train_loader, val_loader


def accuracy_topk(logits: torch.Tensor, target: torch.Tensor, topk: Tuple[int, ...] = (1, 5)) -> Dict[str, float]:
    maxk = max(topk)
    _, pred = logits.topk(maxk, 1, True, True)
    pred = pred.t()
    correct = pred.eq(target.view(1, -1).expand_as(pred))

    result = {}
    for k in topk:
        correct_k = correct[:k].reshape(-1).float().sum(0)
        result[f"acc{k}"] = float(correct_k.mul_(100.0 / target.size(0)).item())
    return result


def cosine_with_warmup(step: int, total_steps: int, warmup_steps: int, base_lr: float) -> float:
    if step < warmup_steps:
        return base_lr * float(step + 1) / float(max(1, warmup_steps))
    progress = float(step - warmup_steps) / float(max(1, total_steps - warmup_steps))
    return base_lr * 0.5 * (1.0 + math.cos(math.pi * progress))


def save_checkpoint(
    output_dir: Path,
    name: str,
    epoch: int,
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    scaler: torch.amp.GradScaler,
    best_acc1: float,
    args: argparse.Namespace,
) -> None:
    payload = {
        "epoch": epoch,
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "scaler": scaler.state_dict(),
        "best_acc1": best_acc1,
        "args": vars(args),
    }
    torch.save(payload, output_dir / name)


def train_one_epoch(
    epoch: int,
    model: IMX500ResNetSupernet,
    train_loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    hard_criterion: nn.Module,
    soft_criterion: nn.Module,
    scaler: torch.amp.GradScaler,
    args: argparse.Namespace,
    device: torch.device,
    logger: logging.Logger,
) -> Dict[str, float]:
    model.train()
    num_steps = len(train_loader)
    total_steps = args.epochs * num_steps
    warmup_steps = args.warmup_epochs * num_steps

    loss_meter = 0.0
    acc1_meter = 0.0
    acc5_meter = 0.0

    pbar = tqdm(train_loader, desc=f"train {epoch:03d}", leave=False)
    for batch_idx, (images, target) in enumerate(pbar):
        global_step = epoch * num_steps + batch_idx
        lr = cosine_with_warmup(global_step, total_steps, warmup_steps, args.lr)
        for group in optimizer.param_groups:
            group["lr"] = lr

        images = images.to(device, non_blocking=True)
        target = target.to(device, non_blocking=True)

        optimizer.zero_grad(set_to_none=True)

        with torch.autocast(device_type=device.type, enabled=args.amp):
            max_config = model.sample_subnet(mode="max")
            _ = max_config
            max_logits = model(images)
            loss = hard_criterion(max_logits, target)

        scaler.scale(loss).backward()

        with torch.no_grad():
            teacher_logits = max_logits.detach()

        num_arches = max(2, args.num_arch_training)
        for arch_idx in range(1, num_arches):
            if args.sandwich_rule and arch_idx == (num_arches - 1):
                model.sample_subnet(mode="min")
            else:
                tolerance_ratio_low = args.target_tolerance_ratio_low
                tolerance_ratio_high = args.target_tolerance_ratio_high
                if args.target_tolerance_ratio is not None:
                    tolerance_ratio_low = args.target_tolerance_ratio
                    tolerance_ratio_high = args.target_tolerance_ratio

                model.sample_subnet(
                    mode="random",
                    target_total_bytes=args.target_total_bytes,
                    tolerance_ratio=args.target_tolerance_ratio,
                    tolerance_ratio_low=tolerance_ratio_low,
                    tolerance_ratio_high=tolerance_ratio_high,
                    firmware_bytes=args.firmware_bytes,
                    working_memory_factor=args.working_memory_factor,
                )

            with torch.autocast(device_type=device.type, enabled=args.amp):
                student_logits = model(images)
                if args.inplace_distill:
                    aux_loss = soft_criterion(student_logits, teacher_logits)
                else:
                    aux_loss = hard_criterion(student_logits, target)

            scaler.scale(aux_loss).backward()
            loss = aux_loss

        if args.grad_clip > 0:
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)

        scaler.step(optimizer)
        scaler.update()

        with torch.no_grad():
            metrics = accuracy_topk(max_logits, target, topk=(1, 5))

        loss_meter += float(loss.item())
        acc1_meter += metrics["acc1"]
        acc5_meter += metrics["acc5"]

        pbar.set_postfix(loss=f"{loss.item():.4f}", acc1=f"{metrics['acc1']:.2f}", lr=f"{lr:.5f}")

    stats = {
        "loss": loss_meter / num_steps,
        "acc1": acc1_meter / num_steps,
        "acc5": acc5_meter / num_steps,
    }
    logger.info(
        "epoch=%d train loss=%.5f acc1=%.3f acc5=%.3f",
        epoch,
        stats["loss"],
        stats["acc1"],
        stats["acc5"],
    )
    return stats


@torch.no_grad()
def evaluate(
    epoch: int,
    model: IMX500ResNetSupernet,
    val_loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
    logger: logging.Logger,
    mode: str,
) -> Dict[str, float]:
    model.eval()
    model.sample_subnet(mode=mode)

    loss_meter = 0.0
    acc1_meter = 0.0
    acc5_meter = 0.0
    steps = 0

    pbar = tqdm(val_loader, desc=f"val-{mode} {epoch:03d}", leave=False)
    for images, target in pbar:
        images = images.to(device, non_blocking=True)
        target = target.to(device, non_blocking=True)

        logits = model(images)
        loss = criterion(logits, target)
        metrics = accuracy_topk(logits, target, topk=(1, 5))

        steps += 1
        loss_meter += float(loss.item())
        acc1_meter += metrics["acc1"]
        acc5_meter += metrics["acc5"]

        pbar.set_postfix(loss=f"{loss.item():.4f}", acc1=f"{metrics['acc1']:.2f}")

    stats = {
        "loss": loss_meter / max(1, steps),
        "acc1": acc1_meter / max(1, steps),
        "acc5": acc5_meter / max(1, steps),
    }

    active = model.active_subnet
    resources = model.estimate_subnet_resources(active)
    logger.info(
        "epoch=%d val mode=%s loss=%.5f acc1=%.3f acc5=%.3f total_bytes=%d params=%d config=%s",
        epoch,
        mode,
        stats["loss"],
        stats["acc1"],
        stats["acc5"],
        resources["total_estimated_bytes"],
        resources["params"],
        json.dumps(active.to_dict(), sort_keys=True),
    )
    return stats


def dump_supernet_profile(model: IMX500ResNetSupernet, output_dir: Path, logger: logging.Logger) -> None:
    max_cfg = model.max_subnet_config()
    min_cfg = model.min_subnet_config()

    summary = {
        "max_subnet": {
            "config": max_cfg.to_dict(),
            "resource_estimate": model.estimate_subnet_resources(max_cfg),
        },
        "min_subnet": {
            "config": min_cfg.to_dict(),
            "resource_estimate": model.estimate_subnet_resources(min_cfg),
        },
        "candidates": {
            "resolution": list(model.resolution_candidates),
            "stem_width": list(model.stem_width_candidates),
            "stage_depths": [list(v) for v in model.stage_depth_candidates],
            "stage_widths": [list(v) for v in model.stage_width_candidates],
        },
    }

    with (output_dir / "supernet_profile.json").open("w", encoding="utf-8") as fp:
        json.dump(summary, fp, indent=2)
    logger.info("Wrote supernet profile: %s", output_dir / "supernet_profile.json")


def main() -> None:
    args = parse_args()
    run_id = time.strftime("%Y%m%d_%H%M%S")
    output_dir = Path(args.output_dir) / run_id
    logger = setup_logging(output_dir)

    set_seed(args.seed)

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    if device.type != "cuda":
        logger.warning("CUDA not available, using CPU. This will be very slow.")

    model = create_default_supernet(num_classes=args.num_classes).to(device)
    hard_criterion = nn.CrossEntropyLoss(label_smoothing=args.label_smoothing)
    soft_criterion = SoftTargetKLLoss(temperature=1.0)

    optimizer = torch.optim.SGD(
        model.parameters(),
        lr=args.lr,
        momentum=args.momentum,
        weight_decay=args.weight_decay,
        nesterov=True,
    )
    scaler = torch.amp.GradScaler(enabled=args.amp and device.type == "cuda")

    train_loader, val_loader = create_loaders(args, max_resolution=max(model.resolution_candidates))

    logger.info("Training configuration: %s", json.dumps(vars(args), sort_keys=True))
    logger.info("Using device: %s", device)
    logger.info("Train samples=%d | Val samples=%d", len(train_loader.dataset), len(val_loader.dataset))

    dump_supernet_profile(model, output_dir, logger)

    best_acc1 = -1.0
    history = []
    for epoch in range(args.epochs):
        train_stats = train_one_epoch(
            epoch,
            model,
            train_loader,
            optimizer,
            hard_criterion,
            soft_criterion,
            scaler,
            args,
            device,
            logger,
        )
        val_max = evaluate(epoch, model, val_loader, hard_criterion, device, logger, mode="max")
        val_min = evaluate(epoch, model, val_loader, hard_criterion, device, logger, mode="min")

        epoch_record = {
            "epoch": epoch,
            "train": train_stats,
            "val_max": val_max,
            "val_min": val_min,
        }
        history.append(epoch_record)

        with (output_dir / "metrics.json").open("w", encoding="utf-8") as fp:
            json.dump(history, fp, indent=2)

        if val_max["acc1"] > best_acc1:
            best_acc1 = val_max["acc1"]
            save_checkpoint(output_dir, "best.pt", epoch, model, optimizer, scaler, best_acc1, args)
            logger.info("Saved new best checkpoint (acc1=%.3f)", best_acc1)

        if epoch % args.save_every == 0 or epoch == args.epochs - 1:
            save_checkpoint(output_dir, f"checkpoint_epoch_{epoch:03d}.pt", epoch, model, optimizer, scaler, best_acc1, args)

    logger.info("Training complete. Best val(max) acc1=%.3f", best_acc1)


if __name__ == "__main__":
    main()
