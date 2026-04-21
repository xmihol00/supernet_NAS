#!/usr/bin/env python3
"""
Full-dataset correlation experiment.

Trains 15 NAS-selected subnet architectures (initially from the supernet checkpoint)
on the full ImageNet training set in a round-robin fashion — 1 epoch per architecture
per cycle — and tests whether the 6-class NAS quantised accuracy (nas_quant_acc1)
correlates with the 1000-class floating-point validation accuracy.

Usage:
    python full_dataset_training_and_analysis.py \\
        --architectures-json selected_architectures.json \\
        --dataset-path /path/to/imagenet/train \\
        --checkpoint /path/to/supernet/best.pt \\
        --output-dir ./full_dataset_experiment

Resume: re-run the exact same command.  Progress is recovered from checkpoints
and progress.json inside output-dir.

Stop: Ctrl-C (or SIGTERM).  All state is already saved — the next run picks up
where this one left off.
"""

from __future__ import annotations

import argparse
import copy
import csv
import json
import logging
import math
import os
import random
import signal
import sys
import time
from dataclasses import asdict, dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Sized, Tuple, cast
import shutil

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
from scipy import stats as scipy_stats
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms

# ── project root on sys.path ──────────────────────────────────────────────────
_PROJECT_ROOT = Path(__file__).resolve().parent.parent
_SUPERNET_DIR = _PROJECT_ROOT / "supernet"
if str(_SUPERNET_DIR) not in sys.path:
    sys.path.insert(0, str(_SUPERNET_DIR))

from imx500_supernet import IMX500ResNetSupernet, SubnetConfig, create_default_supernet  # noqa: E402

try:
    import safe_gpu
except Exception:
    safe_gpu = None

# ─────────────────────────────────────────────────────────────────────────────
# Architecture constants
# ─────────────────────────────────────────────────────────────────────────────
_IMAGENET_MEAN = [0.485, 0.456, 0.406]
_IMAGENET_STD  = [0.229, 0.224, 0.225]

_SSD_MOUNT     = Path("/mnt/ssd")
_SSD_CACHE_DIR = _SSD_MOUNT / "xmihol00"

# ─────────────────────────────────────────────────────────────────────────────
# Static subnet model  (copied verbatim from fully_train_best_subnets.py)
# ─────────────────────────────────────────────────────────────────────────────

class StaticBasicBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, stride: int) -> None:
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, 3, stride=stride, padding=1, bias=False)
        self.bn1   = nn.BatchNorm2d(out_channels)
        self.relu  = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(out_channels, out_channels, 3, stride=1, padding=1, bias=False)
        self.bn2   = nn.BatchNorm2d(out_channels)
        self.downsample: Optional[nn.Sequential] = None
        if stride != 1 or in_channels != out_channels:
            self.downsample = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, 1, stride=stride, bias=False),
                nn.BatchNorm2d(out_channels),
            )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        identity = x
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        if self.downsample is not None:
            identity = self.downsample(x)
        return self.relu(out + identity)


class StaticSubnetModel(nn.Module):
    def __init__(self, config: SubnetConfig, num_classes: int,
                 stage_strides: Sequence[int]) -> None:
        super().__init__()
        self.config = config
        self.stem_conv = nn.Conv2d(3, config.stem_width, 3, stride=2, padding=1, bias=False)
        self.stem_bn   = nn.BatchNorm2d(config.stem_width)
        self.stem_act  = nn.ReLU(inplace=True)
        stages: List[nn.Module] = []
        prev = config.stem_width
        for i in range(4):
            depth, out_w, stride = config.stage_depths[i], config.stage_widths[i], int(stage_strides[i])
            blocks = [StaticBasicBlock(prev if j == 0 else out_w, out_w, stride if j == 0 else 1)
                      for j in range(depth)]
            stages.append(nn.Sequential(*blocks))
            prev = out_w
        self.stages      = nn.ModuleList(stages)
        self.global_pool = nn.AdaptiveAvgPool2d(1)
        self.classifier  = nn.Linear(config.stage_widths[-1], num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.stem_act(self.stem_bn(self.stem_conv(x)))
        for stage in self.stages:
            x = stage(x)
        return self.classifier(torch.flatten(self.global_pool(x), 1))


def _copy_bn(src: nn.BatchNorm2d, dst: nn.BatchNorm2d, ch: int) -> None:
    dst.weight.data.copy_(src.weight.data[:ch])
    dst.bias.data.copy_(src.bias.data[:ch])
    if src.running_mean is not None and dst.running_mean is not None:
        dst.running_mean.data.copy_(src.running_mean.data[:ch])
    if src.running_var is not None and dst.running_var is not None:
        dst.running_var.data.copy_(src.running_var.data[:ch])
    if src.num_batches_tracked is not None and dst.num_batches_tracked is not None:
        dst.num_batches_tracked.data.copy_(src.num_batches_tracked.data)


def build_static_subnet_model(supernet: IMX500ResNetSupernet, config: SubnetConfig,
                               num_classes: int, device: torch.device
                               ) -> tuple[StaticSubnetModel, str | None]:
    """Build a static subnet and transfer supernet weights.

    Returns (model, error_message). On any weight-transfer failure the model is
    returned with whatever weights were successfully copied plus random init for
    the remaining layers, and error_message describes the first error encountered.
    Blocks whose index exceeds the supernet's capacity are left randomly
    initialised (partial transfer) rather than aborting the whole copy.
    """
    model = StaticSubnetModel(config, num_classes, supernet.stage_strides).to(device)
    transfer_error: str | None = None
    try:
        with torch.no_grad():
            model.stem_conv.weight.copy_(supernet.stem_conv.weight.data[:config.stem_width, :3])
            _copy_bn(supernet.stem_bn, model.stem_bn, config.stem_width)
            prev = config.stem_width
            for si in range(4):
                depth, out_w = config.stage_depths[si], config.stage_widths[si]
                stride = supernet.stage_strides[si]
                max_blocks = len(supernet.stages[si].blocks)
                for bi in range(depth):
                    block_in = prev if bi == 0 else out_w
                    if bi >= max_blocks:
                        # Supernet has fewer blocks than this subnet stage requires;
                        # leave the extra blocks randomly initialised.
                        if transfer_error is None:
                            transfer_error = (
                                f"stage {si} block {bi}: supernet only has {max_blocks} "
                                f"block(s) but subnet needs {depth} — extra blocks use random init"
                            )
                        continue
                    db = supernet.stages[si].blocks[bi]
                    sb = cast(StaticBasicBlock, cast(nn.Sequential, model.stages[si])[bi])
                    sb.conv1.weight.copy_(db.conv1.weight.data[:out_w, :block_in])
                    _copy_bn(db.bn1, sb.bn1, out_w)
                    sb.conv2.weight.copy_(db.conv2.weight.data[:out_w, :out_w])
                    _copy_bn(db.bn2, sb.bn2, out_w)
                    needs_proj = (stride != 1 and bi == 0) or (block_in != out_w)
                    if needs_proj and sb.downsample is not None:
                        ds = cast(nn.Sequential, sb.downsample)
                        # Support both attribute-style and sequential-style downsample on supernet
                        if hasattr(db, "downsample_conv"):
                            ds_conv_w = db.downsample_conv.weight.data
                            ds_bn_src = db.downsample_bn
                        elif hasattr(db, "downsample") and db.downsample is not None:
                            ds_conv_w = db.downsample[0].weight.data
                            ds_bn_src = db.downsample[1]
                        else:
                            raise AttributeError(
                                f"stage {si} block {bi}: cannot locate downsample weights on supernet block"
                            )
                        cast(nn.Conv2d, ds[0]).weight.copy_(ds_conv_w[:out_w, :block_in])
                        _copy_bn(ds_bn_src, cast(nn.BatchNorm2d, ds[1]), out_w)
                prev = out_w
            model.classifier.weight.copy_(supernet.classifier.weight.data[:, :config.stage_widths[-1]])
            model.classifier.bias.copy_(supernet.classifier.bias.data)
    except Exception as exc:
        transfer_error = str(exc)

    return model, transfer_error


# ─────────────────────────────────────────────────────────────────────────────
# EMA
# ─────────────────────────────────────────────────────────────────────────────

class ModelEMA:
    def __init__(self, model: nn.Module, decay: float) -> None:
        self.decay = decay
        self.ema = copy.deepcopy(model).eval()
        for p in self.ema.parameters():
            p.requires_grad_(False)

    @torch.no_grad()
    def update(self, model: nn.Module) -> None:
        for key, ev in self.ema.state_dict().items():
            mv = model.state_dict()[key].detach()
            if torch.is_floating_point(ev):
                ev.mul_(self.decay).add_(mv, alpha=1.0 - self.decay)
            else:
                ev.copy_(mv)


# ─────────────────────────────────────────────────────────────────────────────
# Data
# ─────────────────────────────────────────────────────────────────────────────

def create_split_indices(dataset_path: Path, val_frac: float,
                         seed: int, cache_path: Path) -> Tuple[List[int], List[int]]:
    """Create (or reload) reproducible train/val split indices."""
    if cache_path.exists():
        with cache_path.open() as f:
            d = json.load(f)
        return d["train"], d["val"]

    base = datasets.ImageFolder(root=str(dataset_path))
    n = len(base)
    rng = np.random.default_rng(seed)
    idx = np.arange(n)
    rng.shuffle(idx)
    n_val = int(math.floor(n * val_frac))
    val_idx  = idx[:n_val].tolist()
    train_idx = idx[n_val:].tolist()

    cache_path.parent.mkdir(parents=True, exist_ok=True)
    with cache_path.open("w") as f:
        json.dump({"train": train_idx, "val": val_idx,
                   "n_total": n, "val_frac": val_frac, "seed": seed}, f)
    return train_idx, val_idx


def build_loaders(dataset_path: Path, train_idx: List[int], val_idx: List[int],
                  resolution: int, batch_size: int, num_workers: int,
                  randaugment_magnitude: int) -> Tuple[DataLoader, DataLoader]:
    normalize = transforms.Normalize(_IMAGENET_MEAN, _IMAGENET_STD)
    train_tf = transforms.Compose([
        transforms.RandomResizedCrop(resolution, scale=(0.2, 1.0), ratio=(0.75, 1.333),
                                     interpolation=transforms.InterpolationMode.BILINEAR),
        transforms.RandomHorizontalFlip(),
        transforms.RandAugment(num_ops=2, magnitude=randaugment_magnitude),
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.05),
        transforms.ToTensor(), normalize,
        transforms.RandomErasing(p=0.15, scale=(0.02, 0.12), ratio=(0.3, 3.3)),
    ])
    val_tf = transforms.Compose([
        transforms.Resize(int(resolution * 1.14), interpolation=transforms.InterpolationMode.BILINEAR),
        transforms.CenterCrop(resolution),
        transforms.ToTensor(), normalize,
    ])
    root = str(dataset_path)
    train_ds = Subset(datasets.ImageFolder(root=root, transform=train_tf), train_idx)
    val_ds   = Subset(datasets.ImageFolder(root=root, transform=val_tf),   val_idx)
    kw = dict(num_workers=num_workers, pin_memory=True,
              persistent_workers=num_workers > 0)
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True,
                              drop_last=True, **kw)
    val_loader   = DataLoader(val_ds,   batch_size=batch_size, shuffle=False,
                              drop_last=False, **kw)
    return train_loader, val_loader


# ─────────────────────────────────────────────────────────────────────────────
# SSD dataset cache
# ─────────────────────────────────────────────────────────────────────────────

def _get_dir_size(path: Path) -> int:
    """Return total byte size of a directory tree (fast: delegates to `du -sb`)."""
    import subprocess
    try:
        out = subprocess.check_output(["du", "-sb", str(path)], stderr=subprocess.DEVNULL)
        return int(out.split()[0])
    except Exception:
        # Fallback: Python walk (slower on NFS)
        total = 0
        for p in path.rglob("*"):
            if p.is_file():
                try:
                    total += p.stat().st_size
                except OSError:
                    pass
        return total


def try_cache_dataset_on_ssd(dataset_path: Path, logger: logging.Logger) -> Path:
    """Copy the dataset to SSD if there is enough free space.

    Returns the path that should be used for training — either the SSD copy
    (fast local NVMe) or the original path if copying is not possible.
    If the destination already exists it is reused without re-copying.
    """
    ssd_dest = _SSD_CACHE_DIR / dataset_path.name
    
    if ssd_dest.exists() and any(ssd_dest.iterdir()):
        logger.info("SSD cache found at %s — skipping copy.", ssd_dest)
        return ssd_dest

    # Measure dataset size
    logger.info("Measuring dataset size at %s …", dataset_path)
    t0 = time.time()
    dataset_bytes = _get_dir_size(dataset_path)
    logger.info("Dataset size: %.2f GB (measured in %.1f s)", dataset_bytes / 1e9, time.time() - t0)

    # Check free space on SSD
    try:
        free_bytes = shutil.disk_usage(_SSD_MOUNT).free
        logger.info("SSD free space: %.2f GB", free_bytes / 1e9)
    except OSError as exc:
        logger.warning("Cannot check SSD space (%s) — using original path.", exc)
        return dataset_path

    if free_bytes <= dataset_bytes:
        logger.warning(
            "SSD has %.2f GB free but dataset needs %.2f GB — using original path.",
            free_bytes / 1e9, dataset_bytes / 1e9,
        )
        return dataset_path

    # Copy with periodic progress logging
    logger.info("Copying dataset to SSD: %s → %s", dataset_path, ssd_dest)
    _SSD_CACHE_DIR.mkdir(parents=True, exist_ok=True)

    copied_bytes: list[int] = [0]
    copied_files: list[int] = [0]
    last_log:     list[float] = [time.time()]

    def _copy_fn(src: str, dst: str) -> None:
        shutil.copy2(src, dst)
        try:
            copied_bytes[0] += Path(src).stat().st_size
        except OSError:
            pass
        copied_files[0] += 1
        now = time.time()
        if now - last_log[0] >= 30:
            pct = 100.0 * copied_bytes[0] / dataset_bytes if dataset_bytes else 0.0
            logger.info(
                "  Copying… %d files | %.2f / %.2f GB (%.1f%%)",
                copied_files[0], copied_bytes[0] / 1e9, dataset_bytes / 1e9, pct,
            )
            last_log[0] = now

    try:
        shutil.copytree(str(dataset_path), str(ssd_dest), copy_function=_copy_fn)
        logger.info(
            "Dataset copy complete: %d files, %.2f GB → %s",
            copied_files[0], copied_bytes[0] / 1e9, ssd_dest,
        )
        return ssd_dest
    except Exception as exc:
        logger.warning("Dataset copy failed (%s) — removing partial copy and using original path.", exc)
        shutil.rmtree(ssd_dest, ignore_errors=True)
        return dataset_path


# ─────────────────────────────────────────────────────────────────────────────
# Training
# ─────────────────────────────────────────────────────────────────────────────

def _cosine_restart_lr(epoch: int, t0: int, lr_max: float, lr_min: float,
                       warmup: int) -> float:
    """Cosine annealing with warm restarts; linear warmup for the first `warmup` epochs."""
    if epoch < warmup:
        return lr_min + (lr_max - lr_min) * (epoch + 1) / warmup
    t = (epoch - warmup) % t0
    return lr_min + 0.5 * (lr_max - lr_min) * (1.0 + math.cos(math.pi * t / t0))


def _rand_bbox(size: torch.Size, lam: float) -> Tuple[int, int, int, int]:
    _, _, h, w = size
    cut = math.sqrt(max(0.0, 1.0 - lam))
    cw, ch = int(w * cut), int(h * cut)
    cx, cy = np.random.randint(0, w), np.random.randint(0, h)
    return (int(np.clip(cx - cw // 2, 0, w)), int(np.clip(cy - ch // 2, 0, h)),
            int(np.clip(cx + cw // 2, 0, w)), int(np.clip(cy + ch // 2, 0, h)))


def apply_batch_reg(imgs: torch.Tensor, tgt: torch.Tensor,
                    cutmix_alpha: float, cutmix_prob: float,
                    mixup_alpha: float,  mixup_prob: float,
                    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, float]:
    if cutmix_alpha > 0 and random.random() < cutmix_prob:
        lam = float(np.random.beta(cutmix_alpha, cutmix_alpha))
        idx = torch.randperm(imgs.size(0), device=imgs.device)
        ta, tb = tgt, tgt[idx]
        x1, y1, x2, y2 = _rand_bbox(imgs.size(), lam)
        imgs[:, :, y1:y2, x1:x2] = imgs[idx, :, y1:y2, x1:x2]
        lam = 1.0 - (x2 - x1) * (y2 - y1) / (imgs.size(-1) * imgs.size(-2))
        return imgs, ta, tb, float(lam)
    if mixup_alpha > 0 and random.random() < mixup_prob:
        lam = float(np.random.beta(mixup_alpha, mixup_alpha))
        idx = torch.randperm(imgs.size(0), device=imgs.device)
        return lam * imgs + (1 - lam) * imgs[idx], tgt, tgt[idx], float(lam)
    return imgs, tgt, tgt, 1.0


def accuracy_topk(logits: torch.Tensor, target: torch.Tensor) -> Tuple[float, float]:
    _, pred = logits.topk(5, 1, True, True)
    pred = pred.t()
    correct = pred.eq(target.view(1, -1).expand_as(pred))
    acc1 = float(correct[:1].reshape(-1).float().sum()) * 100.0 / target.size(0)
    acc5 = float(correct[:5].reshape(-1).float().sum()) * 100.0 / target.size(0)
    return acc1, acc5


def train_one_epoch(
    arch_idx: int,
    epoch: int,
    model: nn.Module,
    train_loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    criterion: nn.Module,
    scaler: torch.amp.GradScaler,
    ema: ModelEMA,
    device: torch.device,
    args: argparse.Namespace,
    logger: logging.Logger,
) -> Dict[str, float]:
    model.train()
    n_steps = len(train_loader)
    lr = _cosine_restart_lr(epoch, args.lr_restart_period, args.lr, args.min_lr, args.warmup_epochs)
    for g in optimizer.param_groups:
        g["lr"] = lr

    # Unfreeze backbone after first epoch
    if epoch >= args.freeze_backbone_epochs:
        for p in model.parameters():
            p.requires_grad_(True)
    else:
        for name, p in model.named_parameters():
            p.requires_grad_("classifier" in name)

    loss_sum = acc1_sum = acc5_sum = 0.0
    t0 = time.time()

    for step, (imgs, tgt) in enumerate(train_loader):
        imgs = imgs.to(device, non_blocking=True)
        tgt  = tgt.to(device, non_blocking=True)
        imgs, ta, tb, lam = apply_batch_reg(imgs, tgt,
                                            args.cutmix_alpha, args.cutmix_prob,
                                            args.mixup_alpha, args.mixup_prob)
        optimizer.zero_grad(set_to_none=True)
        with torch.autocast(device_type=device.type, enabled=args.amp and device.type == "cuda"):
            logits = model(imgs)
            loss = lam * criterion(logits, ta) + (1 - lam) * criterion(logits, tb) if lam < 1.0 else criterion(logits, tgt)
        scaler.scale(loss).backward()
        if args.grad_clip > 0:
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
        scaler.step(optimizer)
        scaler.update()
        ema.update(model)

        with torch.no_grad():
            a1, a5 = accuracy_topk(logits, tgt)
        loss_sum += float(loss.item()); acc1_sum += a1; acc5_sum += a5

        if (step + 1) % args.log_interval == 0 or step == n_steps - 1:
            elapsed = time.time() - t0
            eta = elapsed / (step + 1) * (n_steps - step - 1)
            logger.info(
                "[arch %02d | ep %d | %5d/%d] loss=%.4f acc1=%5.2f%% acc5=%5.2f%% "
                "lr=%.6f | %.1f s/batch | ETA %.0f s",
                arch_idx, epoch, step + 1, n_steps,
                loss_sum / (step + 1), acc1_sum / (step + 1), acc5_sum / (step + 1),
                lr, elapsed / (step + 1), eta,
            )

    n = max(1, n_steps)
    return {"loss": loss_sum / n, "acc1": acc1_sum / n, "acc5": acc5_sum / n, "lr": lr}


@torch.no_grad()
def evaluate(model: nn.Module, val_loader: DataLoader, criterion: nn.Module,
             device: torch.device) -> Dict[str, float]:
    model.eval()
    loss_sum = acc1_sum = acc5_sum = 0.0
    n = 0
    for imgs, tgt in val_loader:
        imgs, tgt = imgs.to(device, non_blocking=True), tgt.to(device, non_blocking=True)
        logits = model(imgs)
        loss_sum += float(criterion(logits, tgt).item())
        a1, a5 = accuracy_topk(logits, tgt)
        acc1_sum += a1; acc5_sum += a5; n += 1
    n = max(1, n)
    return {"loss": loss_sum / n, "acc1": acc1_sum / n, "acc5": acc5_sum / n}


# ─────────────────────────────────────────────────────────────────────────────
# Checkpointing
# ─────────────────────────────────────────────────────────────────────────────

def save_arch_checkpoint(path: Path, epoch: int, model: nn.Module,
                         optimizer: torch.optim.Optimizer,
                         scaler: torch.amp.GradScaler,
                         ema: ModelEMA, best_acc1: float,
                         arch_record: dict) -> None:
    torch.save({
        "epoch": epoch,
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "scaler": scaler.state_dict(),
        "ema": ema.ema.state_dict(),
        "best_acc1": best_acc1,
        "arch_record": arch_record,
    }, path)


def load_arch_checkpoint(path: Path, model: nn.Module,
                          optimizer: torch.optim.Optimizer,
                          scaler: torch.amp.GradScaler,
                          ema: ModelEMA, device: torch.device,
                          ) -> Tuple[int, float]:
    """Returns (epoch, best_acc1)."""
    ckpt = torch.load(path, map_location=device, weights_only=False)
    model.load_state_dict(ckpt["model"])
    optimizer.load_state_dict(ckpt["optimizer"])
    scaler.load_state_dict(ckpt["scaler"])
    ema.ema.load_state_dict(ckpt["ema"])
    return int(ckpt["epoch"]), float(ckpt.get("best_acc1", -1.0))


# ─────────────────────────────────────────────────────────────────────────────
# Statistical analysis
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class CycleStats:
    cycle: int
    n_archs: int
    spearman_r: float
    spearman_p: float
    pearson_r: float
    pearson_p: float
    kendall_tau: float
    kendall_p: float
    bootstrap_ci_spearman: tuple
    mean_val_acc1: float
    std_val_acc1: float
    per_arch: List[Dict[str, Any]]


def compute_cycle_stats(cycle: int, arch_records: List[dict],
                        n_boot: int = 5000, rng_seed: int = 42) -> CycleStats:
    nas_accs  = np.array([r["nas_quant_acc1"] for r in arch_records])
    val_accs  = np.array([r["current_val_acc1"] for r in arch_records])
    ema_accs  = np.array([r["current_ema_acc1"] for r in arch_records])

    sr, sp = scipy_stats.spearmanr(nas_accs, val_accs)
    pr, pp = scipy_stats.pearsonr(nas_accs, val_accs)
    kt, kp = scipy_stats.kendalltau(nas_accs, val_accs)

    # Bootstrap CI for Spearman ρ using Fisher z-transform
    rng = np.random.default_rng(rng_seed)
    boot_r = []
    n = len(nas_accs)
    for _ in range(n_boot):
        idx = rng.integers(0, n, size=n)
        if len(set(idx)) < 3:
            continue
        r_b, _ = scipy_stats.spearmanr(nas_accs[idx], val_accs[idx])
        boot_r.append(float(r_b))
    if boot_r:
        ci = (float(np.percentile(boot_r, 2.5)), float(np.percentile(boot_r, 97.5)))
    else:
        ci = (float("nan"), float("nan"))

    per_arch = []
    for i, r in enumerate(arch_records):
        per_arch.append({
            "arch_index": r["arch_index"],
            "nas_quant_acc1": r["nas_quant_acc1"],
            "val_acc1": r["current_val_acc1"],
            "ema_acc1": r["current_ema_acc1"],
            "epochs_completed": r["epochs_completed"],
            "nas_rank": int(np.argsort(np.argsort(-nas_accs))[i]),
            "val_rank": int(np.argsort(np.argsort(-val_accs))[i]),
        })

    return CycleStats(
        cycle=cycle,
        n_archs=len(arch_records),
        spearman_r=float(sr), spearman_p=float(sp),
        pearson_r=float(pr),  pearson_p=float(pp),
        kendall_tau=float(kt), kendall_p=float(kp),
        bootstrap_ci_spearman=ci,
        mean_val_acc1=float(np.mean(val_accs)),
        std_val_acc1=float(np.std(val_accs, ddof=1)),
        per_arch=per_arch,
    )


# ─────────────────────────────────────────────────────────────────────────────
# Plotting
# ─────────────────────────────────────────────────────────────────────────────

_CMAP = matplotlib.colormaps.get_cmap("tab20").resampled(20)


def _arch_color(i: int):
    return _CMAP(i % 20)


def plot_training_curves(arch_records: List[dict], histories: Dict[int, List[dict]],
                          out_path: Path) -> None:
    """Multi-panel: loss and val_acc1 per architecture over all completed epochs."""
    n = len(arch_records)
    ncols = min(5, n)
    nrows = math.ceil(n / ncols)
    fig, axes = plt.subplots(nrows, ncols, figsize=(4.5 * ncols, 3.5 * nrows), squeeze=False)

    for idx, rec in enumerate(arch_records):
        ai = rec["arch_index"]
        hist = histories.get(ai, [])
        ax = axes[idx // ncols][idx % ncols]
        if hist:
            eps   = [h["epoch"] for h in hist]
            v_acc = [h["val_acc1"] for h in hist]
            e_acc = [h["ema_acc1"] for h in hist]
            t_acc = [h["train_acc1"] for h in hist]
            ax.plot(eps, t_acc, lw=1.2, alpha=0.7, label="train")
            ax.plot(eps, v_acc, lw=1.5, label="val")
            ax.plot(eps, e_acc, lw=1.5, linestyle="--", label="val-EMA")
        ax.set_title(f"Arch {ai} | NAS={rec['nas_quant_acc1']:.1f}%", fontsize=9)
        ax.set_xlabel("Epoch", fontsize=8)
        ax.set_ylabel("Top-1 Acc (%)", fontsize=8)
        ax.legend(fontsize=7, loc="lower right")
        ax.yaxis.grid(True, alpha=0.3)

    # hide unused panels
    for j in range(n, nrows * ncols):
        axes[j // ncols][j % ncols].set_visible(False)

    fig.suptitle("Training Curves — All Architectures", fontsize=12)
    fig.tight_layout()
    fig.savefig(out_path, dpi=130, bbox_inches="tight")
    plt.close(fig)


def plot_correlation_scatter(stats: CycleStats, arch_records: List[dict],
                              out_path: Path) -> None:
    """NAS quantised acc (x) vs current full-dataset val acc (y) scatter."""
    nas  = [p["nas_quant_acc1"] for p in stats.per_arch]
    val  = [p["val_acc1"] for p in stats.per_arch]
    ema  = [p["ema_acc1"] for p in stats.per_arch]
    idxs = [p["arch_index"] for p in stats.per_arch]

    fig, ax = plt.subplots(figsize=(8, 6))
    for i, (x, y, ye, ai) in enumerate(zip(nas, val, ema, idxs)):
        ax.scatter(x, y, color=_arch_color(i), s=80, zorder=3, edgecolors="white", lw=0.6)
        ax.scatter(x, ye, marker="D", color=_arch_color(i), s=50, zorder=3,
                   edgecolors="black", lw=0.5, alpha=0.7)
        ax.annotate(f"A{ai}", (x, y), textcoords="offset points", xytext=(5, 3), fontsize=7)

    # Regression line (using val)
    if len(nas) >= 2:
        m, b, *_ = scipy_stats.linregress(nas, val)
        xs = np.linspace(min(nas), max(nas), 100)
        ax.plot(xs, m * xs + b, "k--", lw=1.2, alpha=0.6)

    ci_lo, ci_hi = stats.bootstrap_ci_spearman
    ax.set_xlabel("NAS Quantised Accuracy — 6-class subset (%)", fontsize=11)
    ax.set_ylabel("Full ImageNet Val Accuracy — float, 1000 classes (%)", fontsize=11)
    sig = "✓ significant" if stats.spearman_p < 0.05 else "✗ not significant"
    ax.set_title(
        f"Cycle {stats.cycle} | Spearman ρ = {stats.spearman_r:.3f} (p={stats.spearman_p:.4f}) {sig}\n"
        f"Bootstrap 95% CI: [{ci_lo:.3f}, {ci_hi:.3f}] | "
        f"Pearson r = {stats.pearson_r:.3f} | Kendall τ = {stats.kendall_tau:.3f}",
        fontsize=10,
    )
    ax.xaxis.grid(True, alpha=0.3)
    ax.yaxis.grid(True, alpha=0.3)
    ax.set_axisbelow(True)
    # legend: circles = val, diamonds = EMA
    from matplotlib.lines import Line2D
    legend_handles = [
        Line2D([0], [0], marker="o", color="gray", lw=0, markersize=8, label="Val (float)"),
        Line2D([0], [0], marker="D", color="gray", lw=0, markersize=7, label="Val EMA"),
    ]
    ax.legend(handles=legend_handles, fontsize=9)
    fig.tight_layout()
    fig.savefig(out_path, dpi=130, bbox_inches="tight")
    plt.close(fig)


def plot_rank_comparison(stats: CycleStats, out_path: Path) -> None:
    """Side-by-side bar: NAS rank vs full-dataset rank per architecture."""
    n = len(stats.per_arch)
    sorted_by_nas = sorted(stats.per_arch, key=lambda p: p["nas_quant_acc1"])
    ai_labels = [f"A{p['arch_index']}" for p in sorted_by_nas]
    nas_ranks = [p["nas_rank"] for p in sorted_by_nas]
    val_ranks = [p["val_rank"] for p in sorted_by_nas]

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    x = np.arange(n)
    w = 0.38
    axes[0].bar(x - w / 2, nas_ranks, w, label="NAS rank",  color="#0072B2", alpha=0.85)
    axes[0].bar(x + w / 2, val_ranks, w, label="Full-dataset rank", color="#D55E00", alpha=0.85)
    axes[0].set_xticks(x); axes[0].set_xticklabels(ai_labels, fontsize=8)
    axes[0].set_ylabel("Rank (0=best)", fontsize=11)
    axes[0].set_title("NAS rank vs. Full-dataset rank (sorted by NAS acc)", fontsize=11)
    axes[0].legend(fontsize=10)
    axes[0].yaxis.grid(True, alpha=0.3)

    # Rank difference
    diff = [v - n for v, n in zip(val_ranks, nas_ranks)]
    colors = ["#009E73" if d <= 0 else "#CC79A7" for d in diff]
    axes[1].bar(x, diff, color=colors, alpha=0.85, edgecolor="white")
    axes[1].axhline(0, color="black", lw=0.8)
    axes[1].set_xticks(x); axes[1].set_xticklabels(ai_labels, fontsize=8)
    axes[1].set_ylabel("Val rank − NAS rank (negative = moved up)", fontsize=11)
    axes[1].set_title("Rank shift from NAS to Full-dataset", fontsize=11)
    axes[1].yaxis.grid(True, alpha=0.3)

    fig.suptitle(f"Cycle {stats.cycle} | Spearman ρ = {stats.spearman_r:.3f}", fontsize=12)
    fig.tight_layout()
    fig.savefig(out_path, dpi=130, bbox_inches="tight")
    plt.close(fig)


def plot_correlation_over_cycles(all_stats: List[dict], out_path: Path) -> None:
    """Line plot of Spearman ρ, Pearson r, Kendall τ over cycles."""
    if len(all_stats) < 2:
        return
    cycles = [s["cycle"] for s in all_stats]
    sr = [s["spearman_r"] for s in all_stats]
    pr = [s["pearson_r"]  for s in all_stats]
    kt = [s["kendall_tau"] for s in all_stats]
    sp = [s["spearman_p"]  for s in all_stats]

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 7), sharex=True)
    ax1.plot(cycles, sr, "o-", color="#0072B2", lw=2, label="Spearman ρ")
    ax1.plot(cycles, pr, "s-", color="#D55E00", lw=2, label="Pearson r")
    ax1.plot(cycles, kt, "^-", color="#009E73", lw=2, label="Kendall τ")
    ax1.axhline(0, color="black", lw=0.8, linestyle="--")
    ax1.set_ylabel("Correlation coefficient", fontsize=11)
    ax1.set_title("Correlation between NAS accuracy and Full-dataset accuracy over cycles", fontsize=12)
    ax1.legend(fontsize=10)
    ax1.yaxis.grid(True, alpha=0.3)
    ax1.set_ylim(-1.05, 1.05)

    ax2.semilogy(cycles, [max(p, 1e-10) for p in sp], "o-", color="#0072B2", lw=2, label="Spearman p-value")
    ax2.axhline(0.05, color="red", lw=1.2, linestyle="--", label="α=0.05")
    ax2.set_xlabel("Cycle", fontsize=11)
    ax2.set_ylabel("p-value (log scale)", fontsize=11)
    ax2.legend(fontsize=10)
    ax2.yaxis.grid(True, alpha=0.3)

    fig.tight_layout()
    fig.savefig(out_path, dpi=130, bbox_inches="tight")
    plt.close(fig)


def plot_accuracy_progress(arch_records: List[dict], all_stats: List[dict],
                            out_path: Path) -> None:
    """Val accuracy of all architectures over cycles (colored by NAS rank)."""
    if not all_stats:
        return
    cycles = [s["cycle"] for s in all_stats]
    fig, ax = plt.subplots(figsize=(10, 6))
    sorted_by_nas = sorted(arch_records, key=lambda r: r["nas_quant_acc1"])
    for rank, rec in enumerate(sorted_by_nas):
        ai = rec["arch_index"]
        y = []
        for cs in all_stats:
            pa = next((p for p in cs["per_arch"] if p["arch_index"] == ai), None)
            if pa:
                y.append(pa["val_acc1"])
            else:
                y.append(float("nan"))
        ax.plot(cycles[:len(y)], y, lw=1.8, label=f"A{ai} NAS={rec['nas_quant_acc1']:.1f}%",
                color=_arch_color(rank))
    ax.set_xlabel("Cycle (= epochs per architecture)", fontsize=11)
    ax.set_ylabel("Full ImageNet Val Top-1 Accuracy (%)", fontsize=11)
    ax.set_title("All Architectures: Full-Dataset Val Accuracy over Cycles", fontsize=12)
    ax.legend(fontsize=7, ncol=3, loc="lower right")
    ax.yaxis.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_path, dpi=130, bbox_inches="tight")
    plt.close(fig)


# ─────────────────────────────────────────────────────────────────────────────
# Logging
# ─────────────────────────────────────────────────────────────────────────────

def setup_logger(log_path: Path) -> logging.Logger:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    logger = logging.getLogger("NAS_full_train")
    logger.setLevel(logging.DEBUG)
    logger.handlers = []
    fmt = logging.Formatter("%(asctime)s | %(levelname)-8s | %(message)s",
                             datefmt="%Y-%m-%d %H:%M:%S")
    fh = logging.FileHandler(log_path, encoding="utf-8")
    fh.setLevel(logging.DEBUG)
    fh.setFormatter(fmt)
    sh = logging.StreamHandler(sys.stdout)
    sh.setLevel(logging.INFO)
    sh.setFormatter(fmt)
    logger.addHandler(fh)
    logger.addHandler(sh)
    return logger


# ─────────────────────────────────────────────────────────────────────────────
# Argument parsing
# ─────────────────────────────────────────────────────────────────────────────

def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(
        description="Round-robin full-dataset training + NAS correlation analysis",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    ap.add_argument("--architectures-json", type=str, default="selected_architectures.json",
                    help="JSON produced by select_architectures.py")
    ap.add_argument("--dataset-path", type=str, default="/mnt/matylda5/xmihol00/datasets/imagenet/train",
                    help="Path to ImageNet training directory (ImageFolder layout)")
    ap.add_argument("--checkpoint", type=str, default="/mnt/matylda5/xmihol00/EUD/supernet/runs_imx500_supernet/20260402_200233/best.pt",
                    help="Supernet checkpoint (.pt) to initialise subnet weights")
    ap.add_argument("--output-dir", default="./full_dataset_experiment")
    ap.add_argument("--num-classes", type=int, default=1000)
    ap.add_argument("--val-frac", type=float, default=0.2)
    ap.add_argument("--batch-size", type=int, default=512)
    ap.add_argument("--num-workers", type=int, default=6)
    ap.add_argument("--seed", type=int, default=1)
    # Optimiser
    ap.add_argument("--lr",            type=float, default=0.02)
    ap.add_argument("--min-lr",        type=float, default=1e-5)
    ap.add_argument("--weight-decay",  type=float, default=5e-5)
    ap.add_argument("--momentum",      type=float, default=0.9)
    ap.add_argument("--label-smoothing", type=float, default=0.1)
    ap.add_argument("--grad-clip",     type=float, default=1.0)
    ap.add_argument("--warmup-epochs", type=int,   default=2)
    ap.add_argument("--lr-restart-period", type=int, default=30,
                    help="Cosine restart period (epochs); LR resets every T_0 epochs after warmup")
    ap.add_argument("--freeze-backbone-epochs", type=int, default=1)
    ap.add_argument("--ema-decay",     type=float, default=0.9998)
    # Augmentation
    ap.add_argument("--mixup-alpha",   type=float, default=0.2)
    ap.add_argument("--mixup-prob",    type=float, default=0.5)
    ap.add_argument("--cutmix-alpha",  type=float, default=1.0)
    ap.add_argument("--cutmix-prob",   type=float, default=0.2)
    ap.add_argument("--randaugment-magnitude", type=int, default=7)
    # AMP
    ap.add_argument("--amp",    action="store_true", default=True)
    ap.add_argument("--no-amp", dest="amp", action="store_false")
    # Misc
    ap.add_argument("--log-interval", type=int, default=10,
                    help="Log to console every N training batches")
    ap.add_argument("--device", type=str, default="cuda")
    ap.add_argument("--bootstrap-samples", type=int, default=5000)
    return ap.parse_args()


# ─────────────────────────────────────────────────────────────────────────────
# Main experiment
# ─────────────────────────────────────────────────────────────────────────────

# Global flag for graceful shutdown
_STOP_REQUESTED = False

def _signal_handler(sig, frame):
    global _STOP_REQUESTED
    print("\n[INFO] Shutdown requested — will stop after current epoch completes.")
    _STOP_REQUESTED = True

signal.signal(signal.SIGINT,  _signal_handler)
signal.signal(signal.SIGTERM, _signal_handler)


def main() -> None:
    global _STOP_REQUESTED
    args = parse_args()

    # ── output structure ──────────────────────────────────────────────────────
    out_root    = Path(args.output_dir)
    arch_dir    = out_root / "architectures"
    cycles_dir  = out_root / "cycles"
    out_root.mkdir(parents=True, exist_ok=True)
    arch_dir.mkdir(parents=True, exist_ok=True)
    cycles_dir.mkdir(parents=True, exist_ok=True)

    logger = setup_logger(out_root / "training.log")
    logger.info("=" * 70)
    logger.info("Full-dataset NAS correlation experiment")
    logger.info("Output: %s", out_root)
    logger.info("=" * 70)

    # ── seeding ───────────────────────────────────────────────────────────────
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)

    # ── GPU ───────────────────────────────────────────────────────────────────
    if safe_gpu is not None:
        while True:
            try:
                safe_gpu.claim_gpus(1)
                break
            except Exception:
                logger.info("Waiting for free GPU…")
                time.sleep(10)

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    if device.type != "cuda":
        logger.warning("CUDA not available — running on CPU. Expect very slow training.")
    logger.info("Device: %s", device)

    # ── load architectures ────────────────────────────────────────────────────
    with open(args.architectures_json) as f:
        arch_list: List[dict] = json.load(f)
    logger.info("Loaded %d architectures from %s", len(arch_list), args.architectures_json)
    # Copy the JSON into output dir for reference
    shutil.copy2(args.architectures_json, out_root / "selected_architectures.json")

    # Save config
    cfg_path = out_root / "experiment_config.json"
    if not cfg_path.exists():
        with cfg_path.open("w") as f:
            json.dump(vars(args), f, indent=2, default=str)

    # ── dataset split (cached) ───────────────────────────────────────────────
    split_cache = out_root / "dataset_split.json"
    logger.info("Creating/loading dataset split (val_frac=%.2f, seed=%d)…", args.val_frac, args.seed)
    train_idx, val_idx = create_split_indices(
        Path(args.dataset_path), args.val_frac, args.seed, split_cache)
    logger.info("Train samples: %d | Val samples: %d", len(train_idx), len(val_idx))

    # ── SSD dataset cache ─────────────────────────────────────────────────────
    effective_dataset_path = try_cache_dataset_on_ssd(Path(args.dataset_path), logger)
    if effective_dataset_path != Path(args.dataset_path):
        logger.info("Training will read data from SSD: %s", effective_dataset_path)

    # ── DataLoader cache (per resolution) ────────────────────────────────────
    loader_cache: Dict[int, Tuple[DataLoader, DataLoader]] = {}

    def get_loaders(resolution: int) -> Tuple[DataLoader, DataLoader]:
        if resolution not in loader_cache:
            logger.info("Building DataLoaders for resolution=%d px…", resolution)
            loader_cache[resolution] = build_loaders(
                effective_dataset_path, train_idx, val_idx,
                resolution, args.batch_size, args.num_workers,
                args.randaugment_magnitude)
        return loader_cache[resolution]

    # ── load supernet ─────────────────────────────────────────────────────────
    logger.info("Loading supernet checkpoint: %s", args.checkpoint)
    supernet = create_default_supernet(num_classes=args.num_classes)
    ckpt_payload = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    sd = ckpt_payload.get("model") or ckpt_payload.get("state_dict") or ckpt_payload
    missing, unexpected = supernet.load_state_dict(sd, strict=False)
    if missing:
        logger.warning("Supernet missing keys: %d", len(missing))
    if unexpected:
        logger.warning("Supernet unexpected keys: %d", len(unexpected))

    # ── initialise per-architecture state ────────────────────────────────────
    models:     Dict[int, StaticSubnetModel]        = {}
    optimizers: Dict[int, torch.optim.Optimizer]    = {}
    scalers:    Dict[int, torch.amp.GradScaler] = {}
    emas:       Dict[int, ModelEMA]                  = {}
    histories:  Dict[int, List[dict]]                = {}
    best_acc1s: Dict[int, float]                     = {}
    # Track epochs completed per arch (for LR schedule and resume)
    epochs_done: Dict[int, int] = {}

    criterion = nn.CrossEntropyLoss(label_smoothing=args.label_smoothing)

    for arch_rec in arch_list:
        ai = arch_rec["arch_index"]
        config = SubnetConfig.from_dict(arch_rec["config"])
        adir = arch_dir / f"arch_{ai:02d}"
        adir.mkdir(parents=True, exist_ok=True)

        # Save arch metadata once
        meta_path = adir / "arch_config.json"
        if not meta_path.exists():
            with meta_path.open("w") as f:
                json.dump(arch_rec, f, indent=2)

        # Build model and transfer supernet weights
        model, transfer_err = build_static_subnet_model(supernet, config, args.num_classes, device)
        if transfer_err is None:
            logger.info("Arch %02d: supernet weights transferred successfully", ai)
        else:
            logger.warning(
                "Arch %02d: supernet weight transfer incomplete — %s — affected layers use random init",
                ai, transfer_err,
            )
        opt = torch.optim.SGD(model.parameters(), lr=args.lr,
                              momentum=args.momentum, weight_decay=args.weight_decay,
                              nesterov=True)
        scaler = torch.amp.GradScaler("cuda", enabled=args.amp and device.type == "cuda")
        ema = ModelEMA(model, decay=args.ema_decay)

        # Load checkpoint if exists
        last_ckpt = adir / "last.pt"
        if last_ckpt.exists():
            epoch, best = load_arch_checkpoint(last_ckpt, model, opt, scaler, ema, device)
            epochs_done[ai] = epoch + 1   # epoch N is complete
            best_acc1s[ai] = best
            logger.info("Arch %02d: resumed from epoch %d (best_acc1=%.2f%%)", ai, epoch, best)
        else:
            epochs_done[ai] = 0
            best_acc1s[ai]  = -1.0
            logger.info("Arch %02d: fresh start (config=%s, nas_acc=%.2f%%)",
                        ai, arch_rec["config"], arch_rec["nas_quant_acc1"])

        # Load training history if exists
        hist_path = adir / "metrics.json"
        if hist_path.exists():
            with hist_path.open() as f:
                histories[ai] = json.load(f)
        else:
            histories[ai] = []

        models[ai]     = model
        optimizers[ai] = opt
        scalers[ai]    = scaler
        emas[ai]       = ema

    del supernet  # free memory

    # ── load per-cycle stats history ─────────────────────────────────────────
    stats_history_path = out_root / "stats_history.json"
    if stats_history_path.exists():
        with stats_history_path.open() as f:
            all_cycle_stats: List[dict] = json.load(f)
    else:
        all_cycle_stats = []

    # Determine starting cycle number
    current_cycle = max((s["cycle"] for s in all_cycle_stats), default=-1) + 1
    logger.info("Starting from cycle %d", current_cycle)

    # ── CSV header for cycle summary ─────────────────────────────────────────
    cycle_csv_path = out_root / "cycle_summary.csv"
    csv_fields = ["cycle", "timestamp", "spearman_r", "spearman_p", "pearson_r",
                  "pearson_p", "kendall_tau", "kendall_p",
                  "ci_lo", "ci_hi", "mean_val_acc1", "std_val_acc1"]
    if not cycle_csv_path.exists():
        with cycle_csv_path.open("w", newline="") as fh:
            csv.DictWriter(fh, fieldnames=csv_fields).writeheader()

    # ── main loop ─────────────────────────────────────────────────────────────
    logger.info("Entering main training loop (Ctrl-C to stop gracefully).")
    while not _STOP_REQUESTED:
        logger.info("━" * 70)
        logger.info("CYCLE %d  (%s)", current_cycle, datetime.now().strftime("%Y-%m-%d %H:%M:%S"))
        logger.info("━" * 70)

        cycle_val_results: Dict[int, Tuple[float, float]] = {}  # ai → (val_acc1, ema_acc1)

        for arch_rec in arch_list:
            if _STOP_REQUESTED:
                break

            ai     = arch_rec["arch_index"]
            epoch  = epochs_done[ai]
            config = SubnetConfig.from_dict(arch_rec["config"])
            adir   = arch_dir / f"arch_{ai:02d}"

            logger.info("─── Arch %02d | Epoch %d | NAS=%.2f%% | config=%s",
                        ai, epoch, arch_rec["nas_quant_acc1"], arch_rec["config"])

            train_loader, val_loader = get_loaders(config.resolution)

            # ── train one epoch ───────────────────────────────────────────────
            model, opt, scaler, ema = models[ai], optimizers[ai], scalers[ai], emas[ai]
            t_stats = train_one_epoch(
                ai, epoch, model, train_loader, opt, criterion, scaler, ema, device, args, logger)

            # ── evaluate ──────────────────────────────────────────────────────
            logger.info("  Evaluating model and EMA on val set…")
            v_stats   = evaluate(model, val_loader, criterion, device)
            ema_stats = evaluate(ema.ema, val_loader, criterion, device)
            logger.info("  val: loss=%.4f acc1=%.3f%% acc5=%.3f%%",
                        v_stats["loss"], v_stats["acc1"], v_stats["acc5"])
            logger.info("  ema: loss=%.4f acc1=%.3f%% acc5=%.3f%%",
                        ema_stats["loss"], ema_stats["acc1"], ema_stats["acc5"])

            # Use EMA accuracy as selection criterion (better generalisation)
            sel_acc1 = ema_stats["acc1"]
            if sel_acc1 > best_acc1s[ai]:
                best_acc1s[ai] = sel_acc1
                save_arch_checkpoint(adir / "best.pt", epoch, model, opt, scaler, ema,
                                     sel_acc1, arch_rec)
                logger.info("  ★ New best: %.3f%%", sel_acc1)

            # ── history ───────────────────────────────────────────────────────
            row = {
                "epoch": epoch,
                "train_loss": t_stats["loss"], "train_acc1": t_stats["acc1"],
                "train_acc5": t_stats["acc5"], "lr": t_stats["lr"],
                "val_loss": v_stats["loss"],   "val_acc1":  v_stats["acc1"],
                "val_acc5": v_stats["acc5"],
                "ema_loss": ema_stats["loss"], "ema_acc1":  ema_stats["acc1"],
                "ema_acc5": ema_stats["acc5"],
            }
            histories[ai].append(row)
            with (adir / "metrics.json").open("w") as f:
                json.dump(histories[ai], f, indent=2)

            # ── CSV append ────────────────────────────────────────────────────
            arch_csv = adir / "metrics.csv"
            write_header = not arch_csv.exists()
            with arch_csv.open("a", newline="") as fh:
                w = csv.DictWriter(fh, fieldnames=list(row.keys()))
                if write_header:
                    w.writeheader()
                w.writerow(row)

            # ── checkpoint ───────────────────────────────────────────────────
            save_arch_checkpoint(adir / "last.pt", epoch, model, opt, scaler, ema,
                                  best_acc1s[ai], arch_rec)
            # Also save a per-epoch checkpoint to never lose progress
            epoch_ckpt = adir / f"epoch_{epoch:04d}.pt"
            # Only keep every 5th epoch checkpoint to save disk space
            if epoch % 5 == 0:
                save_arch_checkpoint(epoch_ckpt, epoch, model, opt, scaler, ema,
                                     best_acc1s[ai], arch_rec)

            epochs_done[ai] = epoch + 1
            cycle_val_results[ai] = (v_stats["acc1"], ema_stats["acc1"])

        if _STOP_REQUESTED:
            logger.info("Stop requested — skipping cycle stats and exiting.")
            break

        # ── per-cycle statistics ───────────────────────────────────────────────
        logger.info("Computing cycle %d statistics…", current_cycle)
        stat_arch_records = []
        for arch_rec in arch_list:
            ai = arch_rec["arch_index"]
            v1, e1 = cycle_val_results.get(ai, (-1, -1))
            stat_arch_records.append({
                "arch_index": ai,
                "nas_quant_acc1": arch_rec["nas_quant_acc1"],
                "current_val_acc1": v1,
                "current_ema_acc1": e1,
                "epochs_completed": epochs_done[ai],
            })

        cs = compute_cycle_stats(current_cycle, stat_arch_records, args.bootstrap_samples)
        logger.info(
            "  Spearman ρ = %.4f (p=%.4f%s) | Pearson r = %.4f (p=%.4f) | Kendall τ = %.4f (p=%.4f)",
            cs.spearman_r, cs.spearman_p, " ✓" if cs.spearman_p < 0.05 else "",
            cs.pearson_r,  cs.pearson_p,
            cs.kendall_tau, cs.kendall_p,
        )
        logger.info(
            "  Bootstrap 95%% CI [%.4f, %.4f] | Mean val_acc1 = %.3f±%.3f%%",
            cs.bootstrap_ci_spearman[0], cs.bootstrap_ci_spearman[1],
            cs.mean_val_acc1, cs.std_val_acc1,
        )

        # ── save cycle stats ──────────────────────────────────────────────────
        cdir = cycles_dir / f"cycle_{current_cycle:04d}"
        cdir.mkdir(parents=True, exist_ok=True)

        cs_dict = asdict(cs)
        cs_dict["timestamp"] = datetime.utcnow().isoformat()
        with (cdir / "stats.json").open("w") as f:
            json.dump(cs_dict, f, indent=2)

        # Append to rolling history
        all_cycle_stats.append(cs_dict)
        with stats_history_path.open("w") as f:
            json.dump(all_cycle_stats, f, indent=2)

        # Append to CSV
        with cycle_csv_path.open("a", newline="") as fh:
            w2 = csv.DictWriter(fh, fieldnames=csv_fields)
            w2.writerow({
                "cycle": current_cycle,
                "timestamp": cs_dict["timestamp"],
                "spearman_r": cs.spearman_r, "spearman_p": cs.spearman_p,
                "pearson_r":  cs.pearson_r,  "pearson_p":  cs.pearson_p,
                "kendall_tau": cs.kendall_tau, "kendall_p": cs.kendall_p,
                "ci_lo": cs.bootstrap_ci_spearman[0],
                "ci_hi": cs.bootstrap_ci_spearman[1],
                "mean_val_acc1": cs.mean_val_acc1,
                "std_val_acc1":  cs.std_val_acc1,
            })

        # ── plots ─────────────────────────────────────────────────────────────
        logger.info("Generating cycle %d plots…", current_cycle)
        try:
            plot_correlation_scatter(cs, stat_arch_records,
                                     cdir / "correlation_scatter.png")
            plot_rank_comparison(cs, cdir / "rank_comparison.png")
            plot_training_curves(stat_arch_records, histories,
                                 cdir / "training_curves.png")
            # Also update rolling plots at root level
            plot_correlation_over_cycles(all_cycle_stats,
                                         out_root / "correlation_over_cycles.png")
            plot_accuracy_progress(stat_arch_records, all_cycle_stats,
                                   out_root / "accuracy_progress.png")
        except Exception as e:
            logger.warning("Plot generation failed: %s", e)

        logger.info("Cycle %d complete. Output saved to %s", current_cycle, cdir)
        current_cycle += 1

    logger.info("Experiment stopped at cycle %d. All checkpoints saved.", current_cycle)
    logger.info("Resume by re-running with the same --output-dir argument.")


if __name__ == "__main__":
    main()
