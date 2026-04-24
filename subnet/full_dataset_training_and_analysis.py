#!/usr/bin/env python3
"""
Full-dataset correlation experiment.

Trains NAS-selected subnet architectures (initially from the supernet checkpoint)
on the full ImageNet training set in a round-robin fashion — 1 epoch per architecture
per cycle — and tests whether the 6-class NAS quantised accuracy (nas_quant_acc1)
correlates with the 1000-class floating-point validation accuracy.

Usage:
    python full_dataset_training_and_analysis.py \\
        --architectures-json selected_architectures.json \\
        --dataset-path /path/to/imagenet/train \\
        --checkpoint /path/to/supernet/best.pt \\
        --output-dir ./full_dataset_experiment

Resume: re-run the exact same command.  Progress is recovered from per-arch
checkpoints, cycle_progress.json, and metrics inside output-dir.

Stop: Ctrl-C (or SIGTERM).  All state is already saved — the next run picks up
where this one left off (including mid-cycle position).
"""

from __future__ import annotations

import argparse
import copy
import csv
import gc
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
from typing import Any, Dict, List, Optional, Sequence, Set, Tuple, cast
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
        total = 0
        for p in path.rglob("*"):
            if p.is_file():
                try:
                    total += p.stat().st_size
                except OSError:
                    pass
        return total


def try_cache_dataset_on_ssd(dataset_path: Path, cache_dataset: bool, logger: logging.Logger) -> Path:
    """Copy the dataset to SSD if there is enough free space.

    Returns the path that should be used for training — either the SSD copy
    (fast local NVMe) or the original path if copying is not possible.
    If the destination already exists it is reused without re-copying.
    """
    if not cache_dataset:
        logger.info("SSD caching disabled by config — using original dataset path.")
        return dataset_path
    ssd_dest = _SSD_CACHE_DIR / dataset_path.name

    if ssd_dest.exists() and any(ssd_dest.iterdir()):
        logger.info("SSD cache found at %s — skipping copy.", ssd_dest)
        return ssd_dest

    logger.info("Measuring dataset size at %s …", dataset_path)
    t0 = time.time()
    dataset_bytes = _get_dir_size(dataset_path)
    logger.info("Dataset size: %.2f GB (measured in %.1f s)", dataset_bytes / 1e9, time.time() - t0)

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
    tmp = path.with_suffix(".tmp")
    torch.save({
        "epoch": epoch,
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "scaler": scaler.state_dict(),
        "ema": ema.ema.state_dict(),
        "best_acc1": best_acc1,
        "arch_record": arch_record,
    }, tmp)
    tmp.replace(path)


def load_arch_checkpoint(path: Path, model: nn.Module,
                          optimizer: torch.optim.Optimizer,
                          scaler: torch.amp.GradScaler,
                          ema: ModelEMA, device: torch.device,
                          ) -> Tuple[int, float]:
    """Returns (epoch_completed, best_acc1)."""
    ckpt = torch.load(path, map_location=device, weights_only=False)
    model.load_state_dict(ckpt["model"])
    optimizer.load_state_dict(ckpt["optimizer"])
    scaler.load_state_dict(ckpt["scaler"])
    ema.ema.load_state_dict(ckpt["ema"])
    return int(ckpt["epoch"]), float(ckpt.get("best_acc1", -1.0))


def peek_checkpoint_epoch(path: Path) -> Tuple[int, float]:
    """Read epoch and best_acc1 from checkpoint without loading model weights."""
    ckpt = torch.load(path, map_location="cpu", weights_only=False)
    return int(ckpt["epoch"]), float(ckpt.get("best_acc1", -1.0))


# ─────────────────────────────────────────────────────────────────────────────
# I/O utilities
# ─────────────────────────────────────────────────────────────────────────────

def _atomic_json_write(path: Path, data: Any) -> None:
    """Write JSON atomically via a temp file to avoid partial writes on crash."""
    tmp = path.with_suffix(".tmp")
    with tmp.open("w") as f:
        json.dump(data, f, indent=2)
    tmp.replace(path)


def _rebuild_metrics_csv(adir: Path, hist: List[dict]) -> None:
    """Rebuild metrics.csv from the canonical history list."""
    if not hist:
        return
    fields = list(hist[0].keys())
    with (adir / "metrics.csv").open("w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=fields)
        w.writeheader()
        w.writerows(hist)


def _reconcile_arch_state(adir: Path, ai: int, logger: logging.Logger,
                           ) -> Tuple[int, float, List[dict]]:
    """Return (epochs_done, best_acc1, history) reconciled between checkpoint and metrics.json.

    epochs_done is the number of epochs fully completed (= next epoch index to run).
    The checkpoint is authoritative for epochs_done; metrics.json is truncated or
    left as-is to match.  The metrics.csv is rebuilt from metrics.json when truncated.
    """
    last_ckpt = adir / "last.pt"
    if last_ckpt.exists():
        epoch_completed, best_acc1 = peek_checkpoint_epoch(last_ckpt)
        epochs_done = epoch_completed + 1
    else:
        epochs_done = 0
        best_acc1 = -1.0

    hist_path = adir / "metrics.json"
    if hist_path.exists():
        try:
            with hist_path.open() as f:
                hist = json.load(f)
        except Exception as exc:
            logger.warning("Arch %02d: metrics.json unreadable (%s) — resetting history.", ai, exc)
            hist = []

        if len(hist) > epochs_done:
            logger.warning(
                "Arch %02d: metrics.json has %d entries but checkpoint says %d epochs completed "
                "— truncating to match checkpoint (likely a partial write before last crash).",
                ai, len(hist), epochs_done,
            )
            hist = hist[:epochs_done]
            _atomic_json_write(hist_path, hist)
            _rebuild_metrics_csv(adir, hist)
        elif len(hist) < epochs_done:
            logger.warning(
                "Arch %02d: metrics.json has only %d entries but checkpoint says %d epochs — "
                "history is incomplete; will proceed from checkpoint.",
                ai, len(hist), epochs_done,
            )
    else:
        hist = []
        if epochs_done > 0:
            logger.warning(
                "Arch %02d: checkpoint says %d epochs done but no metrics.json found — "
                "history will be empty.",
                ai, epochs_done,
            )

    return epochs_done, best_acc1, hist


# ─────────────────────────────────────────────────────────────────────────────
# Cycle progress  (enables mid-cycle resume after crash)
# ─────────────────────────────────────────────────────────────────────────────

def load_cycle_progress(cdir: Path) -> Tuple[Set[int], Dict[int, Tuple[float, float, float, float]]]:
    """Load in-progress cycle state.

    Returns (completed_arch_set, cycle_val_results) where cycle_val_results maps
    arch_index → (val_acc1, ema_acc1, best_val_acc1, best_ema_acc1).
    """
    progress_path = cdir / "cycle_progress.json"
    if not progress_path.exists():
        return set(), {}
    try:
        with progress_path.open() as f:
            d = json.load(f)
        completed: Set[int] = set(d.get("completed_arch_indices", []))
        raw = d.get("cycle_val_results", {})
        results: Dict[int, Tuple[float, float, float, float]] = {
            int(k): tuple(v) for k, v in raw.items()  # type: ignore[assignment]
        }
        return completed, results
    except Exception:
        return set(), {}


def save_cycle_progress(cdir: Path, cycle: int,
                         completed: Set[int],
                         results: Dict[int, Tuple[float, float, float, float]]) -> None:
    cdir.mkdir(parents=True, exist_ok=True)
    data = {
        "cycle": cycle,
        "completed_arch_indices": sorted(completed),
        "cycle_val_results": {str(k): list(v) for k, v in results.items()},
    }
    _atomic_json_write(cdir / "cycle_progress.json", data)


# ─────────────────────────────────────────────────────────────────────────────
# GPU memory cleanup
# ─────────────────────────────────────────────────────────────────────────────

def _free_gpu(*tensors_or_modules) -> None:
    """Delete objects and release GPU memory."""
    for obj in tensors_or_modules:
        del obj
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.synchronize()


# ─────────────────────────────────────────────────────────────────────────────
# Statistical analysis
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class CycleStats:
    cycle: int
    n_archs: int
    # Correlations on current-epoch val accuracy
    spearman_r: float
    spearman_p: float
    pearson_r: float
    pearson_p: float
    kendall_tau: float
    kendall_p: float
    bootstrap_ci_spearman: tuple
    # Correlations on EMA accuracy
    spearman_r_ema: float
    spearman_p_ema: float
    pearson_r_ema: float
    pearson_p_ema: float
    kendall_tau_ema: float
    kendall_p_ema: float
    bootstrap_ci_spearman_ema: tuple
    # Correlations on best-so-far val accuracy
    spearman_r_best: float
    spearman_p_best: float
    pearson_r_best: float
    pearson_p_best: float
    kendall_tau_best: float
    kendall_p_best: float
    bootstrap_ci_spearman_best: tuple
    # Summary statistics
    mean_val_acc1: float
    std_val_acc1: float
    mean_ema_acc1: float
    mean_best_val_acc1: float
    per_arch: List[Dict[str, Any]]


def _bootstrap_spearman_ci(x: np.ndarray, y: np.ndarray,
                            n_boot: int = 5000, rng_seed: int = 42
                            ) -> Tuple[float, float]:
    rng = np.random.default_rng(rng_seed)
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


def _corr_triple(x: np.ndarray, y: np.ndarray, n_boot: int, seed: int
                 ) -> Tuple[float, float, float, float, float, float, Tuple[float, float]]:
    """Return (sr, sp, pr, pp, kt, kp, bootstrap_ci_spearman)."""
    sr, sp = scipy_stats.spearmanr(x, y)
    pr, pp = scipy_stats.pearsonr(x, y)
    kt, kp = scipy_stats.kendalltau(x, y)
    ci = _bootstrap_spearman_ci(x, y, n_boot, seed)
    return float(sr), float(sp), float(pr), float(pp), float(kt), float(kp), ci


def compute_cycle_stats(cycle: int, arch_records: List[dict],
                        n_boot: int = 5000, rng_seed: int = 42) -> CycleStats:
    nas_accs  = np.array([r["nas_quant_acc1"]   for r in arch_records])
    val_accs  = np.array([r["current_val_acc1"] for r in arch_records])
    ema_accs  = np.array([r["current_ema_acc1"] for r in arch_records])
    best_accs = np.array([r["best_val_acc1"]    for r in arch_records])

    sr, sp, pr, pp, kt, kp, ci_v       = _corr_triple(nas_accs, val_accs,  n_boot, rng_seed)
    sr_e, sp_e, pr_e, pp_e, kt_e, kp_e, ci_e = _corr_triple(nas_accs, ema_accs,  n_boot, rng_seed + 1)
    sr_b, sp_b, pr_b, pp_b, kt_b, kp_b, ci_b = _corr_triple(nas_accs, best_accs, n_boot, rng_seed + 2)

    per_arch = []
    for i, r in enumerate(arch_records):
        per_arch.append({
            "arch_index":      r["arch_index"],
            "nas_quant_acc1":  r["nas_quant_acc1"],
            "val_acc1":        r["current_val_acc1"],
            "ema_acc1":        r["current_ema_acc1"],
            "best_val_acc1":   r["best_val_acc1"],
            "best_ema_acc1":   r["best_ema_acc1"],
            "epochs_completed": r["epochs_completed"],
            "nas_rank":  int(np.argsort(np.argsort(-nas_accs))[i]),
            "val_rank":  int(np.argsort(np.argsort(-val_accs))[i]),
            "ema_rank":  int(np.argsort(np.argsort(-ema_accs))[i]),
            "best_rank": int(np.argsort(np.argsort(-best_accs))[i]),
        })

    return CycleStats(
        cycle=cycle, n_archs=len(arch_records),
        spearman_r=sr, spearman_p=sp, pearson_r=pr, pearson_p=pp,
        kendall_tau=kt, kendall_p=kp, bootstrap_ci_spearman=ci_v,
        spearman_r_ema=sr_e, spearman_p_ema=sp_e,
        pearson_r_ema=pr_e, pearson_p_ema=pp_e,
        kendall_tau_ema=kt_e, kendall_p_ema=kp_e,
        bootstrap_ci_spearman_ema=ci_e,
        spearman_r_best=sr_b, spearman_p_best=sp_b,
        pearson_r_best=pr_b, pearson_p_best=pp_b,
        kendall_tau_best=kt_b, kendall_p_best=kp_b,
        bootstrap_ci_spearman_best=ci_b,
        mean_val_acc1=float(np.mean(val_accs)),
        std_val_acc1=float(np.std(val_accs, ddof=1) if len(val_accs) > 1 else 0.0),
        mean_ema_acc1=float(np.mean(ema_accs)),
        mean_best_val_acc1=float(np.mean(best_accs)),
        per_arch=per_arch,
    )


# ─────────────────────────────────────────────────────────────────────────────
# Plotting
# ─────────────────────────────────────────────────────────────────────────────

_CMAP = matplotlib.colormaps.get_cmap("tab20").resampled(20)


def _arch_color(i: int):
    return _CMAP(i % 20)


def _sig_label(p: float) -> str:
    if p < 0.001:
        return "p<0.001 ✓✓✓"
    if p < 0.01:
        return f"p={p:.4f} ✓✓"
    if p < 0.05:
        return f"p={p:.4f} ✓"
    return f"p={p:.4f} ✗"


def plot_training_curves(arch_records: List[dict], histories: Dict[int, List[dict]],
                          out_path: Path) -> None:
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

    for j in range(n, nrows * ncols):
        axes[j // ncols][j % ncols].set_visible(False)

    fig.suptitle("Training Curves — All Architectures", fontsize=12)
    fig.tight_layout()
    fig.savefig(out_path, dpi=130, bbox_inches="tight")
    plt.close(fig)


def _scatter_with_regression(ax, nas: List[float], acc: List[float],
                              idxs: List[int], label_key: str,
                              marker: str = "o", size: int = 80) -> None:
    """Plot scatter with per-arch colors and an OLS regression line."""
    for i, (x, y, ai) in enumerate(zip(nas, acc, idxs)):
        ax.scatter(x, y, color=_arch_color(i), s=size, zorder=3,
                   edgecolors="white", lw=0.6, marker=marker)
        ax.annotate(f"A{ai}", (x, y), textcoords="offset points", xytext=(5, 3), fontsize=7)
    if len(nas) >= 2:
        m, b, *_ = scipy_stats.linregress(nas, acc)
        xs = np.linspace(min(nas), max(nas), 100)
        ax.plot(xs, m * xs + b, "k--", lw=1.2, alpha=0.6)


def plot_correlation_scatter(stats: CycleStats, arch_records: List[dict],
                              out_path: Path) -> None:
    """Three-panel scatter: val / EMA / best-so-far acc vs NAS acc."""
    nas  = [p["nas_quant_acc1"] for p in stats.per_arch]
    val  = [p["val_acc1"]       for p in stats.per_arch]
    ema  = [p["ema_acc1"]       for p in stats.per_arch]
    best = [p["best_val_acc1"]  for p in stats.per_arch]
    idxs = [p["arch_index"]     for p in stats.per_arch]

    fig, axes = plt.subplots(1, 3, figsize=(18, 6))

    for ax, acc, title, sr, sp, pr, ci in [
        (axes[0], val,  "Val (current epoch)",
         stats.spearman_r, stats.spearman_p, stats.pearson_r, stats.bootstrap_ci_spearman),
        (axes[1], ema,  "EMA val (current epoch)",
         stats.spearman_r_ema, stats.spearman_p_ema, stats.pearson_r_ema, stats.bootstrap_ci_spearman_ema),
        (axes[2], best, "Best val (all epochs so far)",
         stats.spearman_r_best, stats.spearman_p_best, stats.pearson_r_best, stats.bootstrap_ci_spearman_best),
    ]:
        _scatter_with_regression(ax, nas, acc, idxs, title)
        ci_lo, ci_hi = ci
        ax.set_xlabel("NAS Quantised Accuracy — 6-class subset (%)", fontsize=10)
        ax.set_ylabel("Full ImageNet Val Top-1 Acc (%)", fontsize=10)
        ax.set_title(
            f"{title}\nSpearman ρ={sr:.3f} ({_sig_label(sp)}) | Pearson r={pr:.3f}\n"
            f"Bootstrap 95% CI: [{ci_lo:.3f}, {ci_hi:.3f}]",
            fontsize=9,
        )
        ax.xaxis.grid(True, alpha=0.3)
        ax.yaxis.grid(True, alpha=0.3)
        ax.set_axisbelow(True)

    fig.suptitle(f"Cycle {stats.cycle} — NAS accuracy vs Full-dataset accuracy (N={stats.n_archs})",
                 fontsize=13)
    fig.tight_layout()
    fig.savefig(out_path, dpi=130, bbox_inches="tight")
    plt.close(fig)


def plot_rank_comparison(stats: CycleStats, out_path: Path) -> None:
    """Side-by-side bar: NAS rank vs full-dataset rank per architecture."""
    n = len(stats.per_arch)
    sorted_by_nas = sorted(stats.per_arch, key=lambda p: p["nas_quant_acc1"])
    ai_labels = [f"A{p['arch_index']}" for p in sorted_by_nas]
    nas_ranks  = [p["nas_rank"]  for p in sorted_by_nas]
    val_ranks  = [p["val_rank"]  for p in sorted_by_nas]
    best_ranks = [p["best_rank"] for p in sorted_by_nas]

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    x = np.arange(n)
    w = 0.28
    axes[0].bar(x - w, nas_ranks,  w, label="NAS rank",        color="#0072B2", alpha=0.85)
    axes[0].bar(x,     val_ranks,  w, label="Val rank",         color="#D55E00", alpha=0.85)
    axes[0].bar(x + w, best_ranks, w, label="Best val rank",    color="#009E73", alpha=0.85)
    axes[0].set_xticks(x); axes[0].set_xticklabels(ai_labels, fontsize=8)
    axes[0].set_ylabel("Rank (0=best)", fontsize=11)
    axes[0].set_title("NAS rank vs. full-dataset ranks (sorted by NAS acc)", fontsize=11)
    axes[0].legend(fontsize=9)
    axes[0].yaxis.grid(True, alpha=0.3)

    diff = [v - na for v, na in zip(val_ranks, nas_ranks)]
    colors = ["#009E73" if d <= 0 else "#CC79A7" for d in diff]
    axes[1].bar(x, diff, color=colors, alpha=0.85, edgecolor="white")
    axes[1].axhline(0, color="black", lw=0.8)
    axes[1].set_xticks(x); axes[1].set_xticklabels(ai_labels, fontsize=8)
    axes[1].set_ylabel("Val rank − NAS rank  (negative = moved up)", fontsize=11)
    axes[1].set_title("Rank shift: NAS → full-dataset (current epoch)", fontsize=11)
    axes[1].yaxis.grid(True, alpha=0.3)

    fig.suptitle(
        f"Cycle {stats.cycle} | Spearman ρ={stats.spearman_r:.3f} ({_sig_label(stats.spearman_p)})",
        fontsize=12,
    )
    fig.tight_layout()
    fig.savefig(out_path, dpi=130, bbox_inches="tight")
    plt.close(fig)


def plot_correlation_over_cycles(all_stats: List[dict], out_path: Path) -> None:
    """Line plot of all correlation metrics (val, EMA, best) over cycles."""
    if len(all_stats) < 2:
        return
    cycles = [s["cycle"] for s in all_stats]

    fig, axes = plt.subplots(3, 1, figsize=(11, 12), sharex=True)

    for ax, sr_key, sp_key, pr_key, label_suffix, color in [
        (axes[0], "spearman_r",      "spearman_p",      "pearson_r",      "current val", "#0072B2"),
        (axes[1], "spearman_r_ema",  "spearman_p_ema",  "pearson_r_ema",  "EMA val",     "#D55E00"),
        (axes[2], "spearman_r_best", "spearman_p_best", "pearson_r_best", "best val",    "#009E73"),
    ]:
        sr = [s.get(sr_key, float("nan")) for s in all_stats]
        pr = [s.get(pr_key, float("nan")) for s in all_stats]
        sp = [s.get(sp_key, 1.0)          for s in all_stats]
        ax.plot(cycles, sr, "o-", color=color,   lw=2, label=f"Spearman ρ ({label_suffix})")
        ax.plot(cycles, pr, "s--", color=color,  lw=1.5, alpha=0.7, label=f"Pearson r ({label_suffix})")
        ax.axhline(0,    color="black", lw=0.8, linestyle="--")
        # Mark significant cycles
        for c, s_val, s_p in zip(cycles, sr, sp):
            if s_p < 0.05:
                ax.scatter(c, s_val, s=120, marker="*", color="gold", zorder=4, edgecolors="black", lw=0.5)
        ax.set_ylabel("Correlation", fontsize=10)
        ax.legend(fontsize=9)
        ax.yaxis.grid(True, alpha=0.3)
        ax.set_ylim(-1.05, 1.05)
        ax.set_title(f"Correlation on {label_suffix} accuracy", fontsize=11)

    axes[-1].set_xlabel("Cycle (= epochs per architecture)", fontsize=11)
    fig.suptitle("NAS ↔ Full-dataset Correlation over Training  (★ = p<0.05)", fontsize=13)
    fig.tight_layout()
    fig.savefig(out_path, dpi=130, bbox_inches="tight")
    plt.close(fig)


def plot_accuracy_progress(arch_records: List[dict], all_stats: List[dict],
                            out_path: Path) -> None:
    """Val and EMA accuracy of all architectures over cycles."""
    if not all_stats:
        return
    cycles = [s["cycle"] for s in all_stats]
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))

    sorted_by_nas = sorted(arch_records, key=lambda r: r["nas_quant_acc1"])
    for rank, rec in enumerate(sorted_by_nas):
        ai = rec["arch_index"]
        val_y, ema_y = [], []
        for cs in all_stats:
            pa = next((p for p in cs["per_arch"] if p["arch_index"] == ai), None)
            val_y.append(pa["val_acc1"]  if pa else float("nan"))
            ema_y.append(pa["ema_acc1"]  if pa else float("nan"))
        lbl = f"A{ai} NAS={rec['nas_quant_acc1']:.1f}%"
        col = _arch_color(rank)
        axes[0].plot(cycles[:len(val_y)], val_y, lw=1.8, label=lbl, color=col)
        axes[1].plot(cycles[:len(ema_y)], ema_y, lw=1.8, label=lbl, color=col)

    for ax, title in [(axes[0], "Val (model)"), (axes[1], "Val (EMA)")]:
        ax.set_xlabel("Cycle", fontsize=11)
        ax.set_ylabel("Top-1 Accuracy (%)", fontsize=11)
        ax.set_title(f"All Architectures: Full-Dataset {title} Accuracy", fontsize=12)
        ax.legend(fontsize=7, ncol=3, loc="lower right")
        ax.yaxis.grid(True, alpha=0.3)

    fig.tight_layout()
    fig.savefig(out_path, dpi=130, bbox_inches="tight")
    plt.close(fig)


def plot_best_acc_summary(arch_records: List[dict], all_stats: List[dict],
                           out_path: Path) -> None:
    """Best val acc achieved per arch vs NAS acc (updated each cycle)."""
    if not all_stats:
        return
    last = all_stats[-1]
    nas  = [p["nas_quant_acc1"] for p in last["per_arch"]]
    best = [p["best_val_acc1"]  for p in last["per_arch"]]
    idxs = [p["arch_index"]     for p in last["per_arch"]]

    fig, ax = plt.subplots(figsize=(8, 6))
    _scatter_with_regression(ax, nas, best, idxs, "best val")
    sr = last.get("spearman_r_best", float("nan"))
    sp = last.get("spearman_p_best", 1.0)
    ax.set_xlabel("NAS Quantised Accuracy — 6-class subset (%)", fontsize=11)
    ax.set_ylabel("Best Full ImageNet Val Top-1 Acc achieved (%)", fontsize=11)
    ax.set_title(
        f"NAS acc vs. Best val acc after {last['cycle']+1} cycles\n"
        f"Spearman ρ={sr:.3f} ({_sig_label(sp)})",
        fontsize=11,
    )
    ax.xaxis.grid(True, alpha=0.3); ax.yaxis.grid(True, alpha=0.3)
    ax.set_axisbelow(True)
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
    ap.add_argument("--batch-size", type=int, default=650)
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
    ap.add_argument("--cache-dataset", action="store_true", default=False,
                    help="Copy dataset to SSD for faster training (if enough space)")
    ap.add_argument("--max-cycles", type=int, default=0,
                    help="Stop after this many cycles (0 = run until Ctrl-C)")
    return ap.parse_args()


# ─────────────────────────────────────────────────────────────────────────────
# Main experiment
# ─────────────────────────────────────────────────────────────────────────────

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
    out_root   = Path(args.output_dir)
    arch_dir   = out_root / "architectures"
    cycles_dir = out_root / "cycles"
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
    shutil.copy2(args.architectures_json, out_root / "selected_architectures.json")

    cfg_path = out_root / "experiment_config.json"
    if not cfg_path.exists():
        _atomic_json_write(cfg_path, vars(args))

    # ── dataset split (cached) ───────────────────────────────────────────────
    split_cache = out_root / "dataset_split.json"
    logger.info("Creating/loading dataset split (val_frac=%.2f, seed=%d)…", args.val_frac, args.seed)
    train_idx, val_idx = create_split_indices(
        Path(args.dataset_path), args.val_frac, args.seed, split_cache)
    logger.info("Train samples: %d | Val samples: %d", len(train_idx), len(val_idx))

    # ── SSD dataset cache ─────────────────────────────────────────────────────
    effective_dataset_path = try_cache_dataset_on_ssd(Path(args.dataset_path), args.cache_dataset, logger)
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

    # ── load supernet onto CPU (stays there for weight transfer) ──────────────
    logger.info("Loading supernet checkpoint: %s", args.checkpoint)
    supernet = create_default_supernet(num_classes=args.num_classes)
    ckpt_payload = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    sd = ckpt_payload.get("model") or ckpt_payload.get("state_dict") or ckpt_payload
    missing, unexpected = supernet.load_state_dict(sd, strict=False)
    if missing:
        logger.warning("Supernet missing keys: %d", len(missing))
    if unexpected:
        logger.warning("Supernet unexpected keys: %d", len(unexpected))
    supernet.eval()
    # Supernet remains on CPU throughout; individual subnets are moved to device one at a time.

    # ── reconcile per-architecture state on startup ───────────────────────────
    epochs_done:     Dict[int, int]   = {}
    best_acc1s:      Dict[int, float] = {}   # best EMA acc
    best_val_acc1s:  Dict[int, float] = {}   # best val acc (non-EMA)
    histories:       Dict[int, List[dict]] = {}

    for arch_rec in arch_list:
        ai = arch_rec["arch_index"]
        adir = arch_dir / f"arch_{ai:02d}"
        adir.mkdir(parents=True, exist_ok=True)

        meta_path = adir / "arch_config.json"
        if not meta_path.exists():
            _atomic_json_write(meta_path, arch_rec)

        ep, best, hist = _reconcile_arch_state(adir, ai, logger)
        epochs_done[ai]    = ep
        best_acc1s[ai]     = best
        histories[ai]      = hist
        best_val_acc1s[ai] = max((h["val_acc1"] for h in hist), default=-1.0)

        if ep == 0:
            logger.info("Arch %02d: fresh start (nas_acc=%.2f%%)", ai, arch_rec["nas_quant_acc1"])
        else:
            logger.info("Arch %02d: resumed at epoch %d (best_ema=%.2f%%)", ai, ep, best)

    # ── load cycle stats history ─────────────────────────────────────────────
    stats_history_path = out_root / "stats_history.json"
    if stats_history_path.exists():
        with stats_history_path.open() as f:
            all_cycle_stats: List[dict] = json.load(f)
    else:
        all_cycle_stats = []

    current_cycle = max((s["cycle"] for s in all_cycle_stats), default=-1) + 1
    logger.info("Starting from cycle %d", current_cycle)

    # ── CSV header for cycle summary ─────────────────────────────────────────
    cycle_csv_path = out_root / "cycle_summary.csv"
    _CSV_FIELDS = [
        "cycle", "timestamp",
        "spearman_r", "spearman_p", "pearson_r", "pearson_p",
        "kendall_tau", "kendall_p", "ci_lo", "ci_hi",
        "spearman_r_ema", "spearman_p_ema", "pearson_r_ema", "pearson_p_ema",
        "kendall_tau_ema", "kendall_p_ema",
        "spearman_r_best", "spearman_p_best", "pearson_r_best", "pearson_p_best",
        "kendall_tau_best", "kendall_p_best",
        "mean_val_acc1", "std_val_acc1", "mean_ema_acc1", "mean_best_val_acc1",
        "n_archs",
    ]
    if not cycle_csv_path.exists():
        with cycle_csv_path.open("w", newline="") as fh:
            csv.DictWriter(fh, fieldnames=_CSV_FIELDS).writeheader()

    criterion = nn.CrossEntropyLoss(label_smoothing=args.label_smoothing)

    # ── main loop ─────────────────────────────────────────────────────────────
    logger.info("Entering main training loop (Ctrl-C to stop gracefully).")
    while not _STOP_REQUESTED:
        if args.max_cycles > 0 and current_cycle >= args.max_cycles:
            logger.info("Reached max_cycles=%d — stopping.", args.max_cycles)
            break

        logger.info("━" * 70)
        logger.info("CYCLE %d  (%s)", current_cycle, datetime.now().strftime("%Y-%m-%d %H:%M:%S"))
        logger.info("━" * 70)

        cdir = cycles_dir / f"cycle_{current_cycle:04d}"
        cdir.mkdir(parents=True, exist_ok=True)

        # Load in-progress state (non-empty only after a crash mid-cycle)
        completed_in_cycle, cycle_val_results = load_cycle_progress(cdir)
        if completed_in_cycle:
            logger.info("Resuming cycle %d: %d arch(es) already done: %s",
                        current_cycle, len(completed_in_cycle), sorted(completed_in_cycle))

        # ── per-arch training round ───────────────────────────────────────────
        for arch_rec in arch_list:
            if _STOP_REQUESTED:
                break

            ai     = arch_rec["arch_index"]
            epoch  = epochs_done[ai]
            config = SubnetConfig.from_dict(arch_rec["config"])
            adir   = arch_dir / f"arch_{ai:02d}"

            if ai in completed_in_cycle:
                logger.info("  Arch %02d: already completed in cycle %d — skipping.", ai, current_cycle)
                continue

            logger.info("─── Arch %02d | Epoch %d | NAS=%.2f%% | config=%s",
                        ai, epoch, arch_rec["nas_quant_acc1"], arch_rec["config"])

            train_loader, val_loader = get_loaders(config.resolution)

            # Build model — skip supernet weight transfer if we have a checkpoint
            last_ckpt = adir / "last.pt"
            if last_ckpt.exists():
                model = StaticSubnetModel(config, args.num_classes, supernet.stage_strides).to(device)
                transfer_err = None
            else:
                model, transfer_err = build_static_subnet_model(supernet, config, args.num_classes, device)
                if transfer_err is None:
                    logger.info("  Arch %02d: supernet weights transferred successfully", ai)
                else:
                    logger.warning("  Arch %02d: partial supernet transfer — %s", ai, transfer_err)

            opt = torch.optim.SGD(model.parameters(), lr=args.lr,
                                  momentum=args.momentum, weight_decay=args.weight_decay,
                                  nesterov=True)
            scaler = torch.amp.GradScaler("cuda", enabled=args.amp and device.type == "cuda")
            ema = ModelEMA(model, decay=args.ema_decay)

            if last_ckpt.exists():
                ep_ckpt, best = load_arch_checkpoint(last_ckpt, model, opt, scaler, ema, device)
                best_acc1s[ai] = best
                logger.info("  Arch %02d: loaded checkpoint (epoch %d, best_ema=%.2f%%)",
                            ai, ep_ckpt, best)

            # ── train one epoch ───────────────────────────────────────────────
            skipped = False
            try:
                t_stats = train_one_epoch(
                    ai, epoch, model, train_loader, opt, criterion, scaler, ema, device, args, logger)

                # ── evaluate ──────────────────────────────────────────────────
                logger.info("  Evaluating model and EMA on val set…")
                v_stats   = evaluate(model, val_loader, criterion, device)
                ema_stats = evaluate(ema.ema, val_loader, criterion, device)
                logger.info("  val: loss=%.4f acc1=%.3f%% acc5=%.3f%%",
                            v_stats["loss"], v_stats["acc1"], v_stats["acc5"])
                logger.info("  ema: loss=%.4f acc1=%.3f%% acc5=%.3f%%",
                            ema_stats["loss"], ema_stats["acc1"], ema_stats["acc5"])

                # Best acc tracking
                if ema_stats["acc1"] > best_acc1s[ai]:
                    best_acc1s[ai] = ema_stats["acc1"]
                    save_arch_checkpoint(adir / "best.pt", epoch, model, opt, scaler, ema,
                                         best_acc1s[ai], arch_rec)
                    logger.info("  ★ New best EMA: %.3f%%", best_acc1s[ai])
                if v_stats["acc1"] > best_val_acc1s[ai]:
                    best_val_acc1s[ai] = v_stats["acc1"]

                # ── history ───────────────────────────────────────────────────
                row = {
                    "epoch": epoch,
                    "train_loss": t_stats["loss"], "train_acc1": t_stats["acc1"],
                    "train_acc5": t_stats["acc5"], "lr": t_stats["lr"],
                    "val_loss":   v_stats["loss"], "val_acc1":   v_stats["acc1"],
                    "val_acc5":   v_stats["acc5"],
                    "ema_loss":   ema_stats["loss"], "ema_acc1": ema_stats["acc1"],
                    "ema_acc5":   ema_stats["acc5"],
                }
                histories[ai].append(row)
                _atomic_json_write(adir / "metrics.json", histories[ai])

                # Append row to per-arch CSV
                arch_csv = adir / "metrics.csv"
                write_header = not arch_csv.exists()
                with arch_csv.open("a", newline="") as fh:
                    w = csv.DictWriter(fh, fieldnames=list(row.keys()))
                    if write_header:
                        w.writeheader()
                    w.writerow(row)

                # ── checkpoint ───────────────────────────────────────────────
                save_arch_checkpoint(adir / "last.pt", epoch, model, opt, scaler, ema,
                                      best_acc1s[ai], arch_rec)
                if epoch % 5 == 0:
                    save_arch_checkpoint(adir / f"epoch_{epoch:04d}.pt", epoch, model, opt,
                                          scaler, ema, best_acc1s[ai], arch_rec)

                epochs_done[ai] = epoch + 1

                # Record results for this cycle and persist immediately
                cycle_val_results[ai] = (
                    v_stats["acc1"], ema_stats["acc1"],
                    best_val_acc1s[ai], best_acc1s[ai],
                )
                completed_in_cycle.add(ai)
                save_cycle_progress(cdir, current_cycle, completed_in_cycle, cycle_val_results)

            except (torch.cuda.OutOfMemoryError, RuntimeError) as exc:
                logger.error(
                    "  Arch %02d: error during epoch %d — %s\n"
                    "  This arch will be skipped in cycle %d and retried next cycle.",
                    ai, epoch, exc, current_cycle,
                )
                skipped = True

            finally:
                # Always free GPU memory regardless of success or failure
                _free_gpu(model, opt, scaler, ema)
                logger.debug("  Arch %02d: GPU memory freed after epoch %d.", ai, epoch)

        if _STOP_REQUESTED:
            logger.info("Stop requested — saving state and exiting before cycle stats.")
            break

        # ── per-cycle statistics ──────────────────────────────────────────────
        if len(cycle_val_results) < 2:
            logger.warning(
                "Cycle %d: only %d arch(es) have results — skipping stats (need ≥2).",
                current_cycle, len(cycle_val_results),
            )
            current_cycle += 1
            continue

        logger.info("Computing cycle %d statistics…", current_cycle)
        stat_arch_records = []
        for arch_rec in arch_list:
            ai = arch_rec["arch_index"]
            if ai not in cycle_val_results:
                continue
            v1, e1, bv1, be1 = cycle_val_results[ai]
            stat_arch_records.append({
                "arch_index":       ai,
                "nas_quant_acc1":   arch_rec["nas_quant_acc1"],
                "current_val_acc1": v1,
                "current_ema_acc1": e1,
                "best_val_acc1":    bv1,
                "best_ema_acc1":    be1,
                "epochs_completed": epochs_done[ai],
            })

        cs = compute_cycle_stats(current_cycle, stat_arch_records, args.bootstrap_samples)
        logger.info(
            "  [Val]  Spearman ρ=%.4f (%s) | Pearson r=%.4f | Kendall τ=%.4f",
            cs.spearman_r, _sig_label(cs.spearman_p), cs.pearson_r, cs.kendall_tau,
        )
        logger.info(
            "  [EMA]  Spearman ρ=%.4f (%s) | Pearson r=%.4f | Kendall τ=%.4f",
            cs.spearman_r_ema, _sig_label(cs.spearman_p_ema), cs.pearson_r_ema, cs.kendall_tau_ema,
        )
        logger.info(
            "  [Best] Spearman ρ=%.4f (%s) | Pearson r=%.4f | Kendall τ=%.4f",
            cs.spearman_r_best, _sig_label(cs.spearman_p_best), cs.pearson_r_best, cs.kendall_tau_best,
        )
        logger.info(
            "  Bootstrap 95%% CI (val) [%.4f, %.4f] | Mean val=%.3f±%.3f%% | Mean EMA=%.3f%%",
            cs.bootstrap_ci_spearman[0], cs.bootstrap_ci_spearman[1],
            cs.mean_val_acc1, cs.std_val_acc1, cs.mean_ema_acc1,
        )

        # ── save cycle stats ──────────────────────────────────────────────────
        cs_dict = asdict(cs)
        cs_dict["timestamp"] = datetime.utcnow().isoformat()
        _atomic_json_write(cdir / "stats.json", cs_dict)

        all_cycle_stats.append(cs_dict)
        _atomic_json_write(stats_history_path, all_cycle_stats)

        # Append to CSV
        with cycle_csv_path.open("a", newline="") as fh:
            w2 = csv.DictWriter(fh, fieldnames=_CSV_FIELDS)
            w2.writerow({
                "cycle": current_cycle, "timestamp": cs_dict["timestamp"],
                "spearman_r":    cs.spearman_r,    "spearman_p":    cs.spearman_p,
                "pearson_r":     cs.pearson_r,      "pearson_p":     cs.pearson_p,
                "kendall_tau":   cs.kendall_tau,    "kendall_p":     cs.kendall_p,
                "ci_lo":         cs.bootstrap_ci_spearman[0],
                "ci_hi":         cs.bootstrap_ci_spearman[1],
                "spearman_r_ema":  cs.spearman_r_ema,  "spearman_p_ema":  cs.spearman_p_ema,
                "pearson_r_ema":   cs.pearson_r_ema,   "pearson_p_ema":   cs.pearson_p_ema,
                "kendall_tau_ema": cs.kendall_tau_ema, "kendall_p_ema":   cs.kendall_p_ema,
                "spearman_r_best":  cs.spearman_r_best,  "spearman_p_best":  cs.spearman_p_best,
                "pearson_r_best":   cs.pearson_r_best,   "pearson_p_best":   cs.pearson_p_best,
                "kendall_tau_best": cs.kendall_tau_best, "kendall_p_best":   cs.kendall_p_best,
                "mean_val_acc1":  cs.mean_val_acc1, "std_val_acc1":   cs.std_val_acc1,
                "mean_ema_acc1":  cs.mean_ema_acc1, "mean_best_val_acc1": cs.mean_best_val_acc1,
                "n_archs": cs.n_archs,
            })

        # Save per-arch best-acc summary (updated every cycle)
        best_summary = [
            {
                "arch_index":     arch_rec["arch_index"],
                "nas_quant_acc1": arch_rec["nas_quant_acc1"],
                "best_val_acc1":  best_val_acc1s[arch_rec["arch_index"]],
                "best_ema_acc1":  best_acc1s[arch_rec["arch_index"]],
                "epochs_completed": epochs_done[arch_rec["arch_index"]],
            }
            for arch_rec in arch_list
        ]
        _atomic_json_write(out_root / "best_acc_summary.json", best_summary)

        # ── per-cycle plots ───────────────────────────────────────────────────
        logger.info("Generating cycle %d plots…", current_cycle)
        try:
            plot_correlation_scatter(cs, stat_arch_records,
                                     cdir / "correlation_scatter.png")
            plot_rank_comparison(cs, cdir / "rank_comparison.png")
            plot_training_curves(stat_arch_records, histories,
                                 cdir / "training_curves.png")
        except Exception as e:
            logger.warning("Per-cycle plot generation failed: %s", e)

        # ── rolling plots (root level) ────────────────────────────────────────
        try:
            plot_correlation_over_cycles(all_cycle_stats,
                                         out_root / "correlation_over_cycles.png")
            plot_accuracy_progress(stat_arch_records, all_cycle_stats,
                                   out_root / "accuracy_progress.png")
            plot_best_acc_summary(stat_arch_records, all_cycle_stats,
                                  out_root / "best_acc_summary.png")
        except Exception as e:
            logger.warning("Rolling plot generation failed: %s", e)

        logger.info("Cycle %d complete. Output saved to %s", current_cycle, cdir)
        current_cycle += 1

    logger.info("Experiment stopped at cycle %d. All checkpoints saved.", current_cycle)
    logger.info("Resume by re-running with the same --output-dir argument.")


if __name__ == "__main__":
    main()
