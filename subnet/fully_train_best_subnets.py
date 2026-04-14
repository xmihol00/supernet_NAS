from __future__ import annotations

import argparse
import copy
import csv
import json
import logging
import math
import random
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Sequence, Sized, Tuple, cast

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SUPERNET_DIR = PROJECT_ROOT / "supernet"
if str(SUPERNET_DIR) not in sys.path:
	sys.path.append(str(SUPERNET_DIR))

from imx500_supernet import IMX500ResNetSupernet, SubnetConfig, create_default_supernet

try:
	import safe_gpu
except Exception:
	safe_gpu = None


if safe_gpu is not None:
	while True:
		try:
			safe_gpu.claim_gpus(1)
			break
		except Exception:
			print("Waiting for free GPU")
			time.sleep(5)


class StaticBasicBlock(nn.Module):
	def __init__(self, in_channels: int, out_channels: int, stride: int) -> None:
		super().__init__()
		self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False)
		self.bn1 = nn.BatchNorm2d(out_channels)
		self.relu = nn.ReLU(inplace=True)
		self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False)
		self.bn2 = nn.BatchNorm2d(out_channels)

		self.downsample = None
		if stride != 1 or in_channels != out_channels:
			self.downsample = nn.Sequential(
				nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride, padding=0, bias=False),
				nn.BatchNorm2d(out_channels),
			)

	def forward(self, x: torch.Tensor) -> torch.Tensor:
		identity = x
		out = self.conv1(x)
		out = self.bn1(out)
		out = self.relu(out)
		out = self.conv2(out)
		out = self.bn2(out)

		if self.downsample is not None:
			identity = self.downsample(identity)

		out = out + identity
		out = self.relu(out)
		return out


class StaticSubnetModel(nn.Module):
	def __init__(self, config: SubnetConfig, num_classes: int, stage_strides: Sequence[int]) -> None:
		super().__init__()
		self.config = config

		self.stem_conv = nn.Conv2d(3, config.stem_width, kernel_size=3, stride=2, padding=1, bias=False)
		self.stem_bn = nn.BatchNorm2d(config.stem_width)
		self.stem_act = nn.ReLU(inplace=True)

		stages: List[nn.Module] = []
		prev_width = config.stem_width
		for stage_idx in range(4):
			depth = config.stage_depths[stage_idx]
			out_width = config.stage_widths[stage_idx]
			stride = int(stage_strides[stage_idx])

			blocks: List[nn.Module] = []
			for block_idx in range(depth):
				block_stride = stride if block_idx == 0 else 1
				block_in = prev_width if block_idx == 0 else out_width
				blocks.append(StaticBasicBlock(block_in, out_width, block_stride))
			stages.append(nn.Sequential(*blocks))
			prev_width = out_width

		self.stages = nn.ModuleList(stages)
		self.global_pool = nn.AdaptiveAvgPool2d(1)
		self.classifier = nn.Linear(config.stage_widths[-1], num_classes)

	def forward(self, x: torch.Tensor) -> torch.Tensor:
		x = self.stem_conv(x)
		x = self.stem_bn(x)
		x = self.stem_act(x)

		for stage in self.stages:
			x = stage(x)

		x = self.global_pool(x)
		x = torch.flatten(x, 1)
		x = self.classifier(x)
		return x


@dataclass
class SelectedArchitecture:
	run_index: int
	seed: int
	score: float
	config: SubnetConfig
	algorithm: str


def parse_args() -> argparse.Namespace:
	parser = argparse.ArgumentParser("Fully train top-K subnet architectures from multi-run NAS records")

	parser.add_argument("--run-records-json", type=str, default="/mnt/matylda5/xmihol00/EUD/NAS/old_multi_run_parallel/sga_2026-04-05_17-40-43/baseline_sga/run_records.json")
	parser.add_argument("--top-k", type=int, default=3)
	parser.add_argument("--num-classes", type=int, default=1000)

	parser.add_argument("--dataset-path", type=str, default="/mnt/matylda5/xmihol00/datasets/imagenet/train")
	parser.add_argument("--val-split", type=float, default=0.15)

	parser.add_argument("--epochs", type=int, default=120)
	parser.add_argument("--early-stop-patience", type=int, default=5)
	parser.add_argument("--batch-size", type=int, default=96)
	parser.add_argument("--num-workers", type=int, default=8)
	parser.add_argument("--lr", type=float, default=0.02)
	parser.add_argument("--min-lr", type=float, default=1e-4)
	parser.add_argument("--weight-decay", type=float, default=5e-5)
	parser.add_argument("--momentum", type=float, default=0.9)
	parser.add_argument("--label-smoothing", type=float, default=0.1)
	parser.add_argument("--grad-clip", type=float, default=1.0)
	parser.add_argument("--warmup-epochs", type=int, default=5)
	parser.add_argument("--mixup-alpha", type=float, default=0.2)
	parser.add_argument("--mixup-prob", type=float, default=0.5)
	parser.add_argument("--cutmix-alpha", type=float, default=1.0)
	parser.add_argument("--cutmix-prob", type=float, default=0.2)
	parser.add_argument("--mix-reg-off-epoch", type=int, default=0)
	parser.add_argument("--freeze-backbone-epochs", type=int, default=1)
	parser.add_argument("--ema-decay", type=float, default=0.9998)
	parser.add_argument("--randaugment-magnitude", type=int, default=7)

	parser.add_argument("--checkpoint", type=str, default="/mnt/matylda5/xmihol00/EUD/supernet/runs_imx500_supernet/20260402_200233/best.pt")
	parser.add_argument("--seed", type=int, default=42)
	parser.add_argument("--device", type=str, default="cuda")
	parser.add_argument("--amp", action="store_true", default=True)
	parser.add_argument("--no-amp", dest="amp", action="store_false")
	parser.add_argument("--output-dir", type=str, default="./runs_full_subnet_training")
	parser.add_argument("--plot-every", type=int, default=1)

	return parser.parse_args()


def setup_logger(log_file: Path) -> logging.Logger:
	log_file.parent.mkdir(parents=True, exist_ok=True)
	logger = logging.getLogger(str(log_file))
	logger.setLevel(logging.INFO)
	logger.handlers = []

	formatter = logging.Formatter("%(asctime)s | %(levelname)s | %(message)s")

	file_handler = logging.FileHandler(log_file, encoding="utf-8")
	file_handler.setLevel(logging.INFO)
	file_handler.setFormatter(formatter)

	stream_handler = logging.StreamHandler()
	stream_handler.setLevel(logging.INFO)
	stream_handler.setFormatter(formatter)

	logger.addHandler(file_handler)
	logger.addHandler(stream_handler)
	return logger


def set_seed(seed: int) -> None:
	random.seed(seed)
	np.random.seed(seed)
	torch.manual_seed(seed)
	torch.cuda.manual_seed_all(seed)


def _extract_state_dict(payload: object) -> Dict[str, torch.Tensor]:
	if isinstance(payload, dict) and "model" in payload:
		state_dict = payload["model"]
	elif isinstance(payload, dict) and "state_dict" in payload:
		state_dict = payload["state_dict"]
	else:
		state_dict = payload

	if not isinstance(state_dict, dict):
		raise ValueError("Unsupported checkpoint format.")
	return state_dict


def _copy_batchnorm(dynamic_bn: nn.BatchNorm2d, static_bn: nn.BatchNorm2d, channels: int) -> None:
	static_bn.weight.data.copy_(dynamic_bn.weight.data[:channels])
	static_bn.bias.data.copy_(dynamic_bn.bias.data[:channels])
	if dynamic_bn.running_mean is not None and static_bn.running_mean is not None:
		static_bn.running_mean.data.copy_(dynamic_bn.running_mean.data[:channels])
	if dynamic_bn.running_var is not None and static_bn.running_var is not None:
		static_bn.running_var.data.copy_(dynamic_bn.running_var.data[:channels])
	if dynamic_bn.num_batches_tracked is not None and static_bn.num_batches_tracked is not None:
		static_bn.num_batches_tracked.data.copy_(dynamic_bn.num_batches_tracked.data)


def build_static_subnet_model(
	supernet: IMX500ResNetSupernet,
	config: SubnetConfig,
	num_classes: int,
	device: torch.device,
) -> StaticSubnetModel:
	model = StaticSubnetModel(config=config, num_classes=num_classes, stage_strides=supernet.stage_strides).to(device)

	with torch.no_grad():
		model.stem_conv.weight.copy_(supernet.stem_conv.weight.data[: config.stem_width, :3, :, :])
		_copy_batchnorm(supernet.stem_bn, model.stem_bn, config.stem_width)

		prev_width = config.stem_width
		for stage_idx in range(4):
			depth = config.stage_depths[stage_idx]
			out_width = config.stage_widths[stage_idx]
			stride = supernet.stage_strides[stage_idx]

			for block_idx in range(depth):
				block_in = prev_width if block_idx == 0 else out_width
				dynamic_block = supernet.stages[stage_idx].blocks[block_idx]
				static_stage = cast(nn.Sequential, model.stages[stage_idx])
				static_block = cast(StaticBasicBlock, static_stage[block_idx])

				static_block.conv1.weight.copy_(dynamic_block.conv1.weight.data[:out_width, :block_in, :, :])
				_copy_batchnorm(dynamic_block.bn1, static_block.bn1, out_width)

				static_block.conv2.weight.copy_(dynamic_block.conv2.weight.data[:out_width, :out_width, :, :])
				_copy_batchnorm(dynamic_block.bn2, static_block.bn2, out_width)

				needs_projection = (stride != 1 and block_idx == 0) or (block_in != out_width)
				if needs_projection and static_block.downsample is not None:
					downsample = cast(nn.Sequential, static_block.downsample)
					proj_conv = cast(nn.Conv2d, downsample[0])
					proj_bn = cast(nn.BatchNorm2d, downsample[1])
					proj_conv.weight.copy_(dynamic_block.downsample_conv.weight.data[:out_width, :block_in, :, :])
					_copy_batchnorm(dynamic_block.downsample_bn, proj_bn, out_width)

			prev_width = out_width

		model.classifier.weight.copy_(supernet.classifier.weight.data[:, : config.stage_widths[-1]])
		model.classifier.bias.copy_(supernet.classifier.bias.data)

	return model


def create_splits(dataset_path: Path, val_split: float, seed: int) -> Tuple[List[int], List[int]]:
	base_dataset = datasets.ImageFolder(root=str(dataset_path))
	num_samples = len(base_dataset)
	all_indices = np.arange(num_samples)
	rng = np.random.default_rng(seed)
	rng.shuffle(all_indices)

	val_count = int(math.floor(num_samples * val_split))
	val_indices = all_indices[:val_count].tolist()
	train_indices = all_indices[val_count:].tolist()
	return train_indices, val_indices


def create_loaders(args: argparse.Namespace, max_resolution: int) -> Tuple[DataLoader, DataLoader]:
	normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])

	train_transform = transforms.Compose(
		[
			transforms.RandomResizedCrop(
				max_resolution,
				scale=(0.2, 1.0),
				ratio=(0.75, 1.3333333333),
				interpolation=transforms.InterpolationMode.BILINEAR,
			),
			transforms.RandomHorizontalFlip(),
			transforms.RandAugment(num_ops=2, magnitude=args.randaugment_magnitude),
			transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.05),
			transforms.ToTensor(),
			normalize,
			transforms.RandomErasing(p=0.15, scale=(0.02, 0.12), ratio=(0.3, 3.3)),
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

	result: Dict[str, float] = {}
	for k in topk:
		correct_k = correct[:k].reshape(-1).float().sum(0)
		result[f"acc{k}"] = float(correct_k.mul_(100.0 / target.size(0)).item())
	return result


def cosine_with_warmup(step: int, total_steps: int, warmup_steps: int, base_lr: float, min_lr: float) -> float:
	if step < warmup_steps:
		warmup_lr = base_lr * float(step + 1) / float(max(1, warmup_steps))
		return max(min_lr, warmup_lr)
	progress = float(step - warmup_steps) / float(max(1, total_steps - warmup_steps))
	return min_lr + (base_lr - min_lr) * 0.5 * (1.0 + math.cos(math.pi * progress))


def _rand_bbox(size: torch.Size, lam: float) -> Tuple[int, int, int, int]:
	_, _, h, w = size
	cut_ratio = math.sqrt(max(0.0, 1.0 - lam))
	cut_w = int(w * cut_ratio)
	cut_h = int(h * cut_ratio)

	cx = np.random.randint(0, w)
	cy = np.random.randint(0, h)

	x1 = int(np.clip(cx - cut_w // 2, 0, w))
	y1 = int(np.clip(cy - cut_h // 2, 0, h))
	x2 = int(np.clip(cx + cut_w // 2, 0, w))
	y2 = int(np.clip(cy + cut_h // 2, 0, h))
	return x1, y1, x2, y2


def apply_batch_regularization(
	images: torch.Tensor,
	target: torch.Tensor,
	args: argparse.Namespace,
	epoch: int,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, float, str]:
	if args.mix_reg_off_epoch > 0 and epoch >= args.mix_reg_off_epoch:
		return images, target, target, 1.0, "none"

	apply_cutmix = args.cutmix_alpha > 0.0 and random.random() < args.cutmix_prob
	if apply_cutmix:
		lam = float(np.random.beta(args.cutmix_alpha, args.cutmix_alpha))
		index = torch.randperm(images.size(0), device=images.device)
		target_a, target_b = target, target[index]
		x1, y1, x2, y2 = _rand_bbox(images.size(), lam)
		images[:, :, y1:y2, x1:x2] = images[index, :, y1:y2, x1:x2]
		box_area = float((x2 - x1) * (y2 - y1))
		lam = 1.0 - box_area / float(images.size(-1) * images.size(-2))
		return images, target_a, target_b, lam, "cutmix"

	apply_mixup = args.mixup_alpha > 0.0 and random.random() < args.mixup_prob
	if apply_mixup:
		lam = float(np.random.beta(args.mixup_alpha, args.mixup_alpha))
		index = torch.randperm(images.size(0), device=images.device)
		mixed_images = lam * images + (1.0 - lam) * images[index]
		target_a, target_b = target, target[index]
		return mixed_images, target_a, target_b, lam, "mixup"

	return images, target, target, 1.0, "none"


def set_backbone_trainable(model: StaticSubnetModel, trainable: bool) -> None:
	for parameter in model.stem_conv.parameters():
		parameter.requires_grad = trainable
	for parameter in model.stem_bn.parameters():
		parameter.requires_grad = trainable
	for stage in model.stages:
		for parameter in stage.parameters():
			parameter.requires_grad = trainable
	for parameter in model.classifier.parameters():
		parameter.requires_grad = True


class ModelEMA:
	def __init__(self, model: nn.Module, decay: float) -> None:
		self.decay = decay
		self.ema = copy.deepcopy(model).eval()
		for parameter in self.ema.parameters():
			parameter.requires_grad_(False)

	@torch.no_grad()
	def update(self, model: nn.Module) -> None:
		ema_state = self.ema.state_dict()
		model_state = model.state_dict()
		for key, ema_value in ema_state.items():
			model_value = model_state[key].detach()
			if torch.is_floating_point(ema_value):
				ema_value.mul_(self.decay).add_(model_value, alpha=(1.0 - self.decay))
			else:
				ema_value.copy_(model_value)


def train_one_epoch(
	epoch: int,
	model: nn.Module,
	train_loader: DataLoader,
	optimizer: torch.optim.Optimizer,
	criterion: nn.Module,
	scaler: torch.cuda.amp.GradScaler,
	args: argparse.Namespace,
	device: torch.device,
	ema: ModelEMA | None,
	logger: logging.Logger,
) -> Dict[str, float]:
	model.train()
	num_steps = len(train_loader)
	total_steps = args.epochs * num_steps
	warmup_steps = args.warmup_epochs * num_steps

	loss_meter = 0.0
	acc1_meter = 0.0
	acc5_meter = 0.0
	lr = args.lr
	reg_mode = "none"

	for batch_idx, (images, target) in enumerate(train_loader):
		global_step = epoch * num_steps + batch_idx
		lr = cosine_with_warmup(global_step, total_steps, warmup_steps, args.lr, args.min_lr)
		for group in optimizer.param_groups:
			group["lr"] = lr

		images = images.to(device, non_blocking=True)
		target = target.to(device, non_blocking=True)

		mixed_images, target_a, target_b, lam, reg_mode = apply_batch_regularization(images, target, args, epoch)

		optimizer.zero_grad(set_to_none=True)
		with torch.autocast(device_type=device.type, enabled=args.amp and device.type == "cuda"):
			logits = model(mixed_images)
			if lam < 1.0:
				loss = lam * criterion(logits, target_a) + (1.0 - lam) * criterion(logits, target_b)
			else:
				loss = criterion(logits, target)

		scaler.scale(loss).backward()

		if args.grad_clip > 0:
			scaler.unscale_(optimizer)
			torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)

		scaler.step(optimizer)
		scaler.update()
		if ema is not None:
			ema.update(model)

		with torch.no_grad():
			metrics = accuracy_topk(logits, target, topk=(1, 5))

		loss_meter += float(loss.item())
		acc1_meter += metrics["acc1"]
		acc5_meter += metrics["acc5"]

	stats = {
		"loss": loss_meter / max(1, num_steps),
		"acc1": acc1_meter / max(1, num_steps),
		"acc5": acc5_meter / max(1, num_steps),
		"last_lr": lr,
		"reg_mode": reg_mode,
	}
	logger.info(
		"epoch=%d train loss=%.5f acc1=%.3f acc5=%.3f lr=%.6f reg=%s",
		epoch,
		stats["loss"],
		stats["acc1"],
		stats["acc5"],
		stats["last_lr"],
		stats["reg_mode"],
	)
	return stats


@torch.no_grad()
def evaluate(
	epoch: int,
	model: nn.Module,
	val_loader: DataLoader,
	criterion: nn.Module,
	device: torch.device,
	logger: logging.Logger,
	prefix: str = "val",
) -> Dict[str, float]:
	model.eval()

	loss_meter = 0.0
	acc1_meter = 0.0
	acc5_meter = 0.0
	steps = 0

	for images, target in val_loader:
		images = images.to(device, non_blocking=True)
		target = target.to(device, non_blocking=True)

		logits = model(images)
		loss = criterion(logits, target)
		metrics = accuracy_topk(logits, target, topk=(1, 5))

		steps += 1
		loss_meter += float(loss.item())
		acc1_meter += metrics["acc1"]
		acc5_meter += metrics["acc5"]

	stats = {
		"loss": loss_meter / max(1, steps),
		"acc1": acc1_meter / max(1, steps),
		"acc5": acc5_meter / max(1, steps),
	}
	logger.info(
		"epoch=%d %s loss=%.5f acc1=%.3f acc5=%.3f",
		epoch,
		prefix,
		stats["loss"],
		stats["acc1"],
		stats["acc5"],
	)
	return stats


def save_checkpoint(
	path: Path,
	epoch: int,
	model: nn.Module,
	optimizer: torch.optim.Optimizer,
	scaler: torch.cuda.amp.GradScaler,
	best_acc1: float,
	config: SubnetConfig,
	args: argparse.Namespace,
) -> None:
	payload = {
		"epoch": epoch,
		"model": model.state_dict(),
		"optimizer": optimizer.state_dict(),
		"scaler": scaler.state_dict(),
		"best_acc1": best_acc1,
		"subnet_config": config.to_dict(),
		"args": vars(args),
	}
	torch.save(payload, path)


def update_plots(history: List[Dict[str, Any]], out_png: Path) -> None:
	if not history:
		return

	epochs = [int(record["epoch"]) for record in history]
	train_loss = [float(record["train"]["loss"]) for record in history]
	val_loss = [float(record["val"]["loss"]) for record in history]
	val_ema_loss = [float(record.get("val_ema", {}).get("loss", record["val"]["loss"])) for record in history]
	train_acc1 = [float(record["train"]["acc1"]) for record in history]
	val_acc1 = [float(record["val"]["acc1"]) for record in history]
	val_ema_acc1 = [float(record.get("val_ema", {}).get("acc1", record["val"]["acc1"])) for record in history]

	fig, axes = plt.subplots(1, 2, figsize=(12, 4))

	axes[0].plot(epochs, train_loss, label="train_loss", color="tab:blue")
	axes[0].plot(epochs, val_loss, label="val_loss", color="tab:orange")
	axes[0].plot(epochs, val_ema_loss, label="val_ema_loss", color="tab:purple")
	axes[0].set_xlabel("Epoch")
	axes[0].set_ylabel("Loss")
	axes[0].set_title("Loss")
	axes[0].grid(alpha=0.3)
	axes[0].legend()

	axes[1].plot(epochs, train_acc1, label="train_acc1", color="tab:green")
	axes[1].plot(epochs, val_acc1, label="val_acc1", color="tab:red")
	axes[1].plot(epochs, val_ema_acc1, label="val_ema_acc1", color="tab:brown")
	axes[1].set_xlabel("Epoch")
	axes[1].set_ylabel("Top-1 Accuracy (%)")
	axes[1].set_title("Accuracy")
	axes[1].grid(alpha=0.3)
	axes[1].legend()

	fig.tight_layout()
	out_png.parent.mkdir(parents=True, exist_ok=True)
	fig.savefig(out_png, dpi=140)
	plt.close(fig)


def _try_load_json(path: Path) -> object:
	with path.open("r", encoding="utf-8") as handle:
		return json.load(handle)


def _as_int(value: object, default: int = -1) -> int:
	if isinstance(value, bool):
		return int(value)
	if isinstance(value, int):
		return value
	if isinstance(value, float):
		return int(value)
	if isinstance(value, str):
		try:
			return int(value)
		except ValueError:
			return default
	try:
		return int(str(value))
	except (TypeError, ValueError):
		return default


def _as_float(value: object, default: float = 0.0) -> float:
	if isinstance(value, bool):
		return float(value)
	if isinstance(value, (int, float)):
		return float(value)
	if isinstance(value, str):
		try:
			return float(value)
		except ValueError:
			return default
	try:
		return float(str(value))
	except (TypeError, ValueError):
		return default


def _extract_best_from_run_record(record: Dict[str, object]) -> SelectedArchitecture | None:
	summary = record.get("summary", {})
	if not isinstance(summary, dict):
		summary = {}

	algorithm = str(record.get("algorithm", "unknown"))
	run_index = _as_int(record.get("run_index", -1), default=-1)
	seed = _as_int(record.get("seed", -1), default=-1)

	config_dict = summary.get("best_config")
	score = summary.get("best_quant_acc1", summary.get("best_fitness", 0.0))

	if isinstance(config_dict, dict):
		try:
			return SelectedArchitecture(
				run_index=run_index,
				seed=seed,
				score=_as_float(score, default=0.0),
				config=SubnetConfig.from_dict(config_dict),
				algorithm=algorithm,
			)
		except Exception:
			pass

	top3 = summary.get("top_3_architectures")
	if isinstance(top3, list) and top3:
		best = top3[0]
		if isinstance(best, dict):
			candidate_cfg = best.get("config")
			candidate_score = best.get("fitness", best.get("quant_acc1", score))
			if isinstance(candidate_cfg, dict):
				try:
					return SelectedArchitecture(
						run_index=run_index,
						seed=seed,
						score=_as_float(candidate_score, default=0.0),
						config=SubnetConfig.from_dict(candidate_cfg),
						algorithm=algorithm,
					)
				except Exception:
					pass

	run_dir = record.get("run_dir")
	if isinstance(run_dir, str) and run_dir:
		top3_path = Path(run_dir) / "top_3_architectures.json"
		summary_path = Path(run_dir) / "summary.json"

		for candidate_path in [summary_path, top3_path]:
			if not candidate_path.exists():
				continue
			try:
				payload = _try_load_json(candidate_path)
			except Exception:
				continue

			if candidate_path.name == "summary.json" and isinstance(payload, dict):
				cfg = payload.get("best_config")
				sc = payload.get("best_quant_acc1", payload.get("best_fitness", score))
				if isinstance(cfg, dict):
					try:
						return SelectedArchitecture(
							run_index=run_index,
							seed=seed,
							score=_as_float(sc, default=0.0),
							config=SubnetConfig.from_dict(cfg),
							algorithm=algorithm,
						)
					except Exception:
						continue

			if candidate_path.name == "top_3_architectures.json" and isinstance(payload, dict):
				architectures = payload.get("architectures")
				if isinstance(architectures, list) and architectures:
					best = architectures[0]
					if isinstance(best, dict) and isinstance(best.get("config"), dict):
						try:
							return SelectedArchitecture(
								run_index=run_index,
								seed=seed,
								score=_as_float(best.get("fitness", best.get("quant_acc1", score)), default=0.0),
								config=SubnetConfig.from_dict(best["config"]),
								algorithm=algorithm,
							)
						except Exception:
							continue

	return None


def select_top_k_across_runs(run_records_path: Path, top_k: int) -> List[SelectedArchitecture]:
	payload = _try_load_json(run_records_path)
	if not isinstance(payload, list):
		raise ValueError("Expected run records JSON to be a list of run records.")

	per_run_bests: List[SelectedArchitecture] = []
	for record in payload:
		if not isinstance(record, dict):
			continue
		selected = _extract_best_from_run_record(record)
		if selected is not None:
			per_run_bests.append(selected)

	if not per_run_bests:
		raise RuntimeError("No valid architecture configuration found in run records JSON.")

	# Keep one best architecture per run, then rank these globally.
	by_run: Dict[Tuple[str, int], SelectedArchitecture] = {}
	for item in per_run_bests:
		key = (item.algorithm, item.run_index)
		existing = by_run.get(key)
		if existing is None or item.score > existing.score:
			by_run[key] = item

	unique_per_run = list(by_run.values())
	unique_per_run.sort(key=lambda item: item.score, reverse=True)
	return unique_per_run[: max(1, top_k)]


def train_single_architecture(
	arch: SelectedArchitecture,
	arch_rank: int,
	args: argparse.Namespace,
	train_loader: DataLoader,
	val_loader: DataLoader,
	device: torch.device,
	output_root: Path,
) -> Dict[str, object]:
	run_dir = output_root / f"rank_{arch_rank:02d}_run_{arch.run_index:03d}_seed_{arch.seed}"
	run_dir.mkdir(parents=True, exist_ok=True)

	logger = setup_logger(run_dir / "train.log")
	logger.info("Preparing architecture rank=%d from run=%d seed=%d score=%.4f", arch_rank, arch.run_index, arch.seed, arch.score)
	logger.info("Subnet config: %s", json.dumps(arch.config.to_dict(), sort_keys=True))

	supernet = create_default_supernet(num_classes=args.num_classes)
	if args.checkpoint:
		checkpoint_payload = torch.load(args.checkpoint, map_location="cpu")
		state_dict = _extract_state_dict(checkpoint_payload)
		missing, unexpected = supernet.load_state_dict(state_dict, strict=False)
		logger.info("Loaded checkpoint: %s", args.checkpoint)
		if missing:
			logger.info("Missing keys count while loading checkpoint: %d", len(missing))
		if unexpected:
			logger.info("Unexpected keys count while loading checkpoint: %d", len(unexpected))

	model = build_static_subnet_model(supernet, arch.config, args.num_classes, device=device)

	criterion = nn.CrossEntropyLoss(label_smoothing=args.label_smoothing)
	optimizer = torch.optim.SGD(
		model.parameters(),
		lr=args.lr,
		momentum=args.momentum,
		weight_decay=args.weight_decay,
		nesterov=True,
	)
	scaler = torch.cuda.amp.GradScaler(enabled=args.amp and device.type == "cuda")
	ema = ModelEMA(model, decay=args.ema_decay)

	history: List[Dict[str, Any]] = []
	metrics_csv = run_dir / "metrics.csv"
	with metrics_csv.open("w", encoding="utf-8", newline="") as handle:
		writer = csv.DictWriter(
			handle,
			fieldnames=[
				"epoch",
				"train_loss",
				"train_acc1",
				"train_acc5",
				"val_loss",
				"val_acc1",
				"val_acc5",
				"val_ema_loss",
				"val_ema_acc1",
				"val_ema_acc5",
			],
		)
		writer.writeheader()

	best_acc1 = -1.0
	best_epoch = -1
	epochs_without_improvement = 0

	for epoch in range(args.epochs):
		backbone_trainable = epoch >= args.freeze_backbone_epochs
		set_backbone_trainable(cast(StaticSubnetModel, model), trainable=backbone_trainable)
		logger.info("epoch=%d backbone_trainable=%s", epoch, str(backbone_trainable).lower())

		train_stats = train_one_epoch(epoch, model, train_loader, optimizer, criterion, scaler, args, device, ema, logger)
		val_stats = evaluate(epoch, model, val_loader, criterion, device, logger, prefix="val")
		ema_val_stats = evaluate(epoch, ema.ema, val_loader, criterion, device, logger, prefix="val_ema")
		selection_stats = ema_val_stats

		record = {
			"epoch": epoch,
			"train": train_stats,
			"val": val_stats,
			"val_ema": ema_val_stats,
		}
		history.append(record)

		with (run_dir / "metrics.json").open("w", encoding="utf-8") as handle:
			json.dump(history, handle, indent=2)

		with metrics_csv.open("a", encoding="utf-8", newline="") as handle:
			writer = csv.DictWriter(
				handle,
				fieldnames=[
					"epoch",
					"train_loss",
					"train_acc1",
					"train_acc5",
					"val_loss",
					"val_acc1",
					"val_acc5",
					"val_ema_loss",
					"val_ema_acc1",
					"val_ema_acc5",
				],
			)
			writer.writerow(
				{
					"epoch": epoch,
					"train_loss": train_stats["loss"],
					"train_acc1": train_stats["acc1"],
					"train_acc5": train_stats["acc5"],
					"val_loss": val_stats["loss"],
					"val_acc1": val_stats["acc1"],
					"val_acc5": val_stats["acc5"],
					"val_ema_loss": ema_val_stats["loss"],
					"val_ema_acc1": ema_val_stats["acc1"],
					"val_ema_acc5": ema_val_stats["acc5"],
				}
			)

		if args.plot_every > 0 and (epoch % args.plot_every == 0 or epoch == args.epochs - 1):
			update_plots(history, run_dir / "training_curves.png")

		if selection_stats["acc1"] > best_acc1:
			best_acc1 = selection_stats["acc1"]
			best_epoch = epoch
			epochs_without_improvement = 0
			save_checkpoint(run_dir / "best.pt", epoch, model, optimizer, scaler, best_acc1, arch.config, args)
			torch.save({"epoch": epoch, "model": ema.ema.state_dict(), "best_acc1": best_acc1}, run_dir / "best_ema.pt")
			logger.info("Saved new best checkpoint at epoch %d with val_ema_acc1=%.4f", epoch, best_acc1)
		else:
			epochs_without_improvement += 1

		save_checkpoint(run_dir / "last.pt", epoch, model, optimizer, scaler, best_acc1, arch.config, args)

		if epochs_without_improvement >= args.early_stop_patience:
			logger.info(
				"Early stopping activated at epoch %d (patience=%d, best_epoch=%d, best_val_acc1=%.4f)",
				epoch,
				args.early_stop_patience,
				best_epoch,
				best_acc1,
			)
			break

	result = {
		"rank": arch_rank,
		"source_algorithm": arch.algorithm,
		"source_run_index": arch.run_index,
		"source_seed": arch.seed,
		"source_score": arch.score,
		"best_epoch": best_epoch,
		"best_val_acc1": best_acc1,
		"subnet_config": arch.config.to_dict(),
		"run_dir": str(run_dir),
	}

	with (run_dir / "summary.json").open("w", encoding="utf-8") as handle:
		json.dump(result, handle, indent=2)

	logger.info("Training completed. Summary: %s", json.dumps(result, sort_keys=True))
	return result


def main() -> None:
	args = parse_args()
	set_seed(args.seed)

	output_root = Path(args.output_dir) / time.strftime("%Y%m%d_%H%M%S")
	output_root.mkdir(parents=True, exist_ok=True)
	root_logger = setup_logger(output_root / "fully_train_best_subnets.log")

	selected_arches = select_top_k_across_runs(Path(args.run_records_json), args.top_k)
	root_logger.info("Selected %d architectures for full training.", len(selected_arches))
	for idx, arch in enumerate(selected_arches, start=1):
		root_logger.info(
			"Selection rank=%d algorithm=%s run_index=%d seed=%d score=%.4f config=%s",
			idx,
			arch.algorithm,
			arch.run_index,
			arch.seed,
			arch.score,
			json.dumps(arch.config.to_dict(), sort_keys=True),
		)

	device = torch.device(args.device if torch.cuda.is_available() else "cpu")
	if device.type != "cuda":
		root_logger.warning("CUDA not available, using CPU. This will be slow.")
	root_logger.info("Using device: %s", device)

	max_resolution = max(create_default_supernet(num_classes=args.num_classes).resolution_candidates)
	train_loader, val_loader = create_loaders(args, max_resolution=max_resolution)
	root_logger.info(
		"Train samples=%d | Val samples=%d",
		len(cast(Sized, train_loader.dataset)),
		len(cast(Sized, val_loader.dataset)),
	)

	summary_rows: List[Dict[str, object]] = []
	for arch_rank, arch in enumerate(selected_arches, start=1):
		result = train_single_architecture(
			arch=arch,
			arch_rank=arch_rank,
			args=args,
			train_loader=train_loader,
			val_loader=val_loader,
			device=device,
			output_root=output_root,
		)
		summary_rows.append(result)

		with (output_root / "training_summary.json").open("w", encoding="utf-8") as handle:
			json.dump(summary_rows, handle, indent=2)

	summary_rows.sort(key=lambda row: _as_float(row.get("best_val_acc1", 0.0), default=0.0), reverse=True)
	with (output_root / "training_summary_sorted.json").open("w", encoding="utf-8") as handle:
		json.dump(summary_rows, handle, indent=2)

	root_logger.info("All subnet trainings completed. Sorted summary saved to: %s", output_root / "training_summary_sorted.json")


if __name__ == "__main__":
	main()
