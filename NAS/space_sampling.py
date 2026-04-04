from __future__ import annotations

import argparse
import json
import os
import random
import subprocess
import sys
import time
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Dict, Iterable, List, Sequence, Tuple

import numpy as np
import onnx
import onnxruntime as ort
import torch
import torch.nn as nn
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms

import model_compression_toolkit as mct
from edgemdt_tpc import get_target_platform_capabilities


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SUPERNET_DIR = PROJECT_ROOT / "supernet"
if str(SUPERNET_DIR) not in sys.path:
    sys.path.append(str(SUPERNET_DIR))

from imx500_supernet import IMX500ResNetSupernet, SubnetConfig, create_default_supernet

try:
    from mct_quantizers import get_ort_session_options
except Exception:
    get_ort_session_options = None


SUPPORTED_IMAGE_EXTENSIONS = (".jpg", ".jpeg", ".png", ".bmp", ".webp")


def log(message: str) -> None:
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    print(f"[{timestamp}] {message}", flush=True)


def log_duration(step: str, started_at: float) -> None:
    elapsed = time.perf_counter() - started_at
    log(f"{step} finished in {elapsed:.2f}s")


class StaticBasicBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, stride: int) -> None:
        super().__init__()
        self.conv1 = nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size=3,
            stride=stride,
            padding=1,
            bias=False,
        )
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(
            out_channels,
            out_channels,
            kernel_size=3,
            stride=1,
            padding=1,
            bias=False,
        )
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


def _copy_batchnorm(dynamic_bn: nn.Module, static_bn: nn.BatchNorm2d, channels: int) -> None:
    static_bn.weight.data.copy_(dynamic_bn.weight.data[:channels])
    static_bn.bias.data.copy_(dynamic_bn.bias.data[:channels])
    static_bn.running_mean.data.copy_(dynamic_bn.running_mean.data[:channels])
    static_bn.running_var.data.copy_(dynamic_bn.running_var.data[:channels])
    if hasattr(dynamic_bn, "num_batches_tracked") and hasattr(static_bn, "num_batches_tracked"):
        static_bn.num_batches_tracked.data.copy_(dynamic_bn.num_batches_tracked.data)


def build_static_subnet_model(
    supernet: IMX500ResNetSupernet,
    config: SubnetConfig,
    num_classes: int,
    device: torch.device,
) -> StaticSubnetModel:
    model = StaticSubnetModel(
        config=config,
        num_classes=num_classes,
        stage_strides=supernet.stage_strides,
    ).to(device)

    with torch.no_grad():
        model.stem_conv.weight.copy_(
            supernet.stem_conv.weight.data[: config.stem_width, :3, :, :]
        )
        _copy_batchnorm(supernet.stem_bn, model.stem_bn, config.stem_width)

        prev_width = config.stem_width
        for stage_idx in range(4):
            depth = config.stage_depths[stage_idx]
            out_width = config.stage_widths[stage_idx]
            stride = supernet.stage_strides[stage_idx]

            for block_idx in range(depth):
                block_in = prev_width if block_idx == 0 else out_width

                dynamic_block = supernet.stages[stage_idx].blocks[block_idx]
                static_block = model.stages[stage_idx][block_idx]

                static_block.conv1.weight.copy_(
                    dynamic_block.conv1.weight.data[:out_width, :block_in, :, :]
                )
                _copy_batchnorm(dynamic_block.bn1, static_block.bn1, out_width)

                static_block.conv2.weight.copy_(
                    dynamic_block.conv2.weight.data[:out_width, :out_width, :, :]
                )
                _copy_batchnorm(dynamic_block.bn2, static_block.bn2, out_width)

                needs_projection = (stride != 1 and block_idx == 0) or (block_in != out_width)
                if needs_projection and static_block.downsample is not None:
                    proj_conv = static_block.downsample[0]
                    proj_bn = static_block.downsample[1]
                    proj_conv.weight.copy_(
                        dynamic_block.downsample_conv.weight.data[:out_width, :block_in, :, :]
                    )
                    _copy_batchnorm(dynamic_block.downsample_bn, proj_bn, out_width)

            prev_width = out_width

        model.classifier.weight.copy_(
            supernet.classifier.weight.data[:, : config.stage_widths[-1]]
        )
        model.classifier.bias.copy_(supernet.classifier.bias.data)

    model.eval()
    return model


@dataclass
class DatasetEntry:
    image_path: str
    class_name: str
    class_index: int


class ImageFolderCalibrationDataset(Dataset):
    def __init__(
        self,
        folder_path: str,
        transform: transforms.Compose,
        limit: int,
    ) -> None:
        image_paths: List[str] = []
        folder = Path(folder_path)
        for ext in SUPPORTED_IMAGE_EXTENSIONS:
            image_paths.extend(str(p) for p in folder.glob(f"*{ext}"))
            image_paths.extend(str(p) for p in folder.glob(f"*{ext.upper()}"))

        image_paths = sorted(set(image_paths))
        if not image_paths:
            raise ValueError(f"No calibration images found in: {folder_path}")

        self.image_paths = image_paths[: max(1, limit)]
        self.transform = transform

    def __len__(self) -> int:
        return len(self.image_paths)

    def __getitem__(self, idx: int) -> torch.Tensor:
        image = Image.open(self.image_paths[idx]).convert("RGB")
        return self.transform(image)


class RepresentativeDataGenerator:
    def __init__(
        self,
        image_folder_path: str,
        input_shape: Tuple[int, int, int],
        batch_size: int,
        num_images: int,
        device: torch.device,
    ) -> None:
        transform = transforms.Compose(
            [
                transforms.Resize((input_shape[1], input_shape[2])),
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=[0.485, 0.456, 0.406],
                    std=[0.229, 0.224, 0.225],
                ),
            ]
        )
        dataset = ImageFolderCalibrationDataset(
            folder_path=image_folder_path,
            transform=transform,
            limit=num_images,
        )

        self.device = device
        self.dataloader = DataLoader(
            dataset,
            batch_size=max(1, batch_size),
            shuffle=False,
            num_workers=0,
            drop_last=False,
        )

    def __call__(self) -> Iterable[List[torch.Tensor]]:
        for batch in self.dataloader:
            yield [batch.to(self.device)]


def claim_gpu_if_needed(device: torch.device) -> None:
    if device.type != "cuda":
        return

    import safe_gpu

    log("CUDA requested; waiting to claim shared GPU slot.")
    while True:
        try:
            safe_gpu.claim_gpus(1)
            log("Successfully claimed GPU.")
            break
        except Exception:
            log("Waiting for free GPU")
            time.sleep(5)
            pass


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        "Sample supernet architectures and keep IMX500-viable candidates"
    )
    parser.add_argument("--num-target-models", type=int, default=50)
    parser.add_argument("--max-attempts", type=int, default=400)
    parser.add_argument("--seed", type=int, default=42)

    parser.add_argument("--num-classes", type=int, default=1000)
    parser.add_argument("--checkpoint", type=str, default="/mnt/matylda5/xmihol00/EUD/supernet/runs_imx500_supernet/20260402_200233/best.pt")
    parser.add_argument("--device", type=str, default="cuda")

    parser.add_argument("--calibration-dir", type=str, default=str(Path(__file__).resolve().parent / "images"))
    parser.add_argument("--num-calibration-images", type=int, default=50)
    parser.add_argument("--calibration-batch-size", type=int, default=1)

    parser.add_argument("--dataset", type=str, default="/mnt/matylda5/xmihol00/datasets/imagenet/sampled")
    parser.add_argument("--eval-batch-size", type=int, default=20)
    parser.add_argument(
        "--images-per-class",
        type=int,
        default=1,
        help="Maximum images to evaluate per class (0 means no limit).",
    )

    parser.add_argument("--tpc-version", type=str, default="1.0")
    parser.add_argument("--opset-version", type=int, default=15)
    parser.add_argument(
        "--compile-timeout-sec",
        type=int,
        default=1800,
        help="Timeout for imxconv-pt compilation in seconds.",
    )
    parser.add_argument(
        "--eval-log-every",
        type=int,
        default=5,
        help="Print evaluation progress every N batches (0 disables).",
    )

    parser.add_argument("--output-dir", type=str, default=str(Path(__file__).resolve().parent / "space_sampling_runs"))
    return parser.parse_args()


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def _extract_state_dict(checkpoint: object) -> Dict[str, torch.Tensor]:
    if isinstance(checkpoint, dict) and "model" in checkpoint:
        state_dict = checkpoint["model"]
    elif isinstance(checkpoint, dict) and "state_dict" in checkpoint:
        state_dict = checkpoint["state_dict"]
    else:
        state_dict = checkpoint

    if not isinstance(state_dict, dict):
        raise ValueError("Unsupported checkpoint format: expected a state_dict-like dict.")
    return state_dict


def _infer_checkpoint_num_classes(state_dict: Dict[str, torch.Tensor]) -> int | None:
    classifier_weight = state_dict.get("classifier.weight")
    if classifier_weight is None:
        return None
    if not hasattr(classifier_weight, "shape") or len(classifier_weight.shape) != 2:
        return None
    return int(classifier_weight.shape[0])


def load_supernet(args: argparse.Namespace, device: torch.device) -> Tuple[IMX500ResNetSupernet, int]:
    effective_num_classes = int(args.num_classes)
    state_dict: Dict[str, torch.Tensor] | None = None
    ckpt_start: float | None = None

    if args.checkpoint:
        ckpt_start = time.perf_counter()
        log(f"Loading checkpoint: {args.checkpoint}")
        checkpoint = torch.load(args.checkpoint, map_location="cpu")
        state_dict = _extract_state_dict(checkpoint)
        checkpoint_num_classes = _infer_checkpoint_num_classes(state_dict)
        if checkpoint_num_classes is not None and checkpoint_num_classes != args.num_classes:
            log(
                "Detected checkpoint classifier class count mismatch: "
                f"--num-classes={args.num_classes}, checkpoint={checkpoint_num_classes}. "
                f"Using checkpoint class count for model loading."
            )
            effective_num_classes = checkpoint_num_classes

    model = create_default_supernet(num_classes=effective_num_classes)

    if state_dict is not None:
        missing, unexpected = model.load_state_dict(state_dict, strict=False)
        log(f"Loaded checkpoint: {args.checkpoint}")
        if missing:
            log(f"Missing keys count: {len(missing)}")
        if unexpected:
            log(f"Unexpected keys count: {len(unexpected)}")
        if ckpt_start is not None:
            log_duration("Checkpoint loading", ckpt_start)

    model.to(device)
    model.eval()
    return model, effective_num_classes


def export_to_onnx(
    model: nn.Module,
    output_path: Path,
    input_shape: Tuple[int, int, int],
    opset_version: int,
    device: torch.device,
) -> None:
    export_start = time.perf_counter()
    log(f"Exporting float ONNX: {output_path}")
    model.eval()
    dummy_input = torch.randn(1, input_shape[0], input_shape[1], input_shape[2], device=device)

    torch.onnx.export(
        model,
        dummy_input,
        str(output_path),
        input_names=["input"],
        output_names=["output"],
        dynamic_axes={"input": {0: "batch_size"}},
        opset_version=opset_version,
        verbose=False,
    )

    onnx_model = onnx.load(str(output_path))
    onnx.checker.check_model(onnx_model)
    log_duration("Float ONNX export", export_start)


def quantize_and_export_onnx(
    model: nn.Module,
    representative_data_gen: RepresentativeDataGenerator,
    output_path: Path,
    tpc_version: str,
) -> Dict[str, object]:
    quant_start = time.perf_counter()
    log(f"Starting PTQ quantization + ONNX export: {output_path}")
    tpc = get_target_platform_capabilities(tpc_version=tpc_version, device_type="imx500")

    quantized_model, quant_info = mct.ptq.pytorch_post_training_quantization(
        in_module=model,
        representative_data_gen=representative_data_gen,
        target_platform_capabilities=tpc,
    )

    mct.exporter.pytorch_export_model(
        model=quantized_model,
        save_model_path=str(output_path),
        repr_dataset=representative_data_gen,
        serialization_format=mct.exporter.PytorchExportSerializationFormat.ONNX,
    )

    log_duration("PTQ + quantized ONNX export", quant_start)

    return quant_info if isinstance(quant_info, dict) else {"info": str(quant_info)}


def run_imx500_compile(
    quantized_onnx_path: Path,
    compile_output_dir: Path,
    timeout_sec: int,
) -> Tuple[bool, str]:
    compile_output_dir.mkdir(parents=True, exist_ok=True)
    compile_start = time.perf_counter()
    log(
        "Starting IMX500 compilation: "
        f"input={quantized_onnx_path} output={compile_output_dir} timeout={timeout_sec}s"
    )
    cmd = [
        "imxconv-pt",
        "-i",
        str(quantized_onnx_path),
        "-o",
        str(compile_output_dir),
        "--no-input-persistency",
        "--overwrite",
    ]
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=max(1, timeout_sec))
    except subprocess.TimeoutExpired as exc:
        timeout_stdout = str(exc.stdout or "")
        timeout_stderr = str(exc.stderr or "")
        combined_output = timeout_stdout + "\n" + timeout_stderr
        log_duration("IMX500 compilation (timeout)", compile_start)
        return False, f"Compilation timed out after {timeout_sec}s.\n{combined_output.strip()}"

    combined_output = (result.stdout or "") + "\n" + (result.stderr or "")
    log_duration("IMX500 compilation", compile_start)
    return result.returncode == 0, combined_output.strip()


def discover_dataset(
    dataset_root: str,
    max_classes: int,
    images_per_class: int,
) -> Tuple[List[DatasetEntry], List[str], int]:
    root = Path(dataset_root)
    if not root.is_dir():
        raise FileNotFoundError(f"Dataset directory not found: {dataset_root}")

    if max_classes <= 0:
        raise ValueError(f"--num-classes must be > 0, got: {max_classes}")
    if images_per_class < 0:
        raise ValueError(f"--images-per-class must be >= 0, got: {images_per_class}")

    all_class_names = sorted([d.name for d in root.iterdir() if d.is_dir()])
    if not all_class_names:
        raise ValueError(f"No class subdirectories found in: {dataset_root}")

    class_names = all_class_names[:max_classes]

    entries: List[DatasetEntry] = []
    for class_index, class_name in enumerate(class_names):
        class_dir = root / class_name
        class_images = [
            image_path
            for image_path in sorted(class_dir.iterdir())
            if image_path.suffix.lower() in SUPPORTED_IMAGE_EXTENSIONS
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
        raise ValueError(f"No supported images found under dataset: {dataset_root}")

    return entries, class_names, len(all_class_names)


def preprocess_image(
    image_path: str,
    input_height: int,
    input_width: int,
    mean: np.ndarray,
    std: np.ndarray,
) -> np.ndarray:
    image = Image.open(image_path).convert("RGB")
    image = image.resize((input_width, input_height), Image.BILINEAR)
    image_array = np.asarray(image, dtype=np.float32) / 255.0
    image_array = (image_array - mean) / std
    image_array = np.transpose(image_array, (2, 0, 1))
    return image_array


def batched(items: Sequence[DatasetEntry], batch_size: int) -> List[List[DatasetEntry]]:
    return [list(items[i : i + batch_size]) for i in range(0, len(items), max(1, batch_size))]


def evaluate_onnx(
    onnx_path: Path,
    dataset_entries: Sequence[DatasetEntry],
    input_resolution: int,
    batch_size: int,
    selected_num_classes: int,
    eval_log_every: int,
) -> Dict[str, float]:
    eval_start = time.perf_counter()
    log(
        f"Starting evaluation for {onnx_path.name}: "
        f"images={len(dataset_entries)} batch_size={batch_size}"
    )
    available_providers = ort.get_available_providers()
    preferred = ["CUDAExecutionProvider", "CPUExecutionProvider"]
    providers = [p for p in preferred if p in available_providers] or available_providers

    session_options = get_ort_session_options() if get_ort_session_options is not None else None
    session = ort.InferenceSession(str(onnx_path), sess_options=session_options, providers=providers)

    input_name = session.get_inputs()[0].name
    output_name = session.get_outputs()[0].name

    mean = np.array([0.485, 0.456, 0.406], dtype=np.float32).reshape(1, 1, 3)
    std = np.array([0.229, 0.224, 0.225], dtype=np.float32).reshape(1, 1, 3)

    total = 0
    correct = 0
    num_batches = max(1, (len(dataset_entries) + max(1, batch_size) - 1) // max(1, batch_size))

    for batch_idx, batch_entries in enumerate(batched(dataset_entries, batch_size), start=1):
        image_tensors = [
            preprocess_image(
                image_path=entry.image_path,
                input_height=input_resolution,
                input_width=input_resolution,
                mean=mean,
                std=std,
            )
            for entry in batch_entries
        ]
        inputs = np.stack(image_tensors, axis=0).astype(np.float32)
        logits = session.run([output_name], {input_name: inputs})[0]
        logits = np.asarray(logits).reshape(len(batch_entries), -1)
        if logits.shape[1] > selected_num_classes:
            logits = logits[:, :selected_num_classes]
        predictions = np.argmax(logits, axis=1)

        for entry, pred in zip(batch_entries, predictions):
            total += 1
            if int(pred) == entry.class_index:
                correct += 1

        if eval_log_every > 0 and (batch_idx % eval_log_every == 0 or batch_idx == num_batches):
            partial_acc = (100.0 * correct / total) if total else 0.0
            log(
                f"Eval progress {onnx_path.name}: batch {batch_idx}/{num_batches}, "
                f"images={total}/{len(dataset_entries)}, acc1={partial_acc:.2f}%"
            )

    acc1 = (100.0 * correct / total) if total else 0.0
    log_duration(f"Evaluation ({onnx_path.name})", eval_start)
    return {"acc1": acc1, "correct": float(correct), "total": float(total)}


def as_jsonable(obj: object) -> object:
    if isinstance(obj, dict):
        return {str(k): as_jsonable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [as_jsonable(v) for v in obj]
    if isinstance(obj, (str, int, float, bool)) or obj is None:
        return obj
    return str(obj)


def main() -> None:
    run_start = time.perf_counter()
    args = parse_args()
    set_seed(args.seed)

    log(
        "Starting space sampling run: "
        f"targets={args.num_target_models}, max_attempts={args.max_attempts}, "
        f"seed={args.seed}, eval_batch_size={args.eval_batch_size}"
    )

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    log(f"Output directory: {output_dir}")

    dataset_entries, class_names, total_available_classes = discover_dataset(
        dataset_root=args.dataset,
        max_classes=args.num_classes,
        images_per_class=args.images_per_class,
    )
    log(
        "Discovered dataset "
        f"available_classes={total_available_classes} "
        f"selected_classes={len(class_names)} "
        f"images={len(dataset_entries)}"
    )
    if len(class_names) < total_available_classes:
        log(
            f"Using first {len(class_names)} classes alphabetically due to --num-classes={args.num_classes}."
        )

    calibration_dir = Path(args.calibration_dir)
    if not calibration_dir.is_dir():
        raise FileNotFoundError(f"Calibration image directory not found: {calibration_dir}")

    requested_device = torch.device(args.device)
    device = requested_device if (requested_device.type != "cuda" or torch.cuda.is_available()) else torch.device("cpu")
    if requested_device.type == "cuda" and device.type != "cuda":
        log("CUDA requested but not available. Falling back to CPU.")
    log(f"Using torch device: {device}")

    claim_gpu_if_needed(device)

    supernet, model_num_classes = load_supernet(args, device)

    successful = 0
    attempts = 0
    seen_configs = set()
    records: List[Dict[str, object]] = []

    while successful < args.num_target_models and attempts < args.max_attempts:
        attempts += 1
        attempt_start = time.perf_counter()
        sampled_config = supernet.sample_subnet(mode="random")
        cfg_key = supernet.config_to_json(sampled_config)
        if cfg_key in seen_configs:
            log(f"Attempt {attempts}: duplicate sampled config, skipping.")
            continue
        seen_configs.add(cfg_key)

        sample_id = f"sample_{successful:04d}_attempt_{attempts:04d}"
        sample_dir = output_dir / sample_id
        sample_dir.mkdir(parents=True, exist_ok=True)

        supernet.set_active_subnet(sampled_config)
        static_model = build_static_subnet_model(
            supernet=supernet,
            config=sampled_config,
            num_classes=model_num_classes,
            device=device,
        )

        input_shape = (3, sampled_config.resolution, sampled_config.resolution)
        float_onnx = sample_dir / "model_float.onnx"
        quant_onnx = sample_dir / "model_quant.onnx"

        sample_record: Dict[str, object] = {
            "sample_id": sample_id,
            "attempt": attempts,
            "config": sampled_config.to_dict(),
            "float_onnx": str(float_onnx),
            "quant_onnx": str(quant_onnx),
            "compiled": False,
        }

        try:
            log(
                f"[{sample_id}] attempt={attempts} started | "
                f"res={sampled_config.resolution}, stem={sampled_config.stem_width}, "
                f"depths={sampled_config.stage_depths}, widths={sampled_config.stage_widths}"
            )

            export_to_onnx(
                model=static_model,
                output_path=float_onnx,
                input_shape=input_shape,
                opset_version=args.opset_version,
                device=device,
            )

            rep_data_gen = RepresentativeDataGenerator(
                image_folder_path=str(calibration_dir),
                input_shape=input_shape,
                batch_size=args.calibration_batch_size,
                num_images=args.num_calibration_images,
                device=device,
            )

            quant_info = quantize_and_export_onnx(
                model=static_model,
                representative_data_gen=rep_data_gen,
                output_path=quant_onnx,
                tpc_version=args.tpc_version,
            )
            sample_record["quantization_info"] = as_jsonable(quant_info)

            compiled, compile_log = run_imx500_compile(
                quantized_onnx_path=quant_onnx,
                compile_output_dir=sample_dir / "imx500_output",
                timeout_sec=args.compile_timeout_sec,
            )
            sample_record["compiled"] = compiled
            sample_record["compile_log"] = compile_log

            if not compiled:
                sample_record["discard_reason"] = "Compilation failed (likely memory/resource limits)"
                records.append(sample_record)
                log(f"[{sample_id}] discarded: compilation failed or timed out")
                continue

            float_metrics = evaluate_onnx(
                onnx_path=float_onnx,
                dataset_entries=dataset_entries,
                input_resolution=sampled_config.resolution,
                batch_size=args.eval_batch_size,
                selected_num_classes=len(class_names),
                eval_log_every=args.eval_log_every,
            )
            quant_metrics = evaluate_onnx(
                onnx_path=quant_onnx,
                dataset_entries=dataset_entries,
                input_resolution=sampled_config.resolution,
                batch_size=args.eval_batch_size,
                selected_num_classes=len(class_names),
                eval_log_every=args.eval_log_every,
            )
            sample_record["float_eval"] = float_metrics
            sample_record["quant_eval"] = quant_metrics

            records.append(sample_record)
            successful += 1
            log(
                f"[{sample_id}] accepted | acc1 float={float_metrics['acc1']:.2f}% "
                f"quant={quant_metrics['acc1']:.2f}%"
            )
            log_duration(f"[{sample_id}] total attempt", attempt_start)

        except Exception as exc:
            sample_record["discard_reason"] = f"Pipeline exception: {exc}"
            records.append(sample_record)
            log(f"[{sample_id}] discarded due to error: {exc}")
            log_duration(f"[{sample_id}] failed attempt", attempt_start)

        with (output_dir / "sampling_results.json").open("w", encoding="utf-8") as fp:
            json.dump(as_jsonable(records), fp, indent=2)

    summary = {
        "requested_models": args.num_target_models,
        "successful_models": successful,
        "attempts": attempts,
        "unique_sampled": len(seen_configs),
        "dataset": args.dataset,
        "available_classes": total_available_classes,
        "selected_classes": len(class_names),
        "model_num_classes": model_num_classes,
        "images_per_class": args.images_per_class,
        "calibration_dir": str(calibration_dir),
        "results_file": str(output_dir / "sampling_results.json"),
    }

    with (output_dir / "summary.json").open("w", encoding="utf-8") as fp:
        json.dump(summary, fp, indent=2)

    log("Sampling finished.")
    print(json.dumps(summary, indent=2), flush=True)
    log_duration("Full run", run_start)


if __name__ == "__main__":
    main()
