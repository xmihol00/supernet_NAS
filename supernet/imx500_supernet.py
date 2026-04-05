from __future__ import annotations

import json
import math
import random
from dataclasses import asdict, dataclass
from typing import Dict, Sequence, Tuple, cast

import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass(frozen=True)
class SubnetConfig:
    resolution: int
    stem_width: int
    stage_depths: Tuple[int, ...]
    stage_widths: Tuple[int, ...]

    def to_dict(self) -> Dict[str, object]:
        return asdict(self)

    @staticmethod
    def from_dict(config: Dict[str, object]) -> "SubnetConfig":
        stage_depths = tuple(cast(int, v) for v in cast(Sequence[int], config["stage_depths"]))
        stage_widths = tuple(cast(int, v) for v in cast(Sequence[int], config["stage_widths"]))
        return SubnetConfig(
            resolution=int(cast(int, config["resolution"])),
            stem_width=int(cast(int, config["stem_width"])),
            stage_depths=stage_depths,
            stage_widths=stage_widths,
        )


class DynamicBatchNorm2d(nn.BatchNorm2d):
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        channels = x.shape[1]
        if channels == self.num_features:
            return super().forward(x)

        running_mean = self.running_mean[:channels] if self.running_mean is not None else None
        running_var = self.running_var[:channels] if self.running_var is not None else None
        weight = self.weight[:channels] if self.weight is not None else None
        bias = self.bias[:channels] if self.bias is not None else None

        return F.batch_norm(
            x,
            running_mean,
            running_var,
            weight,
            bias,
            self.training or not self.track_running_stats,
            self.momentum,
            self.eps,
        )


class DynamicConv2d(nn.Module):
    def __init__(
        self,
        max_in_channels: int,
        max_out_channels: int,
        kernel_size: int,
        stride: int = 1,
        padding: int = 0,
        groups: int = 1,
        bias: bool = False,
    ) -> None:
        super().__init__()
        self.max_in_channels = max_in_channels
        self.max_out_channels = max_out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.groups = groups
        self.weight = nn.Parameter(
            torch.randn(max_out_channels, max_in_channels // groups, kernel_size, kernel_size)
        )
        self.bias = nn.Parameter(torch.zeros(max_out_channels)) if bias else None
        self.reset_parameters()

    def reset_parameters(self) -> None:
        nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        if self.bias is not None:
            fan_in = self.weight.size(1) * self.weight.size(2) * self.weight.size(3)
            bound = 1 / math.sqrt(fan_in)
            nn.init.uniform_(self.bias, -bound, bound)

    def forward(self, x: torch.Tensor, out_channels: int | None = None) -> torch.Tensor:
        in_channels = x.shape[1]
        if out_channels is None:
            out_channels = self.max_out_channels

        out_channels = min(out_channels, self.max_out_channels)
        weight = self.weight[:out_channels, : in_channels // self.groups]
        bias = self.bias[:out_channels] if self.bias is not None else None

        return F.conv2d(
            x,
            weight,
            bias,
            stride=self.stride,
            padding=self.padding,
            groups=self.groups,
        )


class DynamicLinear(nn.Module):
    def __init__(self, max_in_features: int, out_features: int, bias: bool = True) -> None:
        super().__init__()
        self.max_in_features = max_in_features
        self.out_features = out_features
        self.weight = nn.Parameter(torch.randn(out_features, max_in_features))
        self.bias = nn.Parameter(torch.zeros(out_features)) if bias else None
        self.reset_parameters()

    def reset_parameters(self) -> None:
        nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        if self.bias is not None:
            bound = 1 / math.sqrt(self.max_in_features)
            nn.init.uniform_(self.bias, -bound, bound)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        in_features = x.shape[-1]
        weight = self.weight[:, :in_features]
        return F.linear(x, weight, self.bias)


class DynamicBasicBlock(nn.Module):
    def __init__(self, max_in_channels: int, max_out_channels: int, stride: int) -> None:
        super().__init__()
        self.stride = stride
        self.conv1 = DynamicConv2d(max_in_channels, max_out_channels, 3, stride=stride, padding=1, bias=False)
        self.bn1 = DynamicBatchNorm2d(max_out_channels)
        self.conv2 = DynamicConv2d(max_out_channels, max_out_channels, 3, stride=1, padding=1, bias=False)
        self.bn2 = DynamicBatchNorm2d(max_out_channels)

        self.downsample_conv = DynamicConv2d(
            max_in_channels,
            max_out_channels,
            kernel_size=1,
            stride=stride,
            padding=0,
            bias=False,
        )
        self.downsample_bn = DynamicBatchNorm2d(max_out_channels)
        self.activation = nn.ReLU(inplace=True)

    def forward(self, x: torch.Tensor, out_channels: int) -> torch.Tensor:
        identity = x

        out = self.conv1(x, out_channels)
        out = self.bn1(out)
        out = self.activation(out)

        out = self.conv2(out, out_channels)
        out = self.bn2(out)

        if self.stride != 1 or identity.shape[1] != out_channels:
            identity = self.downsample_conv(identity, out_channels)
            identity = self.downsample_bn(identity)

        out = out + identity
        out = self.activation(out)
        return out


class DynamicStage(nn.Module):
    def __init__(self, max_in_channels: int, max_out_channels: int, max_depth: int, stride: int) -> None:
        super().__init__()
        self.blocks = nn.ModuleList()
        for depth_idx in range(max_depth):
            block_stride = stride if depth_idx == 0 else 1
            block_in = max_in_channels if depth_idx == 0 else max_out_channels
            self.blocks.append(DynamicBasicBlock(block_in, max_out_channels, stride=block_stride))

    def forward(self, x: torch.Tensor, depth: int, out_channels: int) -> torch.Tensor:
        for block_idx in range(depth):
            x = self.blocks[block_idx](x, out_channels)
        return x


class IMX500ResNetSupernet(nn.Module):
    stage_strides = (1, 2, 2, 2)

    def __init__(
        self,
        num_classes: int = 1000,
        resolution_candidates: Sequence[int] = (192, 224, 256, 288),
        stem_width_candidates: Sequence[int] = (24, 32, 40),
        stage_depth_candidates: Sequence[Sequence[int]] = ((1, 2, 3), (1, 2, 3, 4), (1, 2, 3, 4, 5, 6), (1, 2, 3)),
        stage_width_candidates: Sequence[Sequence[int]] = ((48, 64), (96, 128), (160, 192, 224), (224, 256, 288)),
    ) -> None:
        super().__init__()
        if len(stage_depth_candidates) != 4 or len(stage_width_candidates) != 4:
            raise ValueError("Expected 4 stage depth/width candidate sets.")

        self.num_classes = num_classes
        self.resolution_candidates = tuple(sorted(int(v) for v in resolution_candidates))
        self.stem_width_candidates = tuple(sorted(int(v) for v in stem_width_candidates))
        self.stage_depth_candidates = tuple(tuple(sorted(int(v) for v in stage)) for stage in stage_depth_candidates)
        self.stage_width_candidates = tuple(tuple(sorted(int(v) for v in stage)) for stage in stage_width_candidates)

        max_stem = max(self.stem_width_candidates)
        max_widths = [max(stage) for stage in self.stage_width_candidates]
        max_depths = [max(stage) for stage in self.stage_depth_candidates]

        self.stem_conv = DynamicConv2d(3, max_stem, kernel_size=3, stride=2, padding=1, bias=False)
        self.stem_bn = DynamicBatchNorm2d(max_stem)
        self.stem_act = nn.ReLU(inplace=True)

        stages = []
        prev_width = max_stem
        for stage_idx in range(4):
            stages.append(
                DynamicStage(
                    max_in_channels=prev_width,
                    max_out_channels=max_widths[stage_idx],
                    max_depth=max_depths[stage_idx],
                    stride=self.stage_strides[stage_idx],
                )
            )
            prev_width = max_widths[stage_idx]
        self.stages = nn.ModuleList(stages)

        self.global_pool = nn.AdaptiveAvgPool2d(1)
        self.classifier = DynamicLinear(max_widths[-1], num_classes)

        self.active_subnet = self.max_subnet_config()

    def max_subnet_config(self) -> SubnetConfig:
        return SubnetConfig(
            resolution=max(self.resolution_candidates),
            stem_width=max(self.stem_width_candidates),
            stage_depths=tuple(max(stage) for stage in self.stage_depth_candidates),
            stage_widths=tuple(max(stage) for stage in self.stage_width_candidates),
        )

    def min_subnet_config(self) -> SubnetConfig:
        return SubnetConfig(
            resolution=min(self.resolution_candidates),
            stem_width=min(self.stem_width_candidates),
            stage_depths=tuple(min(stage) for stage in self.stage_depth_candidates),
            stage_widths=tuple(min(stage) for stage in self.stage_width_candidates),
        )

    def random_subnet_config(self) -> SubnetConfig:
        return SubnetConfig(
            resolution=random.choice(self.resolution_candidates),
            stem_width=random.choice(self.stem_width_candidates),
            stage_depths=tuple(random.choice(stage) for stage in self.stage_depth_candidates),
            stage_widths=tuple(random.choice(stage) for stage in self.stage_width_candidates),
        )

    def set_active_subnet(self, config: SubnetConfig) -> None:
        self.active_subnet = config

    def sample_subnet(
        self,
        mode: str = "random",
        target_total_bytes: int | None = None,
        tolerance_ratio: float = 0.25,
        tolerance_ratio_low: float | None = None,
        tolerance_ratio_high: float | None = None,
        max_trials: int = 48,
        firmware_bytes: int = 1_572_864,
        activation_bytes: int = 1,
        working_memory_factor: float = 2.0,
    ) -> SubnetConfig:
        if mode == "max":
            config = self.max_subnet_config()
            self.set_active_subnet(config)
            return config
        if mode == "min":
            config = self.min_subnet_config()
            self.set_active_subnet(config)
            return config

        if target_total_bytes is None:
            config = self.random_subnet_config()
            self.set_active_subnet(config)
            return config

        if tolerance_ratio_low is None:
            tolerance_ratio_low = tolerance_ratio
        if tolerance_ratio_high is None:
            tolerance_ratio_high = tolerance_ratio

        tolerance_low = int(target_total_bytes * tolerance_ratio_low)
        tolerance_high = int(target_total_bytes * tolerance_ratio_high)
        lower_bound = target_total_bytes - tolerance_low
        upper_bound = target_total_bytes + tolerance_high

        best_config = None
        best_distance = float("inf")

        for _ in range(max_trials):
            candidate = self.random_subnet_config()
            resources = self.estimate_subnet_resources(
                candidate,
                firmware_bytes=firmware_bytes,
                activation_bytes=activation_bytes,
                working_memory_factor=working_memory_factor,
            )
            candidate_total = int(resources["total_estimated_bytes"])
            if candidate_total < lower_bound:
                distance = float(lower_bound - candidate_total)
            elif candidate_total > upper_bound:
                distance = float(candidate_total - upper_bound)
            else:
                distance = 0.0

            if distance < best_distance:
                best_distance = distance
                best_config = candidate
            if distance == 0.0:
                best_config = candidate
                break

        assert best_config is not None
        self.set_active_subnet(best_config)
        return best_config

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        config = self.active_subnet
        if x.shape[-1] != config.resolution or x.shape[-2] != config.resolution:
            x = F.interpolate(x, size=(config.resolution, config.resolution), mode="bilinear", align_corners=False)

        x = self.stem_conv(x, config.stem_width)
        x = self.stem_bn(x)
        x = self.stem_act(x)

        for stage_idx, stage in enumerate(self.stages):
            x = stage(x, depth=config.stage_depths[stage_idx], out_channels=config.stage_widths[stage_idx])

        x = self.global_pool(x)
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        return x

    def _estimate_param_count(self, config: SubnetConfig) -> int:
        param_count = 0

        stem_width = config.stem_width
        stage_depths = config.stage_depths
        stage_widths = config.stage_widths

        # Stem conv + BN
        param_count += 3 * stem_width * 3 * 3
        param_count += 2 * stem_width

        prev_width = stem_width
        for stage_idx in range(4):
            out_width = stage_widths[stage_idx]
            depth = stage_depths[stage_idx]
            stride = self.stage_strides[stage_idx]

            for block_idx in range(depth):
                block_stride = stride if block_idx == 0 else 1
                in_width = prev_width if block_idx == 0 else out_width

                # conv1 + bn1
                param_count += out_width * in_width * 3 * 3
                param_count += 2 * out_width

                # conv2 + bn2
                param_count += out_width * out_width * 3 * 3
                param_count += 2 * out_width

                # projection path when needed
                if block_stride != 1 or in_width != out_width:
                    param_count += out_width * in_width
                    param_count += 2 * out_width

            prev_width = out_width

        # classifier
        param_count += self.num_classes * stage_widths[-1]
        param_count += self.num_classes

        return int(param_count)

    def estimate_subnet_resources(
        self,
        config: SubnetConfig,
        firmware_bytes: int = 1_572_864,
        weight_bytes: int = 1,
        activation_bytes: int = 1,
        working_memory_factor: float = 2.0,
    ) -> Dict[str, int]:
        params = self._estimate_param_count(config)
        weight_memory = params * int(weight_bytes)

        # Very lightweight feature-map estimate for peak activation usage.
        resolution = config.resolution
        h = resolution // 2
        w = resolution // 2
        peak_elements = h * w * config.stem_width

        for stage_idx in range(4):
            if self.stage_strides[stage_idx] == 2:
                h = max(1, h // 2)
                w = max(1, w // 2)
            stage_elements = h * w * config.stage_widths[stage_idx]
            peak_elements = max(peak_elements, stage_elements)

        peak_activation_memory = peak_elements * int(activation_bytes)
        working_memory = int(working_memory_factor * peak_activation_memory)
        total = firmware_bytes + weight_memory + working_memory

        return {
            "params": int(params),
            "weights_bytes": int(weight_memory),
            "peak_activation_bytes": int(peak_activation_memory),
            "working_memory_bytes": int(working_memory),
            "firmware_bytes": int(firmware_bytes),
            "total_estimated_bytes": int(total),
        }

    @staticmethod
    def config_to_json(config: SubnetConfig) -> str:
        return json.dumps(config.to_dict(), sort_keys=True)


def create_default_supernet(num_classes: int = 1000) -> IMX500ResNetSupernet:
    return IMX500ResNetSupernet(num_classes=num_classes)
