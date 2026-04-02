# IMX500-Oriented Supernet for Image Classification

This directory contains a **single-GPU**, easy-to-read implementation of a dynamic supernet inspired by AlphaNet/AttentiveNAS ideas, but tailored for your thesis objective:

- search candidate subnets around the **IMX500 memory envelope** (about **8,388,480 B total**),
- train one shared supernet once,
- then drive evolutionary / genetic search over sampled subnet architectures.

The implementation is intentionally Python-first (no YAML-driven runtime model creation), with clear command line defaults and file-based logging for long server runs.

---

## Files

- `imx500_supernet.py`: ResNet-inspired dynamic supernet and resource estimator.
- `train_supernet.py`: single-GPU supernet training entry point.
- `sample_subnets.py`: candidate sampler for GA population seeding around memory target.

---

## 1) Architecture design (ResNet-inspired dynamic supernet)

## 1.1 Motivation

Classical ResNets are robust and deployment-friendly. For hardware-aware NAS, they are attractive because:

1. block topology is simple and stable,
2. memory/compute trends are predictable when changing depth/width,
3. quantization is usually easier than with highly exotic operators.

This supernet keeps those strengths while adding dynamic dimensions for NAS.

## 1.2 Supernet macro-structure

The supernet uses a stem + 4 residual stages + classifier:

1. **Stem**: 3x3 convolution, stride 2, dynamic output channels.
2. **Stage 1..4**: dynamic stacks of `BasicBlock` residual units.
3. **Global average pool**.
4. **Dynamic linear classifier**.

Downsampling strides per stage are `(1, 2, 2, 2)`.

## 1.3 Dynamic search dimensions

The current default search space:

- input resolution: `[192, 224, 256]`
- stem width: `[24, 32, 40]`
- stage depths:
	- stage 1: `[1, 2, 3]`
	- stage 2: `[1, 2, 3, 4]`
	- stage 3: `[1, 2, 3, 4, 5]`
	- stage 4: `[1, 2, 3]`
- stage widths:
	- stage 1: `[48, 64]`
	- stage 2: `[96, 128]`
	- stage 3: `[160, 192, 224]`
	- stage 4: `[224, 256, 288]`

Each sampled subnet is represented by:

- `resolution`
- `stem_width`
- `stage_depths` (4 integers)
- `stage_widths` (4 integers)

## 1.4 Dynamic operator implementation

Instead of rebuilding models repeatedly, max-size operators are instantiated once:

- `DynamicConv2d`: slices in/out channels from max weight tensor.
- `DynamicBatchNorm2d`: applies channel-wise slicing for BN stats/params.
- `DynamicLinear`: slices input dimension for classifier.

Residual blocks are dynamic BasicBlocks with optional projection (`1x1`) on mismatch/stride.

This design allows many subnet configurations to share one set of weights (supernet training).

---

## 2) IMX500 memory-aware objective

## 2.1 Constraint target

IMX500 deployment budget in your request:

- total available memory (firmware + weights + working memory):

$$
M_{budget} = 8,388,480 \text{ bytes}
$$

Subnets near this region are preferred for NAS; some oversized candidates are acceptable and may fail compiler checks (as expected).

## 2.2 Resource estimator used during sampling

For a subnet config, `estimate_subnet_resources` computes:

1. **Parameter count** (conv/bn/fc).
2. **Weight bytes** (`params * weight_bytes`, int8 default = 1 B/param).
3. **Peak activation bytes** from a lightweight stage-wise feature map estimate.
4. **Working memory bytes** using:

$$
M_{work} = \alpha \cdot M_{peak\_activation}
$$

where `\alpha = working_memory_factor` (default `2.0`).

5. **Total estimate**:

$$
M_{total} = M_{firmware} + M_{weights} + M_{work}
$$

with default `M_firmware = 1,572,864` B.

This is a fast heuristic for NAS pre-filtering. Final hardware compiler checks remain the ground truth.

## 2.3 Targeted subnet sampling

`sample_subnet(...)` supports random sampling with a memory target:

- repeatedly sample random configs,
- evaluate distance to target bytes,
- keep the closest candidate (or stop early when within tolerance).

This makes training/search biased toward the deployable IMX500 region.

---

## 3) Training strategy (`train_supernet.py`)

## 3.1 Single-GPU, no DDP

The training pipeline is intentionally non-distributed:

- one GPU (`--device cuda` by default),
- optional AMP mixed precision,
- checkpoint + metric logging to output directory.

## 3.2 Dataset handling

Expected dataset format: ImageFolder with class subdirectories.

Default path:

- `/mnt/matylda5/xmihol00/datasets/imagenet/train`

Validation split:

- random split with `--val-split 0.15` (15%).

## 3.3 Augmentation / preprocessing

Training:

- `RandomResizedCrop(max_resolution)`
- `RandomHorizontalFlip`
- ImageNet normalization

Validation:

- resize to `1.14 * max_resolution`
- center crop
- ImageNet normalization

Inside the model, input is resized to active subnet resolution if needed.

## 3.4 Supernet optimization (sandwich rule style)

Per mini-batch:

1. Sample and train **max subnet** with hard labels.
2. Use max subnet logits as teacher for inplace distillation.
3. Train several additional subnets (`--num-arch-training`) including:
	 - random target-aware subnet(s),
	 - minimum subnet at the end when sandwich rule is enabled.

Losses:

- hard loss: `CrossEntropyLoss(label_smoothing)`
- soft loss: KL-style soft-target loss (`SoftTargetKLLoss`)

Other training details:

- SGD + Nesterov,
- cosine LR with warmup,
- gradient clipping,
- top-1/top-5 metrics.

## 3.5 Logging and outputs

For long runs on server:

- progress bars via `tqdm`,
- summary logs via Python `logging`,
- file outputs in `output_dir/run_timestamp/`:
	- `train.log`
	- `metrics.json`
	- `supernet_profile.json`
	- checkpoints (`checkpoint_best.pt`, periodic checkpoints)

---

## 4) Candidate generation for GA (`sample_subnets.py`)

Use `sample_subnets.py` to seed initial GA populations near IMX500 constraints.

Output JSON includes for each candidate:

- subnet config,
- estimated resources,
- distance to target total bytes.

Candidates are sorted by closeness to memory target.

---

## 5) Quantization and deployment notes

Your final deployable subnet should be calibrated and compiled with the target toolchain.

Recommended practical flow:

1. Train supernet.
2. Run GA/evolution using memory estimate + validation score.
3. Export top candidates.
4. Perform PTQ/QAT pipeline to int8.
5. Compile candidates for IMX500; reject those failing memory/layout constraints.
6. Re-rank by real hardware-valid metrics (accuracy + latency + compile pass).

The current estimator is intentionally simple and fast; adjust `firmware_bytes`, `working_memory_factor`, and acceptance tolerance as you gain empirical compiler feedback.

---

## 6) Command line examples

Train supernet (single GPU, defaults suitable for your dataset path):

```bash
cd /home/david/projs/supernet_NAS/supernet
python train_supernet.py
```

Train with custom output and memory targeting:

```bash
cd /home/david/projs/supernet_NAS/supernet
python train_supernet.py \
	--output-dir ./runs_thesis \
	--target-total-bytes 8388480 \
	--target-tolerance-ratio 0.20 \
	--epochs 180 \
	--batch-size 96
```

Sample candidate subnets for GA initialization:

```bash
cd /home/david/projs/supernet_NAS/supernet
python sample_subnets.py \
	--num-samples 500 \
	--target-total-bytes 8388480 \
	--tolerance-ratio 0.20 \
	--output ./ga_seed_candidates.json
```

---

## 7) Hyperparameter tuning suggestions for a 24 GB GPU

- Start with `batch_size=96`; increase if memory allows.
- Increase `num_arch_training` gradually (`3 -> 4 -> 5`) to improve supernet robustness.
- Use `epochs=120` for smoke baseline, then `180-360` for thesis-grade training.
- If random subnets are too often outside useful memory region, reduce tolerance ratio or increase sampling trials in `imx500_supernet.py`.
- Track both min/max subnet validation trends; divergence usually indicates insufficient sandwich coverage or unstable LR.

---

## 8) Limitations and next thesis steps

Current version priorities readability and reproducibility. For higher fidelity hardware co-design, recommended extensions:

1. Replace heuristic memory model with calibrated proxy learned from real compiler outputs.
2. Add latency estimator and multi-objective fitness: accuracy vs. memory vs. latency.
3. Add BN recalibration pass before subnet evaluation.
4. Add optional class-balanced split and deterministic per-class validation protocol.
5. Add export helper producing frozen subnet PyTorch modules for deployment toolchain.

These can be added incrementally without changing the core training script structure.
