# Upgrade Plan: MobileNet-Small → EfficientNet-B0
### What we have · What we need to build

> Baseline: MobileNet-V3-Small on UTKFace → 4.73 MAE (val), poor MAE on real-world images.
> Goal: EfficientNet-B0 on merged multi-dataset → strong real-world age + gender accuracy.

This document maps every stage of the upgrade plan against the actual code that exists today.
Use it as a **gap analysis and execution checklist**.

---

## Legend

| Symbol | Meaning |
|--------|---------|
| ✅ | Fully implemented, production-ready |
| 🟡 | Partially implemented, needs extension |
| ❌ | Does not exist — must be built |
| ⚠️  | Exists but has a known bug or design flaw to fix first |

---

## Stage 0 — Pre-flight: Fix Pipeline Before Switching Architecture

### 0.1 Face crop consistency (train == inference)

| Status | Component | Location |
|--------|-----------|----------|
| ❌ | YOLO face detector integration | Not in codebase anywhere |
| ❌ | Consistent crop margin / aspect-ratio convention | Not codified |
| ❌ | Inference crop pipeline matching training crop | No inference script uses a detector |

**What exists:** Raw UTKFace images are used as-is. `process_data_set.py` has `verify_and_clean_images()` and `stratified_split()` but **no face detection step**. `prepare_split_ds.py` downloads UTKFace from Kaggle and splits into folds — again no crop standardization.

**What to build:** A preprocessing script (`src/scripts/crop_pipeline.py`) that runs a YOLO face detector (e.g. `ultralytics` YOLOv8n-face) over raw dataset images, applies a fixed margin (e.g. +25% around bbox), saves crops, and stores `crop_metadata.json` (bbox, confidence, yaw/pitch/roll estimate per image).

---

### 0.2 Identity-based train/val/test split

| Status | Component | Location |
|--------|-----------|----------|
| ⚠️  | Random split by image index | `data_defs.py:368-390`, `prepare_split_ds.py:80-86` |
| ❌ | Identity-aware split (no person leaks across splits) | Not implemented |

**What exists:** `AgeGenderDataModule.setup()` does an 80/20 split using `torch.utils.data.random_split` on image indices (line 371–375). `prepare_split_ds.py` also does a random fold split by image. UTKFace filenames encode identity in the `_{race}_{number}.jpg` suffix — identity is not used for splitting.

**What to build:** Parse the identity component from each dataset's filename/metadata before splitting. Group all images per identity, then split groups (not images) into train/val/test. For multi-dataset this is critical to prevent leakage.

---

### 0.3 Real-world test set

| Status | Component | Location |
|--------|-----------|----------|
| ❌ | Manually labeled real-world test set (200-500 images) | Not in codebase |
| ❌ | Stratified by pose bucket (frontal/3-quarter/profile) | Not in codebase |

**What to build:** Manual labeling effort + storage convention. This is the only true scoreboard. Nothing in code can substitute for it.

---

## Stage 1 — Data Pipeline

### 1.1 Dataset merging

| Status | Component | Location |
|--------|-----------|----------|
| ✅ | UTKFace download + fold split | `utils/prepare_split_ds.py` |
| ✅ | Duplicate detection (perceptual hash) | `src/scripts/process_data_set.py:119-164` |
| ✅ | Corrupt image verification + removal | `src/scripts/process_data_set.py:29-53` |
| ✅ | Stratified split by category | `src/scripts/process_data_set.py:167-213` |
| ❌ | FairFace download + label adapter | Not in codebase |
| ❌ | APPA-REAL download + label adapter | Not in codebase |
| ❌ | IMDB-WIKI download + quality filter | Not in codebase |
| ❌ | AgeDB download + label adapter | Not in codebase |
| ❌ | Unified label schema (age in years, gender 0/1) | Implied by UTKFace filename convention only |
| ❌ | Per-dataset sampling weights for DataLoader | `WeightedAgeGenderSampler` exists but is broken (see below) |

**What exists:** The data cleaning tools in `process_data_set.py` are solid and reusable — `verify_and_clean_images`, `find_duplicates`, `stratified_split`, and `verify_dataset` can all be adapted for each new dataset.

**What to build:** A `src/scripts/merge_datasets.py` that:
1. Downloads / locates each dataset.
2. Normalizes filenames or generates a master CSV (`image_path, age, gender, dataset_name, identity_id`).
3. Runs `verify_and_clean_images` and `find_duplicates` per dataset.
4. Produces one unified directory (or symlinked structure) with a manifest CSV.

---

### 1.2–1.3 Face cropping & alignment

| Status | Component | Location |
|--------|-----------|----------|
| ❌ | YOLO face detection on training data | Not in codebase |
| ❌ | Roll correction (in-plane rotation via landmarks) | Not in codebase |
| ❌ | Crop metadata storage per image | Not in codebase |

**What to build:** Add to `crop_pipeline.py` (see Stage 0.1). Landmarks can come from `mediapipe` or a lightweight dlib model. Store `yaw`, `pitch`, `roll`, `confidence`, `bbox` in a sidecar CSV.

---

### 1.4 Label harmonization

| Status | Component | Location |
|--------|-----------|----------|
| 🟡 | Age/gender extraction from UTKFace filename | `src/scripts/ds_utils.py:12-25` |
| ❌ | Age extraction for FairFace / APPA-REAL / IMDB-WIKI | Not implemented |
| ❌ | Pre-binned age labels in unified manifest | Not implemented |

**What exists:** `get_image_data()` in `ds_utils.py` parses `age` and `gender` from the UTKFace filename format `{age}_{gender}_...jpg`. This only works for UTKFace. Other datasets use CSV/JSON metadata.

**What to build:** Adapter functions per dataset in a new `src/scripts/label_adapters.py`, each returning a normalized `(image_path, age_years, gender_int)` tuple. The master manifest CSV built by `merge_datasets.py` uses these.

---

### 1.5 Quality filtering (IMDB-WIKI specific)

| Status | Component | Location |
|--------|-----------|----------|
| ✅ | Laplacian variance blur check (can add easily) | `process_data_set.py` — `verify_and_clean_images` resizes, can extend |
| ✅ | Perceptual hash duplicate removal | `process_data_set.py:119-164` |
| ❌ | Face-detector confidence threshold filter | Not implemented |
| ❌ | Age plausibility bounds filter | Not implemented |

**What to build:** Extend `verify_and_clean_images` with a Laplacian variance cutoff and face-detector confidence check (computed during Stage 1.2 crop pipeline). Add age plausibility filter (e.g., drop `age < 0` or `age > 100`) to the manifest builder.

---

### 1.6 Dataset mixing / weighted sampling

| Status | Component | Location |
|--------|-----------|----------|
| ⚠️  | `WeightedAgeGenderSampler` defined but unused | `data_defs.py:222-252` |
| ⚠️  | `train_dataloader` uses `shuffle=True` — incompatible with sampler | `data_defs.py:414` |
| ❌ | Per-dataset sampling weights (not just per-age-bin weights) | Not in codebase |

**What to build:**
1. Fix the `WeightedAgeGenderSampler` integration: remove `shuffle=True` from `train_dataloader`, pass the sampler explicitly.
2. Extend sampler to accept per-dataset weights in addition to per-age-bin weights.
3. The manifest CSV (`dataset_name` column) makes it trivial to compute these weights.

---

## Stage 2 — Augmentation Strategy

### Tier A — Always-on (low-risk)

| Status | Augmentation | Location |
|--------|--------------|----------|
| ✅ | `RandomHorizontalFlip` | `data_defs.py:212` |
| ✅ | `ColorJitter` (brightness/contrast/saturation/hue) | `data_defs.py:216` |
| ✅ | `RandomRotation` ±10° | `data_defs.py:213` |
| ❌ | `RandomResizedCrop` narrow scale (0.9-1.0) | Not present — only `Resize(224,224)` used |

**What to build:** Replace the fixed `Resize(224,224)` at the start of the transform chain with `RandomResizedCrop(224, scale=(0.9, 1.0))` for training only. Val/test keep `Resize + CenterCrop`.

---

### Tier B — Probabilistic (real-world degradation)

| Status | Augmentation | Location |
|--------|--------------|----------|
| ✅ | `GaussianBlur` | `data_defs.py:218` (in config, disabled by default) |
| ✅ | `RandomErasing` light occlusion | `data_defs.py:321-330` |
| ✅ | `RandomPerspective` mild warp | `data_defs.py:214` |
| ✅ | `RandomGrayscale` | `data_defs.py:217` |
| ❌ | JPEG compression artifacts | Not implemented |
| ❌ | Mixup / CutMix (at batch level) | Not implemented |
| ❌ | Stacking cap (max 1-2 Tier-B per image) | No such logic — all active augmentations stack freely |

**What to build:**
- Add JPEG compression simulation: `transforms.Lambda(lambda img: apply_jpeg_compression(img, quality=random.randint(40,90)))`.
- Add Mixup/CutMix as a collate-level transform (not a per-image transform) — can be toggled via config.
- Add augmentation stacking guard: sample from enabled Tier-B augmentations, pick at most 2 per image.

---

### Validation / Ablation Guardrail

| Status | Component | Location |
|--------|-----------|----------|
| ✅ | `TrackingCompose` — validates no aug on val samples | `data_defs.py:17-34` |
| ❌ | Explicit clean-unaugmented val subset separate from real-world test | Not implemented |
| ❌ | Ablation runner (Tier A only vs Tier A+B) | Not implemented |

---

## Stage 3 — Model Design

### 3.1 Backbone

| Status | Component | Location |
|--------|-----------|----------|
| ✅ | MobileNet-V3-Small (current best) | `classifier.py:127` |
| ✅ | MobileNet-V3-Large | `classifier.py:125` |
| ✅ | EfficientNet-B0 backbone loading | `classifier.py:129` — already supported in `_initialize_model` |
| ⚠️  | `pretrained=True` API deprecated (torchvision 0.13+) | `classifier.py:125-129` |
| ❌ | `timm` integration (EfficientNetV2-B0 or broader model zoo) | Not used — only `torchvision.models` |

**What exists:** `_initialize_model` already handles `efficientnet_b0` as a model type via `torchvision.models.efficientnet_b0`. The head surgery (`[:-2]` for EfficientNet) is already implemented at line 147. You can switch to EfficientNet-B0 today by changing `model_type: "efficientnet_b0"` in the YAML config.

**What to fix first:** Replace `pretrained=True` with the `weights=` API before running new experiments.

---

### 3.2 Heads — Multi-task

| Status | Component | Location |
|--------|-----------|----------|
| ✅ | Gender head (binary classification, CrossEntropy) | `classifier.py:155-157` |
| ✅ | Age head (regression, L1Loss) | `classifier.py:158-160` |
| ✅ | Shared dropout before heads | `classifier.py:164-165` |
| ❌ | Age head as DEX-style softmax over age bins (CE instead of regression) | Not implemented |
| ❌ | Pose auxiliary head (yaw/pitch/roll regression) | Not implemented |
| ❌ | Shared FC layer before the two task heads | Currently global_pool goes directly to heads |

**What to build:**
- New `AgeGenderPoseClassifier` class (or extend existing) with:
  - Shared FC: `nn.Linear(num_features, 512)` + ReLU + Dropout
  - Age head: `nn.Linear(512, num_age_bins)` + softmax, decodes as expected value
  - Gender head: `nn.Linear(512, 2)`
  - Pose head: `nn.Linear(512, 3)` (yaw, pitch, roll)

---

### 3.3 Loss function

| Status | Component | Location |
|--------|-----------|----------|
| ✅ | Weighted multi-task loss (gender_loss_weight) | `classifier.py:224-226` |
| ✅ | L1 regularization on head weights | `classifier.py:229-234` |
| ❌ | Mean-Variance Loss for age bins | Not implemented |
| ❌ | Pose SmoothL1 loss term | Not implemented |
| ❌ | KL-divergence distillation loss | Not implemented |

**What to build:** A `src/models/efficientnet/losses.py` file containing:
```python
class MeanVarianceLoss(nn.Module): ...    # penalizes spread of age bin distribution
class DistillationLoss(nn.Module): ...    # KL-div student/teacher soft outputs
```
These integrate into `training_step` with the existing weighted sum pattern.

---

## Stage 4 — Training Strategy

### 4.1 Staged fine-tuning

| Status | Component | Location |
|--------|-----------|----------|
| ✅ | Stage 1: Freeze backbone, train heads only | `classifier.py:99-115` via `freeze_epochs` config |
| 🟡 | Stage 2: Partial unfreeze (top N blocks) | Only full freeze/unfreeze — no partial unfreeze |
| ❌ | Stage 3: Discriminative LR (different LR per layer group) | Single LR for all params |
| ✅ | AdamW optimizer | `classifier.py:295` |
| ❌ | Warmup (first 3-5% of steps) | Not implemented — `OneCycleLR` has `pct_start` but no explicit warmup-only phase |
| ❌ | EMA of model weights (decay 0.999) | Not implemented |
| ✅ | LR schedulers: OneCycleLR, ReduceLROnPlateau, StepLR | `classifier.py:293-368` |
| ✅ | Checkpoint resumption (`ckpt_path`) | `trainer.py:94-100` |
| ✅ | Top-3 ModelCheckpoint by `val_total_loss` | `trainer.py:74-81` |

**What to build:**
- Partial unfreeze: parameterize how many backbone blocks to unfreeze (e.g. `unfreeze_top_n_blocks: 3` in config).
- Discriminative LR: pass `param_groups` with per-group LR to `AdamW` instead of `self.parameters()`.
- EMA: add `torch.optim.swa_utils.AveragedModel` wrapper after training converges, or use a manual EMA tracked in a callback.

---

### 4.2 Knowledge distillation

| Status | Component | Location |
|--------|-----------|----------|
| ❌ | Teacher model (nateraw/vit-age-classifier) inference | Not in codebase |
| ❌ | Soft label generation over training/unlabeled data | Not in codebase |
| ❌ | KL-divergence distillation loss in `training_step` | Not in codebase |

**What to build:** A one-time script (`src/scripts/generate_teacher_labels.py`) that loads the ViT teacher from HuggingFace, runs it over all training images, and saves soft probability vectors to disk. Then extend `AgeGenderDataset.__getitem__` to return teacher soft labels alongside the image, and add the `DistillationLoss` term to `training_step`.

---

## Stage 5 — Pose Robustness

| Status | Component | Location |
|--------|-----------|----------|
| ❌ | Pose estimation (yaw/pitch/roll) per training image | Not in codebase |
| ❌ | Pose-bucket rebalancing of training data | Not in codebase |
| ❌ | Pose auxiliary head (see Stage 3.2) | Not in codebase |
| ❌ | Evaluation stratified by pose bucket | Not in codebase |
| 🟡 | Age-bin rebalancing (exists for UTKFace only) | `WeightedAgeGenderSampler` + `create_augmented_indices` |

**What to build:**
- Pose labels are generated during the crop pipeline (Stage 1.2) — store them in the manifest CSV.
- Extend `WeightedAgeGenderSampler` to also balance by pose bucket.
- Add `pose_bucket` column to evaluation metrics output in `metrics.py`.

---

## Stage 6 — Hyperparameter Search

| Status | Component | Location |
|--------|-----------|----------|
| ✅ | WandB sweep integration | `src/runners/sweeper.py` |
| ✅ | Transform ablation runner (one transform at a time) | `src/runners/transforms_tune.py` |
| 🟡 | Optuna integration | `src/scripts/ml_utils.py:204-232` — old FastAI-based, not compatible with Lightning pipeline |
| ✅ | YAML config system | `config/model/*.yaml` |
| ❌ | Optuna + Lightning integration for new pipeline | Not implemented |
| ❌ | Sweep evaluated against real-world test set MAE | Not possible until real-world test set exists (Stage 0.3) |

**What to build:** Replace the old FastAI-based `objective()` function in `ml_utils.py` with an Optuna `objective()` that calls `trainer.train(config)` (the Lightning-based trainer) and reads `val_total_loss` or `real_world_mae` from WandB or trainer callback metrics.

---

## Stage 7 — Evaluation Protocol

| Status | Component | Location |
|--------|-----------|----------|
| ✅ | Overall gender accuracy, precision, recall, F1, AUC-ROC | `src/models/mobilenet/metrics.py:135-223` |
| ✅ | Overall age MAE, MSE, RMSE, R², MAPE | `src/models/mobilenet/metrics.py:170-183` |
| ✅ | Age-bin stratified MAE | `src/models/mobilenet/metrics.py:193-213` |
| ✅ | Gender accuracy stratified by age bin | `src/models/mobilenet/metrics.py:186-191` |
| ✅ | Binned metrics by arbitrary column | `src/models/mobilenet/metrics.py:71-132` |
| ❌ | MAE stratified by pose bucket | Not implemented — no pose metadata |
| ❌ | MAE stratified by image quality (blur/JPEG) | Not implemented |
| ❌ | Evaluation on real-world test set | Not possible until Stage 0.3 |
| ❌ | Comparison against ViT teacher on real-world set | Not implemented |

**The existing `metrics.py` is the strongest part of the codebase for evaluation.** It just needs pose and quality columns added once the manifest CSV exists.

---

## Stage 8 — Inference-Time Tricks

| Status | Component | Location |
|--------|-----------|----------|
| ✅ | Prediction function with `torch.no_grad()` | `classifier.py:371-427` (`predict_with_model`) |
| ✅ | `model.eval()` correctly set before inference | `classifier.py:387` |
| ❌ | TTA (horizontal flip + crop variants) | Not implemented |
| ❌ | EMA weights at inference | Not implemented |
| ❌ | ONNX / TFLite export | `utils/export.py` is empty (139 bytes) |
| ❌ | Pose-aware crop margin at inference | Not implemented |
| ⚠️  | `num_workers=21` hardcoded in `predict_with_model` | `classifier.py:392` — should be parameterized |
| ⚠️  | `torch.load()` missing `map_location` | `trainer.py:106, 156` |

---

## Execution Checklist (Priority Order)

### Immediate fixes (before any new experiment)

- [ ] Fix `pretrained=True` → `weights=` API in `classifier.py:125-129`
- [ ] Fix `torch.load()` → add `map_location="cpu"` in `trainer.py:106, 156`
- [ ] Fix `WeightedAgeGenderSampler`: remove `shuffle=True` from `train_dataloader`, wire sampler in
- [ ] Fix `num_workers=21` → `min(os.cpu_count(), 8)` in `classifier.py:392`
- [ ] Verify EfficientNet-B0 works today: change `model_type: "efficientnet_b0"` in config and run

### Data pipeline (Stage 0 + Stage 1)

- [ ] Build `src/scripts/crop_pipeline.py` — YOLO face detection, fixed margin, crop metadata CSV
- [ ] Build `src/scripts/label_adapters.py` — per-dataset label normalizers
- [ ] Build `src/scripts/merge_datasets.py` — unified manifest CSV with `image_path, age, gender, dataset_name, identity_id, yaw, pitch, roll, detector_conf, blur_score`
- [ ] Rewrite identity-based split logic using `identity_id` groups
- [ ] Build and manually label real-world test set (200-500 images, pose-stratified)

### Model + training (Stage 3 + Stage 4)

- [ ] Build `src/models/efficientnet/classifier.py` — new model class with shared FC + age/gender/pose heads
- [ ] Build `src/models/efficientnet/losses.py` — `MeanVarianceLoss`, `DistillationLoss`
- [ ] Add partial unfreeze logic (top-N blocks) to model config
- [ ] Add discriminative LR (per-layer-group param groups)
- [ ] Add EMA weights callback
- [ ] Build `src/scripts/generate_teacher_labels.py` — ViT teacher soft label generation

### Augmentation + evaluation (Stage 2 + Stage 7)

- [ ] Add `RandomResizedCrop(224, scale=(0.9, 1.0))` for training
- [ ] Add JPEG compression augmentation
- [ ] Add Mixup/CutMix at collate level
- [ ] Add augmentation stacking cap
- [ ] Add pose bucket + image quality columns to `metrics.py` evaluation

### Search + inference (Stage 6 + Stage 8)

- [ ] Rewrite `ml_utils.objective()` for Optuna + Lightning pipeline
- [ ] Implement TTA in `predict_with_model` (flip + crop variants)
- [ ] Implement EMA at inference
- [ ] Implement `utils/export.py` (ONNX export)

---

## What You Can Reuse Without Changes

These components are solid and carry over to the new pipeline with no or minimal modification:

| File | What to reuse |
|------|--------------|
| `src/scripts/process_data_set.py` | `verify_and_clean_images`, `find_duplicates`, `stratified_split` — apply per dataset |
| `src/models/mobilenet/metrics.py` | All evaluation functions — just add pose/quality columns |
| `src/models/mobilenet/callbacks.py` | `EarlyStoppingCB`, `BestMetricsCallback`, `LRMonitorCallback` — framework-agnostic |
| `src/runners/trainer.py` | `load_config`, `save_model`, `load_model`, `ModelCheckpoint` setup |
| `src/runners/sweeper.py` | WandB sweep runner — just update `objective` |
| `src/runners/transforms_tune.py` | Transform ablation runner — reusable as-is |
| `src/models/mobilenet/data_defs.py` | `get_transforms`, `TrackingCompose`, `collate_fn` patterns |
| `config/model/*.yaml` | Config structure — extend with new keys |

---

*Last updated: July 2026 — based on full codebase scan.*
