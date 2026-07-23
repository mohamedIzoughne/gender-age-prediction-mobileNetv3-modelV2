# PyTorch Learning Curriculum
### Reverse-Engineered from the AgeGenderClassifier Codebase

> This document is a living reference. Every concept below maps directly to code you have already written.
> Read it alongside your source files, not as a replacement for them.

---

## Table of Contents
1. [Codebase Tech Map](#1-codebase-tech-map)
2. [Core Concepts You Must Master](#2-core-concepts-you-must-master-mapped-to-your-code)
3. [Training & Execution Mechanics](#3-the-training--execution-mechanics)
4. [Hidden Pitfalls & Best Practices](#4-hidden-pitfalls--best-practices-in-your-code)
5. [Quick-Reference Cheat Sheet](#5-quick-reference-pytorch-cheat-sheet)

---

## 1. Codebase Tech Map

A full inventory of every PyTorch primitive and abstraction used across the project.

| Category | Exact Symbol Used | Where |
|---|---|---|
| **Core Tensor** | `torch.Tensor`, `torch.tensor()`, `torch.stack()`, `torch.zeros()`, `torch.ones()`, `torch.multinomial()` | `data_defs.py`, `callbacks.py` |
| **nn.Module** | `nn.Sequential`, `nn.Linear`, `nn.Dropout`, `nn.AdaptiveAvgPool2d`, `nn.CrossEntropyLoss`, `nn.L1Loss` | `classifier.py` |
| **Functional** | `F.softmax()` | `classifier.py:413` |
| **Autograd** | `torch.no_grad()` context manager | `classifier.py:403` |
| **Torchvision Models** | `models.mobilenet_v3_large`, `models.mobilenet_v3_small`, `models.efficientnet_b0` | `classifier.py:124-129` |
| **Transforms v2** | `Resize`, `RandomHorizontalFlip`, `ColorJitter`, `RandomGrayscale`, `RandomPerspective`, `RandomAffine`, `RandomRotation`, `RandomErasing`, `RandomAutocontrast`, `RandomAdjustSharpness`, `GaussianBlur`, `Normalize`, `ToImage`, `ToDtype`, `Compose`, `Lambda`, `ToPILImage` | `data_defs.py` |
| **Dataset / DataLoader** | `torch.utils.data.Dataset`, `DataLoader`, `Sampler`, `random_split` | `data_defs.py` |
| **Optimizer** | `torch.optim.AdamW` | `classifier.py:295` |
| **Schedulers** | `OneCycleLR` (extended), `ReduceLROnPlateau`, `StepLR` | `classifier.py:33,331,358` |
| **Device / CUDA** | `torch.device()`, `torch.cuda.is_available()`, `torch.cuda.empty_cache()`, `.to(device)` | `classifier.py:385`, `ml_utils.py:230` |
| **Serialization** | `torch.save()`, `torch.load()`, `model.state_dict()`, `model.load_state_dict()` | `trainer.py:106,137,156,166` |
| **Seeding** | `torch.manual_seed()`, `torch.cuda.manual_seed()`, `torch.cuda.manual_seed_all()`, `torch.Generator().manual_seed()` | `ml_utils.py:98-100`, `data_defs.py:374` |
| **cuDNN flags** | `torch.backends.cudnn.benchmark`, `.deterministic` | `classifier.py:20`, `ml_utils.py:101` |
| **torchmetrics** | `Accuracy(task="binary")`, `MeanAbsoluteError()` | `classifier.py:78-80` |
| **Lightning** | `pl.LightningModule`, `pl.LightningDataModule`, `pl.Trainer`, `Callback`, `ModelCheckpoint`, `TQDMProgressBar`, `Timer`, `EarlyStopping`, `TensorBoardLogger` | `classifier.py`, `trainer.py` |
| **Mixed Precision** | `precision="16-mixed"` (AMP) | `trainer.py:90` |

---

## 2. Core Concepts You Must Master (Mapped to Your Code)

### 2.1 — Custom `nn.Module` & Transfer Learning Surgery

**File:** `src/models/mobilenet/classifier.py:117–162`

```python
# Strip the original classifier head
self.base_model = nn.Sequential(*list(self.base_model.children())[:-1])
# Attach two new task heads
self.gender_classifier = nn.Sequential(nn.Dropout(p=dropout_rate), nn.Linear(num_features, 2))
self.age_regressor     = nn.Sequential(nn.Dropout(p=dropout_rate), nn.Linear(num_features, 1))
```

**What PyTorch does under the hood:**

`model.children()` returns immediate child modules as a Python generator. Slicing `[:-1]` drops the
last child (the pretrained linear head) and wraps the rest in a new `nn.Sequential`. The critical
detail: `nn.Sequential` calls each sub-module's `forward()` in order; there is no graph surgery—the
computation graph is rebuilt fresh on each `forward()` call.

`num_features` is read from `self.base_model.classifier[0].in_features`—the output channel count
of the backbone's pooling stage (960 for `mobilenet_v3_large`).

**Minimal mental model:**

```
# Weight tensor shape for gender head Linear:
# weight: [2, 960], bias: [2]
# Output after Linear: [batch_size, 2]  <- raw logits, NOT probabilities

# Weight tensor shape for age head Linear:
# weight: [1, 960], bias: [1]
# Output before squeeze: [batch_size, 1]
# After .squeeze(1):      [batch_size]   <- scalar prediction per sample
```

---

### 2.2 — Multi-Task Forward Pass & `AdaptiveAvgPool2d`

**File:** `src/models/mobilenet/classifier.py:201–207`

```python
def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    features = self.base_model(x)                              # [B, 960, H', W']
    features = self.global_pool(features).view(x.size(0), -1) # [B, 960]
    gender_output = self.gender_classifier(features)           # [B, 2]
    age_output    = self.age_regressor(features).squeeze(1)    # [B]
    return gender_output, age_output
```

**What PyTorch does under the hood:**

`nn.AdaptiveAvgPool2d(1)` reduces any spatial `H'xW'` to `1x1` by computing the mean of each
feature map. After pooling, `features` shape is `[B, 960, 1, 1]`. The `.view(x.size(0), -1)`
reshapes (no data copy if memory is contiguous) to `[B, 960]`.

Both task heads share the same `features` tensor — this is **hard parameter sharing**: one
backbone, two gradient signals flowing back simultaneously.

**Autograd graph sketch:**

```
x (input)
  └─ base_model (backbone, frozen or not)
       └─ global_pool
            └─ view / flatten
                 ├─ gender_classifier -> gender_output -> CrossEntropyLoss -+
                 └─ age_regressor    -> age_output    -> L1Loss            -+
                                                                            v
                                                                      total_loss
                                                                      .backward()
```

---

### 2.3 — Freezing / Unfreezing Parameters

**File:** `src/models/mobilenet/classifier.py:99–115`

```python
for param in self.base_model.parameters():
    param.requires_grad = False   # freeze
    # OR
    param.requires_grad = True    # unfreeze
```

**What PyTorch does under the hood:**

`requires_grad=False` tells the autograd engine to **not** build a gradient node for this parameter
in the computation graph. When `loss.backward()` runs, it skips those parameters—they receive no
gradient, so `optimizer.step()` does not update them.

`freeze_epochs` is checked inside `on_train_epoch_start()`, a Lightning hook that fires before each
epoch. When `self.current_epoch == freeze_epochs`, the backbone is thawed and all parameters again
participate in gradient flow.

---

### 2.4 — Custom `Dataset.__getitem__` and the Augmentation Flag

**File:** `src/models/mobilenet/data_defs.py:187–206`

```python
def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int, int, bool, str]:
    if idx < len(self.valid_images):
        is_augmented = False
    else:
        aug_idx = idx - len(self.valid_images)
        orig_idx, is_augmented = self.augmented_indices[aug_idx]
    image = Image.open(img_path).convert("RGB")
    image = self.transform(image)   # CPU transform always applied here
    return image, age, gender, is_augmented, source_image
```

**What PyTorch does under the hood:**

`__len__` returns `len(self.valid_images) + len(self.augmented_indices)`. The DataLoader uses this
to generate indices `0 ... N-1`. Indices `>= len(valid_images)` are **virtual** — they point back to
real images but flag themselves for GPU-side augmentation later. This is your custom oversampling
strategy for rare age bins, without duplicating images on disk.

The base `self.transform` is always applied on the CPU worker process (inside `__getitem__`).
Heavy GPU augmentations are deferred to `on_after_batch_transfer`.

---

### 2.5 — `collate_fn` and Batch Assembly

**File:** `src/models/mobilenet/data_defs.py:443–451`

```python
def collate_fn(self, batch):
    images, ages, genders, is_augmented, image_paths = zip(*batch)
    images       = torch.stack(images)                           # [B, 3, 224, 224]
    ages         = torch.tensor(ages)                            # [B]  int64 by default
    genders      = torch.tensor(genders)                         # [B]  int64
    is_augmented = torch.tensor(is_augmented, dtype=torch.bool) # [B]  bool
    return images, ages, genders, is_augmented, image_paths
```

**What PyTorch does under the hood:**

`zip(*batch)` transposes a list of tuples into separate iterables — one per field.
`torch.stack(images)` concatenates a list of `[3, 224, 224]` tensors along a **new** first
dimension, producing `[B, 3, 224, 224]`. This is the standard way to batch image tensors.

`torch.tensor(ages)` creates a new tensor from Python ints. This happens in the DataLoader
worker, running in a subprocess when `num_workers > 0`.

---

### 2.6 — GPU-Side Augmentation via `on_after_batch_transfer`

**File:** `src/models/mobilenet/classifier.py:175–187`

```python
def on_after_batch_transfer(self, batch, dataloader_idx):
    if self.trainer.training:
        x, age, gender, is_augmented, image_paths = batch
        if self.dynamic_augment_transform is not None and is_augmented.any():
            augmented_x = self.dynamic_augment_transform(x[is_augmented])
            x[is_augmented] = augmented_x
        return x, age, gender, image_paths
```

**What PyTorch does under the hood:**

After Lightning moves the batch to the GPU (via `pin_memory=True` + CUDA DMA), this hook fires
**before** `training_step`. `x[is_augmented]` is **advanced boolean indexing** — it returns a
view (or copy, depending on contiguity) of rows where `is_augmented == True`, applies
`dynamic_augment_transform` to those images on the GPU, then writes them back.

This pattern offloads expensive augmentations from CPU workers to the GPU, avoiding CPU
bottlenecks on the augmentation ops themselves.

---

### 2.7 — Custom LR Scheduler: `OneCycleWithDecay`

**File:** `src/models/mobilenet/classifier.py:33–66`

```python
class OneCycleWithDecay(torch.optim.lr_scheduler.OneCycleLR):
    def get_lr(self) -> List[float]:
        if self.last_epoch < self.total_steps:
            return super().get_lr()            # standard one-cycle phase
        return [_calc_lr(group["lr"]) for group in self.optimizer.param_groups]  # decay phase

    def step(self, epoch=None):
        if self.last_epoch >= self.total_steps:
            self.last_epoch += 1
            for param_group, lr in zip(self.optimizer.param_groups, self.get_lr()):
                param_group["lr"] = lr
        else:
            super().step(epoch)
```

**What PyTorch does under the hood:**

`OneCycleLR` internally tracks `last_epoch` as the step counter (not a real epoch). It computes
the LR via cosine annealing between `base_lr/div_factor` -> `max_lr` -> `base_lr/final_div_factor`.

Your `OneCycleWithDecay` extension kicks in **after** `total_steps` is exhausted, applying a
multiplicative decay (`* 1.0055`) capped at `0.001`. Registered with `"interval": "step"` in
`configure_optimizers`, Lightning calls `scheduler.step()` after **every batch**, not every epoch.

```python
# In configure_optimizers:
return {"optimizer": optimizer, "lr_scheduler": {"scheduler": scheduler, "interval": "step"}}
```

---

### 2.8 — `WeightedAgeGenderSampler`: Custom Sampler

**File:** `src/models/mobilenet/data_defs.py:222–252`

```python
def _calculate_weights(self) -> torch.Tensor:
    weights = torch.zeros(len(self.dataset))
    max_freq = max(self.bin_frequencies)
    for bin_idx, indices in enumerate(self.bins):
        bin_weight = max_freq / self.bin_frequencies[bin_idx]
        for idx in indices:
            weights[idx] = bin_weight
    return weights

def __iter__(self):
    return iter(torch.multinomial(self.weights, len(self.dataset), self.replacement).tolist())
```

**What PyTorch does under the hood:**

`torch.multinomial(weights, n, replacement=True)` performs weighted sampling. It produces a 1-D
integer tensor of sampled indices. Rare age bins get `max_freq / small_count` weight, so they are
oversampled proportionally. The DataLoader consumes `__iter__` to determine sample order per epoch.
When a custom `sampler` is provided, the DataLoader's own `shuffle` must be `False`.

> **Note:** `WeightedAgeGenderSampler` is defined but never passed to `train_dataloader` (which
> uses `shuffle=True` on line 414). The sampler is unused in the active training path.

---

### 2.9 — Loss Functions: CrossEntropy vs L1 in a Multi-Task Setup

**File:** `src/models/mobilenet/classifier.py:76–77, 221–226`

```python
self.gender_loss = nn.CrossEntropyLoss()   # classification
self.age_loss    = nn.L1Loss()             # regression

total_loss = (
    gender_loss_weight * gender_loss
    + (1 - gender_loss_weight) * age_loss
)
```

**What PyTorch does under the hood:**

`nn.CrossEntropyLoss` combines `log_softmax + nll_loss` in one numerically stable operation.
It expects **raw logits** of shape `[B, num_classes]` and integer targets `[B]`.

`nn.L1Loss` computes `mean(|y_pred - y_true|)`. The `.float()` cast on `age` is required because
`torch.tensor(ages)` defaults to `int64`, and L1Loss requires matching dtypes.

`total_loss` is a scalar tensor. Lightning calls `.backward()` on it, differentiating through
both loss branches simultaneously. `gender_loss_weight=0.9` prioritises gender accuracy.

---

### 2.10 — L1 Weight Regularisation

**File:** `src/models/mobilenet/classifier.py:229–234`

```python
if l1_lambda > 0:
    l1_norm = sum(p.abs().sum() for p in self.gender_classifier[1].parameters()) \
            + sum(p.abs().sum() for p in self.age_regressor[1].parameters())
    total_loss += l1_lambda * l1_norm
```

**What PyTorch does under the hood:**

`p.abs().sum()` is a differentiable tensor operation — PyTorch builds an autograd node for it.
When `total_loss.backward()` fires, gradients flow through this term, pushing linear layer weights
towards zero (sparsity pressure). `[1]` indexes the `nn.Linear` inside the `nn.Sequential` (index
`[0]` is `nn.Dropout`). Applying L1 only to the heads avoids regularising the pretrained backbone.

---

### 2.11 — Serialization: `state_dict` vs Full Model

**File:** `src/runners/trainer.py:137–143`

```python
torch.save({
    "model_state_dict": model.state_dict(),
    "config": model.config,
}, f"model_store/{save_path}")
```

**What PyTorch does under the hood:**

`state_dict()` returns an `OrderedDict` of all `nn.Parameter` and registered buffer tensors,
keyed by dotted path (e.g. `"gender_classifier.1.weight"`). It does **not** save the model class
definition — you must re-instantiate `AgeGenderClassifier(config)` first, then call
`model.load_state_dict(checkpoint["model_state_dict"])`.

This is the correct PyTorch pattern. Saving the entire model object (`torch.save(model, ...)`)
pickles the class itself, which breaks if you rename or move files.

---

### 2.12 — Transforms v2: `ToImage` + `ToDtype` Pipeline

**File:** `src/models/mobilenet/data_defs.py:258–261`

```python
transforms.Compose([transforms.ToImage(), transforms.ToDtype(torch.float32, scale=True)]),
transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
```

**What PyTorch does under the hood:**

`torchvision.transforms.v2` is the newer API (replaces v1). `ToImage()` converts a PIL Image
to a `torch.Tensor` of dtype `uint8` with shape `[C, H, W]`. `ToDtype(torch.float32, scale=True)`
both casts to float32 **and** scales from `[0, 255]` to `[0.0, 1.0]` in one pass (replaces v1's
`ToTensor()`).

`Normalize(mean, std)` applies `output = (input - mean) / std` channel-wise. The values
`[0.485, 0.456, 0.406]` and `[0.229, 0.224, 0.225]` are ImageNet statistics, required because
your backbone was pretrained on ImageNet.

---

## 3. The Training & Execution Mechanics

### Full Data Flow

```
Disk (JPG files)  {age}_{gender}_*.jpg
  |
  v
AgeGenderDataset.__getitem__()              [CPU Worker Process]
  | PIL.Image.open().convert("RGB")
  | TrackingCompose (train_transform):
  |   Resize(224,224) -> aug transforms -> ToImage -> ToDtype(float32) -> Normalize
  | Returns: (Tensor[3,224,224], int age, int gender, bool is_augmented, str path)
  |
  v
collate_fn()                                [CPU Worker Process]
  | torch.stack(images)   -> Tensor[B, 3, 224, 224]  float32
  | torch.tensor(ages)    -> Tensor[B]                int64
  | torch.tensor(genders) -> Tensor[B]                int64
  | torch.tensor(flags)   -> Tensor[B]                bool
  |
  v
DataLoader (pin_memory=True)               [Main Process]
  | Batch pinned to page-locked CPU memory
  | Enables async DMA transfer to GPU
  |
  v
Lightning on_after_batch_transfer()        [CUDA GPU]
  | GPU-side augmentation on is_augmented rows
  | Returns (x, age, gender, paths)
  |
  v
training_step()                            [CUDA GPU]
  | forward(x) -> (gender_logits[B,2], age_pred[B])
  | gender_loss = CrossEntropyLoss(gender_logits, gender)   scalar
  | age_loss    = L1Loss(age_pred, age.float())             scalar
  | total_loss  = 0.9*gender_loss + 0.1*age_loss + L1_reg  scalar
  | self.log(...) -> TensorBoard
  | return total_loss
  |
  v
Lightning (automatic)
  | optimizer.zero_grad()
  | total_loss.backward()     <- autograd graph, compute grads
  | optimizer.step()          <- AdamW updates weights
  | scheduler.step()          <- OneCycleWithDecay updates LR (every step)
  |
  v
validation_step()  (end of epoch)
  | forward(x) -> predictions
  | Log val_gender_loss, val_age_loss, val_total_loss, val_gender_acc, val_age_mae
  |
  v
ModelCheckpoint
  | Saves top-3 checkpoints by val_total_loss
```

### Device & Precision Management

| Mechanism | Where | Effect |
|---|---|---|
| `precision="16-mixed"` | `trainer.py:90` | AMP: forward in float16, backward in float32. Uses `GradScaler` internally. ~2x speedup on modern GPUs. |
| `accelerator="gpu", devices=1` | `trainer.py:88-89` | Lightning moves model + batch to `cuda:0` automatically |
| `pin_memory=True` | `data_defs.py:417,427,440` | Allocates batch tensors in pinned CPU RAM enabling async DMA to GPU |
| `cudnn.benchmark = True` | `classifier.py:20` | cuDNN auto-tunes convolution kernels for fixed input size (224x224). Speeds up after first forward pass. |
| `persistent_workers=True` | `data_defs.py:416,426,439` | Worker subprocesses survive between epochs. Requires `num_workers > 0`. |
| Manual device in `predict_with_model` | `classifier.py:385-386` | Old-style device management used outside Lightning context |

---

## 4. Hidden Pitfalls & Best Practices in Your Code

### PITFALL 1 — `shuffle=True` with a custom `Sampler` is mutually exclusive

**File:** `src/models/mobilenet/data_defs.py:414`

```python
return DataLoader(
    self.train_dataset,
    shuffle=True,       # <- PROBLEM if you ever pass sampler=
    ...
)
```

`WeightedAgeGenderSampler` is defined (line 222) but never passed to `train_dataloader`.
If you add `sampler=WeightedAgeGenderSampler(...)`, PyTorch raises `ValueError` because
`shuffle=True` and a custom `sampler` are mutually exclusive.

```python
# Correct pattern:
DataLoader(dataset, sampler=WeightedAgeGenderSampler(...), shuffle=False, ...)
```

---

### PITFALL 2 — `pretrained=True` is deprecated in modern torchvision

**File:** `src/models/mobilenet/classifier.py:125-129`

```python
self.base_model = models.mobilenet_v3_large(pretrained=pretrained)  # <- DeprecationWarning
```

Since torchvision 0.13+, the correct API is:

```python
weights = models.MobileNet_V3_Large_Weights.IMAGENET1K_V2 if pretrained else None
self.base_model = models.mobilenet_v3_large(weights=weights)
```

---

### PITFALL 3 — `num_workers=21` in `predict_with_model` is a magic number

**File:** `src/models/mobilenet/classifier.py:392`

```python
DataLoader(datamodule.test_dataset, batch_size=batch_size, num_workers=21, ...)
```

Hard-coded `21` (vs `num_workers=2` in the DataModule) can exhaust OS resources.
Best practice: `num_workers = min(os.cpu_count(), 8)` or add it to config.

---

### PITFALL 4 — Inplace ops on tensors in the autograd graph

**File:** `src/models/mobilenet/classifier.py:181-182`

```python
augmented_x = self.dynamic_augment_transform(x[is_augmented])
x[is_augmented] = augmented_x   # <- in-place write to x
```

This is **safe here** because it happens in `on_after_batch_transfer`, before `forward()` is
called. If you ever move this logic inside `forward()` or after any gradient-requiring op,
you would corrupt the autograd graph. The `inplace=False` on `RandomErasing` (line 329) is the
correct habit to maintain.

---

### PITFALL 5 — `torch.load()` without `map_location` crashes on CPU machines

**File:** `src/runners/trainer.py:106, 156`

```python
checkpoint = torch.load(best_model_path)       # <- dangerous
checkpoint = torch.load(f"model_store/{path}") # <- dangerous
```

If a checkpoint was saved on GPU and loaded on a CPU-only machine, PyTorch fails.
Always specify:

```python
checkpoint = torch.load(path, map_location="cpu")  # safe everywhere
```

---

### PITFALL 6 (CORRECT) — `.item()` in `BestMetricsCallback` prevents memory leaks

**File:** `src/models/mobilenet/callbacks.py:35`

```python
v.item() if isinstance(v, torch.Tensor) else v   # <- correctly handled
```

Without `.item()`, storing a tensor in `self.best_metrics` keeps the entire computation
graph alive across epochs, accumulating GPU memory indefinitely. This is done correctly.

---

### PITFALL 7 (CORRECT) — `model.eval()` is set before inference in `predict_with_model`

**File:** `src/models/mobilenet/classifier.py:387`

```python
model.eval()   # <- correctly set before inference loop
```

`model.eval()` disables `nn.Dropout` and switches `nn.BatchNorm` to use running statistics.
The complementary `model.train()` is handled by Lightning automatically inside `trainer.fit()`.

---

### PITFALL 8 (CORRECT) — `torch.Generator().manual_seed(42)` for reproducible splits

**File:** `src/models/mobilenet/data_defs.py:374`

```python
torch.utils.data.random_split(
    range(len(temp_dataset)), [train_size, val_size],
    generator=torch.Generator().manual_seed(42),
)
```

`torch.Generator` provides a local, isolated random state for the split. This is preferred over
setting the global `torch.manual_seed()` before the call — correctly handled.

---

### PITFALL 9 — `TrackingCompose` validation guard fires at runtime only

**File:** `src/models/mobilenet/data_defs.py:31–33`

```python
if self.transform_type == "val" and not self.applied_transforms.issubset(allowed_transforms):
    raise ValueError(...)
```

Excellent defensive programming. However, the check only fires when a sample is fetched, not
at `setup()` time. With `num_workers > 0`, the error surfaces in a subprocess and may appear
as a confusing `DataLoader` crash. Consider adding a setup-time validation check.

---

## 5. Quick-Reference PyTorch Cheat Sheet

### Tensors

```python
torch.tensor([1, 2, 3])              # Create from Python list; dtype inferred
torch.zeros(N)                       # Zero-filled float32 tensor of length N
torch.ones(N)                        # One-filled float32 tensor
torch.stack([t1, t2], dim=0)         # Stack list of [C,H,W] -> [B,C,H,W]
torch.multinomial(w, n, replacement) # Sample n indices from weight tensor w

t.view(B, -1)                        # Reshape; -1 is inferred. Requires contiguous memory.
t.squeeze(1)                         # Remove dim 1 if size==1: [B,1] -> [B]
t[bool_mask]                         # Advanced boolean indexing; returns subset rows
t.abs().sum()                        # Differentiable L1 norm
t.float()                            # Cast to float32
t.cpu().numpy()                      # Move to CPU, detach from graph, convert to ndarray
t.to(device)                         # Move tensor/module to device
t.any()                              # Returns True if any element is nonzero/True
t.item()                             # Extract Python scalar from 0-dim tensor (avoids memory leak)
```

### `nn` Modules

```python
nn.Sequential(*layers)               # Ordered container; calls each layer forward() in order
nn.Linear(in_features, out_features) # Fully-connected: y = xW^T + b
nn.Dropout(p=0.1)                    # Zeroes p% of activations randomly (train mode only)
nn.AdaptiveAvgPool2d(1)              # Reduce [B,C,H,W] -> [B,C,1,1] via spatial mean
nn.CrossEntropyLoss()                # log_softmax + nll_loss; expects logits + int targets
nn.L1Loss()                          # mean(|pred - target|); expects matching dtypes
```

### Autograd

```python
param.requires_grad = False          # Exclude parameter from gradient computation
param.requires_grad = True           # Include parameter in gradient computation
torch.no_grad()                      # Context: disable gradient tracking (use for inference)
# Lightning calls loss.backward() and optimizer.step() automatically in training_step
```

### Optimizers

```python
torch.optim.AdamW(
    model.parameters(),
    lr=base_lr,
    weight_decay=weight_decay
)
optimizer.param_groups[0]["lr"]      # Read current LR from first parameter group
```

### Schedulers

```python
# One Cycle (step-level):
torch.optim.lr_scheduler.OneCycleLR(
    optimizer, max_lr=..., total_steps=...,
    pct_start=..., anneal_strategy="cos",
    div_factor=..., final_div_factor=...
)
# Plateau (epoch-level):
torch.optim.lr_scheduler.ReduceLROnPlateau(
    optimizer, mode="min", factor=..., patience=..., threshold=...
)
# Step (epoch-level):
torch.optim.lr_scheduler.StepLR(optimizer, step_size=..., gamma=...)

# Lightning registration:
{"optimizer": opt, "lr_scheduler": {"scheduler": sched, "interval": "step"}}
```

### DataLoader / Dataset

```python
class MyDataset(Dataset):
    def __len__(self): ...           # Total samples (incl. virtual augmented indices)
    def __getitem__(self, idx): ...  # Returns single sample as tuple

DataLoader(
    dataset,
    batch_size=256,
    shuffle=True,                    # Mutually exclusive with sampler=
    num_workers=2,
    persistent_workers=True,         # Keep workers alive between epochs
    pin_memory=True,                 # Page-lock CPU batch memory for fast GPU DMA
    collate_fn=fn,                   # Custom batch assembly function
)
torch.utils.data.random_split(
    dataset, [train_size, val_size],
    generator=torch.Generator().manual_seed(42)
)
```

### Device & CUDA

```python
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)                          # Move all model parameters to device
tensor.to(device)                         # Move tensor to device
torch.cuda.empty_cache()                  # Release GPU memory cache (not actual allocations)
torch.backends.cudnn.benchmark = True     # Auto-tune kernels for fixed input sizes
torch.backends.cudnn.deterministic = True # Reproducible results (slower)
```

### Serialization

```python
torch.save({"model_state_dict": model.state_dict(), "config": ...}, path)
checkpoint = torch.load(path, map_location="cpu")   # ALWAYS specify map_location!
model.load_state_dict(checkpoint["model_state_dict"])
```

### Seeding for Reproducibility

```python
torch.manual_seed(42)
torch.cuda.manual_seed(42)
torch.cuda.manual_seed_all(42)       # For multi-GPU
torch.Generator().manual_seed(42)    # Isolated generator for a single operation
```

### Lightning-Specific Patterns

```python
class MyModule(pl.LightningModule):
    def training_step(self, batch, batch_idx) -> torch.Tensor: ...  # return loss
    def validation_step(self, batch, batch_idx): ...
    def configure_optimizers(self) -> dict: ...
    def on_train_epoch_start(self): ...
    def on_after_batch_transfer(self, batch, idx): ...  # hook after batch moved to GPU

self.log("metric_name", value,
    on_step=True/False,
    on_epoch=True/False,
    prog_bar=True)

pl.Trainer(
    max_epochs=N,
    accelerator="gpu", devices=1,
    precision="16-mixed",            # AMP: fp16 forward, fp32 backward
    callbacks=[...],
    logger=TensorBoardLogger(...),
    log_every_n_steps=50,
)
trainer.fit(model, datamodule=dm, ckpt_path=resume_ckpt)
```

### `torchvision.transforms.v2` (your pipeline)

```python
transforms.Resize((224, 224))
transforms.ToImage()                           # PIL -> uint8 Tensor [C,H,W]
transforms.ToDtype(torch.float32, scale=True)  # uint8->float32, divides by 255
transforms.Normalize(mean, std)                # (x - mean) / std, channel-wise

transforms.RandomHorizontalFlip(p=0.5)
transforms.ColorJitter(brightness, contrast, saturation, hue)
transforms.RandomGrayscale(p=0.25)
transforms.RandomPerspective(distortion_scale=0.2, p=0.2)
transforms.RandomAffine(degrees=..., translate=..., scale=...)
transforms.RandomRotation(degrees=(-20, 20))
transforms.RandomErasing(p=0.1, scale=..., ratio=..., inplace=False)
transforms.RandomAutocontrast(p=0.2)
transforms.RandomAdjustSharpness(sharpness_factor=..., p=0.2)
transforms.GaussianBlur(kernel_size=3, sigma=(0.1, 2.0))
transforms.RandomApply([transform], p=0.1)
transforms.Compose([t1, t2, ...])
transforms.Lambda(lambda x: x + torch.randn_like(x) * 0.1)   # Gaussian noise
```

---

*Generated from codebase scan — AgeGenderClassifier project, July 2026.*
