# Age and Gender Classifier: Technical Techniques Overview

This document outlines the advanced technical methodologies implemented in this project to achieve high precision in Age and Gender classification using MobileNet architectures.

---

## 1. Parameter Tuning & Optimization

The project utilizes a highly structured and automated approach to hyperparameter tuning, evident from the YAML configuration sweeps (e.g., `swept-sweep-34_improved_DYNAMIC_AUG.yaml`).

### **A. Hyperparameter Sweeps**
- **Systematic Searching:** The naming conventions (`swept-sweep-34`) indicate that automated hyperparameter sweeps (likely using Weights & Biases or a similar tool) were run to find the absolute optimal combination of learning rates, dropout rates, and batch sizes.
- **Loss Weighting (`gender_loss_weight`):** Since this is a multi-task learning problem, the overall loss is calculated as a weighted sum of the Gender Loss and Age Loss. Tuning this parameter ensures the model doesn't over-prioritize gender classification (which is an easier binary task) at the expense of age regression.

### **B. Advanced Learning Rate Scheduling**
- **One Cycle Policy with Decay (`OneCycleWithDecay`):** The custom scheduler in `classifier.py` starts with a low learning rate, warms up to a maximum (`max_lr`), and then anneals down. Crucially, a custom *decay factor* is added to continue reducing the learning rate by a small percentage (e.g., 1%) every step *after* the cycle completes, allowing for ultra-fine-tuning in the final epochs.
- **ReduceLROnPlateau / StepLR:** Configurable fallbacks exist to drop the learning rate dynamically when validation loss stagnates.

### **C. Regularization & Precision**
- **L1 Regularization (`l1_lambda`):** Applied specifically to the weights of the final classification and regression layers to enforce sparsity and prevent overfitting on the training set.
- **Mixed Precision Training (16-bit):** The PyTorch Lightning trainer is configured with `precision="16-mixed"`, which drastically reduces memory footprint and accelerates training on modern GPUs without sacrificing model accuracy.

---

## 2. Advanced Data Augmentation & Balancing

Standard datasets for age estimation (like UTKFace) are notoriously unbalanced—they have an overwhelming number of young adults (20-30 years old) and very few infants or elderly individuals. The project combats this with several clever techniques.

### **A. Dynamic Age-Bin Balancing**
- **Binning:** The dataset (`AgeGenderDataset`) categorizes all ages into bins (e.g., 0-9, 10-19, etc.) and calculates the frequency of each bin.
- **Dynamic Duplication:** It identifies the majority class (most frequent age bin). For all under-represented bins, it dynamically creates synthetic duplicate indices (`augmented_indices`) to artificially balance the dataset.

### **B. Heavy "Dynamic" Augmentation on Duplicates**
To prevent the model from simply memorizing the duplicated images from minority age groups, a massive, randomized augmentation pipeline is applied **only to the duplicated samples**. This pipeline includes:
- **Spatial distortions:** `RandomHorizontalFlip`, `RandomPerspective`, `RandomAffine`, `RandomRotation`.
- **Pixel-level distortions:** `RandomAutocontrast`, `RandomAdjustSharpness`, `ColorJitter`, `RandomGrayscale`, `GaussianBlur`.
- **Random Erasing:** Blacking out random blocks of the image forces the model to rely on multiple facial features (e.g., wrinkles, hair, jawline) rather than a single feature that might be occluded.

### **C. Weighted Sampling (`WeightedAgeGenderSampler`)**
As an alternative or addition to dynamic augmentation, the project implements a custom `WeightedAgeGenderSampler` that uses `torch.multinomial`. It assigns a higher sampling probability to images from rare age bins, ensuring every training batch has a relatively uniform distribution of ages.

---

## 3. Techniques for Best Precision (Especially for Age)

Achieving high precision on Age estimation is notoriously difficult. The following architectural and mathematical choices were made to optimize specifically for this:

### **A. Multi-Task Learning Architecture**
Instead of having two separate models, the project uses a single base feature extractor (MobileNetV3). The network splits at the end into two heads:
1. **Gender Head:** A Linear layer outputting 2 classes (Binary Cross Entropy).
2. **Age Head:** A Linear layer outputting 1 continuous value.
*Why it works:* Sharing the base weights forces the model to learn generalized, robust facial features (like bone structure and skin texture) that are mutually beneficial for predicting both age and gender, resulting in better generalization.

### **B. Age as Regression (L1 Loss/MAE)**
Age is treated as a continuous regression problem using `L1Loss` (Mean Absolute Error) rather than bucketing ages into classification categories. 
*Why it works:* If a person is 40, guessing 41 is a small error, but guessing 80 is a huge error. L1 loss mathematically penalizes the network proportional to *how far off* the guess was in actual years, which is crucial for precise age estimation.

### **C. Transfer Learning & Staged Unfreezing**
- The MobileNetV3 base is initialized with pre-trained ImageNet weights (`pretrained=True`), starting the model with a strong foundational understanding of edges, textures, and shapes.
- **`freeze_epochs`:** The base model can be temporarily frozen at the start of training. This allows the newly initialized random weights in the Age and Gender heads to "warm up" without corrupting the valuable pre-trained features of the base model via massive backpropagation gradients.

---

## 4. End-to-End Project Execution Flow (Deep Technical Breakdown)

Understanding the sequence of operations is key to navigating the codebase. Here is the step-by-step flow from raw data to a fully trained and evaluated model, detailing the exact mathematical and programmatic steps.

### **Phase 1: Data Preparation & Cleaning** (`src/scripts/process_data_set.py`)
Before training begins, raw datasets (like UTKFace) must be cleaned and structured.
1. **Validation & Repair:** The `verify_and_clean_images` function opens every image using `Image.open(img_path)`. It attempts a `.convert("RGB")` and a `.resize((224, 224))`. Any image that throws a `PIL.UnidentifiedImageError` or `IOError` is physically deleted via `img_path.unlink()` and recorded in `broken_files`.
2. **Deduplication:** To prevent data leakage, `find_duplicates` runs across a multiprocessing pool (`Pool(processes=cpu_count())`). It converts images to RGB and generates an 8x8 hash using `imagehash.dhash(img, 8)`. If `hash_val` matches an existing entry in the `hash_dict`, the image is deemed a duplicate and unlinked.
3. **Stratified Splitting:** `stratified_split` iterates through the cleaned dataset, categorizing by the folder structure. It utilizes `sklearn.model_selection.train_test_split` with `test_size=0.12`, maintaining the exact `stratify=categories` parameter, and saves files into explicit `train` and `test` directories using `shutil.copy`.

### **Phase 2: Dataset Construction & On-the-Fly Balancing** (`src/models/MobileNet/data_defs.py`)
The `AgeGenderDataModule` (PyTorch Lightning) handles feeding data to the model.
1. **Parsing & Memory Loading:** Inside `AgeGenderDataset.__init__`, image names (e.g., `25_1_...`) are split by underscores: `age = int(splits[0])` and `gender = int(splits[1])`. The paths, ages, and genders are loaded into parallel lists in memory.
2. **Binning & Distribution Analysis:** The `calculate_age_bins` function executes a mathematical mapping: `bin_idx = min(age // 10, num_bins - 1)`. With `num_bins=9`, this creates decades (0-9, 10-19, up to 80+). It stores the raw indices belonging to each bin and tabulates `bin_frequencies`.
3. **Synthetic Duplication (The Math):** If `use_dynamic_augmentation=True`, `create_augmented_indices` calculates the maximum frequency (`max_freq = max(self.bin_frequencies)`). For every under-represented bin, it calculates the number of synthetic samples to inject: 
   `num_to_add = int(int((max_freq - len(indices)) * mult) + max_freq * 0.1)` 
   *(Note the 10% structural boost past max frequency)*. It then generates virtual indices mapping back to the original index `orig_idx` but flagged as a duplicate.
4. **Data Loader Flagging (`__getitem__`):** During iteration, the dataset checks if the requested `idx < len(self.valid_images)`. If true, it loads the original and sets `is_augmented = False`. If false, it maps to `orig_idx` from the synthetic list and sets `is_augmented = True`. At this stage on the CPU, only `transforms.Resize((224, 224))` and `transforms.Normalize` (to ImageNet standards) are executed.

### **Phase 3: Model Initialization** (`src/models/MobileNet/classifier.py`)
The architecture is dynamically constructed inside `AgeGenderClassifier._initialize_model`.
1. **Base Extraction:** The backbone (e.g., `models.mobilenet_v3_large(pretrained=True)`) is loaded. To remove the 1000-class ImageNet head, it executes:
   `self.base_model = nn.Sequential(*list(self.base_model.children())[:-1])`
2. **Pooling:** An `nn.AdaptiveAvgPool2d(1)` is added to squash the spatial dimensions into a 1D tensor.
3. **Dual Heads:** Two distinct modules are attached, incorporating configurable `nn.Dropout(p=dropout_rate)`:
   - `self.gender_classifier = nn.Linear(num_features, 2)`
   - `self.age_regressor = nn.Linear(num_features, 1)`
4. **Freezing Logic:** If `config["freeze_epochs"] > 0`, the loop `for param in self.base_model.parameters(): param.requires_grad = False` executes, locking the backbone gradients.

### **Phase 4: The Training Loop** (`src/models/MobileNet/runner_scripts/trainer.py` & `classifier.py`)
The core training leverages PyTorch Lightning's optimization strategies.
1. **GPU Batch Transfer:** A batch (images, ages, genders, and boolean `is_augmented` tensor) is pushed to the GPU.
2. **GPU-Accelerated Augmentation:** Inside the `on_after_batch_transfer` hook, the exact code `if self.dynamic_augment_transform is not None and is_augmented.any():` triggers. It executes a heavy `v2.Compose` (ColorJitter, Sharpness(p=0.2), RandomErasing(value=0, scale=(0.02, 0.33))) directly on the boolean-sliced tensor: `x[is_augmented] = augmented_x`. This is a massive optimization preventing CPU dataloader bottlenecks.
3. **Forward Pass:** The batched tensor traverses `self.base_model`, is flattened by `.view(x.size(0), -1)`, and splits into `gender_output` and `age_output.squeeze(1)`.
4. **Loss Computation:** 
   - `gender_loss = nn.CrossEntropyLoss()(gender_pred, gender)`
   - `age_loss = nn.L1Loss()(age_pred, age.float())`
   - **Merge:** `total_loss = (weight * gender_loss) + ((1 - weight) * age_loss)`
   - **L1 Penalty:** If `l1_lambda > 0`, `p.abs().sum()` across the parameters of the dual heads is added to the loss to enforce sparsity.
5. **Scheduler Step:** `OneCycleWithDecay` executes. Once the primary OneCycle policy finishes, the custom `step()` method kicks in: `new_lr = g_lr * 1.0055`, artificially forcing a ~0.5% decay per step until the floor of `0.001` is hit.
6. **Unfreezing Hook:** At the start of an epoch (`on_train_epoch_start`), `check_unfreeze_base_model` fires. If `current_epoch == freeze_epochs`, it sets `param.requires_grad = True` across the backbone.

### **Phase 5: Evaluation & Metrics** (`src/models/MobileNet/metrics.py`)
After training, `metrics.evaluate_predictions` parses the raw model outputs.
1. **Probability Conversion:** Gender logits are pushed through `F.softmax(gender_logits, dim=1)`, extracting the probability for the positive class (e.g., Male).
2. **Binned Metrics Analysis:** `bin_column` groups true ages using `DEFAULT_AGE_BINS = [0, 4, 14, 24, 30, 40, 50, ... 80, np.inf]`. It calculates accuracy mathematically `true_genders == gender_pred_labels` for every specific bin, outputting highly granular statistics.
3. **Advanced Scoring:** 
   - Uses `roc_auc_score` and `average_precision_score` for gender bounds.
   - Computes `MAPE` (Mean Absolute Percentage Error) for age: `np.mean(np.abs((true_ages - age_preds) / true_ages)) * 100`, determining exactly how far off the age guess is relative to the person's true lifespan.
