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
