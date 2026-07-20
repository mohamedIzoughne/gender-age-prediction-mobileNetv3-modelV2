import os
import shutil
from collections import Counter
from typing import List, Tuple, Dict, Any, Optional

import pytorch_lightning as pl
import torch
from PIL import Image
from fastai.data.load import DataLoader
from torch.utils.data import Dataset
from torchvision.transforms import v2 as transforms
import torchvision.utils as vutils
import random as rnd

SAFE_SLOW_VALIDATE_IMAGES = False


class TrackingCompose(transforms.Compose):
    """Tracks applied transforms and validates them for train/val samples."""

    def __init__(self, transforms: List[Any], transform_type: str):
        super().__init__(transforms)
        self.transform_type = transform_type
        self.applied_transforms = set()

    def __call__(self, img: Image.Image) -> torch.Tensor:
        allowed_transforms = {"Resize", "ToTensor", "Normalize", "Compose"}
        self.applied_transforms.clear()
        for t in self.transforms:
            img = t(img)
            self.applied_transforms.add(type(t).__name__)
        if self.transform_type == "val" and not self.applied_transforms.issubset(allowed_transforms):
            raise ValueError(
                f"Validation sample processed with non-allowed transforms: {self.applied_transforms - allowed_transforms}")
        return img


def get_dynamic_augmentations(include_normalize: bool = True, boost: bool = False) -> List[Tuple[str, Any]]:
    """Generate dynamic augmentation configurations based on settings."""
    base_transforms_pre, base_transforms_post = get_transforms(get_compose=False)

    p_01, p_02, p_025, p_05, p_2 = (0.1, 0.2, 0.25, 0.5, 2) if boost else (0.2, 0.4, 0.5, 0.7, 2)
    m_pt = 1 if boost else 1.75

    configs = [
        ("include_random_horizontal_flip", transforms.RandomHorizontalFlip(p=p_05)),
        ("include_random_autocontrast", transforms.RandomAutocontrast(p=p_02)),
        ("include_random_sharpness", transforms.RandomAdjustSharpness(p=p_02, sharpness_factor=p_2 * m_pt)),
        ("include_random_perspective", transforms.RandomPerspective(p=0.1 * m_pt, distortion_scale=p_02)),
        ("include_color_jitter", transforms.ColorJitter(brightness=0.35, contrast=p_01, saturation=0.3, hue=0.3)),
        ("include_random_grayscale", transforms.RandomGrayscale(p=0.25)),
        ("include_random_affine",
         transforms.RandomApply([transforms.RandomAffine(degrees=(20 * m_pt, 20 * m_pt))], p=p_01)),
        ("include_random_rotation", transforms.RandomApply([transforms.RandomRotation(degrees=(-20, 20))], p=p_01)),
    ]

    if include_normalize:
        base_pre_configs = [("base_pre_" + str(i), transform) for i, transform in enumerate(base_transforms_pre)]
        base_post_configs = [("base_post_" + str(i), transform) for i, transform in enumerate(base_transforms_post)]
    else:
        base_pre_configs, base_post_configs = [], []

    final_configs = base_pre_configs + configs + base_post_configs
    final_configs.append(("include_random_erasing",
                          transforms.RandomErasing(p=0.1 * m_pt, scale=(0.02, 0.33), ratio=(0.3, 3.3), value=0,
                                                   inplace=False)))
    return final_configs


class AgeGenderDataset(Dataset):
    """Dataset for age and gender classification tasks."""

    def __init__(
            self,
            root_dir: str,
            transform: Optional[Any] = None,
            use_dynamic_augmentation: bool = False,
            indices: Optional[List[int]] = None,
            num_aug_bins: Optional[int] = None,
            dynamic_augmentation_mult: Optional[float] = None,
    ):
        self.root_dir = root_dir
        self.transform = transform
        self.augmented_indices: List[Tuple[int, bool]] = []
        self.use_dynamic_augmentation = use_dynamic_augmentation

        self.image_files = [f for f in os.listdir(root_dir) if f.endswith(".jpg")]
        self.valid_images: List[str] = []
        self.ages: List[int] = []
        self.genders: List[int] = []

        print(f"Sample images found: {len(self.image_files)}")

        for img_name in self.image_files:
            img_path = os.path.join(root_dir, img_name)
            try:
                if SAFE_SLOW_VALIDATE_IMAGES:
                    with Image.open(img_path) as img:
                        img.verify()
                splits = img_name.split("_")
                age = int(splits[0])
                gender = int(splits[1])
                self.valid_images.append(img_path)
                self.ages.append(age)
                self.genders.append(gender)
            except Exception as e:
                print(f"Error with image {img_path}: {e}")

        if indices is not None:
            self.valid_images = [self.valid_images[i] for i in indices]
            self.ages = [self.ages[i] for i in indices]
            self.genders = [self.genders[i] for i in indices]

        self.bins, self.bin_frequencies = self.calculate_age_bins(self.ages)
        self.log_distribution("Initial")

        if self.use_dynamic_augmentation:
            self.augmented_indices = self.create_augmented_indices(mult=dynamic_augmentation_mult)
            self.bins, self.bin_frequencies = self.calculate_age_bins(self.ages, include_augmented=True)
            self.log_distribution("After Augmentation")

        print(f"Valid images: {len(self.valid_images)}")
        print(f"Gender distribution: {self.gender_distribution()}")
        print(f"Age range: {min(self.ages)} - {max(self.ages)}")

    def get_image_files(self) -> List[str]:
        """Return the list of image files."""
        return self.image_files

    def calculate_age_bins(self, ages: List[int], num_bins: int = 9, include_augmented: bool = False) -> Tuple[
        List[List[int]], List[int]]:
        """Calculate age bins and their frequencies."""
        bins = [[] for _ in range(num_bins)]
        bin_frequencies = [0] * num_bins

        for idx, age in enumerate(ages):
            bin_idx = min(age // 10, num_bins - 1)
            bins[bin_idx].append(idx)
            bin_frequencies[bin_idx] += 1

        if include_augmented and self.use_dynamic_augmentation:
            for orig_idx, _ in self.augmented_indices:
                age = ages[orig_idx]
                bin_idx = min(age // 10, num_bins - 1)
                bin_frequencies[bin_idx] += 1

        return bins, bin_frequencies

    def create_augmented_indices(self, mult: float = 0.25) -> List[Tuple[int, bool]]:
        """Create indices for augmented samples to balance the dataset."""
        max_freq = max(self.bin_frequencies)
        augmented_indices = []
        for bin_idx, indices in enumerate(self.bins):
            num_to_add = int(int((max_freq - (len(indices))) * mult) + max_freq * 0.1)
            augmented_indices.extend([(idx, True) for idx in indices * (num_to_add // len(indices) + 1)][:num_to_add])
        return augmented_indices

    def get_bin_info(self) -> Tuple[List[List[int]], List[int]]:
        """Return age bins and their frequencies."""
        return self.bins, self.bin_frequencies

    def gender_distribution(self) -> Dict[int, int]:
        """Return the gender distribution of the dataset."""
        return {0: self.genders.count(0), 1: self.genders.count(1)}

    def log_distribution(self, stage: str) -> None:
        """Log the age and gender distribution of the dataset."""
        bins, frequencies = self.get_bin_info()
        print(f"\n{stage} age distribution:")
        for i, count in enumerate(frequencies):
            print(f"Bin {i * 10}-{(i + 1) * 10 - 1}: {count}")
        print(f"Gender distribution: {self.gender_distribution()}")
        print(f"Age distribution: {self.age_distribution()}")
        print(f"Total samples: {len(self)}")

    def age_distribution(self) -> List[Tuple[int, int]]:
        """Return the age distribution of the dataset."""
        return Counter(self.ages).most_common()

    def __len__(self) -> int:
        return len(self.valid_images) + len(self.augmented_indices)

    def apply_dynamic_augmentation(self, img: Image.Image) -> torch.Tensor:
        """Apply dynamic augmentation to an image."""
        augmentation_configs = get_dynamic_augmentations()
        transforms_list = [transform for _, transform in augmentation_configs]
        augment_transform = transforms.Compose(transforms_list)
        return augment_transform(img)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int, int, int, str]:
        """Get a sample from the dataset."""
        orig_idx = None
        if idx < len(self.valid_images):
            img_path = self.valid_images[idx]
            age = self.ages[idx]
            gender = self.genders[idx]
            is_augmented = False
        else:
            aug_idx = idx - len(self.valid_images)
            orig_idx, is_augmented = self.augmented_indices[aug_idx]
            img_path = self.valid_images[orig_idx]
            age = self.ages[orig_idx]
            gender = self.genders[orig_idx]

        image = Image.open(img_path).convert("RGB")

        if is_augmented:
            image = self.apply_dynamic_augmentation(image)
        else:
            image = self.transform(image)

        source_image = self.image_files[orig_idx] if orig_idx is not None else self.image_files[idx]
        return image, age, gender, idx, source_image


def get_transforms_configs() -> List[Tuple[str, Any]]:
    """Return a list of transform configurations."""
    return [
        ("include_random_horizontal_flip", transforms.RandomHorizontalFlip(p=0.5)),
        ("include_random_rotation", transforms.RandomRotation(degrees=(-10, 10))),
        ("include_random_perspective", transforms.RandomPerspective(distortion_scale=0.2, p=0.2)),
        ("include_random_affine", transforms.RandomAffine(degrees=(-10, 10), translate=(0.1, 0.1), scale=(0.9, 1.1))),
        ("include_color_jitter", transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1)),
        ("include_random_grayscale", transforms.RandomGrayscale(p=0.1)),
        ("include_gaussian_blur", transforms.GaussianBlur(kernel_size=3, sigma=(0.1, 2.0))),
    ]


class WeightedAgeGenderSampler(torch.utils.data.Sampler):
    """Weighted sampler for balancing age and gender distribution."""

    def __init__(self, dataset: AgeGenderDataset, bins: List[List[int]], bin_frequencies: List[int],
                 replacement: bool = True):
        self.dataset = dataset
        self.bins = bins
        self.bin_frequencies = bin_frequencies
        self.replacement = replacement
        self.weights = self._calculate_weights()

    def _calculate_weights(self) -> torch.Tensor:
        """Calculate sampling weights based on bin frequencies."""
        if not self.bins or not self.bin_frequencies:
            return torch.ones(len(self.dataset))

        weights = torch.zeros(len(self.dataset))
        max_freq = max(self.bin_frequencies)
        for bin_idx, indices in enumerate(self.bins):
            bin_weight = max_freq / self.bin_frequencies[bin_idx]
            for idx in indices:
                weights[idx] = bin_weight

        print(f"WeightedAgeGenderSampler:\nweights=\n{weights}")
        return weights

    def __iter__(self):
        return iter(torch.multinomial(self.weights, len(self.dataset), self.replacement).tolist())

    def __len__(self) -> int:
        return len(self.dataset)


def get_transforms(get_compose: bool = False) -> Any:
    """Get the list of transforms or a composed transform."""
    base_transforms_pre = [transforms.Resize((224, 224))]
    base_transforms_post = [
        transforms.Compose([transforms.ToImage(), transforms.ToDtype(torch.float32, scale=True)]),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ]

    if get_compose:
        return transforms.Compose(base_transforms_pre + base_transforms_post)
    return base_transforms_pre, base_transforms_post


class AgeGenderDataModule(pl.LightningDataModule):
    """PyTorch Lightning data module for age and gender classification."""

    def __init__(self, config: Dict[str, Any], mode: str = "train"):
        super().__init__()
        self.config = config
        self.mode = mode
        self.bins: Optional[List[List[int]]] = None
        self.bin_frequencies: Optional[List[int]] = None

    def save_sample_images(self, N: int = 100, debug_folder: str = "_debug_images") -> None:
        """Save sample images from the dataset for debugging."""
        if os.path.exists(debug_folder):
            shutil.rmtree(debug_folder)
        os.makedirs(debug_folder, exist_ok=True)

        total_samples = len(self.train_dataset)
        sample_indices = rnd.sample(range(total_samples), min(N, total_samples))

        for idx in sample_indices:
            image, age, gender, original_idx, _ = self.train_dataset[idx]
            is_augmented = idx >= len(self.train_dataset.valid_images)
            img = vutils.make_grid(image, normalize=True, scale_each=True)
            img = transforms.ToPILImage()(img)
            filename = f"sample_{idx}_age{age}_gender{gender}_{'augmented' if is_augmented else 'original'}.png"
            img.save(os.path.join(debug_folder, filename))

        print(f"Saved {N} sample images to {debug_folder}")

    def setup(self, stage: Optional[str] = None) -> None:
        """Set up the datasets for each stage."""
        base_transforms_pre, base_transforms_post = get_transforms(get_compose=False)

        val_transforms = base_transforms_pre + base_transforms_post
        val_transform = TrackingCompose(val_transforms, "val")

        if self.mode == "test":
            self.test_dataset = AgeGenderDataset(
                self.config["ds_path"], transform=val_transform
            )
        else:
            transform_list = []
            transform_configs = get_transforms_configs()

            for config_key, transform in transform_configs:
                if self.config.get(config_key, False):
                    print(f"+{transform.__class__.__name__}")
                    transform_list.append(transform)

            train_transforms: List[transforms.Transform] = (
                    base_transforms_pre + transform_list + base_transforms_post
            )

            if self.config.get("include_random_erasing", False):
                print("+transforms.RandomErasing")
                train_transforms.append(
                    transforms.RandomErasing(
                        p=0.1,
                        scale=(0.02, 0.33),
                        ratio=(0.3, 3.3),
                        value=0,
                        inplace=False,
                    )
                )

            if self.config.get("include_gaussian_noise", False):
                print("+Lambda(add_gaussian_noise)")
                train_transforms.append(
                    transforms.Lambda(lambda x: x + torch.randn_like(x) * 0.1)
                )

            train_transform = TrackingCompose(train_transforms, "train")

            use_dynamic_augmentation = self.config.get("use_dynamic_augmentation", False)
            num_aug_bins = self.config.get("num_aug_bins", False)

            if self.config.get("train_path") and self.config.get("val_path"):
                self.train_dataset = AgeGenderDataset(
                    self.config.get("train_path"),
                    transform=train_transform,
                    use_dynamic_augmentation=use_dynamic_augmentation,
                    num_aug_bins=num_aug_bins,
                    dynamic_augmentation_mult=self.config.get("dynamic_augmentation_mult", 1),
                )
                self.val_dataset = AgeGenderDataset(
                    self.config.get("val_path"), transform=val_transform
                )

                print(f"Not splitting train/val samples, using:")
                print(f"train_path={self.config.get('train_path')}")
                print(f"val_path={self.config.get('val_path')}")

            else:
                temp_dataset = AgeGenderDataset(
                    self.config["ds_path"],
                    transform=None,
                    use_dynamic_augmentation=False,
                )
                print(f"Gender Distribution:\n{temp_dataset.gender_distribution()}")
                print(f"Age Distribution:\n{temp_dataset.age_distribution()}")
                train_size = int(0.8 * len(temp_dataset))
                val_size = len(temp_dataset) - train_size

                train_indices, val_indices = torch.utils.data.random_split(
                    range(len(temp_dataset)),
                    [train_size, val_size],
                    generator=torch.Generator().manual_seed(42),
                )

                self.train_dataset = AgeGenderDataset(
                    self.config["ds_path"],
                    transform=train_transform,
                    use_dynamic_augmentation=use_dynamic_augmentation,
                    num_aug_bins=num_aug_bins,
                    dynamic_augmentation_mult=self.config.get("dynamic_augmentation_mult", 1),
                    indices=train_indices,
                )
                self.val_dataset = AgeGenderDataset(
                    self.config["ds_path"],
                    transform=val_transform,
                    use_dynamic_augmentation=False,
                    indices=val_indices,
                )

                print(f"Splitting train/val samples, using:")
                print(f"ds_path={self.config.get('ds_path')}")

            print(f"Train dataset size: {len(self.train_dataset)}")
            print(f"Validation dataset size: {len(self.val_dataset)}")

            print("\nTrain Dataset Statistics:")
            print(f"Gender Distribution:\n{self.train_dataset.gender_distribution()}")
            print(f"Age Distribution:\n{self.train_dataset.age_distribution()}")

            print("\nValidation Dataset Statistics:")
            print(f"Gender Distribution:\n{self.val_dataset.gender_distribution()}")
            print(f"Age Distribution:\n{self.val_dataset.age_distribution()}")

            if stage == "fit" or stage is None:
                self.save_sample_images(debug_folder="debug_images")

    def train_dataloader(self) -> DataLoader:
        """Create and return the train DataLoader."""
        return DataLoader(
            self.train_dataset,
            batch_size=self.config["batch_size"],
            shuffle=True,
            num_workers=16,
            persistent_workers=True,
            collate_fn=self.collate_fn,
        )

    def test_dataloader(self) -> DataLoader:
        """Create and return the test DataLoader."""
        return DataLoader(
            self.test_dataset,
            batch_size=self.config["batch_size"],
            num_workers=16,
            persistent_workers=True,
            collate_fn=self.collate_fn,
        )

    def val_dataloader(self) -> DataLoader:
        """Create and return the validation DataLoader."""
        return DataLoader(
            self.val_dataset,
            batch_size=self.config["batch_size"],
            num_workers=16,
            persistent_workers=True,
            collate_fn=self.collate_fn,
        )

    def collate_fn(self, batch: List[Tuple[torch.Tensor, int, int, int, str]]) -> Tuple[
        torch.Tensor, torch.Tensor, torch.Tensor, Tuple[str, ...]]:
        """Custom collate function for DataLoader."""
        images, ages, genders, _, image_paths = zip(*batch)
        images = torch.stack(images)
        ages = torch.tensor(ages)
        genders = torch.tensor(genders)
        return images, ages, genders, image_paths
