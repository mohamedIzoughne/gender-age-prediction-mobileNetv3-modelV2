import optuna
from optuna.integration import FastAIPruningCallback
import numpy as np
import pandas as pd
import gc
from enum import Enum, auto
import pytorch_lightning as pl
import torchvision.transforms as transforms
from torch.utils.data import Dataset
from typing import List, Optional, Tuple, Union

FIXED_SEED = 42


class ModelType(Enum):
    """Enum for supported model types."""
    RESNET18 = auto()
    RESNET34 = auto()
    RESNET50 = auto()
    MOBILENET_V3_LARGE = auto()


class GenderDataset(Dataset):
    """Dataset for gender classification from images."""

    def __init__(self, root_dir: str, transform: Optional[transforms.Compose] = None):
        self.root_dir = root_dir
        self.transform = transform
        self.image_files = [f for f in os.listdir(root_dir) if f.endswith(".jpg")]

    def __len__(self) -> int:
        return len(self.image_files)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int]:
        img_name = self.image_files[idx]
        img_path = os.path.join(self.root_dir, img_name)
        image = Image.open(img_path).convert("RGB")
        gender = int(img_name.split("_")[1])
        if self.transform:
            image = self.transform(image)
        return image, gender


class GenderDataModule(pl.LightningDataModule):
    """PyTorch Lightning DataModule for gender classification."""

    def __init__(self, data_dir: str, batch_size: int = 32, num_workers: int = 4):
        super().__init__()
        self.data_dir = data_dir
        self.batch_size = batch_size
        self.num_workers = num_workers

    def setup(self, stage: Optional[str] = None):
        """Prepare datasets for training and validation."""
        self.transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.RandomHorizontalFlip(),
            transforms.RandomRotation(25),
            transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
            transforms.RandomAffine(degrees=0, translate=(0.1, 0.1), scale=(0.9, 1.1)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ])
        full_dataset = GenderDataset(self.data_dir, transform=self.transform)
        train_size = int(0.8 * len(full_dataset))
        val_size = len(full_dataset) - train_size
        self.train_dataset, self.val_dataset = random_split(full_dataset, [train_size, val_size])

    def train_dataloader(self) -> DataLoader:
        """Create training dataloader."""
        return DataLoader(self.train_dataset, batch_size=self.batch_size, num_workers=self.num_workers, shuffle=True)

    def val_dataloader(self) -> DataLoader:
        """Create validation dataloader."""
        return DataLoader(self.val_dataset, batch_size=self.batch_size, num_workers=self.num_workers)


def _get_base_model(model_type: ModelType):
    """Get the base model based on the model type."""
    if model_type == ModelType.RESNET18:
        return models.resnet18
    elif model_type == ModelType.RESNET34:
        return models.resnet34
    elif model_type == ModelType.RESNET50:
        return models.resnet50
    elif model_type == ModelType.MOBILENET_V3_LARGE:
        return models.mobilenet_v3_large
    else:
        raise ValueError(f"Unsupported model type: {model_type}")


def utils_set_seed(seed: int = FIXED_SEED):
    """Set seeds for reproducibility."""
    set_seed(seed)
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


utils_set_seed(FIXED_SEED)


def create_dataloaders(batch_size: int, aug_mode: str, path: str, is_test: bool = False,
                       valid_pct: float = 0.2) -> DataLoaders:
    """Create DataLoaders for training and validation."""
    utils_set_seed(FIXED_SEED)
    normalization = Normalize.from_stats(*imagenet_stats)
    print(f"Augmentation Mode: {aug_mode} | Data Type: {'Test' if is_test else 'Train'}")

    if is_test:
        valid_pct = 0.001
        aug_mode = None

    batch_tfms = []
    if aug_mode == "mult_2":
        batch_tfms = aug_transforms(mult=2.0, do_flip=True, flip_vert=True, max_rotate=25.0)
    elif aug_mode == "mult_2_no_trans":
        batch_tfms = aug_transforms(mult=2.0)
    elif aug_mode in ("mult_1.25_more_trans", "mult_1.0_more_trans", "mult_0.75_more_trans", "mult_2.0_more_trans"):
        mult = float(aug_mode.split("_")[1])
        batch_tfms = aug_transforms(
            mult=mult, do_flip=True, flip_vert=True, max_rotate=25.0, min_zoom=0.75, max_zoom=1.8, max_warp=0.15,
            p_affine=0.05, xtra_tfms=[RandomErasing(p=0.2, sl=0.0, sh=0.3)]
        )
    elif aug_mode == "min_aug":
        batch_tfms = aug_transforms(size=224, min_scale=0.75)
    else:
        print("No augmentations applied.")

    batch_tfms = [*batch_tfms, normalization] if batch_tfms else [normalization]

    return ImageDataLoaders.from_folder(
        path, valid_pct=valid_pct, seed=FIXED_SEED, item_tfms=Resize(224), shuffle=False,
        batch_tfms=batch_tfms, num_workers=20, bs=batch_size
    )


def compute_weights(data: DataLoaders, path: str) -> torch.Tensor:
    """Compute class weights for balancing."""
    label_dict = {v: k for k, v in enumerate(data.vocab)}
    counts = np.zeros(len(label_dict))
    for label in label_dict:
        dir_path = os.path.join(path, label)
        counts[label_dict[label]] = len(os.listdir(dir_path))
    total_counts = np.sum(counts)
    class_weights = total_counts / (len(label_dict) * counts)
    return torch.tensor(class_weights, dtype=torch.float, device="cuda")


def create_learner(data: DataLoaders, weight_decay: float, balancing: str,
                   model_type: ModelType = ModelType.RESNET34) -> Learner:
    """Create a FastAI Learner for training."""
    base_model = _get_base_model(model_type)
    print(f"using {model_type.name}")
    utils_set_seed(FIXED_SEED)
    loss_func = nn.CrossEntropyLoss(
        weight=compute_weights(data, path)) if balancing == "class_weight" else LabelSmoothingCrossEntropy()
    return vision_learner(
        data, base_model,
        metrics=[accuracy, F1Score(average="weighted"), F1Score(average="micro"), F1Score(average="macro")],
        wd=weight_decay, loss_func=loss_func
    )


def train_model(learn: Learner, base_lr: float, lr_mult: float, lr_scheduler_type: str,
                freeze_epochs: int, num_epochs: int, pct_start: float,
                callbacks: Optional[List[Callback]] = None) -> Learner:
    """Train the model using specified parameters."""
    utils_set_seed(FIXED_SEED)
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is not available. Training on GPU is not possible.")
    learn.model.to("cuda")
    if not next(learn.model.parameters()).is_cuda:
        raise Exception("Model is not on GPU")

    if lr_scheduler_type == "one_cycle":
        learn.fit_one_cycle(freeze_epochs, slice(base_lr), pct_start=0.99, cbs=callbacks)
        learn.unfreeze()
        learn.fit_one_cycle(num_epochs, slice(base_lr / lr_mult, base_lr), pct_start=pct_start, cbs=callbacks)
    elif lr_scheduler_type == "flat_cos":
        learn.fit_flat_cos(freeze_epochs, slice(base_lr), pct_start=0.99, cbs=callbacks)
        learn.unfreeze()
        learn.fit_flat_cos(num_epochs, slice(base_lr / lr_mult, base_lr), pct_start=pct_start, cbs=callbacks)
    return learn


def extract_metrics(learn: Learner) -> dict:
    """Extract training metrics from the learner."""
    return {
        "train_loss": learn.recorder.final_record[0],
        "valid_loss": learn.recorder.final_record[1],
        "accuracy": learn.recorder.final_record[2],
        "f1_weighted": learn.recorder.final_record[3],
        "f1_micro": learn.recorder.final_record[4],
        "f1_macro": learn.recorder.final_record[5],
    }


def objective(trial: optuna.Trial) -> float:
    """Objective function for Optuna optimization."""
    utils_set_seed(FIXED_SEED)
    batch_size = trial.suggest_categorical("batch_size", [64, 128, 224, 256])
    base_lr = trial.suggest_float("base_lr", 0.001, 0.01, log=True)
    weight_decay = trial.suggest_float("weight_decay", 0.00005, 0.1, log=True)
    lr_mult = trial.suggest_categorical("lr_mult", [10, 50, 100])
    lr_scheduler_type = trial.suggest_categorical("lr_scheduler", ["one_cycle", "flat_cos"])
    freeze_epochs = trial.suggest_int("freeze_epochs", 1, 6)
    pct_start = trial.suggest_float("pct_start", 0.01, 0.99)
    aug_mode = trial.suggest_categorical("aug_mode", [None, "mult_2", "mult_2.0_more_trans", "mult_1.25_more_trans"])
    num_epochs = 25

    data = create_dataloaders(batch_size, aug_mode=aug_mode)
    learn = create_learner(data, weight_decay, balancing=None)
    callbacks = [FastAIPruningCallback(trial, monitor="valid_loss"),
                 EarlyStoppingCallback(monitor="valid_loss", patience=5)]
    learn = train_model(learn, base_lr, lr_mult, lr_scheduler_type, freeze_epochs, num_epochs, pct_start,
                        callbacks=callbacks)
    metrics = extract_metrics(learn)

    print(f'Trial {trial.number}: train_loss: {metrics["train_loss"]}, valid_loss: {metrics["valid_loss"]}, '
          f'Accuracy {metrics["accuracy"]}, F1 Weighted {metrics["f1_weighted"]}, '
          f'F1 Micro {metrics["f1_micro"]}, F1 Macro {metrics["f1_macro"]}')

    del data, learn
    torch.cuda.empty_cache()
    gc.collect()
    return metrics["f1_weighted"]


def get_study_params(row: pd.Series) -> dict:
    """Extract study parameters from a pandas Series."""
    return {col.replace("params_", ""): None if isinstance(value, float) and math.isnan(value) else value
            for col, value in row.items() if col.startswith("params")}