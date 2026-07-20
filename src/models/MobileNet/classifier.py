"""
This module implements an Age and Gender Classifier using PyTorch Lightning.
It includes custom callbacks, learning rate schedulers, and a prediction function.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import models
import pytorch_lightning as pl
from pytorch_lightning.callbacks import Callback
from torchmetrics import Accuracy, MeanAbsoluteError
from tqdm import tqdm
import numpy as np
from src.models.MobileNet.data_defs import AgeGenderDataModule
from typing import Dict, Any, List, Tuple, Optional

torch.backends.cudnn.benchmark = True


class ProgressBarToggleCallback(Callback):
    """Toggles the progress bar on for training and off for other phases."""

    def on_train_start(self, trainer: pl.Trainer, pl_module: pl.LightningModule) -> None:
        trainer.enable_progress_bar = True  # Enable for training

    def on_train_end(self, trainer: pl.Trainer, pl_module: pl.LightningModule) -> None:
        trainer.enable_progress_bar = False  # Disable after training


class OneCycleWithDecay(torch.optim.lr_scheduler.OneCycleLR):
    """One Cycle learning rate scheduler with decay after cycle completion."""

    def __init__(self, optimizer: torch.optim.Optimizer, decay_factor: float = 1.01, *args: Any, **kwargs: Any) -> None:
        super().__init__(optimizer, *args, **kwargs)
        self.decay_factor = decay_factor

    def get_lr(self) -> List[float]:
        def _calc_lr(g_lr: float) -> float:
            new_lr = g_lr * self.decay_factor
            if new_lr > 0.001:
                return 0.001
            return new_lr

        if self.last_epoch < self.total_steps:
            return super().get_lr()
        return [_calc_lr(group["lr"]) for group in self.optimizer.param_groups]

    def step(self, epoch: Optional[int] = None) -> None:
        if self.last_epoch >= self.total_steps:
            self.last_epoch += 1
            for param_group, lr in zip(self.optimizer.param_groups, self.get_lr()):
                param_group["lr"] = lr
        else:
            super().step(epoch)

    def state_dict(self) -> Dict[str, Any]:
        state_dict = super().state_dict()
        state_dict["decay_factor"] = self.decay_factor
        return state_dict

    def load_state_dict(self, state_dict: Dict[str, Any]) -> None:
        self.decay_factor = state_dict.pop("decay_factor")
        super().load_state_dict(state_dict)


class AgeGenderClassifier(pl.LightningModule):
    """PyTorch Lightning module for age and gender classification."""

    def __init__(self, config: Dict[str, Any]) -> None:
        super().__init__()
        self.config = config
        self._initialize_model()
        self.gender_loss = nn.CrossEntropyLoss()
        self.age_loss = nn.L1Loss()
        self.gender_accuracy = Accuracy(task="binary")
        self.train_gender_accuracy = Accuracy(task="binary")
        self.age_mae = MeanAbsoluteError()

    def get_param(self, key: str, default: Any = None) -> Any:
        """Safely get a parameter from the config."""
        if key not in self.config:
            if default is None:
                raise ValueError(
                    f"Missing required parameter:\n'{key}'.\n Full config: {self.config}"
                )
            return default
        return self.config[key]

    def check_freeze_base_model(self) -> None:
        """Freeze the base model if specified in config."""
        if self.get_param("freeze_epochs") > 0:
            print(f'Freezing base model for: {self.get_param("freeze_epochs")}')
            for param in self.base_model.parameters():
                param.requires_grad = False

    def check_unfreeze_base_model(self) -> None:
        """Unfreeze the base model after specified epochs."""
        if self.current_epoch == self.get_param("freeze_epochs"):
            print(
                f"Unfreezing base model:\n  after self.current_epoch({self.current_epoch}) == self.get_param('freeze_epochs')({self.get_param('freeze_epochs')})"
            )
            for param in self.base_model.parameters():
                param.requires_grad = True

    def _initialize_model(self) -> None:
        """Initialize the model architecture based on config."""
        model_type = self.get_param("model_type")
        pretrained = self.get_param("pretrained", True)

        print(f"using model_type = {model_type} || pretrained = {pretrained} ||")

        if model_type == "mobilenet_v3_large":
            print("using mobilenet_v3_large")
            self.base_model = models.mobilenet_v3_large(pretrained=pretrained)
        elif model_type == "mobilenet_v3_small":
            self.base_model = models.mobilenet_v3_small(pretrained=pretrained)
        elif model_type == "efficientnet_b0":
            self.base_model = models.efficientnet_b0(pretrained=pretrained)
        else:
            raise ValueError(f"Unsupported model type: {model_type}")

        if "mobilenet" in model_type:
            if hasattr(self.base_model, "classifier"):
                if isinstance(self.base_model.classifier, nn.Sequential):
                    num_features = self.base_model.classifier[0].in_features
                else:
                    num_features = self.base_model.classifier.in_features
            else:
                num_features = self.base_model.last_channel  # Fallback if classifier is not present

            self.base_model = nn.Sequential(*list(self.base_model.children())[:-1])

        elif "efficientnet" in model_type:
            num_features = self.base_model.classifier[1].in_features
            self.base_model = nn.Sequential(*list(self.base_model.children())[:-2])
        else:
            raise ValueError(f"Unexpected model type: {model_type}")

        self.global_pool = nn.AdaptiveAvgPool2d(1)

        dropout_rate = self.get_param("dropout", 0)

        self.gender_classifier = nn.Sequential(
            nn.Dropout(p=dropout_rate), nn.Linear(num_features, 2)
        )
        self.age_regressor = nn.Sequential(
            nn.Dropout(p=dropout_rate), nn.Linear(num_features, 1)
        )

        self.check_freeze_base_model()

        if dropout_rate > 0:
            self.dropout = nn.Dropout(p=dropout_rate)

    def on_train_epoch_start(self) -> None:
        """Check if base model should be unfrozen at the start of each epoch."""
        self.check_unfreeze_base_model()

    def on_train_epoch_end(self) -> None:
        """Placeholder for any end-of-epoch operations."""
        pass

    def test_step(self, batch: Tuple[torch.Tensor, torch.Tensor, torch.Tensor, Any, Any], batch_idx: int) -> Dict[str, torch.Tensor]:
        """Perform a test step."""
        x, age, gender, _, _ = batch
        gender_pred, age_pred = self(x)

        return {
            "gender_pred": gender_pred,
            "age_pred": age_pred,
            "true_gender": gender,
            "true_age": age,
        }

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Forward pass through the model."""
        features = self.base_model(x)
        features = self.global_pool(features).view(x.size(0), -1)
        gender_output = self.gender_classifier(features)
        age_output = self.age_regressor(features).squeeze(1)
        return gender_output, age_output

    def get_current_lr(self) -> float:
        """Get the current learning rate."""
        return self.optimizers().param_groups[0]["lr"]

    def training_step(self, batch: Tuple[torch.Tensor, torch.Tensor, torch.Tensor, Any, Any], batch_idx: int) -> torch.Tensor:
        """Perform a training step."""
        current_lr = self.get_current_lr()
        self.log("step_LR", current_lr, on_step=True, on_epoch=False, prog_bar=True)

        x, age, gender, _, _ = batch
        gender_pred, age_pred = self(x)

        gender_loss = self.gender_loss(gender_pred, gender)
        age_loss = self.age_loss(age_pred, age.float())

        total_loss = (
            self.get_param("gender_loss_weight") * gender_loss
            + (1 - self.get_param("gender_loss_weight")) * age_loss
        )

        l1_lambda = self.get_param("l1_lambda", 0)
        if l1_lambda > 0:
            l1_norm = sum(
                p.abs().sum() for p in self.gender_classifier[1].parameters()
            ) + sum(p.abs().sum() for p in self.age_regressor[1].parameters())
            total_loss += l1_lambda * l1_norm

        train_gender_acc = self.train_gender_accuracy(
            torch.argmax(gender_pred, dim=1), gender
        )

        self.log(
            "train_gender_loss",
            gender_loss,
            on_step=False,
            on_epoch=True,
            prog_bar=True,
        )
        self.log(
            "train_age_loss", age_loss, on_step=False, on_epoch=True, prog_bar=True
        )
        self.log(
            "train_total_loss", total_loss, on_step=False, on_epoch=True, prog_bar=True
        )
        self.log(
            "train_gender_acc",
            train_gender_acc,
            on_step=False,
            on_epoch=True,
            prog_bar=True,
        )

        return total_loss

    def validation_step(self, batch: Tuple[torch.Tensor, torch.Tensor, torch.Tensor, Any, Any], batch_idx: int) -> torch.Tensor:
        """Perform a validation step."""
        x, age, gender, _, _ = batch
        gender_pred, age_pred = self(x)

        gender_loss = self.gender_loss(gender_pred, gender)
        age_loss = self.age_loss(age_pred, age.float())

        total_loss = (
            self.get_param("gender_loss_weight") * gender_loss
            + (1 - self.get_param("gender_loss_weight")) * age_loss
        )

        gender_acc = self.gender_accuracy(torch.argmax(gender_pred, dim=1), gender)
        age_mae = self.age_mae(age_pred, age.float())

        self.log(
            "val_gender_loss", gender_loss, prog_bar=True, on_epoch=True, on_step=False
        )
        self.log("val_age_loss", age_loss, prog_bar=True, on_epoch=True, on_step=False)
        self.log(
            "val_total_loss", total_loss, prog_bar=True, on_epoch=True, on_step=False
        )
        self.log(
            "val_gender_acc", gender_acc, prog_bar=True, on_epoch=True, on_step=False
        )
        self.log("val_age_mae", age_mae, prog_bar=True, on_epoch=True, on_step=False)

        return total_loss

    def configure_optimizers(self) -> Dict[str, Any]:
        """Configure optimizers and learning rate schedulers."""
        optimizer = torch.optim.AdamW(
            self.parameters(),
            lr=self.get_param("base_lr"),
            weight_decay=self.get_param("weight_decay"),
        )
        if self.get_param("lr_scheduler") == "one_cycle":
            # Calculate total steps based on override_cycle_epoch_count if available
            override_epochs = self.get_param("override_cycle_epoch_count", None)
            if override_epochs is not None:
                dataloader = self.trainer.datamodule.train_dataloader()
                steps_per_epoch = len(dataloader)
                total_steps = steps_per_epoch * override_epochs

                print(
                    f"override_cycle_epoch_count = {override_epochs} total_steps={total_steps} estimated real = {self.trainer.estimated_stepping_batches}"
                )
            else:
                total_steps = self.trainer.estimated_stepping_batches

            scheduler = OneCycleWithDecay(
                optimizer,
                decay_factor=1.0055,  # This will reduce LR by 1% every step after the cycle
                max_lr=self.get_param("max_lr"),
                total_steps=total_steps,
                pct_start=self.get_param("pct_start"),
                anneal_strategy=self.get_param("anneal_strategy"),
                div_factor=self.get_param("div_factor"),
                final_div_factor=self.get_param("final_div_factor"),
            )

            return {
                "optimizer": optimizer,
                "lr_scheduler": {"scheduler": scheduler, "interval": "step"},
            }

        elif self.get_param("lr_scheduler") == "reduce_on_plateau":
            scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
                optimizer,
                mode="min",
                threshold_mode="rel",
                factor=self.get_param("factor"),
                patience=self.get_param("patience"),
                threshold=self.get_param("threshold"),
            )

            p = {
                "optimizer": optimizer,
                "lr_scheduler": {
                    "scheduler": scheduler,
                    "monitor": "val_total_loss",
                    "interval": "epoch",
                },
            }
            print(
                f"ReduceLROnPlateau params:"
                f"factor={self.get_param('factor')}\n"
                f"patience={self.get_param('patience')}\n"
                f"threshold={self.get_param('threshold')}\n"
                f"\n{p}\n--\n"
            )

            return p
        elif self.get_param("lr_scheduler") == "step_lr":
            scheduler = torch.optim.lr_scheduler.StepLR(
                optimizer,
                step_size=self.get_param("step_size"),
                gamma=self.get_param("gamma"),
            )
            return {
                "optimizer": optimizer,
                "lr_scheduler": {"scheduler": scheduler, "interval": "epoch"},
            }
        else:
            return optimizer


def predict_with_model(
    model: AgeGenderClassifier, datamodule: AgeGenderDataModule, batch_size: int = 32
) -> Dict[str, np.ndarray]:
    """
    Generate predictions using the trained model.

    Args:
        model (AgeGenderClassifier): The trained model.
        datamodule (AgeGenderDataModule): The data module containing the test dataset.
        batch_size (int): Batch size for predictions.

    Returns:
        Dict[str, np.ndarray]: Dictionary containing predictions and true values.
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    model.eval()

    dataloader = DataLoader(
        datamodule.test_dataset,
        batch_size=batch_size,
        num_workers=21,
        persistent_workers=True,
        collate_fn=datamodule.collate_fn,
    )

    all_gender_preds = []
    all_age_preds = []
    all_true_genders = []
    all_true_ages = []
    all_image_paths = []

    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Processing batches", unit="batch"):
            x, true_age, true_gender, image_paths = batch
            x, true_age, true_gender = [
                b.to(device) for b in (x, true_age, true_gender)
            ]

            gender_logits, age_pred = model(x)

            #  logits to probabilities
            gender_probs = F.softmax(gender_logits, dim=1)

            all_gender_preds.extend(gender_probs.cpu().numpy())
            all_age_preds.extend(age_pred.cpu().numpy())
            all_true_genders.extend(true_gender.cpu().numpy())
            all_true_ages.extend(true_age.cpu().numpy())
            all_image_paths.extend(image_paths)

    return {
        "gender_pred": np.array(all_gender_preds),
        "age_pred": np.array(all_age_preds),
        "true_gender": np.array(all_true_genders),
        "true_age": np.array(all_true_ages),
        "image_paths": all_image_paths,
    }
