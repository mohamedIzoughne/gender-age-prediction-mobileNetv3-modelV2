import torch
import torch.nn as nn
import pytorch_lightning as pl
from torchmetrics import Accuracy, MeanAbsoluteError
from typing import Dict, Any, Tuple, List, Optional
from src.age_gender_model.model import AgeGenderModel

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

class AgeGenderLightningModule(pl.LightningModule):
    def __init__(self, config: Dict[str, Any]) -> None:
        super().__init__()
        self.config = config
        
        # Instantiate pure PyTorch model from demographics package
        self.model = AgeGenderModel(config)
        
        self.gender_loss = nn.CrossEntropyLoss()
        self.age_loss = nn.L1Loss()
        self.gender_accuracy = Accuracy(task="binary")
        self.train_gender_accuracy = Accuracy(task="binary")
        self.age_mae = MeanAbsoluteError()

        if self.get_param("use_dynamic_augmentation", False):
            from src.training.dataset import get_dynamic_augmentations
            from torchvision.transforms import v2 as transforms
            augmentation_configs = get_dynamic_augmentations(include_normalize=False)
            transforms_list = [transform for _, transform in augmentation_configs]
            self.dynamic_augment_transform = transforms.Compose(transforms_list)
        else:
            self.dynamic_augment_transform = None
            
        self.check_freeze_base_model()

    def get_param(self, key: str, default: Any = None) -> Any:
        return self.config.get(key, default)

    def check_freeze_base_model(self) -> None:
        freeze_epochs = self.get_param("freeze_epochs", 0)
        if freeze_epochs > 0:
            print(f"Freezing base model for: {freeze_epochs} epochs")
            for param in self.model.base_model.parameters():
                param.requires_grad = False

    def check_unfreeze_base_model(self) -> None:
        freeze_epochs = self.get_param("freeze_epochs", 0)
        if self.current_epoch == freeze_epochs:
            print(f"Unfreezing base model after epoch {freeze_epochs}")
            for param in self.model.base_model.parameters():
                param.requires_grad = True

    def on_train_epoch_start(self) -> None:
        self.check_unfreeze_base_model()

    def on_after_batch_transfer(self, batch: Any, dataloader_idx: int) -> Any:
        if self.trainer.training:
            x, age, gender, is_augmented, image_paths = batch
            if self.dynamic_augment_transform is not None and is_augmented.any():
                augmented_x = self.dynamic_augment_transform(x[is_augmented])
                x[is_augmented] = augmented_x
            return x, age, gender, image_paths
        else:
            x, age, gender, is_augmented, image_paths = batch
            return x, age, gender, image_paths

    def forward(self, x: torch.Tensor):
        return self.model(x)

    def get_current_lr(self) -> float:
        return self.optimizers().param_groups[0]["lr"]

    def training_step(self, batch: Tuple[torch.Tensor, torch.Tensor, torch.Tensor, Any], batch_idx: int) -> torch.Tensor:
        current_lr = self.get_current_lr()
        self.log("step_LR", current_lr, on_step=True, on_epoch=False, prog_bar=True)

        x, age, gender, _ = batch
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
                p.abs().sum() for p in self.model.gender_classifier[1].parameters()
            ) + sum(p.abs().sum() for p in self.model.age_regressor[1].parameters())
            total_loss += l1_lambda * l1_norm

        train_gender_acc = self.train_gender_accuracy(
            torch.argmax(gender_pred, dim=1), gender
        )
        train_age_mae = self.age_mae(age_pred, age.float())

        self.log("train_age_mae", train_age_mae, on_step=False, on_epoch=True, prog_bar=True)
        self.log("train_gender_loss", gender_loss, on_step=False, on_epoch=True, prog_bar=True)
        self.log("train_age_loss", age_loss, on_step=False, on_epoch=True, prog_bar=True)
        self.log("train_total_loss", total_loss, on_step=False, on_epoch=True, prog_bar=True)
        self.log("train_gender_acc", train_gender_acc, on_step=False, on_epoch=True, prog_bar=True)

        return total_loss

    def validation_step(self, batch: Tuple[torch.Tensor, torch.Tensor, torch.Tensor, Any], batch_idx: int) -> torch.Tensor:
        x, age, gender, _ = batch
        gender_pred, age_pred = self(x)

        gender_loss = self.gender_loss(gender_pred, gender)
        age_loss = self.age_loss(age_pred, age.float())

        total_loss = (
            self.get_param("gender_loss_weight") * gender_loss
            + (1 - self.get_param("gender_loss_weight")) * age_loss
        )

        gender_acc = self.gender_accuracy(torch.argmax(gender_pred, dim=1), gender)
        age_mae = self.age_mae(age_pred, age.float())

        self.log("val_gender_loss", gender_loss, prog_bar=True, on_epoch=True, on_step=False)
        self.log("val_age_loss", age_loss, prog_bar=True, on_epoch=True, on_step=False)
        self.log("val_total_loss", total_loss, prog_bar=True, on_epoch=True, on_step=False)
        self.log("val_gender_acc", gender_acc, prog_bar=True, on_epoch=True, on_step=False)
        self.log("val_age_mae", age_mae, prog_bar=True, on_epoch=True, on_step=False)

        return total_loss

    def configure_optimizers(self) -> Dict[str, Any]:
        optimizer = torch.optim.AdamW(
            self.parameters(),
            lr=self.get_param("base_lr"),
            weight_decay=self.get_param("weight_decay"),
        )
        if self.get_param("lr_scheduler") == "one_cycle":
            total_steps = self.trainer.estimated_stepping_batches
            
            override_epochs = self.get_param("override_cycle_epoch_count", None)
            if override_epochs is not None and self.trainer.max_epochs is not None:
                ratio = override_epochs / self.trainer.max_epochs
                total_steps = int(total_steps * ratio)

            scheduler = OneCycleWithDecay(
                optimizer,
                decay_factor=1.0055,
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
            return {
                "optimizer": optimizer,
                "lr_scheduler": {
                    "scheduler": scheduler,
                    "monitor": "val_total_loss",
                    "interval": "epoch",
                },
            }
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
            return {"optimizer": optimizer}
