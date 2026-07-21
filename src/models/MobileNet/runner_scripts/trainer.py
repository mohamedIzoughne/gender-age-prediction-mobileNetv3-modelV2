import os
import sys

# Add the project root to sys.path to resolve 'src' module imports
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../../../"))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

import torch
import pytorch_lightning as pl
from pytorch_lightning.callbacks import Timer, ModelCheckpoint, TQDMProgressBar
from pytorch_lightning.loggers import TensorBoardLogger
from typing import Dict, Any
import yaml

from src.models.MobileNet.callbacks import (
    EarlyStoppingCB,
    BestMetricsCallback,
    LRMonitorCallback,
)
from src.models.MobileNet.data_loader import create_dataloaders
from src.models.MobileNet.classifier import AgeGenderClassifier

PROJECT_NAME = "ag_classifier_main"


def load_config(config_path: str) -> Dict[str, Any]:
    with open(config_path, "r") as f:
        return yaml.safe_load(f)


def train(config: Dict[str, Any], sweep_run=False, serialize_final=False):
    print(f"\n - - - \nConfig:\n{dict(config)}\n\n - - - \n")

    data = create_dataloaders(config)
    model = AgeGenderClassifier(config)
    tb_logger = TensorBoardLogger(save_dir="logs/", name=PROJECT_NAME)

    callbacks = [
        BestMetricsCallback(),
        Timer(duration=None, interval="epoch"),
        LRMonitorCallback(),
        TQDMProgressBar(refresh_rate=50),
    ]

    if os.path.exists("/content/"):
        if not os.path.exists("/content/drive/MyDrive/"):
            raise FileNotFoundError("Google Drive is not mounted! Please mount it to save checkpoints before running.")
        ckpt_dir = "/content/drive/MyDrive/AgeGenderCheckpoints/"
    else:
        ckpt_dir = "checkpoints/"
    checkpoint_callback = ModelCheckpoint(
        dirpath=ckpt_dir,
        filename="mobilenet-{epoch:02d}-{val_total_loss:.4f}",
        save_top_k=3,
        monitor="val_total_loss",
        mode="min",
        save_last=True
    )
    callbacks.append(checkpoint_callback)

    trainer = pl.Trainer(
        max_epochs=config["num_epochs"],
        callbacks=callbacks,
        logger=tb_logger,
        accelerator="gpu",
        devices=1,
        precision="16-mixed",
        log_every_n_steps=50,
    )

    last_ckpt = os.path.join(ckpt_dir, "last.ckpt")
    resume_ckpt = last_ckpt if os.path.exists(last_ckpt) else None

    if resume_ckpt:
        print(f"Resuming training from checkpoint: {resume_ckpt}")

    trainer.fit(model, datamodule=data, ckpt_path=resume_ckpt)

    if serialize_final:
        best_model_path = checkpoint_callback.best_model_path
        if best_model_path and os.path.exists(best_model_path):
            print(f"Loading best model from {best_model_path} for final serialization...")
            checkpoint = torch.load(best_model_path)
            model.load_state_dict(checkpoint['state_dict'])
            best_loss = checkpoint_callback.best_model_score
            best_loss = best_loss.item() if best_loss is not None else 0
        else:
            print("No best model found. Serializing the last epoch instead.")
            best_loss = trainer.callback_metrics.get("val_total_loss", 0)
            if isinstance(best_loss, torch.Tensor):
                best_loss = best_loss.item()

        epochs_run = trainer.current_epoch + 1

        if "prefix" in config:
            save_path = f"{PROJECT_NAME}_{config['prefix']}_epoch{epochs_run}_loss{best_loss:.4f}.pth"
        else:
            save_path = f"{PROJECT_NAME}_epoch{epochs_run}_loss{best_loss:.4f}.pth"

        save_model(model, save_path)


def save_model(
    model: pl.LightningModule, save_path: str = "model_checkpoint.pth"
) -> None:
    """Saves the model state dict and full configuration."""
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "config": model.config,  # Save the full configuration
        },
        f"model_store/{save_path}",
    )


def load_model(path: str = "model_checkpoint.pth") -> AgeGenderClassifier:
    """Loads a saved model checkpoint and returns an initialized AgeGenderClassifier."""
    checkpoint = torch.load(f"model_store/{path}")
    config = checkpoint.get("config", {})

    if not config:
        print(
            "Warning: No configuration found in the checkpoint. Using default configuration."
        )
        config = {"model_type": "mobilenet_v3_large"}  # Example default

    model = AgeGenderClassifier(config)
    model.load_state_dict(checkpoint["model_state_dict"])
    return model


if __name__ == "__main__":
    config_path = os.path.join(project_root, "config/model/swept-sweep-34_improved_DYNAMIC_AUG.yaml")
    config = load_config(config_path)
    train(config, serialize_final=True)
