"""
Training script for the Age & Gender classification model.
Supports logging via TensorBoard and model checkpointing.
"""

import os
import sys

# Add the project root to sys.path to resolve 'src' module imports
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../"))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

import torch
import pytorch_lightning as pl
from pytorch_lightning.callbacks import Timer, ModelCheckpoint, TQDMProgressBar, Callback
from pytorch_lightning.loggers import TensorBoardLogger
from typing import Dict, Any
import yaml
import csv
from datetime import datetime

class MetricsCSVCallback(Callback):
    def __init__(self, filepath):
        super().__init__()
        self.filepath = filepath
        self.file = None
        self.writer = None
        
    def on_train_start(self, trainer, pl_module):
        os.makedirs(os.path.dirname(self.filepath), exist_ok=True)
        file_exists = os.path.exists(self.filepath)
        self.file = open(self.filepath, 'a', newline='')
        self.writer = csv.writer(self.file)
        if not file_exists or os.path.getsize(self.filepath) == 0:
            self.writer.writerow(['epoch', 'stage', 'train_loss', 'val_loss', 'train_age_mae', 'val_age_mae', 'train_gender_acc', 'val_gender_acc', 'lr', 'timestamp'])
            
    def on_train_epoch_end(self, trainer, pl_module):
        self._log_epoch(trainer, 'train')
        
    def on_validation_epoch_end(self, trainer, pl_module):
        if not trainer.sanity_checking:
            self._log_epoch(trainer, 'val')
            
    def _log_epoch(self, trainer, stage):
        metrics = trainer.callback_metrics
        if not metrics:
            return
            
        epoch = trainer.current_epoch
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        
        def get_metric(name):
            val = metrics.get(name, "")
            if isinstance(val, torch.Tensor):
                return f"{val.item():.4f}"
            if isinstance(val, float):
                return f"{val:.4f}"
            return val
            
        train_loss = get_metric('train_total_loss')
        val_loss = get_metric('val_total_loss')
        train_age_mae = get_metric('train_age_mae')
        val_age_mae = get_metric('val_age_mae')
        train_gender_acc = get_metric('train_gender_acc')
        val_gender_acc = get_metric('val_gender_acc')
        lr = ""
        if trainer.optimizers and len(trainer.optimizers[0].param_groups) > 0:
            lr = f"{trainer.optimizers[0].param_groups[0]['lr']:.6f}"
        
        self.writer.writerow([epoch, stage, train_loss, val_loss, train_age_mae, val_age_mae, train_gender_acc, val_gender_acc, lr, timestamp])
        self.file.flush()
        
    def on_train_end(self, trainer, pl_module):
        if self.file:
            self.file.close()

from src.models.mobilenet.callbacks import (
    EarlyStoppingCB,
    BestMetricsCallback,
    LRMonitorCallback,
)
from src.models.mobilenet.data_loader import create_dataloaders
from src.models.mobilenet.classifier import AgeGenderClassifier

PROJECT_NAME = "ag_classifier_main"


def load_config(config_path: str) -> Dict[str, Any]:
    """
    Load YAML configuration from path.

    Args:
        config_path (str): Filepath to the config.

    Returns:
        dict: Parsed configurations.
    """
    with open(config_path, "r") as f:
        return yaml.safe_load(f)


def train(config: Dict[str, Any], sweep_run=False, serialize_final=False):
    """
    Initialize data, model, and trainer to run fitting pipeline.

    Args:
        config (dict): Configuration containing model & training options.
        sweep_run (bool): Flag denoting whether execution is part of a sweep.
        serialize_final (bool): Whether to serialize model to model_store/ upon completion.
    """
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

    model_type = config.get("model_type", "mobilenet_v3")
    is_aug = "aug" if config.get("use_dynamic_augmentation", False) else "no_aug"
    run_name = f"{model_type}_{is_aug}"

    if os.path.exists("/content/"):
        if not os.path.exists("/content/drive/MyDrive/"):
            raise FileNotFoundError("Google Drive is not mounted! Please mount it to save checkpoints before running.")
        ckpt_dir = f"/content/drive/MyDrive/AgeGenderCheckpoints/{run_name}/"
        metrics_csv_path = f"/content/drive/MyDrive/AgeGenderMetrics/{run_name}_metrics.csv"
    else:
        ckpt_dir = f"checkpoints/{run_name}/"
        metrics_csv_path = f"logs/{run_name}_metrics.csv"
        
    checkpoint_callback = ModelCheckpoint(
        dirpath=ckpt_dir,
        filename=run_name + "-{epoch:02d}-{val_total_loss:.4f}",
        save_top_k=-1, # Save every epoch
        monitor="val_total_loss",
        mode="min",
        save_last=True
    )
    callbacks.append(checkpoint_callback)
    
    callbacks.append(MetricsCSVCallback(filepath=metrics_csv_path))

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
            import shutil
            best_ckpt_dest = os.path.join(ckpt_dir, f"{run_name}-best.ckpt")
            shutil.copy2(best_model_path, best_ckpt_dest)
            print(f"Copied best PyTorch Lightning checkpoint to {best_ckpt_dest}")

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
    """
    Saves the model state dict and full configuration.

    Args:
        model (pl.LightningModule): The trained classifier module.
        save_path (str): Destination filename inside 'model_store/'.
    """
    os.makedirs("model_store", exist_ok=True)
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "config": model.config,
        },
        f"model_store/{save_path}",
    )


def load_model(path: str = "model_checkpoint.pth") -> AgeGenderClassifier:
    """
    Loads a saved model checkpoint and returns an initialized AgeGenderClassifier.

    Args:
        path (str): Filename of target pth checkpoint in 'model_store/'.

    Returns:
        AgeGenderClassifier: Initialized PyTorch Lightning classifier model.
    """
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
    config_path = os.path.join(project_root, "config/model/my-configs/mobilenet_v3_large_aug.yaml")
    config = load_config(config_path)
    train(config, serialize_final=True)

