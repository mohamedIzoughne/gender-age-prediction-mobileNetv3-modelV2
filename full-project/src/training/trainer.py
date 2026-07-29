import os
import argparse
import yaml
import csv
from datetime import datetime
from typing import Dict, Any
import torch
import pytorch_lightning as pl
from pytorch_lightning.callbacks import Timer, ModelCheckpoint, TQDMProgressBar, Callback
from pytorch_lightning.loggers import TensorBoardLogger

from src.training.lightning_model import AgeGenderLightningModule
from src.training.dataset import AgeGenderDataModule
from src.training.callbacks import EarlyStoppingCB, BestMetricsCallback, LRMonitorCallback

PROJECT_NAME = "ag_classifier_main"

class MetricsCSVCallback(Callback):
    def __init__(self, filepath):
        super().__init__()
        self.filepath = filepath
        self.file = None
        self.writer = None
        
    def on_train_start(self, trainer, pl_module):
        os.makedirs(os.path.dirname(self.filepath), exist_ok=True)
        file_exists = os.path.exists(self.filepath)
        
        if file_exists and os.path.getsize(self.filepath) > 0:
            try:
                with open(self.filepath, 'r', newline='') as f:
                    reader = csv.reader(f)
                    header = next(reader, None)
                    rows = []
                    for row in reader:
                        try:
                            if int(row[0]) < trainer.current_epoch:
                                rows.append(row)
                        except ValueError:
                            pass
                
                with open(self.filepath, 'w', newline='') as f:
                    writer = csv.writer(f)
                    if header:
                        writer.writerow(header)
                    writer.writerows(rows)
            except Exception as e:
                print(f"Warning: Failed to clean up CSV before appending: {e}")
                
        self.file = open(self.filepath, 'a', newline='')
        self.writer = csv.writer(self.file)
        if not file_exists or os.path.getsize(self.filepath) == 0:
            self.writer.writerow(['epoch', 'stage', 'train_loss', 'val_loss', 'train_age_mae', 'val_age_mae', 'train_gender_acc', 'val_gender_acc', 'lr', 'timestamp'])
            
    def on_train_epoch_end(self, trainer, pl_module):
        self._log_epoch(trainer, 'epoch_end')
        
    def on_validation_epoch_end(self, trainer, pl_module):
        pass
            
    def _log_epoch(self, trainer, stage):
        metrics = trainer.callback_metrics
        if not metrics:
            return
            
        epoch = trainer.current_epoch
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        
        def get_metric(name):
            val = metrics.get(name, "")
            if isinstance(val, torch.Tensor):
                if val.numel() == 1:
                    return str(val.item())
                else:
                    return str(val.tolist())
            if isinstance(val, float):
                return str(val)
            return str(val) if val != "" else ""
            
        train_loss = get_metric('train_total_loss')
        val_loss = get_metric('val_total_loss')
        train_age_mae = get_metric('train_age_mae')
        val_age_mae = get_metric('val_age_mae')
        train_gender_acc = get_metric('train_gender_acc')
        val_gender_acc = get_metric('val_gender_acc')
        lr = ""
        if trainer.optimizers and len(trainer.optimizers[0].param_groups) > 0:
            lr = str(trainer.optimizers[0].param_groups[0]['lr'])
        
        self.writer.writerow([epoch, stage, train_loss, val_loss, train_age_mae, val_age_mae, train_gender_acc, val_gender_acc, lr, timestamp])
        self.file.flush()
        
    def on_train_end(self, trainer, pl_module):
        if self.file:
            self.file.close()

def load_config(config_path: str) -> Dict[str, Any]:
    with open(config_path, "r") as f:
        return yaml.safe_load(f)

def train(config: Dict[str, Any], serialize_final=False):
    print(f"\n - - - \nConfig:\n{dict(config)}\n\n - - - \n")
    pl.seed_everything(42, workers=True)

    data = AgeGenderDataModule(config)
    model = AgeGenderLightningModule(config)
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
        ckpt_dir = f"/content/drive/MyDrive/AgeGenderCheckpoints/{run_name}/"
        metrics_csv_path = f"/content/drive/MyDrive/AgeGenderMetrics/{run_name}_metrics.csv"
    else:
        ckpt_dir = f"checkpoints/{run_name}/"
        metrics_csv_path = f"logs/{run_name}_metrics.csv"
        
    checkpoint_callback = ModelCheckpoint(
        dirpath=ckpt_dir,
        filename=run_name + "-{epoch:02d}-{val_total_loss:.4f}",
        save_top_k=-1,
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
        gradient_clip_val=1.0,
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
        save_path = f"{PROJECT_NAME}_{config.get('prefix', '')}_epoch{epochs_run}_loss{best_loss:.4f}.pth"
        
        save_model(model, save_path)


def save_model(model: AgeGenderLightningModule, save_path: str = "model_checkpoint.pth") -> None:
    os.makedirs("model_store", exist_ok=True)
    # Important: We save `model.model.state_dict()` which represents the pure nn.Module,
    # stripping away the PyTorch Lightning wrapper attributes. This matches what inference.py expects.
    torch.save(
        {
            "model_state_dict": model.model.state_dict(),
            "config": model.config,
        },
        f"model_store/{save_path}",
    )

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train Age & Gender Classifier")
    parser.add_argument("--config", type=str, default="configs/model/my-configs/mobilenet_v3_large_aug.yaml", help="Path to config file")
    args = parser.parse_args()

    config = load_config(args.config)
    train(config, serialize_final=True)
