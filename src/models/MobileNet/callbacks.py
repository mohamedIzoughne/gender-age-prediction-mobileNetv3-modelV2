import torch
from pytorch_lightning import Callback
from pytorch_lightning.callbacks.early_stopping import EarlyStopping

class EarlyStoppingCB(EarlyStopping):
    def on_train_end(self, trainer, pl_module):
        if self.stopped_epoch > 0:
            print(f"Early stopping triggered at epoch {self.stopped_epoch}")
        else:
            print("Training completed normally.")


class BestMetricsCallback(Callback):
    """
    Recording best metric over full run
    """

    def __init__(self):
        self.best_metrics = {}

    def on_validation_epoch_end(self, trainer, pl_module):
        for k, v in trainer.callback_metrics.items():
            if k.startswith("val_"):
                if k not in self.best_metrics or v > self.best_metrics[k]:
                    self.best_metrics[k] = (
                        v.item() if isinstance(v, torch.Tensor) else v
                    )

    def on_fit_end(self, trainer, pl_module):
        for k, v in self.best_metrics.items():
            trainer.logger.experiment.summary[f"best_{k}"] = v


class LRMonitorCallback(Callback):
    def on_validation_epoch_end(self, trainer, pl_module):
        current_lr = pl_module.get_current_lr()
        val_loss = trainer.callback_metrics.get("val_total_loss")
        print(
            f"Epoch {trainer.current_epoch}: LR = {current_lr}, Val Loss = {val_loss}"
        )

        pl_module.log(
            "callback_epoch_LR", current_lr, on_step=False, on_epoch=True, prog_bar=True
        )
