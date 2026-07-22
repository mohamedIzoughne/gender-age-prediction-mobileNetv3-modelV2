"""
Sweeper integration script for Hyperparameter Tuning via Wandb.
"""

import os
import wandb
import yaml
from typing import Dict, Any
from trainer import train, PROJECT_NAME, load_config


def sweep_train() -> None:
    """
    Callback function executed by the wandb agent for each trial run.
    """
    wandb.init(project=PROJECT_NAME)
    train(wandb.config, sweep_run=True, serialize_final=True)


def run_agent(args) -> None:
    """
    Start a Wandb agent on a specific sweep run.
    """
    sweep_id, count = args
    wandb.agent(sweep_id, function=sweep_train, count=count)


def run_sweep(sweep_config: Dict[str, Any]) -> None:
    """
    Create a Wandb sweep instance and start execution.
    """
    wandb.finish()

    sweep_id = wandb.sweep(sweep_config, project=PROJECT_NAME)
    total_runs = sweep_config["count"]

    wandb.agent(sweep_id, function=sweep_train, count=total_runs)


if __name__ == "__main__":
    sweep_config = load_config("config/sweep/sweep_optimal_only_freeze.yaml")
    run_sweep(sweep_config)

