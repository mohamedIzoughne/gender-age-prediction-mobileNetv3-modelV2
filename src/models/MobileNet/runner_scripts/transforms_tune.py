"""
Used for selecting transformers and measuring their impact in a more structured way.
"""

import yaml
import wandb
from typing import Dict, Any

import src.models.MobileNet.runner_scripts.trainer as trainer

PREFIX = "select_transforms"


def run_individual_trials(config: Dict[str, Any]):
    # Extract only transform parameters
    include_params = {k: v for k, v in config.items() if k.startswith("include_")}

    # Run individual trials for each include_* parameter
    for i, param in enumerate(include_params):
        trial_config = config.copy()

        for include_param in include_params:
            trial_config[include_param] = False

        trial_config[param] = True

        print(f"\nRunning trial with only {param} = True")
        wandb.init(
            project=trainer.PROJECT_NAME,
            config=trial_config,
            reinit=True,
            name=f"{PREFIX}only_{param}",
        )
        trainer.train(trial_config, sweep_run=True)
        wandb.finish()

    print("\nRunning trial with original include_* settings")
    wandb.init(
        project=trainer.PROJECT_NAME,
        config=config,
        reinit=True,
        name=f"{PREFIX}all_original",
    )
    trainer.train(config, sweep_run=True)
    wandb.finish()


if __name__ == "__main__":
    wandb.finish()
    config = trainer.load_config("config/model/swept-sweep-34.yaml")

    run_individual_trials(config)
