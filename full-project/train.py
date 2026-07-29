import os
import sys
import argparse

# Add the root directory to PYTHONPATH so `src.` imports work
project_root = os.path.abspath(os.path.dirname(__file__))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from src.training.trainer import load_config, train

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train Age & Gender Classifier")
    parser.add_argument("--config", type=str, default="configs/model/my-configs/mobilenet_v3_large_aug.yaml", help="Path to config file")
    args = parser.parse_args()

    config = load_config(args.config)
    train(config, serialize_final=True)
