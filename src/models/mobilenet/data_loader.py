"""
DataLoader creation helpers for the MobileNet model training pipeline.
"""

from src.models.mobilenet.data_defs import AgeGenderDataModule

FIXED_SEED = 42  # Retained for seeding consistency across scripts


def create_dataloaders(config, mode="train"):
    """
    Instantiate and return the AgeGenderDataModule.

    Args:
        config (dict): Configuration options for the DataModule.
        mode (str): Execution mode ('train' or 'test').

    Returns:
        AgeGenderDataModule: Configured PyTorch Lightning DataModule.
    """
    return AgeGenderDataModule(config, mode)

