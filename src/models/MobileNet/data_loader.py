from src.models.MobileNet.data_defs import AgeGenderDataModule

FIXED_SEED = 42
def create_dataloaders(config, mode="train"):
    return AgeGenderDataModule(config, mode)
