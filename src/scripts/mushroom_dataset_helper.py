"""
Helper script to download and extract the mushroom classification dataset.
This dataset is separate from the main UTKFace age and gender model.
"""

import os
import zipfile
import subprocess
import shutil

TEMP_DATASET_NAME = "../dataset/full"


def download_ds(path):
    """
    Download the mushrooms classification dataset from Kaggle and extract it.
    """
    def download_dataset():
        subprocess.run(
            [
                "kaggle",
                "datasets",
                "download",
                "-d",
                "maysee/mushrooms-classification-common-genuss-images",
            ],
            check=True,
        )

    # TODO: Migrate these configurations to environment variables / config file
    os.environ["KAGGLE_USERNAME"] = "fdsfdssfd"
    os.environ["KAGGLE_KEY"] = "01ae24651b00fa183e6b84bf135d8d84"

    DS_ZIP_FILE = "mushrooms-classification-common-genuss-images.zip"

    download_dataset()

    with zipfile.ZipFile(DS_ZIP_FILE, "r") as zip_ref:
        zip_ref.extractall(path)

    dup_path = os.path.join(os.path.join(path, "Mushrooms"), "Mushrooms")
    if os.path.exists(dup_path):
        shutil.rmtree(dup_path)


if __name__ == "__main__":
    download_ds(TEMP_DATASET_NAME)
