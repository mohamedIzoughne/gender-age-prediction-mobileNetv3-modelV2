import os, zipfile, subprocess, random, shutil
from pathlib import Path
from typing import List, Tuple

SEED = 42
PROJECT_ROOT = Path(__file__).resolve().parent.parent


def set_env_vars():
    return
    env_path = PROJECT_ROOT / ".env"
    if not env_path.exists():
        raise FileNotFoundError(f"Environment file not found: {env_path}")

    load_dotenv(env_path)

    username = os.getenv("KAGGLE_USERNAME")
    key = os.getenv("KAGGLE_KEY")

    if not username or not key:
        raise ValueError("Kaggle credentials not found in .env file.")

    os.environ["KAGGLE_USERNAME"] = username
    os.environ["KAGGLE_KEY"] = key


def download_dataset(
    target_folder: str = "dataset", n_folds: int = 10, subfolder: str = "UTKFace"
):
    set_env_vars()
    target_path = PROJECT_ROOT / target_folder

    if target_path.exists():
        shutil.rmtree(target_path)
    target_path.mkdir(parents=True)

    subprocess.run(["pip", "install", "kaggle"], check=True)
    subprocess.run(
        ["kaggle", "datasets", "download", "-d", "jangedoo/utkface-new"], check=True
    )

    with zipfile.ZipFile("utkface-new.zip", "r") as zip_ref:
        zip_ref.extractall(target_path)
    os.remove("utkface-new.zip")

    source_path = target_path / subfolder
    files = list(source_path.glob("*.jpg.chip.jpg"))
    if not files:
        files = list(source_path.glob("*.jpg.chip"))

    print(f"Number of files: {len(files)}")
    random.Random(SEED).shuffle(files)

    # Create 'full' subfolder and copy all files
    full_path = target_path / "full"
    full_path.mkdir(parents=True, exist_ok=True)
    for file in files:
        shutil.copy2(file, full_path / file.name)

    folds_path = target_path / "folds"
    folds_path.mkdir(parents=True, exist_ok=True)
    for i, file in enumerate(files):
        fold_dir = folds_path / f"fold_{i % n_folds}"
        fold_dir.mkdir(exist_ok=True)
        shutil.copy2(file, fold_dir / file.name)

    for folder in target_path.iterdir():
        if folder.is_dir() and folder.name not in ["full", "folds"]:
            shutil.rmtree(folder)


def get_cv_folds(
    n_folds: int = 5, dataset_path: str = "dataset"
) -> List[Tuple[List[Path], Path]]:
    folds_path = PROJECT_ROOT / dataset_path / "folds"
    all_folds = sorted(list(folds_path.glob("fold_*")))
    return [(all_folds[:i] + all_folds[i + 1 :], all_folds[i]) for i in range(n_folds)]


def get_train_test_split(
    train_ratio: float = 0.8, dataset_path: str = "dataset"
) -> Tuple[List[Path], List[Path]]:
    folds_path = PROJECT_ROOT / dataset_path / "folds"
    all_folds = sorted(list(folds_path.glob("fold_*")))
    split_point = int(train_ratio * len(all_folds))
    return all_folds[:split_point], all_folds[split_point:]


if __name__ == "__main__":
    try:
        download_dataset()
        # print("Dataset downloaded and split into full and folds")
        # cv_folds = get_cv_folds(5)
        # print(f"Cross-validation folds: {len(cv_folds)}")
        # train, test = get_train_test_split(0.8)
        # print(f"Train-test split: {len(train)} train, {len(test)} test")
    except Exception as e:
        print(f"An error occurred: {e}")
