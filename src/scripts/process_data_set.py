import subprocess
import zipfile
from typing import Optional, List
import os
import shutil
from sklearn.model_selection import train_test_split

from PIL import Image, UnidentifiedImageError
from pathlib import Path
import imagehash
import pandas as pd
from multiprocessing import Pool, cpu_count

UNPROC_DATASET_LOC = "../dataset/full"
UNPROC_FOLD_1_LOC = "../dataset/folds/fold_0"
TEMP_DATASET_NAME = "../dataset/full"
OUTPUT_NAME = "dataset"
FIXED_SEED = 42


def download_dataset():
    """
    Download from kaggle and extract zip
    """

    raise NotImplemented


def verify_and_clean_images(path):
    path = Path(path)
    broken_files = []

    x = 0
    for img_path in path.rglob("*.*"):
        x += 1
        try:
            img = Image.open(img_path)
            img.convert("RGB")
            img.resize((224, 224))
            img.close()
        except Exception as e:
            img.close()
            broken_files.append(img_path)
            print(f"Removing corrupt image: {img_path} due to {e}")
            img_path.unlink()
    return broken_files


def analyze_images(path):
    path = Path(path)
    path = Path(path)
    metrics = {"icc_profile_present": [], "dimensions": [], "file_format": []}
    broken_files = []
    for img_path in path.rglob("*.*"):
        try:
            with Image.open(img_path) as img:
                img.verify()  # Verify the integrity of the image
                metrics["icc_profile_present"].append("TODO")
                metrics["dimensions"].append(img.size)
                metrics["file_format"].append(img.format)

        except (UnidentifiedImageError, IOError, Exception) as e:
            raise e

    df_icc_profile = pd.DataFrame(
        {"Count": pd.Series(metrics["icc_profile_present"]).value_counts()}
    )
    df_dimensions = pd.DataFrame(
        {"Count": pd.Series(metrics["dimensions"]).value_counts()}
    )
    df_file_format = pd.DataFrame(
        {"Count": pd.Series(metrics["file_format"]).value_counts()}
    )

    return df_icc_profile, df_dimensions, df_file_format


def collect_results(result, hash_dict, class_duplicates):
    class_name, hash_val, img_path = result
    if class_name not in class_duplicates:
        class_duplicates[class_name] = {"duplicates": 0, "total": 0}
    if hash_val is None:
        return  # skip corrupt images

    class_duplicates[class_name]["total"] += 1
    if hash_val in hash_dict:
        class_duplicates[class_name]["duplicates"] += 1
        img_path.unlink()  # remove duplicates
    else:
        hash_dict[hash_val] = img_path


def process_image(img_path):
    with Image.open(img_path) as img:
        try:
            img = img.convert("RGB")
            hash_val = str(imagehash.dhash(img, 8))
        except Exception:
            hash_val = None

    return img_path.parent.name, hash_val, img_path


def find_duplicates(path):
    path = Path(path)
    hash_dict = {}
    class_duplicates = {}

    img_paths = list(path.rglob("*.*"))
    with Pool(processes=cpu_count()) as pool:
        results = pool.map(process_image, img_paths)

    for result in results:
        collect_results(result, hash_dict, class_duplicates)

    data = []
    for class_name, metrics in class_duplicates.items():
        data.append(
            {
                "Class": class_name,
                "Duplicate Count": metrics["duplicates"],
                "Total Images": metrics["total"],
                "Proportion": (
                    metrics["duplicates"] / metrics["total"]
                    if metrics["total"] > 0
                    else 0
                ),
            }
        )

    df = pd.DataFrame(data)
    if not df.empty:
        total_row = pd.DataFrame(
            [
                {
                    "Class": "Total",
                    "Duplicate Count": df["Duplicate Count"].sum(),
                    "Total Images": df["Total Images"].sum(),
                    "Proportion": (
                        df["Duplicate Count"].sum() / df["Total Images"].sum()
                        if df["Total Images"].sum() > 0
                        else 0
                    ),
                }
            ]
        )
        df = pd.concat([df, total_row], ignore_index=True)

    return df


def stratified_split(
    base_dir, output_dir, test_size=0.12, invalid_images: Optional[List[str]] = None
):
    os.makedirs(output_dir, exist_ok=True)
    train_dir = os.path.join(output_dir, "train")
    test_dir = os.path.join(output_dir, "test")

    all_files = []
    categories = []

    for subdir, dirs, files in os.walk(base_dir):
        print(f"{subdir}: {dirs}: {len(files)}")
        for file in files:
            img_not_valid = False
            if invalid_images is not None:
                for invalid_img in invalid_images:
                    if invalid_img in file:
                        print(f"Dropping invalid_img: {invalid_img}")
                        img_not_valid = True
                        break
            if not img_not_valid and file.endswith(".jpg"):
                file_path = os.path.join(subdir, file)
                category = subdir[len(base_dir) + 1 :].split(os.sep)[0]
                all_files.append(file_path)
                categories.append(category)

    category_set = set(categories)
    print("File count per category before split:")
    for cat in category_set:
        print(f"{cat}: {categories.count(cat)}")

    train_files, test_files = train_test_split(
        all_files, test_size=test_size, stratify=categories, random_state=FIXED_SEED
    )

    def copy_files(files, dest_dir):
        for file in files:
            category = file.split(os.sep)[-2]
            category_dir = os.path.join(dest_dir, category)
            os.makedirs(category_dir, exist_ok=True)
            shutil.copy(file, category_dir)

    copy_files(train_files, train_dir)
    copy_files(test_files, test_dir)

    # shutil.rmtree(base_dir)
    print(f"Data split into {train_dir} and {test_dir}")


def verify_dataset(output_dir):
    train_dir = os.path.join(output_dir, "train")
    test_dir = os.path.join(output_dir, "test")

    def count_samples(directory):
        counts = {}
        for class_folder in os.listdir(directory):
            class_path = os.path.join(directory, class_folder)
            if os.path.isdir(class_path):
                counts[class_folder] = len(
                    [name for name in os.listdir(class_path) if name.endswith(".jpg")]
                )
        return counts

    train_counts = count_samples(train_dir)
    test_counts = count_samples(test_dir)

    total_counts = {
        k: train_counts.get(k, 0) + test_counts.get(k, 0)
        for k in set(train_counts) | set(test_counts)
    }

    data = {}
    for class_name in total_counts:
        train_count = train_counts.get(class_name, 0)
        test_count = test_counts.get(class_name, 0)
        data[class_name] = {
            "Training Samples": train_count,
            "Training Proportion (%)": f"{((train_count / total_counts[class_name]) if total_counts[class_name] > 0 else 0):.2%}",
            "Testing Samples": test_count,
            "Testing Proportion (%)": f"{((test_count / total_counts[class_name]) if total_counts[class_name] > 0 else 0):.2%}",
            "Total Samples": train_count + test_count,
        }

    df = pd.DataFrame.from_dict(data, orient="index")

    train_files = {
        os.path.join(dp, f)
        for dp, dn, filenames in os.walk(train_dir)
        for f in filenames
        if f.endswith(".jpg")
    }
    test_files = {
        os.path.join(dp, f)
        for dp, dn, filenames in os.walk(test_dir)
        for f in filenames
        if f.endswith(".jpg")
    }
    overlaps = train_files & test_files

    if overlaps:
        print("Error: Some files are present in both train and test sets:")
        for overlap in overlaps:
            print(overlap)
    else:
        print("Verification Passed: No overlapping files between train and test sets.")

    return df


def download_ds(path):
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

    # TODO: REMOVE THIS AND MOVE TO ENV FILE
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
