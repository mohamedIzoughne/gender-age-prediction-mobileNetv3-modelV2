from os import path
from typing import Tuple, Optional, List

import numpy as np
import pandas as pd


def get_image_data(image_path) -> Tuple[int, int]:
    """
    Get image age and gender from name

    :param image_path:
    :return: age, gender
    """

    prts = path.basename(image_path).split("_")

    return prts[0], prts[1]


def process_image_metadata(
    df: pd.DataFrame, image_path_column: str, age_bins: Optional[List[int]] = None
) -> pd.DataFrame:
    """
    Process image metadata to extract age and gender, with optional age binning.

    :param df: Input DataFrame
    :param image_path_column: Name of the column containing image paths
    :param age_bins: List of age bin edges. If provided, age binning will be performed
    :return: DataFrame with added 'age' and 'gender' columns, and optionally 'age_group'
    """

    if age_bins is None:
        age_bins = [0, 18, 30, 45, 60, np.inf]

    age_bins_2 = [i * 10 for i in range(0, 10)]
    age_bins_2.append(np.inf)

    df = df.copy()
    df[["age", "gender"]] = df[image_path_column].apply(
        lambda x: pd.Series(get_image_data(x))
    )

    df["age"] = df["age"].astype(int)
    df["gender"] = df["gender"].astype(int)

    if age_bins is not None:
        df["age_group"] = pd.cut(
            df["age"],
            bins=age_bins,
            labels=[
                f"{age_bins[i]}-{age_bins[i + 1]}" for i in range(len(age_bins) - 1)
            ],
            include_lowest=True,
        )

    df["age_bin_raw"] = pd.cut(
        df["age"],
        bins=age_bins_2,
        labels=[
            f"{age_bins_2[i]}-{age_bins_2[i + 1]}" for i in range(len(age_bins_2) - 1)
        ],
        include_lowest=True,
    )

    return df
