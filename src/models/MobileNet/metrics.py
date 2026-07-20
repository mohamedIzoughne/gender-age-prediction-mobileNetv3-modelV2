"""
Various evaluation and related utility functions are included here
e.g. methods for
binning data, calculating metrics, visualizing results etc.
"""

import numpy as np
import pandas as pd
from sklearn.metrics import (
    confusion_matrix, accuracy_score, precision_recall_fscore_support,
    roc_auc_score, average_precision_score, log_loss, mean_absolute_error,
    mean_squared_error, r2_score, f1_score
)
import matplotlib.pyplot as plt
from matplotlib.image import imread
from IPython.display import display

# Constants
DEFAULT_AGE_BINS = [0, 4, 14, 24, 30, 40, 50, 60, 70, 80, np.inf]
DEFAULT_LUMINANCE_BINS = [0, 85, 105, 120, 135, 150, np.inf]
DEFAULT_BRISQUE_BINS = [-np.inf, 25, 35, 45, 55, np.inf]


def create_column_bins_and_labels(bins):
    """
    Create labels for given bin edges.

    Args:
        bins (list): List of bin edges.

    Returns:
        tuple: Bins and corresponding labels.
    """
    labels = [f"{bins[i]}+" if np.isinf(bins[i + 1]) else f"{bins[i]}-{bins[i + 1]}"
              for i in range(len(bins) - 1)]
    return bins, labels


def bin_column(col_values, bins):
    """
    Bin column values based on provided bin edges.

    Args:
        col_values (pd.Series): Column to bin.
        bins (list): List of bin edges.

    Returns:
        tuple: Binned column and labels.
    """
    bins, labels = create_column_bins_and_labels(bins)
    return pd.cut(col_values, bins=bins, labels=labels, include_lowest=True), labels


def cast_numeric(df):
    """
    Attempt to cast object columns to numeric type.

    Args:
        df (pd.DataFrame): Input dataframe.

    Returns:
        pd.DataFrame: Dataframe with numeric columns where possible.
    """
    for col in df.select_dtypes(include=["object"]):
        df[col] = pd.to_numeric(df[col], errors='ignore')
    return df


def calculate_binned_metrics(df, target_vars, vars_to_bin):
    """
    Calculate metrics for binned variables.

    Args:
        df (pd.DataFrame): Input dataframe.
        target_vars (list): List of target variables.
        vars_to_bin (list): List of tuples (variable, bins) to be binned.

    Returns:
        dict: Dictionary of calculated metrics.
    """
    result = {}

    for var, bins in vars_to_bin:
        binned_col, labels = bin_column(df[var], bins)
        df[f"{var}_binned"] = binned_col

        for target in target_vars:
            is_classifier = df[f"true_{target}"].nunique() <= 2
            metrics = []

            for bin_name in labels:
                bin_mask = df[f"{var}_binned"] == bin_name
                y_true = df.loc[bin_mask, f"true_{target}"]
                y_pred = df.loc[bin_mask, f"{target}_pred"]
                bin_size = y_true.shape[0]

                if bin_size == 0:
                    continue

                metric = {
                    "bin": bin_name,
                    "sample_size": bin_size,
                    "mean gender (std)": f"{df.loc[bin_mask, 'true_gender'].mean():.2f}({df.loc[bin_mask, 'true_gender'].std():.2f})",
                    "mean age (std)": f"{df.loc[bin_mask, 'true_age'].mean():.2f}({df.loc[bin_mask, 'true_age'].std():.2f})",
                }

                if is_classifier:
                    metric.update({
                        "accuracy": accuracy_score(y_true, y_pred.round()),
                        "F1": f1_score(y_true, y_pred.round(), average="binary"),
                        "log_loss": log_loss(y_true, y_pred)
                    })
                else:
                    metric.update({
                        "MAE": mean_absolute_error(y_true, y_pred),
                        "RMSE": np.sqrt(mean_squared_error(y_true, y_pred))
                    })

                metrics.append(metric)

            metrics_df = cast_numeric(pd.DataFrame(metrics)).round(3)
            metrics_df.set_index("bin", inplace=True)
            result.setdefault(target, {})[f"{var}_binned"] = metrics_df

    return result


def evaluate_predictions(predictions):
    """
    Evaluate predictions for gender and age.

    Args:
        predictions (dict): Dictionary containing predictions and true values.

    Returns:
        dict: Dictionary of evaluation metrics.
    """
    gender_preds = predictions["gender_pred"]
    age_preds = predictions["age_pred"]
    true_genders = predictions["true_gender"]
    true_ages = predictions["true_age"]

    gender_pred_probs = gender_preds[:, 1]
    gender_pred_labels = (gender_pred_probs > 0.5).astype(int)

    epsilon = 1e-15
    gender_pred_probs = np.clip(gender_pred_probs, epsilon, 1 - epsilon)

    auc_roc = roc_auc_score(true_genders, gender_pred_probs)
    precision, recall, f1, support = precision_recall_fscore_support(true_genders, gender_pred_labels)
    accuracy = accuracy_score(true_genders, gender_pred_labels)

    gender_metrics = pd.DataFrame({
        "Female": [support[1], accuracy, precision[1], recall[1], f1[1], np.nan, np.nan, np.nan],
        "Male": [support[0], accuracy, precision[0], recall[0], f1[0], np.nan, np.nan, np.nan],
        "Overall": [
            len(true_genders), accuracy, np.mean(precision), np.mean(recall), np.mean(f1),
            auc_roc, average_precision_score(true_genders, gender_pred_probs),
            log_loss(true_genders, gender_pred_probs)
        ]
    }, index=["Support", "Accuracy", "Precision", "Recall", "F1-score", "AUC-ROC", "PR-AUC", "Log Loss"])

    age_metrics = pd.DataFrame({
        "Value": [
            mean_absolute_error(true_ages, age_preds),
            mean_squared_error(true_ages, age_preds),
            np.sqrt(mean_squared_error(true_ages, age_preds)),
            r2_score(true_ages, age_preds),
            np.mean(np.abs((true_ages - age_preds) / true_ages)) * 100
        ]
    }, index=["MAE", "MSE", "RMSE", "R-squared", "MAPE"])

    age_stats = pd.DataFrame({
        "True Age": [np.mean(true_ages), np.median(true_ages), np.min(true_ages), np.max(true_ages)],
        "Predicted Age": [np.mean(age_preds), np.median(age_preds), np.min(age_preds), np.max(age_preds)]
    }, index=["Mean", "Median", "Min", "Max"])

    true_ages_binned, _ = bin_column(true_ages, DEFAULT_AGE_BINS)
    accuracy_by_age = pd.DataFrame({
        "Age_Group": true_ages_binned,
        "Correct_Predictions": true_genders == gender_pred_labels
    }).groupby("Age_Group", observed=False).agg({"Correct_Predictions": ["count", "sum", "mean"]})
    accuracy_by_age.columns = ["Total", "Correct", "Accuracy"]
    accuracy_by_age["Accuracy"] = accuracy_by_age["Accuracy"].round(4)

    performance_by_age_bin = []
    for age_label in pd.unique(true_ages_binned):
        mask = true_ages_binned == age_label
        bin_support = np.sum(mask)
        if bin_support == 0:
            continue

        bin_true_ages = true_ages[mask]
        bin_age_preds = age_preds[mask]

        performance_by_age_bin.append({
            "Age_Group": age_label,
            "Support": bin_support,
            "Age_MAE": mean_absolute_error(bin_true_ages, bin_age_preds),
            "Age_MSE": mean_squared_error(bin_true_ages, bin_age_preds),
            "Age_RMSE": np.sqrt(mean_squared_error(bin_true_ages, bin_age_preds)),
            "Age_R-squared": r2_score(bin_true_ages, bin_age_preds),
            "Age_MAPE": np.mean(np.abs((bin_true_ages - bin_age_preds) / bin_true_ages)) * 100
        })

    performance_by_age_bin = pd.DataFrame(performance_by_age_bin)

    return {
        "gender_metrics": gender_metrics,
        "age_metrics": age_metrics,
        "age_statistics": age_stats,
        "gender_accuracy_by_age": accuracy_by_age,
        "gender_pred_probs": gender_pred_probs,
        "true_genders": true_genders,
        "performance_by_age_bin": performance_by_age_bin
    }


def display_binned_samples(df, column_to_bin="luminance", bins=DEFAULT_LUMINANCE_BINS,
                           image_path_column="image_path", base_path="dataset/test_2_folds_last",
                           samples_per_bin=5):
    """
    Display sample images for each bin of a specified column.

    Args:
        df (pd.DataFrame): Input dataframe.
        column_to_bin (str): Column to bin.
        bins (list): List of bin edges.
        image_path_column (str): Column containing image paths.
        base_path (str): Base path for images.
        samples_per_bin (int): Number of samples to display per bin.
    """
    binned_column, bin_labels = bin_column(df[column_to_bin], bins)

    n_bins = len(bin_labels)
    fig, axes = plt.subplots(n_bins, 1, figsize=(10, 1.5 * n_bins))
    axes = np.atleast_1d(axes)

    for ax, bin_label in zip(axes, bin_labels):
        bin_mask = binned_column == bin_label
        sample = df[bin_mask].sample(min(samples_per_bin, bin_mask.sum()), random_state=4)

        if len(sample) > 0:
            images = [imread(f"{base_path}/{path}") for path in sample[image_path_column]]
            combined_image = np.hstack(images)
            ax.imshow(combined_image)

        ax.axis("off")
        ax.set_title(f"{column_to_bin} Bin: {bin_label} (n={len(sample)})", fontsize=8)

    plt.tight_layout()
    display(plt.gcf())
    plt.close()