from collections import namedtuple

import torch
from scipy import stats
from sklearn.metrics import accuracy_score, mean_absolute_error
import Notebooks.utils as utils
import random
import src.process_data_set as process_data_set
import importlib
import os
import cv2
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
import time
import os
import cv2
import pandas as pd
from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm import tqdm
import time
from skimage.measure import shannon_entropy

# from ml_tools_utils import utils
import src.ds_utils as ds_utils

import src.models.MobileNet.runner_scripts.trainer as trainer
import src.models.MobileNet.classifier as classifier
import src.models.MobileNet.data_loader as data_loader
import src.models.MobileNet.metrics as metrics
from skimage.segmentation import mark_boundaries

import src.models.MobileNet.data_defs as data_defs
from PIL import Image
import lime.lime_image as lime_image


def analyze_misclassifications(predictions, data_module):
    results = []

    for idx, (gender_pred, age_pred, true_gender, true_age) in enumerate(
        zip(
            predictions["gender_pred"],
            predictions["age_pred"],
            predictions["true_gender"],
            predictions["true_age"],
        )
    ):
        filename = data_module.test_dataset.valid_images[idx]

        predicted_gender = np.argmax(gender_pred)
        gender_prob = gender_pred[predicted_gender]
        correct_gender_prob = gender_pred[true_gender]
        gender_error = (
            1 - correct_gender_prob
        )  # Error is 1 minus the probability of the correct class

        age_error = abs(age_pred - true_age)

        results.append(
            {
                "filename": os.path.basename(filename),
                "true_gender": "Male" if true_gender == 0 else "Female",
                "predicted_gender": "Male" if predicted_gender == 0 else "Female",
                "gender_prob": gender_prob,
                "gender_error": gender_error,
                "true_age": true_age,
                "predicted_age": age_pred,
                "age_error": age_error,
            }
        )

    df = pd.DataFrame(results)

    max_age_error = df["age_error"].max()
    df["normalized_age_error"] = df["age_error"] / max_age_error

    df["combined_error"] = (df["gender_error"] + df["normalized_age_error"]) / 2

    return df


def get_top_misclassifications(df, error_type, n=20):
    if error_type == "gender":
        return df.sort_values("gender_error", ascending=False).head(n)
    elif error_type == "age":
        return df.sort_values("age_error", ascending=False).head(n)
    elif error_type == "combined":
        return df.sort_values("combined_error", ascending=False).head(n)
    else:
        raise ValueError("error_type must be 'gender', 'age', or 'combined'")


def sync_predictions_with_image_data(predictions, image_data):
    pred_df = pd.DataFrame(
        {
            "gender_pred": predictions["gender_pred"][
                :, 1
            ],  # Assuming second column is the positive class
            "age_pred": predictions["age_pred"],
            "true_gender": predictions["true_gender"],
            "true_age": predictions["true_age"],
            "image_path": [
                os.path.basename(path) for path in predictions["image_paths"]
            ],
        }
    )

    image_data["image_path"] = image_data["image_path"].apply(os.path.basename)

    # Merge predictions with image data
    merged_df = pd.merge(pred_df, image_data, on="image_path", how="inner")

    print(f"Total predictions: {len(pred_df)}")
    print(f"Total image quality data: {len(image_data)}")
    print(f"Matched data points: {len(merged_df)}")

    return merged_df


def evaluate_by_image_quality(merged_df, metrics_to_bin=None, bins=160):
    results = {}

    if metrics_to_bin is None:
        metrics_to_bin = [
            "entropy",
            "brisque_score",
            "laplacian_variance",
            "fft_blur_score",
        ]
    for metric in metrics_to_bin:
        if metric not in merged_df.columns:
            print(f"Warning: {metric} not found in the data. Skipping.")
            continue

        merged_df[f"{metric}_bin"] = pd.qcut(merged_df[metric], q=bins, labels=False)

        performance_by_bin = []
        for bin_num in range(bins):
            bin_data = merged_df[merged_df[f"{metric}_bin"] == bin_num]

            bin_gender_accuracy = accuracy_score(
                bin_data["true_gender"], (bin_data["gender_pred"] > 0.5).astype(int)
            )
            bin_age_mae = mean_absolute_error(
                bin_data["true_age"], bin_data["age_pred"]
            )

            performance_by_bin.append(
                {
                    "Bin": bin_num,
                    "Range": f"{bin_data[metric].min():.2f} - {bin_data[metric].max():.2f}",
                    "Count": len(bin_data),
                    "Gender_Accuracy": bin_gender_accuracy,
                    "Age_MAE": bin_age_mae,
                }
            )

        results[metric] = pd.DataFrame(performance_by_bin)

    return results


def evaluate_age_prediction(true_ages, predicted_ages, bins=metrics.DEFAULT_AGE_BINS):
    df = pd.DataFrame({"True_Age": true_ages, "Predicted_Age": predicted_ages})
    df["Error"] = df["Predicted_Age"] - df["True_Age"]
    df["Absolute_Error"] = np.abs(df["Error"])

    df["Age_Group"], labels = metrics.bin_column(df["True_Age"], bins)

    fig, axs = plt.subplots(3, 2, figsize=(20, 30))

    # 1. Residual Plot with Density
    sns.scatterplot(x="True_Age", y="Error", data=df, ax=axs[0, 0], alpha=0.5)
    sns.kdeplot(
        x="True_Age",
        y="Error",
        data=df,
        ax=axs[0, 0],
        cmap="YlOrRd",
        shade=True,
        cbar=True,
    )
    axs[0, 0].axhline(y=0, color="r", linestyle="--")
    axs[0, 0].set_title("Residual Plot with Density")
    axs[0, 0].set_xlabel("True Age")
    axs[0, 0].set_ylabel("Error (Predicted - True)")

    # 2. Q-Q Plot
    stats.probplot(df["Error"], dist="norm", plot=axs[0, 1])
    axs[0, 1].set_title("Q-Q Plot of Residuals")

    # 3. Predicted vs Actual Plot with Density
    sns.scatterplot(x="True_Age", y="Predicted_Age", data=df, ax=axs[1, 0], alpha=0.5)
    sns.kdeplot(
        x="True_Age",
        y="Predicted_Age",
        data=df,
        ax=axs[1, 0],
        cmap="YlOrRd",
        shade=True,
        cbar=True,
    )
    axs[1, 0].plot(
        [df["True_Age"].min(), df["True_Age"].max()],
        [df["True_Age"].min(), df["True_Age"].max()],
        "r--",
    )
    axs[1, 0].set_title("Predicted vs Actual Age with Density")
    axs[1, 0].set_xlabel("True Age")
    axs[1, 0].set_ylabel("Predicted Age")

    # 4. Error Distribution Plot
    sns.histplot(df["Error"], kde=True, ax=axs[1, 1])
    axs[1, 1].set_title("Distribution of Prediction Errors")
    axs[1, 1].set_xlabel("Error")

    # 5. CDF of Absolute Errors
    sorted_errors = np.sort(df["Absolute_Error"])
    cumulative = np.arange(1, len(sorted_errors) + 1) / len(sorted_errors)
    axs[2, 0].plot(sorted_errors, cumulative)
    axs[2, 0].set_title("CDF of Absolute Errors")
    axs[2, 0].set_xlabel("Absolute Error")
    axs[2, 0].set_ylabel("Cumulative Probability")

    # 6. Absolute Error by Age Group
    # sns.boxplot(x='Age_Group', y='Absolute_Error', data=df, ax=axs[2, 1], fill=False, gap=.1)

    bp = sns.boxplot(
        x="Age_Group",
        y="Absolute_Error",
        data=df,
        ax=axs[2, 1],
        fill=False,
        gap=0.1,
        linewidth=0.75,
    )
    axs[2, 1].set_ylim(0, 20)

    age_group_stats = df.groupby("Age_Group")["Absolute_Error"].agg(
        [
            ("median", "median"),
            ("q1", lambda x: x.quantile(0.25)),
            ("q3", lambda x: x.quantile(0.75)),
        ]
    )
    age_group_stats["iqr"] = age_group_stats["q3"] - age_group_stats["q1"]

    for i, (age_group, row) in enumerate(age_group_stats.iterrows()):
        # Median
        axs[2, 1].text(
            i,
            row["median"],
            f"{row['median']:.2f}",
            ha="center",
            va="bottom",
            fontsize=8,
        )

        # Q1
        axs[2, 1].text(
            i, row["q1"], f"Q1={row['q1']:.2f}", ha="center", va="bottom", fontsize=8
        )

        # Q3
        axs[2, 1].text(
            i, row["q3"] - 0.1, f"Q3={row['q3']:.2f}", ha="center", va="top", fontsize=8
        )

        axs[2, 1].text(
            i + 0.2,
            row["median"] + 1,
            f"IQR: {row['iqr']:.2f}",
            ha="left",
            va="center",
            fontsize=8,
            rotation=-90,
        )

    axs[2, 1].set_title("Absolute Error by Age Group")
    axs[2, 1].set_ylabel("Absolute Error")

    group_counts = df["Age_Group"].value_counts().sort_index()

    new_labels = [f"{group}\nn={count}" for group, count in group_counts.items()]

    axs[2, 1].set_xticklabels(new_labels)

    for label in axs[2, 1].get_xticklabels():
        label.set_fontsize(10)

    axs[2, 1].set_xlabel("")

    plt.subplots_adjust(bottom=0.2)

    plt.tight_layout()
    plt.show()


def predict_single_image(model, image):
    model.eval()
    with torch.no_grad():
        gender_pred, age_pred = model(image.unsqueeze(0))
    return gender_pred.squeeze(0), age_pred.item()


def parse_filename(filename):
    parts = os.path.basename(filename).split("_")
    age = int(parts[0])
    gender = "Male" if parts[1] == "0" else "Female"
    return age, gender


def process_image_for_models(image_file, models):
    image = Image.open(image_file).convert("RGB")
    real_age, real_gender = parse_filename(image_file)
    results = []

    for model in models:
        explainer, predict_fn = get_lime_explainer(model)
        gender_exp, age_exp, gender_pred, age_pred = explain_prediction(
            model, image, explainer, predict_fn
        )

        gender_prob = torch.softmax(gender_pred, dim=0)
        predicted_gender = "Male" if gender_prob[0] > gender_prob[1] else "Female"

        results.append(
            {
                "gender_exp": gender_exp,
                "age_exp": age_exp,
                "pred_age": age_pred,
                "pred_gender": predicted_gender,
                "gender_prob": gender_prob.tolist(),
            }
        )

    return {
        "image": np.array(image),
        "real_age": real_age,
        "real_gender": real_gender,
        "model_results": results,
    }


def process_images(model, image_files):
    return [process_image_for_models(img_file, [model]) for img_file in image_files]


#


def _base_grid_display(
    results,
    columns,
    title_func,
    base_width=12,
    base_row_height=4,
    scale=0.35,
    dpi=200,
    base_font_size=12,
):
    nrows = len(results)
    scaled_width = base_width * scale
    scaled_row_height = base_row_height * scale

    fig, axes = plt.subplots(
        nrows, columns, figsize=(scaled_width, scaled_row_height * nrows), dpi=dpi
    )

    if nrows == 1:
        axes = axes.reshape(1, -1)

    scaled_font_size = int(base_font_size * scale)

    for idx, result in enumerate(results):
        for col in range(columns):
            img, title = title_func(result, col)
            axes[idx, col].imshow(img)
            axes[idx, col].set_title(title, fontsize=scaled_font_size)
            axes[idx, col].axis("off")

    plt.tight_layout(pad=1.0 * scale, h_pad=1.5 * scale, w_pad=1.5 * scale)
    plt.show()

    fig_size_inches = fig.get_size_inches()
    print(
        f"Figure size: {fig_size_inches[0] * dpi:.0f}x{fig_size_inches[1] * dpi:.0f} px"
    )


def display_grid(
    results, base_width=12, base_row_height=4, scale=0.35, dpi=200, base_font_size=12
):
    def title_func(result, col):
        if col == 0:
            return (
                result["image"],
                f"Original\nReal: {result['real_age']} y.o. {result['real_gender']}",
            )
        elif col == 1:
            model_result = result["model_results"][0]
            return (
                model_result["gender_exp"],
                f"Gender.\nPred: {model_result['pred_gender']}\nProb: M:{model_result['gender_prob'][0]:.2f} / F:{model_result['gender_prob'][1]:.2f}",
            )
        else:
            model_result = result["model_results"][0]
            return (
                model_result["age_exp"],
                f"Age.\nPred: {model_result['pred_age']:.1f} y.o.",
            )

    _base_grid_display(
        results, 3, title_func, base_width, base_row_height, scale, dpi, base_font_size
    )


def display_grid_comparison(
    results,
    model_names,
    comparison_type="gender",
    base_width=12,
    base_row_height=4,
    scale=0.35,
    dpi=200,
    base_font_size=12,
):
    def title_func(result, col):
        if col == 0:
            return (
                result["image"],
                f"Original\nReal: {result['real_age']} y.o. {result['real_gender']}",
            )
        else:
            model_result = result["model_results"][col - 1]
            if comparison_type == "gender":
                return (
                    model_result["gender_exp"],
                    f"{model_names[col-1]}\nGender: {model_result['pred_gender']}\nProb: M:{model_result['gender_prob'][0]:.2f} / F:{model_result['gender_prob'][1]:.2f}",
                )
            else:
                return (
                    model_result["age_exp"],
                    f"{model_names[col-1]}\nAge: {model_result['pred_age']:.1f} y.o.",
                )

    _base_grid_display(
        results, 3, title_func, base_width, base_row_height, scale, dpi, base_font_size
    )


def get_lime_explainer(model):
    transforms = data_defs.get_transforms(get_compose=True)
    device = next(model.parameters()).device

    _, age_labels = metrics.create_column_bins_and_labels(metrics.DEFAULT_AGE_BINS)
    num_age_categories = len(age_labels)

    def predict_fn(images):
        batch = torch.stack(
            [transforms(Image.fromarray(img.astype("uint8"))) for img in images]
        )
        batch = batch.to(device)
        with torch.no_grad():
            gender_preds, age_preds = model(batch)
            age_bins, _ = metrics.bin_column(
                age_preds.cpu().numpy(), metrics.DEFAULT_AGE_BINS
            )
            age_categories = pd.Categorical(age_bins).codes

        # Combine gender and age predictions
        combined_preds = np.zeros((len(images), 2 + num_age_categories))
        combined_preds[:, :2] = gender_preds.cpu().numpy()
        combined_preds[np.arange(len(images)), 2 + age_categories] = 1

        return combined_preds

    explainer = lime_image.LimeImageExplainer()
    return explainer, predict_fn


def explain_prediction(model, image, explainer, predict_fn):
    transforms = data_defs.get_transforms(get_compose=True)
    device = next(model.parameters()).device

    _, age_labels = metrics.create_column_bins_and_labels(metrics.DEFAULT_AGE_BINS)
    num_age_categories = len(age_labels)

    explanation = explainer.explain_instance(
        np.array(image),
        predict_fn,
        top_labels=2 + num_age_categories,  # 2 for gender, rest for age categories
        hide_color=0,
        num_samples=500,
    )

    transformed_image = transforms(image).unsqueeze(0).to(device)
    with torch.no_grad():
        gender_pred, age_pred = model(transformed_image)

    gender_label = gender_pred.argmax().item()
    gender_temp, gender_mask = explanation.get_image_and_mask(
        gender_label, positive_only=False, num_features=5, hide_rest=False
    )
    gender_img_boundary = mark_boundaries(gender_temp / 255.0, gender_mask)

    age_bin, _ = metrics.bin_column([age_pred.item()], metrics.DEFAULT_AGE_BINS)
    age_category = pd.Categorical(age_bin).codes[0]
    age_temp, age_mask = explanation.get_image_and_mask(
        age_category + 2,  # +2 because gender takes first two indices
        positive_only=False,
        num_features=5,
        hide_rest=False,
    )
    age_img_boundary = mark_boundaries(age_temp / 255.0, age_mask)

    return (
        gender_img_boundary,
        age_img_boundary,
        gender_pred.cpu().squeeze(0),
        age_pred.cpu().item(),
    )


import matplotlib.pyplot as plt


def display_explanations(
    original_image, gender_explanation, age_explanation, gender_pred, age_pred
):
    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(15, 5))

    ax1.imshow(original_image)
    ax1.set_title("Original Image")
    ax1.axis("off")

    ax2.imshow(gender_explanation)
    ax2.set_title(
        f'Gender Explanation\nPredicted: {"Male" if gender_pred.argmax() == 0 else "Female"}'
    )
    ax2.axis("off")

    ax3.imshow(age_explanation)
    ax3.set_title(f"Age Explanation\nPredicted Age: {age_pred:.2f}")
    ax3.axis("off")

    plt.tight_layout()
    plt.show()


def get_worst_quality_images(df, n_samples=20, weights=None):
    if weights is None:
        weights = {
            "brisque_score": 0.40,
            "laplacian_variance": 0.40,
            "fft_blur_score_bin": 0.15,
            "entropy": 0.05,
        }

    total_weight = sum(weights.values())
    weights = {k: v / total_weight for k, v in weights.items()}

    df_copy = df.copy()

    normalized_brisque = 1 - (
        df_copy["brisque_score"] - df_copy["brisque_score"].min()
    ) / (df_copy["brisque_score"].max() - df_copy["brisque_score"].min())
    normalized_laplacian = (
        df_copy["laplacian_variance"] - df_copy["laplacian_variance"].min()
    ) / (df_copy["laplacian_variance"].max() - df_copy["laplacian_variance"].min())
    normalized_fft = (
        df_copy["fft_blur_score_bin"] - df_copy["fft_blur_score_bin"].min()
    ) / (df_copy["fft_blur_score_bin"].max() - df_copy["fft_blur_score_bin"].min())
    normalized_entropy = (df_copy["entropy"] - df_copy["entropy"].min()) / (
        df_copy["entropy"].max() - df_copy["entropy"].min()
    )

    # Combine scores (lower combined score is worse)
    combined_score = (
        weights["brisque_score"] * normalized_brisque
        + weights["laplacian_variance"] * normalized_laplacian
        + weights["fft_blur_score_bin"] * normalized_fft
        + weights["entropy"] * normalized_entropy
    )

    # Add combined score to the DataFrame
    df_copy["combined_score"] = combined_score

    # Sort by combined score (ascending) and select top n_samples
    worst_images = df_copy.sort_values("combined_score").head(n_samples)

    return worst_images


def format_image_paths(image_paths):
    formatted_paths = [f"dataset/full/{path}" for path in image_paths]

    return formatted_paths


def get_misclassified_from_predictions(predictions, data_module, test_config, n=5):
    misclassified_df = analyze_misclassifications(predictions, data_module)

    worst_gender = get_top_misclassifications(misclassified_df, "gender", n)
    worst_age = get_top_misclassifications(misclassified_df, "age", n)
    worst_combined = get_top_misclassifications(misclassified_df, "combined", n)

    MisclassifiedFiles = namedtuple("MisclassifiedFiles", ["gender", "age", "combined"])

    def prefix_full_path(filenames, base_path):
        return [os.path.join(base_path, filename) for filename in filenames]

    # Get the filenames and prefix with full path
    base_path = test_config["ds_path"]
    misclassified_files = MisclassifiedFiles(
        gender=prefix_full_path(worst_gender["filename"].tolist(), base_path),
        age=prefix_full_path(worst_age["filename"].tolist(), base_path),
        combined=prefix_full_path(worst_combined["filename"].tolist(), base_path),
    )

    return misclassified_files
    # # Print the results
    # print("\nTop 20 Gender Misclassification Filenames:")
    # print(misclassified_files.gender)
    # print("\nTop 20 Age Misclassification Filenames:")
    # print(misclassified_files.age)
    # print("\nTop 20 Combined Misclassification Filenames:")
    # print(misclassified_files.combined)


def compare_model_df_predictions(merged_data_base, merged_data_improved):
    base_data_wrong_pred_df_good_on_improved = merged_data_base[
        (
            (merged_data_base["gender_pred"] > 0.5)
            & (merged_data_base["true_gender"] == 0)
        )
        | (
            (merged_data_base["gender_pred"] <= 0.5)
            & (merged_data_base["true_gender"] == 1)
        )
    ]

    base_data_wrong_pred_df_good_on_improved = pd.merge(
        base_data_wrong_pred_df_good_on_improved,
        merged_data_improved[["image_path", "true_gender", "gender_pred"]],
        on="image_path",
        how="left",
    )

    base_data_wrong_pred_df_good_on_improved = base_data_wrong_pred_df_good_on_improved[
        (
            (
                (base_data_wrong_pred_df_good_on_improved["true_gender_x"] == 0)
                & (base_data_wrong_pred_df_good_on_improved["gender_pred_x"] >= 0.5)
            )
            | (
                (base_data_wrong_pred_df_good_on_improved["true_gender_x"] == 1)
                & (base_data_wrong_pred_df_good_on_improved["gender_pred_x"] < 0.5)
            )
        )
        & (
            (
                (base_data_wrong_pred_df_good_on_improved["true_gender_y"] == 0)
                & (base_data_wrong_pred_df_good_on_improved["gender_pred_y"] < 0.5)
            )
            | (
                (base_data_wrong_pred_df_good_on_improved["true_gender_y"] == 1)
                & (base_data_wrong_pred_df_good_on_improved["gender_pred_y"] >= 0.5)
            )
        )
    ]
    # Calculate error magnitude
    base_data_wrong_pred_df_good_on_improved["base_error"] = abs(
        base_data_wrong_pred_df_good_on_improved["gender_pred_x"]
        - base_data_wrong_pred_df_good_on_improved["true_gender_x"]
    )

    improved_gender_samples_df = base_data_wrong_pred_df_good_on_improved.sort_values(
        "base_error", ascending=False
    )

    merged_data_base["age_error"] = abs(
        merged_data_base["age_pred"] - merged_data_base["true_age"]
    )
    merged_data_improved["age_error"] = abs(
        merged_data_improved["age_pred"] - merged_data_improved["true_age"]
    )

    age_comparison = pd.merge(
        merged_data_base[["image_path", "true_age", "age_pred", "age_error"]],
        merged_data_improved[["image_path", "age_pred", "age_error"]],
        on="image_path",
        suffixes=("_base", "_improved"),
    )

    age_comparison["error_reduction"] = (
        age_comparison["age_error_base"] - age_comparison["age_error_improved"]
    )

    improved_age_samples_df = age_comparison.sort_values(
        "error_reduction", ascending=False
    )

    return improved_gender_samples_df, improved_age_samples_df


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, precision_recall_fscore_support
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.colors import ListedColormap


def confusion_matrix_plot_v2(
    df,
    true_col,
    pred_col,
    class_labels=None,
    threshold=0.5,
    ax=None,
    title=None,
    subtitle=None,
    annotations=None,
):
    if ax is None:
        fig, ax = plt.subplots(figsize=(4 * len(class_labels), 4 * len(class_labels)))

    y_true = df[true_col]
    y_pred = df[pred_col]

    is_binary = len(set(y_true)) == 2

    if is_binary:
        y_pred = (y_pred > threshold).astype(int)
    else:
        y_pred = y_pred.astype(int)

    y_true = y_true.astype(int)

    cm = confusion_matrix(y_true, y_pred)
    cm_normalized = cm.astype("float") / cm.sum(axis=1)[:, np.newaxis]

    class_accuracies = np.diag(cm_normalized)
    precision, recall, f1, _ = precision_recall_fscore_support(
        y_true, y_pred, average=None
    )

    mask_correct = np.eye(cm.shape[0], dtype=bool)
    mask_incorrect = ~mask_correct

    cmap_incorrect = sns.diverging_palette(220, 20, as_cmap=True)
    cmap_correct = cmap_incorrect.copy().reversed()

    sns.heatmap(
        cm_normalized,
        vmin=0,
        vmax=1,
        mask=mask_incorrect,
        cmap=cmap_correct,
        annot=False,
        cbar=False,
        ax=ax,
    )

    sns.heatmap(
        cm_normalized,
        mask=mask_correct,
        cmap=cmap_incorrect,
        annot=False,
        alpha=0.5,
        cbar=False,
        ax=ax,
    )
    total_rows = len(y_true)

    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            expected = y_true.value_counts()[i]

            headline_anno = None
            annotation_1 = ""
            annotation_2 = ""
            annotation_2_1 = ""
            annotation_3 = ""

            if i == j:
                headline_anno = f"Recall:\n{class_accuracies[i]:.1%}\n"
                annotation_2 += f"F1: {f1[i]:.1%}"
                annotation_2_1 += f"Precision: {precision[i]:.1%}, "
            else:
                annotation_2 = f"Missed:\n{cm_normalized[i, j]:.1%}\n"

            annotation_3 += f"n={cm[i, j]} / {expected}"
            color = "black" if i == j else "white"

            if i == j:
                random_guess_accuracy = y_true.value_counts()[j] / total_rows
                annotation_3 += (
                    f"\n\nExp. (random) Recall: {random_guess_accuracy:.1%}\n"
                )

            if headline_anno:
                ax.text(
                    j + 0.5,
                    i + 0.3,
                    headline_anno,
                    horizontalalignment="center",
                    verticalalignment="center",
                    color=color,
                    fontsize=18,
                )
                ax.text(
                    j + 0.5,
                    i + 0.45,
                    annotation_2,
                    horizontalalignment="center",
                    verticalalignment="center",
                    color=color,
                    fontsize=16,
                )
                ax.text(
                    j + 0.5,
                    i + 0.525,
                    f"({annotation_2_1})",
                    horizontalalignment="center",
                    verticalalignment="center",
                    color=color,
                    fontsize=14,
                )
                ax.text(
                    j + 0.5,
                    i + 0.755,
                    annotation_3,
                    horizontalalignment="center",
                    verticalalignment="center",
                    color=color,
                    fontsize=14,
                )
            else:
                ax.text(
                    j + 0.5,
                    i + 0.5,
                    annotation_2,
                    horizontalalignment="center",
                    verticalalignment="center",
                    color=color,
                    fontsize=16,
                )
                ax.text(
                    j + 0.5,
                    i + 0.65,
                    annotation_3,
                    horizontalalignment="center",
                    verticalalignment="center",
                    color=color,
                    fontsize=16,
                )

    # Adding diagonal stripes pattern for incorrect predictions
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            if mask_incorrect[i, j]:
                ax.add_patch(
                    plt.Rectangle(
                        (j, i),
                        1,
                        1,
                        fill=False,
                        hatch="//",
                        edgecolor="black",
                        lw=0,
                        alpha=0.5,
                    )
                )

    if title and subtitle:
        ax.set_title(title, pad=35)
        ax.text(
            0.5,
            1.055,
            f"(Model: {subtitle})",
            ha="center",
            va="center",
            transform=ax.transAxes,
            fontsize="medium",
        )
    else:
        ax.set_title(title if title else "Confusion Matrix with Percentage Accuracy")

    ax.set_xlabel("Predicted label")
    ax.set_ylabel("True label")

    if class_labels:
        ax.set_xticks(np.arange(len(class_labels)) + 0.5)
        ax.set_yticks(np.arange(len(class_labels)) + 0.5)
        ax.set_xticklabels(class_labels)
        ax.set_yticklabels(class_labels)

    if annotations:
        ax.text(
            0.5, -0.1, annotations, ha="center", va="center", transform=ax.transAxes
        )

    return ax


def confusion_matrix_plot_v3(
    df,
    true_col,
    pred_col,
    class_labels=None,
    threshold=0.5,
    ax=None,
    title=None,
    subtitle=None,
    annotations=None,
    simplified=False,
):
    if ax is None:
        fig, ax = plt.subplots(figsize=(4 * len(class_labels), 4 * len(class_labels)))

    y_true = df[true_col]
    y_pred = df[pred_col]

    is_binary = len(set(y_true)) == 2

    if is_binary:
        y_pred = (y_pred > threshold).astype(int)
    else:
        y_pred = y_pred.astype(int)

    y_true = y_true.astype(int)

    cm = confusion_matrix(y_true, y_pred)
    cm_normalized = cm.astype("float") / cm.sum(axis=1)[:, np.newaxis]

    class_accuracies = np.diag(cm_normalized)
    precision, recall, f1, _ = precision_recall_fscore_support(
        y_true, y_pred, average=None
    )

    mask_correct = np.eye(cm.shape[0], dtype=bool)
    mask_incorrect = ~mask_correct

    cmap_incorrect = sns.diverging_palette(220, 20, as_cmap=True)
    cmap_correct = cmap_incorrect.copy().reversed()

    sns.heatmap(
        cm_normalized,
        vmin=0,
        vmax=1,
        mask=mask_incorrect,
        cmap=cmap_correct,
        annot=False,
        cbar=False,
        ax=ax,
    )

    sns.heatmap(
        cm_normalized,
        mask=mask_correct,
        cmap=cmap_incorrect,
        annot=False,
        alpha=0.5,
        cbar=False,
        ax=ax,
    )
    total_rows = len(y_true)

    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            if simplified:
                if i == j:
                    expected = y_true.value_counts()[i]
                    ax.text(
                        j + 0.5,
                        i + 0.3,
                        f"F1 = {f1[i]:.2f}",
                        horizontalalignment="center",
                        verticalalignment="center",
                        color="black",
                        fontsize=16,
                    )
                    ax.text(
                        j + 0.5,
                        i + 0.5,
                        f"Prec/Recall",
                        horizontalalignment="center",
                        verticalalignment="center",
                        color="black",
                        fontsize=14,
                    )
                    ax.text(
                        j + 0.5,
                        i + 0.65,
                        f"{precision[i]:.2f}/{recall[i]:.2f}",
                        horizontalalignment="center",
                        verticalalignment="center",
                        color="black",
                        fontsize=14,
                    )
                    ax.text(
                        j + 0.5,
                        i + 0.8,
                        f"{cm[i, j]}/{expected}",
                        horizontalalignment="center",
                        verticalalignment="center",
                        color="black",
                        fontsize=14,
                    )
            else:
                expected = y_true.value_counts()[i]

                headline_anno = None
                annotation_1 = ""
                annotation_2 = ""
                annotation_2_1 = ""
                annotation_3 = ""

                if i == j:
                    headline_anno = f"Recall:\n{class_accuracies[i]:.1%}\n"
                    annotation_2 += f"F1: {f1[i]:.1%}"
                    annotation_2_1 += f"Precision: {precision[i]:.1%}, "
                else:
                    annotation_2 = f"Missed:\n{cm_normalized[i, j]:.1%}\n"

                annotation_3 += f"n={cm[i, j]} / {expected}"
                color = "black" if i == j else "white"

                if i == j:
                    random_guess_accuracy = y_true.value_counts()[j] / total_rows
                    annotation_3 += (
                        f"\n\nExp. (random) Recall: {random_guess_accuracy:.1%}\n"
                    )

                if headline_anno:
                    ax.text(
                        j + 0.5,
                        i + 0.3,
                        headline_anno,
                        horizontalalignment="center",
                        verticalalignment="center",
                        color=color,
                        fontsize=18,
                    )
                    ax.text(
                        j + 0.5,
                        i + 0.45,
                        annotation_2,
                        horizontalalignment="center",
                        verticalalignment="center",
                        color=color,
                        fontsize=16,
                    )
                    ax.text(
                        j + 0.5,
                        i + 0.525,
                        f"({annotation_2_1})",
                        horizontalalignment="center",
                        verticalalignment="center",
                        color=color,
                        fontsize=14,
                    )
                    ax.text(
                        j + 0.5,
                        i + 0.755,
                        annotation_3,
                        horizontalalignment="center",
                        verticalalignment="center",
                        color=color,
                        fontsize=14,
                    )
                else:
                    ax.text(
                        j + 0.5,
                        i + 0.5,
                        annotation_2,
                        horizontalalignment="center",
                        verticalalignment="center",
                        color=color,
                        fontsize=16,
                    )
                    ax.text(
                        j + 0.5,
                        i + 0.65,
                        annotation_3,
                        horizontalalignment="center",
                        verticalalignment="center",
                        color=color,
                        fontsize=16,
                    )

    # Adding diagonal stripes pattern for incorrect predictions
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            if mask_incorrect[i, j]:
                ax.add_patch(
                    plt.Rectangle(
                        (j, i),
                        1,
                        1,
                        fill=False,
                        hatch="//",
                        edgecolor="black",
                        lw=0,
                        alpha=0.5,
                    )
                )

    if title and subtitle:
        ax.set_title(title, pad=35)
        ax.text(
            0.5,
            1.055,
            f"(Model: {subtitle})",
            ha="center",
            va="center",
            transform=ax.transAxes,
            fontsize="medium",
        )
    else:
        ax.set_title(title if title else "Confusion Matrix with Percentage Accuracy")

    ax.set_xlabel("Predicted label")
    ax.set_ylabel("True label")

    if class_labels:
        ax.set_xticks(np.arange(len(class_labels)) + 0.5)
        ax.set_yticks(np.arange(len(class_labels)) + 0.5)
        ax.set_xticklabels(class_labels)
        ax.set_yticklabels(class_labels)

    if annotations:
        ax.text(
            0.5, -0.1, annotations, ha="center", va="center", transform=ax.transAxes
        )

    return ax
