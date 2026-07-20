import os, sys

import matplotlib.pyplot as plt
from PIL import Image
import pandas as pd


def render_image(img):
    img1 = Image.open(f"Notebooks/imgs/{img}.png")

    # Create a figure with two subplots side by side
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 10))

    # Display images
    ax1.imshow(img1)
    ax2.imshow(img1)

    # Remove axes
    ax1.axis("off")
    ax2.axis("off")

    # Adjust layout and display
    plt.tight_layout()
    plt.show()


def fix_cwd():
    if "Notebooks" in os.getcwd():
        notebook_dir = os.getcwd()

        project_root = os.path.dirname(notebook_dir)

        if os.getcwd() != project_root:
            os.chdir(project_root)


def model_desc_table():
    data = {
        "Metric": [
            "Parameter Count",
            "Model Size (PyTorch, FP32)",
            "Inference Speed (relative)",
            "FLOPs",
            "Approx. Memory Usage (inference)",
        ],
        "VGG16": [
            "~138 million",
            "~528 MB",
            "1x (baseline)",
            "~15.5 billion",
            "1x",
        ],
        "ResNet50": [
            "~25.6 million",
            "~98 MB",
            "~2.5x faster",
            "~4.1 billion",
            "~0.6x",
        ],
        "MobileNetV3-Small": [
            "~2.5 million",
            "~10 MB",
            "~10x faster",
            "~56 million",
            "~0.15x",
        ],
    }

    df = pd.DataFrame(data)
    df.set_index("Metric", inplace=True)

    return df


def get_baselines_table():
    baseline_table = pd.DataFrame(
        {
            "Model": [
                "XGBoost (+feat. extraction)",
                "SVC(..)",
                "VGG_f",
                "ResNet50_f",
                "SENet50_f",
            ],
            "Age Estimation (MAE)": [5.89, 5.49, 4.86, 4.65, 4.58],
            "Gender Classification (Accuracy)": [93.80, 94.64, 93.42, 94.64, 94.9],
        }
    )
    return baseline_table
