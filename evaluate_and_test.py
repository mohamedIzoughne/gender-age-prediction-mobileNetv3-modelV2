import os
import sys
import torch
import torchvision.transforms as transforms
from PIL import Image
import pandas as pd
import matplotlib.pyplot as plt

# Ensure the src module can be imported
if os.getcwd() not in sys.path:
    sys.path.append(os.getcwd())

from src.runners.trainer import load_model

def plot_metrics(csv_path: str):
    print(f"Loading metrics from {csv_path}...")
    if not os.path.exists(csv_path):
        print(f"Error: {csv_path} not found.")
        return

    df = pd.read_csv(csv_path)

    # Clean the dataframe: merge the train and val rows per epoch
    df_clean = df.groupby('epoch').apply(lambda x: x.bfill().ffill().iloc[0]).reset_index(drop=True)
    # The groupby puts 'epoch' in the index. We need to add it back as a column.
    df_clean['epoch'] = df_clean.index
    if 'stage' in df_clean.columns:
        df_clean = df_clean.drop(columns=['stage'])

    # Plot Training vs Validation Loss
    plt.figure(figsize=(10, 6))
    plt.plot(df_clean['epoch'], df_clean['train_loss'], label='Train Loss', marker='o')
    plt.plot(df_clean['epoch'], df_clean['val_loss'], label='Val Loss', marker='o')
    plt.title('Training and Validation Loss over Epochs')
    plt.xlabel('Epoch')
    plt.ylabel('Total Loss')
    plt.legend()
    plt.grid(True)
    
    # Save the plot instead of just showing it so it's easy to view without an interactive display
    plot_path = "loss_plot.png"
    plt.savefig(plot_path)
    print(f"Plot saved successfully to {plot_path}")


def predict_image(image_path: str, checkpoint_name: str):
    print(f"\nLoading model from {checkpoint_name}...")
    try:
        model = load_model(checkpoint_name)
    except Exception as e:
        print(f"Failed to load model: {e}")
        return

    model.eval()
    model.to('cuda' if torch.cuda.is_available() else 'cpu')

    if not os.path.exists(image_path):
        print(f"Error: Test image '{image_path}' not found.")
        return

    # Define Image Transforms
    val_transforms = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    img = Image.open(image_path).convert("RGB")
    
    # Preprocess and prepare for model
    img_tensor = val_transforms(img).unsqueeze(0).to(model.device)
    
    print(f"Running inference on {image_path}...")
    # Inference
    with torch.no_grad():
        gender_logits, age_pred = model(img_tensor)
        
        # Process outputs
        age = age_pred.item()
        gender_prob = torch.softmax(gender_logits, dim=1)
        gender_pred = torch.argmax(gender_prob, dim=1).item()
        
        gender_label = "Female" if gender_pred == 1 else "Male"
        confidence = gender_prob[0][gender_pred].item()
        
        print("-" * 50)
        print(f"Prediction Results:")
        print(f"Age:    {age:.1f} years")
        print(f"Gender: {gender_label} (Confidence: {confidence:.2f})")
        print("-" * 50)


if __name__ == "__main__":
    # 1. Update this to your actual metrics CSV
    csv_file = "mobilenet_v3_large_aug_metrics.csv"
    # plot_metrics(csv_file)

    # 2. Update these to your actual model and test image
    checkpoint_file = "ag_classifier_main_mobilenet_v3_large_aug_epoch25_loss1.1129.pth" # Needs to be inside model_store/
    test_image = "test-images/face14.jpg"
    
    predict_image(test_image, checkpoint_file)
