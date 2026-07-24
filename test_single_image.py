import argparse
import torch
from PIL import Image
import torch.nn.functional as F
import os
import math
from facenet_pytorch import MTCNN

# Import your model loading function and transform logic
from src.models.mobilenet.classifier import AgeGenderClassifier
from src.models.MobileNet.data_defs import get_transforms

def main(image_path: str, model_path: str):
    print(f"Loading model from {model_path}...")
    # Load directly to CPU, handling path properly
    checkpoint = torch.load(model_path, map_location=torch.device('cpu'))
    
    # Standard PyTorch Lightning checkpoints often don't have a "config" key
    # or the user might have saved it differently. Fallback with required parameters.
    config = checkpoint.get("config", {})
    if not config:
        print("Warning: No configuration found in the checkpoint. Using default configuration.")
        config = {
            "model_type": "mobilenet_v3_large",
            "freeze_epochs": 0,
            "use_dynamic_augmentation": False
        }
        
    model = AgeGenderClassifier(config)
    
    # Try PyTorch Lightning's standard "state_dict" first, then custom "model_state_dict"
    if "state_dict" in checkpoint:
        model.load_state_dict(checkpoint["state_dict"])
    elif "model_state_dict" in checkpoint:
        model.load_state_dict(checkpoint["model_state_dict"])
    else:
        model.load_state_dict(checkpoint) # raw fallback

    model.eval()  # Set the model to evaluation mode
    print("Model loaded successfully!")

    # 2. Load the image
    print(f"Loading image from {image_path}...")
    try:
        # Open the image
        image = Image.open(image_path).convert("RGB")
    except FileNotFoundError:
        print(f"Could not find image at {image_path}. Please provide a valid image path.")
        return

    # Setup MTCNN for face detection and alignment
    print("Detecting and aligning face...")
    mtcnn = MTCNN(image_size=224, margin=20, keep_all=False, post_process=False, device='cpu')
    
    # Define where to save the aligned check image
    os.makedirs("aligned_faces", exist_ok=True)
    base_name = os.path.basename(image_path)
    aligned_path = os.path.join("aligned_faces", base_name)
    
    # 1. Detect face and landmarks to get strict rotation alignment
    boxes, probs, landmarks = mtcnn.detect(image, landmarks=True)
    
    if boxes is None or landmarks is None:
        print(f"Warning: No face detected in {image_path}. Using original image directly.")
        image = image.resize((224, 224))
    else:
        # Landmarks: 0: left eye, 1: right eye (from observer's perspective)
        left_eye = landmarks[0][0]
        right_eye = landmarks[0][1]
        
        # Calculate angle for alignment
        dy = right_eye[1] - left_eye[1]
        dx = right_eye[0] - left_eye[0]
        angle = math.degrees(math.atan2(dy, dx))
        
        # Calculate eye center
        eye_center = ((left_eye[0] + right_eye[0]) / 2, (left_eye[1] + right_eye[1]) / 2)
        
        # Rotate image around the eye center to make eyes horizontal
        aligned_full_image = image.rotate(angle, center=eye_center, resample=Image.BILINEAR)
        
        # 2. Re-detect and crop the now strictly aligned face
        face_tensor = mtcnn(aligned_full_image, save_path=aligned_path)
        
        if face_tensor is None:
            # Fallback if re-detection fails after rotation
            print("Warning: Failed to crop aligned face. Using un-aligned crop.")
            face_tensor = mtcnn(image, save_path=aligned_path)
            
        print(f"Strictly aligned face saved to {aligned_path} for checking!")
        # We reload the aligned image to pass it smoothly to your transforms
        image = Image.open(aligned_path).convert("RGB")

    # 3. Preprocess the image
    # This applies normalization
    transform = get_transforms(get_compose=True)
    input_tensor = transform(image).unsqueeze(0)  # Add batch dimension

    # 4. Run the prediction
    print("Running prediction...")
    with torch.no_grad():
        gender_logits, age_pred = model(input_tensor)
        
        # Convert gender logits to probabilities
        gender_probs = F.softmax(gender_logits, dim=1)
        predicted_gender_idx = torch.argmax(gender_probs, dim=1).item()
        
        # Assuming UTKFace standard: 0 = Male, 1 = Female
        gender_label = "Female" if predicted_gender_idx == 1 else "Male"
        gender_confidence = gender_probs[0][predicted_gender_idx].item() * 100
        
        predicted_age = age_pred.item()

    print("\n--- Predictions ---")
    print(f"Predicted Age: {predicted_age:.1f} years")
    print(f"Predicted Gender: {gender_label} ({gender_confidence:.1f}% confidence)")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Test Age/Gender Model on a Single Image")
    parser.add_argument("--image", type=str, default="test_image.jpg", help="Path to the image to test")
    parser.add_argument("--model", type=str, default="model_checkpoint.ckpt", help="Filename of the model checkpoint in model_store/ or relative path depending on load_model implementation")
    args = parser.parse_args()

    main(args.image, args.model)
