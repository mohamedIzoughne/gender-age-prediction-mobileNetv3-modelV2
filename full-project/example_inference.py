import argparse
from PIL import Image
from src.age_gender_model.inference import AgeGenderPredictor
from src.age_gender_model.alignment import detect_and_align_face

def main(image_path: str, model_path: str):
    print(f"Loading age/gender predictor from {model_path}...")
    predictor = AgeGenderPredictor(checkpoint_path=model_path)
    
    print(f"Loading image from {image_path}...")
    image = Image.open(image_path).convert("RGB")
    
    print("Aligning and cropping face...")
    face_image = detect_and_align_face(image, device=predictor.device)
    
    print("Running inference...")
    results = predictor.predict(face_image)
    
    print("\n--- Predictions ---")
    print(f"Predicted Age: {results['age']:.1f} years")
    print(f"Predicted Gender: {results['gender_label']} ({results['gender_confidence']*100:.1f}% confidence)")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Test Demographics Model")
    parser.add_argument("--image", type=str, required=True, help="Path to the image to test")
    parser.add_argument("--model", type=str, required=True, help="Path to the model checkpoint (.pth or .ckpt)")
    args = parser.parse_args()

    main(args.image, args.model)
