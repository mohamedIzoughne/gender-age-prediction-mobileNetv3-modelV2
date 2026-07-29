# Demographics Inference Service

This directory contains the cleaned, modularized code for Age and Gender inference, designed to be easily integrated into a larger monorepo or production service.

All training dependencies (PyTorch Lightning, metrics, TensorBoard, custom data samplers) have been completely removed.

## Structure
- `demographics/model.py`: Pure PyTorch `nn.Module` definition.
- `demographics/transforms.py`: Standard inference transformations.
- `demographics/inference.py`: Wrapper class to handle checkpoint loading and predictions.
- `demographics/alignment.py`: (Optional) MTCNN-based face alignment for raw images.
- `example_usage.py`: Example script tying it all together.

## Installation
```bash
pip install -r requirements.txt
```

## Quick Start
```python
from PIL import Image
from demographics import DemographicsPredictor
from demographics.alignment import detect_and_align_face

# 1. Load the predictor (handles config parsing & weights)
predictor = DemographicsPredictor(checkpoint_path="path/to/model.pth")

# 2. Load and align an image
raw_image = Image.open("person.jpg")
face_image = detect_and_align_face(raw_image) # Requires facenet-pytorch

# 3. Predict
results = predictor.predict(face_image)
print(results)
# {'age': 28.5, 'gender_id': 1, 'gender_label': 'Female', 'gender_confidence': 0.98}
```
