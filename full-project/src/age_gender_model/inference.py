import os
import torch
import torch.nn.functional as F
from PIL import Image
from typing import Dict, Union

from .model import AgeGenderModel
from .transforms import get_inference_transforms

class AgeGenderPredictor:
    """
    End-to-end predictor for Age and Gender.
    """
    def __init__(self, checkpoint_path: str, device: str = None):
        """
        Initializes the predictor by loading the model weights.
        """
        self.device = device or ('cuda' if torch.cuda.is_available() else 'cpu')
        
        if not os.path.exists(checkpoint_path):
            raise FileNotFoundError(f"Checkpoint not found at {checkpoint_path}")
            
        # Load checkpoint directly to the target device
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        
        # Extract config. Fallback to default if missing.
        config = checkpoint.get("config", {
            "model_type": "mobilenet_v3_large",
            "dropout": 0,
        })
        
        self.model = AgeGenderModel(config)
        
        # Support PyTorch Lightning state_dict keys and custom save formats
        state_dict = checkpoint.get("model_state_dict") or checkpoint.get("state_dict") or checkpoint
        self.model.load_state_dict(state_dict)
        
        self.model.to(self.device)
        self.model.eval()
        
        self.transform = get_inference_transforms()
        
    def predict(self, image: Union[Image.Image, torch.Tensor]) -> Dict[str, Union[float, int, str]]:
        """
        Predicts age and gender for a given cropped face image.
        For best results, input image should be a 224x224 cropped face.
        """
        if isinstance(image, Image.Image):
            image = image.convert("RGB")
            tensor = self.transform(image).unsqueeze(0).to(self.device)
        elif isinstance(image, torch.Tensor):
            tensor = image.to(self.device)
            if tensor.dim() == 3:
                tensor = tensor.unsqueeze(0)
        else:
            raise TypeError("Input must be a PIL Image or torch Tensor")
            
        with torch.no_grad():
            gender_logits, age_pred = self.model(tensor)
            
            # Process gender
            gender_probs = F.softmax(gender_logits, dim=1)
            gender_idx = int(torch.argmax(gender_probs, dim=1).item())
            
            # UTKFace mapping: 0 -> Male, 1 -> Female
            gender_label = "Female" if gender_idx == 1 else "Male"
            gender_confidence = float(gender_probs[0][gender_idx].item())
            
            # Process age
            age = float(age_pred.item())
            
        return {
            "age": age,
            "gender_id": gender_idx,
            "gender_label": gender_label,
            "gender_confidence": gender_confidence
        }
