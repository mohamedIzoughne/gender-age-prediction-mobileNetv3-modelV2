import torch
from torchvision.transforms import v2 as transforms

def get_inference_transforms():
    """
    Returns the composed transforms required for model inference.
    Matches the validation transforms used during training.
    """
    return transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.Compose([
            transforms.ToImage(), 
            transforms.ToDtype(torch.float32, scale=True)
        ]),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])
