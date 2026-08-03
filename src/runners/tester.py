import os
import sys
import argparse
import math
from PIL import Image
import torch
import pytorch_lightning as pl
from facenet_pytorch import MTCNN

# Add the project root to sys.path to resolve 'src' module imports
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../"))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from src.models.mobilenet.data_loader import create_dataloaders
from src.runners.trainer import load_config
from src.models.mobilenet.classifier import AgeGenderClassifier
from src.models.mobilenet.data_defs import AgeGenderDataModule

class FaceAlignTransform:
    def __init__(self, base_transform):
        self.base_transform = base_transform
        self.mtcnn = None

    def __call__(self, img):
        if self.mtcnn is None:
            self.mtcnn = MTCNN(image_size=224, margin=20, keep_all=False, post_process=False, device='cpu')

        boxes, probs, landmarks = self.mtcnn.detect(img, landmarks=True)
        if boxes is not None and landmarks is not None:
            left_eye = landmarks[0][0]
            right_eye = landmarks[0][1]
            dy = right_eye[1] - left_eye[1]
            dx = right_eye[0] - left_eye[0]
            angle = math.degrees(math.atan2(dy, dx))
            eye_center = ((left_eye[0] + right_eye[0]) / 2, (left_eye[1] + right_eye[1]) / 2)
            
            aligned_img = img.rotate(angle, center=eye_center, resample=Image.BILINEAR)
            boxes2, probs2 = self.mtcnn.detect(aligned_img)
            if boxes2 is not None:
                box = boxes2[0]
                aligned_img = aligned_img.crop((box[0], box[1], box[2], box[3]))
                img = aligned_img
                
        return self.base_transform(img)

class AlignedTestDataModule(AgeGenderDataModule):
    def setup(self, stage=None):
        super().setup(stage)
        if self.mode == "test":
            original_transform = self.test_dataset.transform
            self.test_dataset.transform = FaceAlignTransform(original_transform)

def load_pth_model(path: str, config: dict) -> AgeGenderClassifier:
    checkpoint = torch.load(path)
    # Use the config stored in the checkpoint if available, as it has the correct model_type
    ckpt_config = checkpoint.get("config", config)
    model = AgeGenderClassifier(ckpt_config)
    model.load_state_dict(checkpoint["model_state_dict"])
    return model

def test(config, ckpt_path):
    print(f"\n - - - \nTesting with Config:\n{dict(config)}\n\n - - - \n")
    pl.seed_everything(42, workers=True)

    data = AlignedTestDataModule(config, mode="test")
    
    print(f"Loading model from {ckpt_path}")
    if ckpt_path.endswith('.ckpt'):
        model = AgeGenderClassifier.load_from_checkpoint(ckpt_path, config=config)
    elif ckpt_path.endswith('.pth'):
        model = load_pth_model(ckpt_path, config)
    else:
        raise ValueError("Invalid checkpoint format. Must be .ckpt or .pth")
        
    trainer = pl.Trainer(
        accelerator="auto",
        devices=1,
    )
    
    results = trainer.test(model, datamodule=data)
    
    print("\n" + "="*50)
    print("FINAL TEST RESULTS:")
    print("="*50)
    if results:
        for k, v in results[0].items():
            print(f"{k}: {v:.4f}")
    print("="*50 + "\n")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Test Age & Gender Classifier")
    parser.add_argument("--config", type=str, default="config/model/my-configs/sweep-34_improved_DYNAMIC_AUG_small.yaml", help="Path to config file (e.g., config/model/my-configs/sweep-34_improved_DYNAMIC_AUG_small.yaml, config/model/my-configs/swept-sweep-34_improved_DYNAMIC_AUG_v3_large.yaml, config/model/my-configs/mobilenet_v3_large_aug.yaml, config/model/my-configs/mobilenet_v3_large_no_aug.yaml, config/model/my-configs/efficientnet_b0_aug.yaml, config/model/my-configs/efficientnet_b0_no_aug.yaml)")
    parser.add_argument("--ckpt", type=str, required=True, help="Path to model checkpoint (.ckpt or .pth)")
    parser.add_argument("--ds_path", type=str, help="Override path to test dataset (e.g. /content/data/UTKFace)")
    args = parser.parse_args()

    config_path = os.path.join(project_root, args.config) if not os.path.isabs(args.config) else args.config
    config = load_config(config_path)
    
    if args.ds_path:
        config["ds_path"] = args.ds_path
        
    test(config, args.ckpt)
