import torch
import torch.nn as nn
from torchvision import models
from typing import Dict, Any

class AgeGenderModel(nn.Module):
    """
    Clean PyTorch module for age and gender classification.
    Stripped of PyTorch Lightning dependencies for pure inference.
    """

    def __init__(self, config: Dict[str, Any]) -> None:
        super().__init__()
        self.config = config
        self._initialize_model()

    def get_param(self, key: str, default: Any = None) -> Any:
        return self.config.get(key, default)

    def _initialize_model(self) -> None:
        model_type = self.get_param("model_type", "mobilenet_v3_large")
        
        # We set weights to None since we load our own checkpoint weights for inference
        if model_type == "mobilenet_v3_large":
            self.base_model = models.mobilenet_v3_large(weights=None)
        elif model_type == "mobilenet_v3_small":
            self.base_model = models.mobilenet_v3_small(weights=None)
        elif model_type == "efficientnet_b0":
            self.base_model = models.efficientnet_b0(weights=None)
        else:
            raise ValueError(f"Unsupported model type: {model_type}")

        # Extract features and strip the original classification heads
        if "mobilenet" in model_type:
            if hasattr(self.base_model, "classifier"):
                if isinstance(self.base_model.classifier, nn.Sequential):
                    num_features = self.base_model.classifier[0].in_features
                else:
                    num_features = self.base_model.classifier.in_features
            else:
                num_features = self.base_model.last_channel

            self.base_model = nn.Sequential(*list(self.base_model.children())[:-1])

        elif "efficientnet" in model_type:
            num_features = self.base_model.classifier[1].in_features
            self.base_model = nn.Sequential(*list(self.base_model.children())[:-2])
        else:
            raise ValueError(f"Unexpected model type: {model_type}")

        self.global_pool = nn.AdaptiveAvgPool2d(1)
        dropout_rate = self.get_param("dropout", 0)

        # Multi-task outputs: age regression + gender classification
        self.gender_classifier = nn.Sequential(
            nn.Dropout(p=dropout_rate), nn.Linear(num_features, 2)
        )
        self.age_regressor = nn.Sequential(
            nn.Dropout(p=dropout_rate), nn.Linear(num_features, 1)
        )

        if dropout_rate > 0:
            self.dropout = nn.Dropout(p=dropout_rate)

    def forward(self, x: torch.Tensor):
        features = self.base_model(x)
        features = self.global_pool(features).view(x.size(0), -1)
        gender_output = self.gender_classifier(features)
        age_output = self.age_regressor(features).squeeze(1)
        return gender_output, age_output
