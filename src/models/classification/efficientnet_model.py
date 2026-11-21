import torch.nn as nn
from torchvision.models import EfficientNet_B3_Weights, efficientnet_b3

from src.config.config import NUM_CLASSES


def get_model(pretrained: bool = False) -> nn.Module:
    """EfficientNet-B3 with a 5-class DR head.

    pretrained=True downloads ImageNet weights (use for training only).
    For inference the trained weights are loaded on top, so no download is needed.
    """
    weights = EfficientNet_B3_Weights.DEFAULT if pretrained else None
    model = efficientnet_b3(weights=weights)

    in_features = model.classifier[1].in_features

    model.classifier = nn.Sequential(
        nn.Dropout(0.5),
        nn.Linear(in_features, NUM_CLASSES),
    )

    return model
