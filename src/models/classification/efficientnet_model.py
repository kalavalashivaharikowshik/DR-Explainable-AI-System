import torch
import torch.nn as nn
from torchvision import models
from torchvision.models import efficientnet_b3, EfficientNet_B3_Weights

NUM_CLASSES = 5

def get_model():
    model = efficientnet_b3(weights=EfficientNet_B3_Weights.DEFAULT)

    in_features = model.classifier[1].in_features

    model.classifier = nn.Sequential(
        nn.Dropout(0.5),
        nn.Linear(in_features, NUM_CLASSES)
    )

    return model
