import torch.nn as nn
from torchvision import models

NUM_CLASSES = 4

def build_model():
    # Pretrained MobileNetV2 (lightweight)
    model = models.mobilenet_v2(weights=models.MobileNet_V2_Weights.DEFAULT)

    # Freeze feature extractor
    for param in model.features.parameters():
        param.requires_grad = False

    # Replace classifier
    model.classifier[1] = nn.Linear(model.last_channel, NUM_CLASSES)

    return model