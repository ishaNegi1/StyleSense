import torch.nn as nn
from torchvision import models

NUM_CLASSES = 4

def build_model():
    # ✅ Load pretrained MobileNetV2
    model = models.mobilenet_v2(weights=None)

    # ✅ Freeze backbone — only train the classifier head
    for param in model.features.parameters():
        param.requires_grad = False

    # ✅ Replace classifier for our 4 classes
    model.classifier[1] = nn.Linear(model.last_channel, NUM_CLASSES)

    return model