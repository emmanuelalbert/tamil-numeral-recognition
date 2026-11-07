"""
Model definitions for the architecture comparison:
  - ResNet-18   (pretrained, fine-tuned) -- primary/headline model
  - MobileNetV2 (pretrained, fine-tuned) -- lightweight comparison
  - SimpleCNN   (trained from scratch)   -- shallow baseline, no transfer learning
  - VGG11       (pretrained, fine-tuned) -- classical deep-CNN baseline

Both ResNet-18 and MobileNetV2 use ImageNet-pretrained weights. Training from
scratch on ~4,000 images with an 11M-parameter ResNet-18 invites overfitting;
pretrained weights give the early/mid layers generic edge and stroke features
that transfer well to handwritten digits, so fine-tuning only really adapts
the later layers.

SimpleCNN is trained from scratch deliberately -- it answers a different
question than the pretrained models: does this task need deep or transferred
features at all, or can a small, purpose-built network trained only on the
~4,000 available images do the job?
"""

import torch.nn as nn
from torchvision import models

import config


def build_resnet18(num_classes: int = config.NUM_CLASSES, pretrained: bool = True):
    weights = models.ResNet18_Weights.IMAGENET1K_V1 if pretrained else None
    model = models.resnet18(weights=weights)
    in_features = model.fc.in_features
    model.fc = nn.Linear(in_features, num_classes)  # raw logits, no softmax
    return model


def build_mobilenet_v2(num_classes: int = config.NUM_CLASSES, pretrained: bool = True):
    weights = models.MobileNet_V2_Weights.IMAGENET1K_V1 if pretrained else None
    model = models.mobilenet_v2(weights=weights)
    in_features = model.classifier[1].in_features
    model.classifier[1] = nn.Linear(in_features, num_classes)  # raw logits
    return model


def build_vgg11(num_classes: int = config.NUM_CLASSES, pretrained: bool = True):
    weights = models.VGG11_Weights.IMAGENET1K_V1 if pretrained else None
    model = models.vgg11(weights=weights)
    in_features = model.classifier[6].in_features
    model.classifier[6] = nn.Linear(in_features, num_classes)  # raw logits
    return model


class SimpleCNN(nn.Module):
    """
    Shallow CNN trained from scratch -- no pretrained weights, no transfer learning.
    Four conv blocks (conv -> ReLU -> maxpool) followed by two dense layers,
    matching the architecture description in the manuscript's original
    (pre-ResNet-18) Model Building section (Sec. 6.3): four conv layers with
    ReLU + max pooling, then a 128-unit dense layer before the output layer.

    Input: 3x224x224 (same preprocessing pipeline as the pretrained models,
    for a fair apples-to-apples data comparison).
    """

    def __init__(self, num_classes: int = config.NUM_CLASSES):
        super().__init__()

        self.features = nn.Sequential(
            # Block 1: 224x224 -> 112x112
            nn.Conv2d(3, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),

            # Block 2: 112x112 -> 56x56
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),

            # Block 3: 56x56 -> 28x28
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),

            # Block 4: 28x28 -> 14x14
            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
        )

        self.classifier = nn.Sequential(
            nn.AdaptiveAvgPool2d((1, 1)),  # 256x14x14 -> 256x1x1, avoids a huge FC layer
            nn.Flatten(),
            nn.Linear(256, 128),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),  # from-scratch training on ~4k images needs regularization
            nn.Linear(128, num_classes),  # raw logits, no softmax
        )

    def forward(self, x):
        x = self.features(x)
        x = self.classifier(x)
        return x


def build_simple_cnn(num_classes: int = config.NUM_CLASSES, pretrained: bool = True):
    # pretrained arg accepted for interface consistency with the registry,
    # but SimpleCNN has no pretrained weights available -- always trained from scratch.
    if pretrained:
        print("[NOTE] SimpleCNN has no pretrained weights; training from scratch regardless.")
    return SimpleCNN(num_classes=num_classes)


MODEL_REGISTRY = {
    "resnet18": build_resnet18,
    "mobilenet_v2": build_mobilenet_v2,
    "vgg11": build_vgg11,
    "simple_cnn": build_simple_cnn,
}


def get_model(name: str, num_classes: int = config.NUM_CLASSES, pretrained: bool = True):
    if name not in MODEL_REGISTRY:
        raise ValueError(f"Unknown model '{name}'. Choose from {list(MODEL_REGISTRY)}")
    return MODEL_REGISTRY[name](num_classes=num_classes, pretrained=pretrained)


def count_trainable_params(model) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def count_total_params(model) -> int:
    return sum(p.numel() for p in model.parameters())