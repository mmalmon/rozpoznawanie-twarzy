"""
Definicja modelu do rozpoznawania twarzy (klasyfikacja obrazu twarzy -> osoba).

Uzywamy transfer learningu: bierzemy siec MobileNetV3-Small wytrenowana na
ImageNet (1000 klas ogolnych obiektow) i podmieniamy jej ostatnia warstwe
(klasyfikator) na nowa, dopasowana do liczby osob w naszej bazie. Dzieki temu:
  - potrzebujemy dużo mniej danych treningowych (setki, a nie miliony zdjec),
  - trening jest szybki nawet na laptopowym GPU z 6 GB VRAM,
  - siec jest mala i szybka w inferencji (nadaje sie do pracy w czasie
    rzeczywistym na strumieniu z kamery).
"""

from __future__ import annotations

import torch
from torch import nn
from torchvision import models

IMAGE_SIZE = 224  # standardowy rozmiar wejscia dla sieci trenowanych na ImageNet

# Srednia i odchylenie standardowe kanalow RGB uzyte przy treningu ImageNet -
# musimy znormalizowac nasze obrazy dokladnie tak samo, jak zrobiono to przy
# oryginalnym treningu backbone'u, inaczej wagi pretrenowane nie beda dzialac.
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]


def build_model(num_classes: int, pretrained: bool = True, freeze_backbone: bool = True) -> nn.Module:
    """Tworzy model MobileNetV3-Small z podmienionym klasyfikatorem koncowym."""
    weights = models.MobileNet_V3_Small_Weights.DEFAULT if pretrained else None
    model = models.mobilenet_v3_small(weights=weights)

    if freeze_backbone:
        # Zamrazamy wagi "ekstraktora cech" - trenujemy tylko nowy klasyfikator.
        # To przyspiesza trening i zmniejsza ryzyko przeuczenia przy malym
        # zbiorze danych (kilkaset zdjec na osobe).
        for param in model.features.parameters():
            param.requires_grad = False

    in_features = model.classifier[-1].in_features
    model.classifier[-1] = nn.Linear(in_features, num_classes)

    return model


def unfreeze_backbone(model: nn.Module, last_n_blocks: int = 3) -> None:
    """Odmraza ostatnie N blokow ekstraktora cech (fine-tuning drugiego etapu).

    Wywolywane po kilku epokach treningu samego klasyfikatora - pozwala
    lekko doszkolic gorne warstwy backbone'u pod nasze konkretne twarze,
    zwykle podnoszac dokladnosc o kilka punktow procentowych.
    """
    blocks = list(model.features.children())
    for block in blocks[-last_n_blocks:]:
        for param in block.parameters():
            param.requires_grad = True


def get_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    print(
        "[model] UWAGA: nie wykryto GPU z obsluga CUDA - trening/inferencja "
        "beda dzialac na CPU i moga byc znacznie wolniejsze."
    )
    return torch.device("cpu")
