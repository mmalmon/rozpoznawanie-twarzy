"""
Krok 3: Trening modelu rozpoznawania twarzy (transfer learning, PyTorch).

Wymaga wczesniejszego uruchomienia:
    01_zbieranie_danych.py (dla kazdej osoby)
    02_podzial_danych.py

Uzycie:
    python 03_trenowanie_modelu.py
    python 03_trenowanie_modelu.py --epoki 20 --batch-size 32

Wynik:
    models/model_twarzy.pt   - wagi wytrenowanego modelu
    models/klasy.json        - mapowanie indeks -> nazwa osoby
    models/krzywe_uczenia.png - wykres accuracy/loss w czasie
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import torch
from torch import nn, optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

from common.model import IMAGE_SIZE, IMAGENET_MEAN, IMAGENET_STD, build_model, get_device, unfreeze_backbone

PROCESSED_DIR = Path(__file__).parent / "data" / "processed"
MODELS_DIR = Path(__file__).parent / "models"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Trening modelu rozpoznawania twarzy.")
    parser.add_argument("--epoki", type=int, default=15, help="Liczba epok treningu klasyfikatora")
    parser.add_argument(
        "--epoki-finetuning", type=int, default=8, help="Liczba epok fine-tuningu backbone'u"
    )
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=1e-3, help="Learning rate klasyfikatora")
    parser.add_argument("--lr-finetuning", type=float, default=1e-4, help="Learning rate fine-tuningu")
    return parser.parse_args()


def build_dataloaders(batch_size: int) -> tuple[DataLoader, DataLoader, list[str]]:
    # Augmentacje tylko dla zbioru treningowego - sztucznie "powiekszaja" dane,
    # ucza siec ignorowac drobne odchylenia w oswietleniu/kadrowaniu/obrocie.
    train_transform = transforms.Compose(
        [
            transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomRotation(10),
            transforms.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.2),
            transforms.ToTensor(),
            transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
        ]
    )
    val_transform = transforms.Compose(
        [
            transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
            transforms.ToTensor(),
            transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
        ]
    )

    train_ds = datasets.ImageFolder(PROCESSED_DIR / "train", transform=train_transform)
    val_ds = datasets.ImageFolder(PROCESSED_DIR / "val", transform=val_transform)

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=2)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=2)

    return train_loader, val_loader, train_ds.classes


def run_epoch(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
    criterion: nn.Module,
    optimizer: optim.Optimizer | None,
) -> tuple[float, float]:
    is_train = optimizer is not None
    model.train(is_train)

    total_loss, correct, total = 0.0, 0, 0
    with torch.set_grad_enabled(is_train):
        for images, labels in loader:
            images, labels = images.to(device), labels.to(device)

            outputs = model(images)
            loss = criterion(outputs, labels)

            if is_train:
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

            total_loss += loss.item() * images.size(0)
            correct += (outputs.argmax(1) == labels).sum().item()
            total += images.size(0)

    return total_loss / total, correct / total


def main() -> None:
    args = parse_args()
    MODELS_DIR.mkdir(parents=True, exist_ok=True)

    if not (PROCESSED_DIR / "train").exists():
        raise SystemExit(
            f"Brak danych w {PROCESSED_DIR}. Najpierw uruchom 02_podzial_danych.py."
        )

    device = get_device()
    train_loader, val_loader, classes = build_dataloaders(args.batch_size)
    print(f"[info] Klasy ({len(classes)}): {classes}")

    model = build_model(num_classes=len(classes), pretrained=True, freeze_backbone=True)
    model.to(device)

    criterion = nn.CrossEntropyLoss()

    history = {"train_loss": [], "train_acc": [], "val_loss": [], "val_acc": []}
    best_val_acc = 0.0

    # --- Etap 1: trening samego klasyfikatora (backbone zamrozony) ---
    optimizer = optim.Adam(
        (p for p in model.parameters() if p.requires_grad), lr=args.lr
    )
    print("\n[etap 1/2] Trening klasyfikatora (backbone zamrozony)")
    for epoch in range(1, args.epoki + 1):
        t0 = time.time()
        train_loss, train_acc = run_epoch(model, train_loader, device, criterion, optimizer)
        val_loss, val_acc = run_epoch(model, val_loader, device, criterion, None)
        dt = time.time() - t0

        history["train_loss"].append(train_loss)
        history["train_acc"].append(train_acc)
        history["val_loss"].append(val_loss)
        history["val_acc"].append(val_acc)

        print(
            f"epoka {epoch:02d}/{args.epoki} | "
            f"train_loss={train_loss:.3f} train_acc={train_acc:.3f} | "
            f"val_loss={val_loss:.3f} val_acc={val_acc:.3f} | {dt:.1f}s"
        )

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(model.state_dict(), MODELS_DIR / "model_twarzy.pt")

    # --- Etap 2: fine-tuning ostatnich blokow backbone'u ---
    if args.epoki_finetuning > 0:
        print("\n[etap 2/2] Fine-tuning ostatnich warstw backbone'u")
        unfreeze_backbone(model, last_n_blocks=3)
        optimizer = optim.Adam(
            (p for p in model.parameters() if p.requires_grad), lr=args.lr_finetuning
        )

        for epoch in range(1, args.epoki_finetuning + 1):
            t0 = time.time()
            train_loss, train_acc = run_epoch(model, train_loader, device, criterion, optimizer)
            val_loss, val_acc = run_epoch(model, val_loader, device, criterion, None)
            dt = time.time() - t0

            history["train_loss"].append(train_loss)
            history["train_acc"].append(train_acc)
            history["val_loss"].append(val_loss)
            history["val_acc"].append(val_acc)

            print(
                f"epoka ft {epoch:02d}/{args.epoki_finetuning} | "
                f"train_loss={train_loss:.3f} train_acc={train_acc:.3f} | "
                f"val_loss={val_loss:.3f} val_acc={val_acc:.3f} | {dt:.1f}s"
            )

            if val_acc > best_val_acc:
                best_val_acc = val_acc
                torch.save(model.state_dict(), MODELS_DIR / "model_twarzy.pt")

    with open(MODELS_DIR / "klasy.json", "w", encoding="utf-8") as f:
        json.dump(classes, f, ensure_ascii=False, indent=2)

    print(f"\n[gotowe] Najlepsza dokladnosc walidacyjna: {best_val_acc:.3f}")
    print(f"[gotowe] Model zapisany w: {MODELS_DIR / 'model_twarzy.pt'}")

    try:
        import matplotlib.pyplot as plt

        epochs_range = range(1, len(history["train_acc"]) + 1)
        fig, axes = plt.subplots(1, 2, figsize=(10, 4))

        axes[0].plot(epochs_range, history["train_loss"], label="train")
        axes[0].plot(epochs_range, history["val_loss"], label="val")
        axes[0].set_title("Loss")
        axes[0].set_xlabel("epoka")
        axes[0].legend()

        axes[1].plot(epochs_range, history["train_acc"], label="train")
        axes[1].plot(epochs_range, history["val_acc"], label="val")
        axes[1].set_title("Accuracy")
        axes[1].set_xlabel("epoka")
        axes[1].legend()

        fig.tight_layout()
        fig.savefig(MODELS_DIR / "krzywe_uczenia.png")
        print(f"[gotowe] Wykres krzywych uczenia: {MODELS_DIR / 'krzywe_uczenia.png'}")
    except ImportError:
        print("[uwaga] matplotlib niedostepny - pomijam wykres krzywych uczenia.")


if __name__ == "__main__":
    main()
