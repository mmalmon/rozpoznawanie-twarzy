"""
Krok 4: Rozpoznawanie twarzy na zywo z kamery RoWave RC16.

Wymaga wczesniejszego wytrenowania modelu (03_trenowanie_modelu.py).

Uzycie:
    python 04_rozpoznawanie_na_zywo.py
    python 04_rozpoznawanie_na_zywo.py --prog-pewnosci 0.75

Jesli pewnosc modelu dla najlepszej klasy jest ponizej progu (domyslnie 0.6),
osoba oznaczana jest jako "Nieznana osoba" - to bardzo wazny mechanizm:
siec zawsze zwroci "najbardziej prawdopodobna" klase spomiedzy tych, ktore
widziala w treningu, nawet jesli w kadrze jest ktos zupelnie inny. Prog
pewnosci pozwala odrzucic takie niepewne dopasowania.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import cv2
import torch
import torch.nn.functional as F
from torchvision import transforms

from common.camera import choose_camera_interactive, open_camera
from common.face_detector import FaceDetector
from common.model import IMAGE_SIZE, IMAGENET_MEAN, IMAGENET_STD, build_model, get_device

MODELS_DIR = Path(__file__).parent / "models"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Rozpoznawanie twarzy na zywo.")
    parser.add_argument(
        "--kamera",
        type=int,
        default=None,
        help="Indeks kamery. Jesli pominiety, program zapyta interaktywnie.",
    )
    parser.add_argument("--szerokosc", type=int, default=1920)
    parser.add_argument("--wysokosc", type=int, default=1080)
    parser.add_argument(
        "--prog-pewnosci",
        type=float,
        default=0.6,
        help="Minimalna pewnosc (0-1), ponizej ktorej osoba jest 'Nieznana'",
    )
    return parser.parse_args()


def load_model(device: torch.device) -> tuple[torch.nn.Module, list[str]]:
    classes_path = MODELS_DIR / "klasy.json"
    weights_path = MODELS_DIR / "model_twarzy.pt"

    if not classes_path.exists() or not weights_path.exists():
        raise SystemExit(
            "Brak wytrenowanego modelu. Najpierw uruchom 03_trenowanie_modelu.py."
        )

    with open(classes_path, "r", encoding="utf-8") as f:
        classes = json.load(f)

    model = build_model(num_classes=len(classes), pretrained=False)
    model.load_state_dict(torch.load(weights_path, map_location=device))
    model.to(device)
    model.eval()

    return model, classes


def main() -> None:
    args = parse_args()
    device = get_device()

    model, classes = load_model(device)
    detector = FaceDetector()

    preprocess = transforms.Compose(
        [
            transforms.ToPILImage(),
            transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
            transforms.ToTensor(),
            transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
        ]
    )

    cap = open_camera(
        args.kamera if args.kamera is not None else choose_camera_interactive(),
        width=args.szerokosc,
        height=args.wysokosc,
    )

    print("[info] Nacisnij 'q' lub ESC, aby zakonczyc.")
    prev_time = time.time()

    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                print("[blad] Nie udalo sie odczytac klatki z kamery.")
                break

            faces = detector.detect(frame)

            for face in faces:
                crop = face.crop(frame, margin=0.2)
                if crop.size == 0:
                    continue

                crop_rgb = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)
                tensor = preprocess(crop_rgb).unsqueeze(0).to(device)

                with torch.no_grad():
                    logits = model(tensor)
                    probs = F.softmax(logits, dim=1)[0]
                    confidence, pred_idx = torch.max(probs, dim=0)

                confidence = confidence.item()
                if confidence >= args.prog_pewnosci:
                    label = f"{classes[pred_idx.item()]} ({confidence * 100:.0f}%)"
                    color = (0, 200, 0)
                else:
                    label = f"Nieznana osoba ({confidence * 100:.0f}%)"
                    color = (0, 0, 220)

                cv2.rectangle(
                    frame, (face.x, face.y), (face.x + face.w, face.y + face.h), color, 2
                )
                cv2.putText(
                    frame,
                    label,
                    (face.x, max(0, face.y - 10)),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.7,
                    color,
                    2,
                )

            now = time.time()
            fps = 1.0 / max(1e-6, now - prev_time)
            prev_time = now
            cv2.putText(
                frame,
                f"FPS: {fps:.1f}",
                (20, 40),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                (255, 255, 0),
                2,
            )

            cv2.imshow("Rozpoznawanie twarzy - RoWave RC16 (q=koniec)", frame)

            key = cv2.waitKey(1) & 0xFF
            if key in (ord("q"), 27):
                break
    finally:
        cap.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
