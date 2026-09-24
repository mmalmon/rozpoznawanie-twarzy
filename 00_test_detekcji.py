"""
Krok 0 (test): sprawdzenie, czy kamera RoWave RC16 i detekcja twarzy dzialaja.

To NIE jest jeszcze rozpoznawanie osob - tylko wykrywanie "czy i gdzie na
obrazie jest jakas twarz". Kazda wykryta twarz oznaczana jest zolta ramka
BEZ imienia. Po wytrenowaniu modelu (03_trenowanie_modelu.py) uzyj
docelowego skryptu 04_rozpoznawanie_na_zywo.py, ktory:
  - rysuje ZIELONA ramke + imie, gdy osoba zostanie rozpoznana z wystarczajaca
    pewnoscia,
  - rysuje CZERWONA ramke + "Nieznana osoba", gdy pewnosc jest za niska albo
    osoby nie ma w bazie.

Uzycie:
    python 00_test_detekcji.py
    python 00_test_detekcji.py --kamera 0

Nacisnij 'q' lub ESC, aby zakonczyc.
"""

from __future__ import annotations

import argparse
import time

import cv2

from common.camera import choose_camera_interactive, open_camera
from common.face_detector import FaceDetector


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Test detekcji twarzy na zywo.")
    parser.add_argument(
        "--kamera",
        type=int,
        default=None,
        help="Indeks kamery. Jesli pominiety, program zapyta interaktywnie.",
    )
    parser.add_argument("--szerokosc", type=int, default=1920)
    parser.add_argument("--wysokosc", type=int, default=1080)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    kamera_index = args.kamera if args.kamera is not None else choose_camera_interactive()

    cap = open_camera(kamera_index, width=args.szerokosc, height=args.wysokosc)
    detector = FaceDetector()

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
                cv2.rectangle(
                    frame,
                    (face.x, face.y),
                    (face.x + face.w, face.y + face.h),
                    (0, 220, 220),  # zolty (BGR) - to tylko test detekcji, bez rozpoznawania
                    2,
                )

            now = time.time()
            fps = 1.0 / max(1e-6, now - prev_time)
            prev_time = now

            cv2.putText(
                frame,
                f"Wykryto twarzy: {len(faces)}  FPS: {fps:.1f}",
                (20, 40),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                (0, 220, 220),
                2,
            )

            cv2.imshow("Test detekcji twarzy - RoWave RC16 (q=koniec)", frame)

            key = cv2.waitKey(1) & 0xFF
            if key in (ord("q"), 27):
                break
    finally:
        cap.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
