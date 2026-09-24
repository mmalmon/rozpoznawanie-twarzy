"""
Krok 1: Zbieranie materialu treningowego z kamery RoWave RC16.

Uzycie:
    python 01_zbieranie_danych.py --osoba jan_kowalski
    python 01_zbieranie_danych.py --osoba jan_kowalski --kamera 1 --cel 250

Jesli nie podasz --kamera, program wykryje dostepne kamery i zapyta,
ktorej uzyc (przydatne, gdy laptop ma zarowno kamere wbudowana, jak i
RoWave RC16).

Sterowanie w oknie podgladu:
    SPACJA / 's' - zapisz aktualnie wykryta twarz jako zdjecie treningowe
    'a'          - wlacz/wylacz automatyczne zapisywanie (co ~0.3 s, gdy wykryto twarz)
    'q' / ESC    - zakoncz

Zdjecia zapisywane sa jako wyciete i przeskalowane kwadraty twarzy (224x224)
w folderze data/raw/<osoba>/.

Wskazowki dla dobrej jakosci danych (patrz tez README.md):
  - zbierz min. 150-300 zdjec na osobe,
  - rozne kąty głowy (lekko w lewo/prawo/gore/dol), rozne wyrazy twarzy,
  - rozne oswietlenie (dzien/wieczor, swiatlo z roznych stron),
  - z okularami i bez, jesli osoba je czasem nosi,
  - staraj sie NIE zbierac serii identycznych, nieruchomych klatek - lepiej
    ruszac lekko glowa miedzy kolejnymi zdjeciami.
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import cv2

from common.camera import choose_camera_interactive, open_camera
from common.face_detector import FaceDetector
from common.imgio import imwrite_unicode

DATA_DIR = Path(__file__).parent / "data" / "raw"
TARGET_SIZE = (224, 224)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Zbieranie zdjec twarzy do treningu.")
    parser.add_argument("--osoba", required=True, help="Identyfikator osoby, np. jan_kowalski")
    parser.add_argument(
        "--kamera",
        type=int,
        default=None,
        help="Indeks kamery. Jesli pominiety, program zapyta interaktywnie.",
    )
    parser.add_argument("--cel", type=int, default=250, help="Docelowa liczba zdjec do zebrania")
    parser.add_argument(
        "--szerokosc", type=int, default=1920, help="Szerokosc podgladu z kamery (px)"
    )
    parser.add_argument(
        "--wysokosc", type=int, default=1080, help="Wysokosc podgladu z kamery (px)"
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    person_dir = DATA_DIR / args.osoba
    person_dir.mkdir(parents=True, exist_ok=True)
    existing = len(list(person_dir.glob("*.jpg")))

    kamera_index = args.kamera if args.kamera is not None else choose_camera_interactive()
    cap = open_camera(kamera_index, width=args.szerokosc, height=args.wysokosc)
    detector = FaceDetector()

    saved = existing
    auto_save = False
    last_auto_save = 0.0

    print(f"[info] Zapisuje do: {person_dir}")
    print(f"[info] Juz zebranych zdjec: {existing}. Cel: {args.cel}.")
    print("[info] SPACJA=zapisz  a=auto-zapis  q/ESC=koniec")

    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                print("[blad] Nie udalo sie odczytac klatki z kamery.")
                break

            faces = detector.detect(frame)
            preview = frame.copy()

            for face in faces:
                cv2.rectangle(
                    preview,
                    (face.x, face.y),
                    (face.x + face.w, face.y + face.h),
                    (0, 200, 0),
                    2,
                )

            status = f"Zebrano: {saved}/{args.cel}  auto={'ON' if auto_save else 'OFF'}"
            cv2.putText(
                preview, status, (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 200, 0), 2
            )
            cv2.imshow("Zbieranie danych - RoWave RC16 (q=koniec)", preview)

            key = cv2.waitKey(1) & 0xFF
            should_save = key in (32, ord("s"))

            if key == ord("a"):
                auto_save = not auto_save
            if auto_save and faces and (time.time() - last_auto_save) > 0.3:
                should_save = True

            if should_save and faces:
                # Bierzemy najwieksza wykryta twarz (najblizej kamery) na wypadek,
                # gdyby w kadrze bylo kilka osob.
                face = max(faces, key=lambda f: f.w * f.h)
                crop = face.crop(frame)
                crop = cv2.resize(crop, TARGET_SIZE)

                out_path = person_dir / f"{args.osoba}_{saved:04d}.jpg"
                imwrite_unicode(out_path, crop)
                saved += 1
                last_auto_save = time.time()
                print(f"[zapisano] {out_path.name} ({saved}/{args.cel})")

            if saved >= args.cel:
                print("[info] Osiagnieto docelowa liczbe zdjec.")
                break
            if key in (ord("q"), 27):
                break
    finally:
        cap.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
