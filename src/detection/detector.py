"""
Detekcja twarzy - cienka warstwa API nad detektorem projektu.

Użycie:
    from src.detection.detector import detect_face

Funkcja korzysta z tego samego detektora co reszta projektu
(common/face_detector.py): najpierw szybka sieć YuNet, a gdy modelu brak -
kaskada Haara. Dzięki temu GUI i skrypty 00-04 wykrywają twarze w identyczny
sposób i nie duplikują kodu detekcji.
"""

from __future__ import annotations

import sys
from pathlib import Path

# Ten plik może być uruchamiany także bezpośrednio (python
# src/detection/detector.py), dlatego sam dbamy o to, żeby katalog główny
# projektu był widoczny dla importów "common.*".
KORZEN_PROJEKTU = Path(__file__).resolve().parents[2]
if str(KORZEN_PROJEKTU) not in sys.path:
    sys.path.insert(0, str(KORZEN_PROJEKTU))

from common.uruchom_w_venv import przelacz_na_venv_jesli_trzeba

przelacz_na_venv_jesli_trzeba()

import cv2
import numpy as np

from common.camera import dopasuj_do_ekranu, otworz_kamere
from common.face_detector import DetektorTwarzy

# Detektory tworzymy raz i cache'ujemy - zwłaszcza YuNet wczytuje model z
# dysku, więc nie wolno robić tego przy każdej klatce (był to wcześniej
# poważny spadek wydajności).
_DETEKTORY: dict[float, DetektorTwarzy] = {}


def _pobierz_detektor(threshold: float) -> DetektorTwarzy:
    threshold = min(0.99, max(0.05, float(threshold)))
    if threshold not in _DETEKTORY:
        _DETEKTORY[threshold] = DetektorTwarzy(prog_yunet=threshold)
    return _DETEKTORY[threshold]


def detect_face(
    frame: np.ndarray, threshold: float = 0.6, aspect_ratio: float = 0.5
) -> list[dict]:
    """Wykrywa twarze na obrazie w formacie BGR.

    Args:
        frame: obraz w formacie BGR (np. klatka z kamery OpenCV).
        threshold: minimalna pewność detekcji (0-1). Wpływa na próg sieci
            YuNet; kaskada Haara nie zwraca oceny pewności, więc w trybie
            awaryjnym parametr jest ignorowany.
        aspect_ratio: minimalny stosunek szerokości do wysokości twarzy.
            Wartość 0.5 oznacza akceptację proporcji od 0.5 do 2.0.

    Returns:
        Lista słowników:
        {
            "bbox": [x, y, w, h],
            "confidence": pewność detekcji (0-1)
        }
    """
    if frame is None or not isinstance(frame, np.ndarray) or frame.size == 0:
        return []
    if frame.ndim != 3 or frame.shape[2] != 3:
        raise ValueError("Oczekiwano obrazu BGR z trzema kanałami koloru.")
    if aspect_ratio <= 0:
        raise ValueError("aspect_ratio musi być większy od 0.")

    detektor = _pobierz_detektor(threshold)
    wynik: list[dict] = []
    for twarz in detektor.wykryj(frame):
        if twarz.h <= 0:
            continue
        proporcja = twarz.w / twarz.h
        if aspect_ratio <= proporcja <= 1.0 / aspect_ratio:
            wynik.append(
                {
                    "bbox": [twarz.x, twarz.y, twarz.w, twarz.h],
                    "confidence": float(twarz.pewnosc),
                }
            )
    return wynik


def main() -> None:
    """Prosty test detekcji na żywo (do uruchomienia bez GUI)."""
    kamera = otworz_kamere(0, szerokosc=1280, wysokosc=720)
    nazwa_okna = "Detekcja twarzy (q=koniec)"
    cv2.namedWindow(nazwa_okna, cv2.WINDOW_NORMAL)
    try:
        while True:
            czy_odczytano, klatka = kamera.read()
            if not czy_odczytano:
                print("[błąd] Nie udało się odczytać klatki z kamery.")
                break

            klatka = cv2.flip(klatka, 1)
            for twarz in detect_face(klatka, threshold=0.6, aspect_ratio=0.5):
                x, y, w, h = twarz["bbox"]
                cv2.rectangle(klatka, (x, y), (x + w, y + h), (0, 255, 0), 2)

            cv2.imshow(nazwa_okna, dopasuj_do_ekranu(klatka))
            if cv2.waitKey(1) & 0xFF in (ord("q"), 27):
                break
    finally:
        kamera.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
