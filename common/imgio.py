"""
Bezpieczny odczyt/zapis obrazow, odporny na polskie znaki w sciezkach.

DLA UCZNIOW - kontekst problemu: biblioteka OpenCV na Windows przy
operacjach cv2.imread/cv2.imwrite korzysta wewnetrznie z funkcji systemowych,
ktore przy sciezkach zawierajacych znaki spoza ASCII (np. polskie "l", "o" -
jak w folderze uzytkownika typu "OneDrive - Zespol Szkol") czesto zawodza
(po cichu zwracaja False / None, bez rzucenia wyjatku!). To bardzo typowy,
podchwytliwy blad przy pracy z OpenCV na polskich komputerach.

Ponizsze funkcje omijaja ten problem w dwoch krokach:
  1. cv2.imencode/cv2.imdecode - kodują/dekoduja obraz w pamieci (bez
     dotykania dysku), wiec nie maja tego problemu ze sciezkami.
  2. numpy.ndarray.tofile / np.fromfile - to juz zwykly, w pelni obslugujacy
     Unicode odczyt/zapis pliku przez Pythona (a nie przez OpenCV).
"""

from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np


def zapisz_obraz(sciezka: str | Path, obraz: np.ndarray) -> bool:
    """Odpowiednik cv2.imwrite(), ktory poprawnie dziala rowniez ze
    sciezkami zawierajacymi polskie znaki. Zwraca True, jesli zapis sie
    powiodl."""
    sciezka = Path(sciezka)
    zakodowano, dane = cv2.imencode(sciezka.suffix or ".jpg", obraz)
    if not zakodowano:
        return False
    dane.tofile(str(sciezka))
    return True


def wczytaj_obraz(sciezka: str | Path, flagi: int = cv2.IMREAD_COLOR) -> np.ndarray | None:
    """Odpowiednik cv2.imread(), ktory poprawnie dziala rowniez ze sciezkami
    zawierajacymi polskie znaki. Zwraca None, jesli plik nie istnieje lub
    nie da sie go zdekodowac jako obrazu."""
    sciezka = Path(sciezka)
    if not sciezka.exists():
        return None
    dane = np.fromfile(str(sciezka), dtype=np.uint8)
    return cv2.imdecode(dane, flagi)
