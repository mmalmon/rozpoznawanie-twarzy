"""
Bezpieczny odczyt/zapis obrazow, odporny na polskie znaki w sciezkach.

DLA UCZNIOW - kontekst problemu: biblioteka OpenCV na Windows przy
operacjach cv2.imread/cv2.imwrite korzysta wewnetrznie z funkcji systemowych,
ktore przy sciezkach zawierajacych znaki spoza ASCII (np. polskie "l", "o" -
jak w folderze uzytkownika typu "OneDrive - Zespol Szkol") czesto zawodza
(po cichu zwracaja False / None, bez rzucenia wyjatku!). To bardzo typowy,
podchwytliwy blad przy pracy z OpenCV na polskich komputerach.

Ponizsze funkcje omijaja ten problem w dwoch krokach:
  1. cv2.imencode/cv2.imdecode - koduja/dekoduja obraz w pamieci (bez
     dotykania dysku), wiec nie maja tego problemu ze sciezkami.
  2. numpy.ndarray.tofile / np.fromfile - to juz zwykly, w pelni obslugujacy
     Unicode odczyt/zapis pliku przez Pythona (a nie przez OpenCV).
"""

from __future__ import annotations

# Path - obiektowa reprezentacja sciezki pliku (patrz wyjasnienie w
# common/face_detector.py).
from pathlib import Path

# cv2 - biblioteka OpenCV, tutaj uzywana do kodowania/dekodowania obrazow
# (imencode/imdecode dzialaja na danych w pamieci, wiec omijaja blad z
# polskimi znakami w sciezce, opisany w docstringu modulu ponizej).
import cv2
# numpy - biblioteka do tablic liczbowych; jej funkcje tofile()/fromfile()
# to zwykly, w pelni obslugujacy Unicode zapis/odczyt pliku przez Pythona
# (w odroznieniu od zawodnych funkcji cv2.imwrite/cv2.imread na Windows).
import numpy as np


def zapisz_obraz(sciezka: str | Path, obraz: np.ndarray) -> bool:
    """Odpowiednik cv2.imwrite(), ktory poprawnie dziala rowniez ze
    sciezkami zawierajacymi polskie znaki. Zwraca True, jesli zapis sie
    powiodl."""
    sciezka = Path(sciezka)
    # cv2.imencode koduje obraz (tablice numpy) do formatu pliku (np. .jpg)
    # w PAMIECI, bez zapisywania niczego na dysk - zwraca dwie wartosci:
    # flage powodzenia oraz same zakodowane bajty obrazu.
    zakodowano, dane = cv2.imencode(sciezka.suffix or ".jpg", obraz)
    if not zakodowano:
        return False
    # dane.tofile(...) to metoda numpy zapisujaca surowe bajty do pliku pod
    # wskazana sciezka, korzystajac z wewnetrznych mechanizmow Pythona
    # (ktore poprawnie obsluguja Unicode) zamiast mechanizmow OpenCV.
    dane.tofile(str(sciezka))
    return True


def wczytaj_obraz(sciezka: str | Path, flagi: int = cv2.IMREAD_COLOR) -> np.ndarray | None:
    """Odpowiednik cv2.imread(), ktory poprawnie dziala rowniez ze sciezkami
    zawierajacymi polskie znaki. Zwraca None, jesli plik nie istnieje lub
    nie da sie go zdekodowac jako obrazu."""
    sciezka = Path(sciezka)
    if not sciezka.exists():
        return None
    # np.fromfile czyta surowe bajty pliku (rowniez poprawnie z Unicode w
    # sciezce) i zamienia je na jednowymiarowa tablice liczb typu uint8
    # (bajtow bez znaku, wartosci 0-255) - dokladnie takich, z jakich
    # sklada sie kazdy plik graficzny na dysku.
    dane = np.fromfile(str(sciezka), dtype=np.uint8)
    # cv2.imdecode zamienia te surowe bajty z powrotem w normalny obraz
    # (tablice numpy [wysokosc, szerokosc, kanaly]), tak jakby zostal on
    # wczytany funkcja cv2.imread - ale dziejac sie w calosci w pamieci.
    return cv2.imdecode(dane, flagi)
