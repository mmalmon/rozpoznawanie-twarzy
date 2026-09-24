"""
Prosty detektor twarzy oparty o kaskade Haara z OpenCV.

DLA UCZNIOW - jak dziala kaskada Haara (algorytm Violi-Jonesa, rok 2001):
  1. Obraz przeszukiwany jest "oknem" o rosnacym rozmiarze, przesuwanym po
     calej klatce (tzw. sliding window).
  2. Kazde okno jest oceniane przez ciag prostych filtrow ("cechy Haara") -
     porownuja one sume jasnosci pikseli w sasiadujacych prostokatach
     (np. "okolice oczu sa ciemniejsze niz okolice policzkow").
  3. Filtry ulozone sa w kaskade: pierwsze, bardzo szybkie filtry od razu
     odrzucaja wiekszosc okien, ktore na pewno NIE sa twarza. Tylko nieliczne
     okna przechodza przez wszystkie etapy kaskady i zostaja uznane za twarz.
  4. Dzieki temu kaskada jest bardzo szybka (dziala plynnie na CPU), kosztem
     nieco nizszej skutecznosci niz nowoczesne detektory oparte o sieci
     neuronowe (np. MediaPipe Face Detector, RetinaFace, MTCNN).

Wybralismy kaskade Haara do tego projektu edukacyjnego, poniewaz:
  - jest wbudowana w opencv-python (zero dodatkowych pobran modeli),
  - dziala szybko na CPU (GPU zostawiamy dla sieci rozpoznajacej OSOBE),
  - jej dzialanie da sie prosto narysowac na tablicy, w przeciwienstwie do
    "czarnej skrzynki" sieci neuronowej.

Pomysl na zadanie dodatkowe dla uczniow: podmienic ten detektor na
nowoczesniejszy (np. z biblioteki MediaPipe) i porownac szybkosc/skutecznosc.
"""

from __future__ import annotations

# atexit pozwala zarejestrowac funkcje, ktora ma sie wykonac automatycznie
# tuz przed zakonczeniem programu (uzywamy tego do posprzatania plikow
# tymczasowych, patrz funkcja _wczytaj_kaskade_bezpiecznie ponizej).
import atexit
# shutil (od "shell utilities") dostarcza operacje na plikach/folderach
# wyzszego poziomu niz podstawowy Python, np. kopiowanie plikow (copyfile)
# czy usuwanie calych folderow razem z zawartoscia (rmtree).
import shutil
# tempfile tworzy tymczasowe pliki/foldery zarzadzane przez system
# operacyjny - idealne na potrzeby chwilowej kopii pliku kaskady.
import tempfile
# dataclass - patrz wyjasnienie w common/camera.py; automatycznie generuje
# konstruktor i inne metody dla prostej klasy przechowujacej dane.
from dataclasses import dataclass
# Path to obiektowa reprezentacja sciezki do pliku/folderu z modulu
# pathlib - wygodniejsza i bezpieczniejsza w uzyciu niz skladanie sciezek
# recznie jako zwykle napisy (stringi).
from pathlib import Path

# cv2 - biblioteka OpenCV, tutaj uzywana do wczytania kaskady Haara i
# przetworzenia obrazu (konwersja na szarosc, wyrownanie histogramu).
import cv2
# numpy (importowana pod skrocona nazwa "np" - to powszechna konwencja w
# calym swiecie Pythona) to biblioteka do szybkich obliczen na tablicach
# liczbowych. Kazda klatka obrazu z kamery jest w Pythonie reprezentowana
# jako tablica numpy (np.ndarray) - trojwymiarowa: [wysokosc, szerokosc, kanaly_koloru].
import numpy as np


@dataclass
class WykrytaTwarz:
    """Prostokat (bounding box) wokol jednej wykrytej twarzy na obrazie.

    Wspolrzedne x, y to lewy-gorny rog prostokata, a w (szerokosc) i
    h (wysokosc) to jego rozmiary - dokladnie tak, jak zwraca je OpenCV.
    """

    x: int
    y: int
    w: int
    h: int

    def wytnij(self, klatka: np.ndarray, margines: float = 0.2) -> np.ndarray:
        """Wycina fragment klatki odpowiadajacy tej twarzy, z dodatkowym
        marginesem wokol niej (domyslnie 20% szerokosci/wysokosci twarzy
        z kazdej strony).

        Margines jest wazny: sama twarz wykryta przez kaskade Haara bywa
        "ciasno" dopasowana (czasem ucina fragment brody albo czola) - kilka
        dodatkowych pikseli tla sprawia, ze siec klasyfikujaca dostaje
        troche wiecej kontekstu.
        """
        wysokosc_klatki, szerokosc_klatki = klatka.shape[:2]
        margines_x = int(self.w * margines)
        margines_y = int(self.h * margines)

        x1 = max(0, self.x - margines_x)
        y1 = max(0, self.y - margines_y)
        x2 = min(szerokosc_klatki, self.x + self.w + margines_x)
        y2 = min(wysokosc_klatki, self.y + self.h + margines_y)

        return klatka[y1:y2, x1:x2]


def _wczytaj_kaskade_bezpiecznie(nazwa_pliku: str) -> cv2.CascadeClassifier:
    """Wczytuje kaskade Haara w sposob odporny na polskie znaki w sciezce.

    UWAGA TECHNICZNA: klasa cv2.CascadeClassifier na Windows potrafi nie
    wczytac pliku, jesli jego sciezka zawiera znaki spoza ASCII (np. gdy
    projekt lezy w folderze typu "OneDrive - Zespol Szkol..."). Dlatego w
    razie niepowodzenia kopiujemy plik XML do katalogu tymczasowego systemu
    (jego sciezka jest zazwyczaj czysto ASCII) i wczytujemy kaskade juz
    stamtad.
    """
    sciezka_zrodlowa = Path(cv2.data.haarcascades) / nazwa_pliku
    # cv2.data.haarcascades to sciezka do folderu WEWNATRZ zainstalowanego
    # pakietu opencv-python, w ktorym producent OpenCV dolacza gotowe pliki
    # kaskad Haara (wytrenowane juz przez tworcow biblioteki) - nie musimy
    # ich sami trenowac ani pobierac osobno z internetu.

    kaskada = cv2.CascadeClassifier(str(sciezka_zrodlowa))
    if not kaskada.empty():
        return kaskada

    katalog_tymczasowy = Path(tempfile.mkdtemp(prefix="haarcascade_"))
    sciezka_tymczasowa = katalog_tymczasowy / nazwa_pliku
    shutil.copyfile(sciezka_zrodlowa, sciezka_tymczasowa)
    # Sprzatamy po sobie katalog tymczasowy przy zamknieciu programu.
    atexit.register(shutil.rmtree, katalog_tymczasowy, True)

    kaskada = cv2.CascadeClassifier(str(sciezka_tymczasowa))
    if kaskada.empty():
        raise RuntimeError(
            f"Nie udalo sie wczytac kaskady Haara ani z {sciezka_zrodlowa}, "
            f"ani z kopii {sciezka_tymczasowa}"
        )
    return kaskada


class DetektorTwarzy:
    """Wykrywa polozenie twarzy na obrazie z kamery (nie rozpoznaje, KTO to
    jest - tym zajmuje sie osobny model z pliku common/model.py)."""

    def __init__(self, wspolczynnik_skali: float = 1.1, min_sasiadow: int = 6):
        """
        wspolczynnik_skali: o ile procent zmniejszane jest okno przeszukiwania
            miedzy kolejnymi probami (mniejsza wartosc = dokladniej, ale
            wolniej). Wartosc 1.1 oznacza zmniejszanie o 10% za kazdym razem.
        min_sasiadow: ile razy dany obszar musi zostac zakwalifikowany jako
            twarz w roznych skalach/przesunieciach, zeby uznac go za
            "prawdziwe" wykrycie, a nie przypadkowy szum. Wieksza wartosc
            = mniej falszywych wykryc, ale mozna przeoczyc twarze pod katem.
        """
        self.kaskada = _wczytaj_kaskade_bezpiecznie("haarcascade_frontalface_default.xml")
        self.wspolczynnik_skali = wspolczynnik_skali
        self.min_sasiadow = min_sasiadow

    def wykryj(self, klatka: np.ndarray) -> list[WykrytaTwarz]:
        """Zwraca liste wszystkich twarzy wykrytych na podanej klatce (obrazie
        BGR, tak jak dostarcza go OpenCV z kamery)."""
        # Kaskada Haara dziala na obrazach w skali szarosci - kolor nie jest
        # jej potrzebny, a jego pominiecie przyspiesza obliczenia.
        szarosc = cv2.cvtColor(klatka, cv2.COLOR_BGR2GRAY)
        # Wyrownanie histogramu poprawia kontrast, co pomaga w slabym/nierownym
        # oswietleniu (typowa sytuacja w sali lekcyjnej).
        szarosc = cv2.equalizeHist(szarosc)

        wykrycia = self.kaskada.detectMultiScale(
            szarosc,
            scaleFactor=self.wspolczynnik_skali,
            minNeighbors=self.min_sasiadow,
            minSize=(80, 80),  # ignorujemy bardzo male wykrycia (dalekie twarze / szum)
        )

        return [WykrytaTwarz(x, y, w, h) for (x, y, w, h) in wykrycia]

