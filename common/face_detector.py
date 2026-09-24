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

    def __init__(
        self,
        wspolczynnik_skali: float = 1.1,
        min_sasiadow: int = 9,
        katy_awaryjne: tuple[int, ...] = (-20, -35, -50, 20, 35, 50),
        min_sasiadow_awaryjny: int = 6,
        min_rozmiar_wzgledny: float = 0.15,
    ):
        """
        wspolczynnik_skali: o ile procent zmniejszane jest okno przeszukiwania
            miedzy kolejnymi probami (mniejsza wartosc = dokladniej, ale
            wolniej). Wartosc 1.1 oznacza zmniejszanie o 10% za kazdym razem.
        min_sasiadow: ile razy dany obszar musi zostac zakwalifikowany jako
            twarz w roznych skalach/przesunieciach, zeby uznac go za
            "prawdziwe" wykrycie, a nie przypadkowy szum. Wieksza wartosc
            = mniej falszywych wykryc, ale mozna przeoczyc twarze pod katem.
        katy_awaryjne: kaskada Haara wykrywa dobrze tylko twarze "pionowe"
            (odchylone od pionu o najwyzej kilka stopni) - przechylenie
            glowy w bok (np. oparcie jej na ramieniu) czesto sprawia, ze
            wykrywanie "gubi" twarz. Jesli detekcja na wprost nie znajdzie
            zadnej twarzy, probujemy dodatkowo obrocic caly obraz o kazdy
            z podanych tu katow (w stopniach) i szukac ponownie - dzieki
            temu przechylona twarz staje sie w obroconym obrazie "pionowa"
            i kaskada moze ja wykryc. Kolejnosc katow ma znaczenie - probujemy
            najpierw mniejszych przechylen (najczestsze w praktyce), a dopiero
            potem wiekszych. Krotka pusta () wylacza ta probe awaryjna
            (szybsze dzialanie, ale bez odpornosci na przechylenie).
        min_sasiadow_awaryjny: prog pewnosci uzywany WYLACZNIE przy probach
            na obroconych kopiach obrazu. Jest celowo nizszy niz
            min_sasiadow, poniewaz obrot (zwlaszcza o wieksze katy) lekko
            znieksztalca obraz (interpolacja pikseli) i naturalna kaskada
            Haara "widzi" wtedy nieco mniej pewnych dopasowan nawet dla
            prawdziwej twarzy - obnizamy wiec prog, zeby jej nie przeoczyc,
            akceptujac w zamian odrobine wiecej falszywych alarmow.
        min_rozmiar_wzgledny: minimalny rozmiar wykrywanego prostokata,
            wyrazony jako uamek KROTSZEGO boku klatki (np. 0.15 = 15%
            szerokosci/wysokosci obrazu). DLA UCZNIOW: bez tego ograniczenia
            kaskada Haara potrafi bledie uznac za "twarz" nawet mala plamke
            (np. sam fragment czola czy dloni) - takie male wykrycia rzadko
            odpowiadaja calej, prawdziwej twarzy w typowej scenie z jedna
            osoba blisko kamery. Wymuszajac wiekszy minimalny rozmiar
            (wzgledny do rozdzielczosci, a nie sztywna liczbe pikseli - bo
            klatka bywa raz w Full HD, raz w 4K) eliminujemy wiekszosc takich
            falszywych, czesciowych wykryc.
        """
        self.kaskada = _wczytaj_kaskade_bezpiecznie("haarcascade_frontalface_default.xml")
        self.wspolczynnik_skali = wspolczynnik_skali
        self.min_sasiadow = min_sasiadow
        self.katy_awaryjne = katy_awaryjne
        self.min_sasiadow_awaryjny = min_sasiadow_awaryjny
        self.min_rozmiar_wzgledny = min_rozmiar_wzgledny

    def _wykryj_na_obrazie(self, obraz_szary: np.ndarray, min_sasiadow: int) -> np.ndarray:
        """Uruchamia surowa detekcje kaskady Haara na juz przygotowanym
        (jednokanalowym) obrazie i zwraca tablice wykryc w formacie OpenCV
        - lista czwórek (x, y, w, h). Wydzielone do osobnej metody, zeby nie
        powtarzac tych samych parametrow w kazdym miejscu, gdzie wywolujemy
        detekcje (obraz na wprost oraz kazda z obroconych kopii). Przyjmuje
        min_sasiadow jako parametr (a nie od razu self.min_sasiadow), bo
        proby awaryjne na obroconych kopiach obrazu uzywaja innego,
        nizszego progu - patrz min_sasiadow_awaryjny w __init__."""
        # Minimalny rozmiar wykrycia liczymy WZGLEDEM rozdzielczosci samej
        # klatki (a nie na sztywno w pikselach) - dzieki temu dziala
        # spojnie zarowno przy podgladzie 720p, jak i przy pelnym 4K z
        # kamery RoWave RC16.
        krotszy_bok = min(obraz_szary.shape[:2])
        min_rozmiar_px = max(60, int(krotszy_bok * self.min_rozmiar_wzgledny))
        return self.kaskada.detectMultiScale(
            obraz_szary,
            scaleFactor=self.wspolczynnik_skali,
            minNeighbors=min_sasiadow,
            minSize=(min_rozmiar_px, min_rozmiar_px),
        )

    def wykryj(self, klatka: np.ndarray) -> list[WykrytaTwarz]:
        """Zwraca liste wszystkich twarzy wykrytych na podanej klatce (obrazie
        BGR, tak jak dostarcza go OpenCV z kamery)."""
        # Kaskada Haara dziala na obrazach w skali szarosci - kolor nie jest
        # jej potrzebny, a jego pominiecie przyspiesza obliczenia.
        szarosc = cv2.cvtColor(klatka, cv2.COLOR_BGR2GRAY)
        # Wyrownanie histogramu poprawia kontrast, co pomaga w slabym/nierownym
        # oswietleniu (typowa sytuacja w sali lekcyjnej).
        szarosc = cv2.equalizeHist(szarosc)

        wykrycia = self._wykryj_na_obrazie(szarosc, self.min_sasiadow)
        if len(wykrycia) > 0:
            return [WykrytaTwarz(x, y, w, h) for (x, y, w, h) in wykrycia]

        # Nic nie znaleziono na wprost - probujemy awaryjnie po kolei kazdy
        # z zadanych katow obrotu (np. przechylona w bok glowa). Zatrzymujemy
        # sie na pierwszym kacie, ktory cokolwiek wykryje - to wystarczajaco
        # dobre przyblizenie dla naszych celow (nie musimy wiedziec ILE
        # dokladnie stopni przechylenia ma glowa, tylko znalezc jej polozenie).
        wysokosc, szerokosc = szarosc.shape[:2]
        srodek_obrazu = (szerokosc / 2, wysokosc / 2)
        for kat in self.katy_awaryjne:
            macierz_obrotu = cv2.getRotationMatrix2D(srodek_obrazu, kat, 1.0)
            obrocony_obraz = cv2.warpAffine(szarosc, macierz_obrotu, (szerokosc, wysokosc))
            wykrycia_obrocone = self._wykryj_na_obrazie(obrocony_obraz, self.min_sasiadow_awaryjny)
            if len(wykrycia_obrocone) == 0:
                continue

            # Znaleziono twarz na obroconej kopii - trzeba przeliczyc jej
            # wspolrzedne z powrotem na uklad oryginalnej (nieobroconej)
            # klatki, zeby ramka na podgladzie z kamery byla we wlasciwym
            # miejscu. Uzywamy do tego macierzy ODWROTNEJ do tej, ktora
            # obrocila obraz.
            macierz_odwrotna = cv2.invertAffineTransform(macierz_obrotu)
            wyniki = []
            for (x, y, w, h) in wykrycia_obrocone:
                # UPROSZCZENIE: przeksztalcamy tylko srodek prostokata z
                # powrotem do ukladu oryginalnego obrazu, a jego szerokosc
                # i wysokosc zostawiamy bez zmian. Dokladne przeksztalcenie
                # calego prostokata zamienilo by go w rownolegobok (bo obrot
                # nie zachowuje prostych katow) - a to niepotrzebnie
                # komplikowaloby dalsze wycinanie/rysowanie ramki.
                srodek_x = x + w / 2
                srodek_y = y + h / 2
                nowy_srodek = macierz_odwrotna @ np.array([srodek_x, srodek_y, 1.0])
                nowy_x = int(nowy_srodek[0] - w / 2)
                nowy_y = int(nowy_srodek[1] - h / 2)
                # Przycinamy wspolrzedne do granic obrazu, na wypadek gdyby
                # przeliczony prostokat wystawal lekko poza krawedz klatki.
                nowy_x = max(0, min(nowy_x, szerokosc - w))
                nowy_y = max(0, min(nowy_y, wysokosc - h))
                wyniki.append(WykrytaTwarz(nowy_x, nowy_y, w, h))
            return wyniki

        return []

