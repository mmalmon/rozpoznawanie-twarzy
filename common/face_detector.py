"""
Detektor twarzy: szybka siec YuNet (OpenCV DNN) z awaryjna kaskada Haara.

DLA UCZNIOW - dlaczego dwa detektory?

Podstawowym detektorem jest mala siec neuronowa YuNet
(common/modele/face_detection_yunet_2023mar.onnx). Jest odporna na
przechylenie glowy i rozne oswietlenie, a na klatce 4K dziala rzad
wielkosci szybciej niz kaskada Haara w pelnej rozdzielczosci. Gdy pliku
modelu brak (albo OpenCV nie ma modulu FaceDetectorYN), program
automatycznie korzysta z klasycznej kaskady Haara.

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

Kaskada Haara pozostaje w projekcie jako:
  - awaryjny tryb dzialania, gdy brak pliku YuNet,
  - material dydaktyczny: jej dzialanie da sie prosto narysowac na tablicy,
    w przeciwienstwie do "czarnej skrzynki" sieci neuronowej,
  - punkt wyjscia do zadania "podmien detektor i porownaj szybkosc".
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


SCIEZKA_YUNET = Path(__file__).parent / "modele" / "face_detection_yunet_2023mar.onnx"


@dataclass
class WykrytaTwarz:
    """Prostokat (bounding box) wokol jednej wykrytej twarzy na obrazie.

    Wspolrzedne x, y to lewy-gorny rog prostokata, a w (szerokosc) i
    h (wysokosc) to jego rozmiary - dokladnie tak, jak zwraca je OpenCV.
    `pewnosc` to ocena pewnosci detekcji (0-1); dla kaskady Haara, ktora
    nie zwraca takiej oceny, przyjmujemy 1.0.
    """

    x: int
    y: int
    w: int
    h: int
    pewnosc: float = 1.0

    def wytnij(self, klatka: np.ndarray, margines: float = 0.2) -> np.ndarray:
        """Wycina fragment klatki odpowiadajacy tej twarzy, z dodatkowym
        marginesem wokol niej (domyslnie 20% szerokosci/wysokosci twarzy
        z kazdej strony).

        Margines jest wazny: samo wykrycie bywa "ciasno" dopasowane (czasem
        ucina fragment brody albo czola) - kilka dodatkowych pikseli tla
        sprawia, ze ekstraktor cech dostaje troche wiecej kontekstu.
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

    MIN_ROZMIAR_DETEKCJI = 24
    DETEKCJA_SZEROKOSC = 2560
    PROG_YUNET = 0.7
    MIN_IOU_POTWIERDZENIA = 0.1

    def __init__(
        self,
        wspolczynnik_skali: float = 1.05,
        min_sasiadow: int = 7,
        katy_awaryjne: tuple[int, ...] = (-20, -35, -50, 20, 35, 50),
        min_sasiadow_awaryjny: int = 5,
        min_rozmiar_wzgledny: float = 0.015,
        prog_yunet: float = PROG_YUNET,
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
        min_rozmiar_wzgledny: minimalny rozmiar wykrycia wzgledem krotszego
            boku obrazu detekcji. Kandydaci musza dodatkowo zostac
            potwierdzeni niezalezna kaskada, co ogranicza falszywe wykrycia
            przy malym rozmiarze twarzy.
        prog_yunet: minimalna pewnosc detekcji sieci YuNet (0-1). Wiecej =
            mniej falszywych ramek, ale wieksze ryzyko pominiecia twarzy w
            trudnych warunkach. Ma znaczenie tylko, gdy YuNet jest dostepny.
        """
        self.kaskada = _wczytaj_kaskade_bezpiecznie("haarcascade_frontalface_default.xml")
        self.kaskada_potwierdzajaca = _wczytaj_kaskade_bezpiecznie(
            "haarcascade_frontalface_alt.xml"
        )
        self.wspolczynnik_skali = wspolczynnik_skali
        self.min_sasiadow = min_sasiadow
        self.katy_awaryjne = katy_awaryjne
        self.min_sasiadow_awaryjny = min_sasiadow_awaryjny
        self.min_rozmiar_wzgledny = min_rozmiar_wzgledny
        self.prog_yunet = prog_yunet
        self.yunet = self._wczytaj_yunet()
        self._rozmiar_yunet = None

    def _wczytaj_yunet(self):
        """Wczytuje model YuNet z common/modele (None, gdy go brak - wtedy
        dziala wolniejszy Haar)."""
        if not hasattr(cv2, "FaceDetectorYN") or not SCIEZKA_YUNET.exists():
            return None
        try:
            dane = np.frombuffer(SCIEZKA_YUNET.read_bytes(), dtype=np.uint8)
            return cv2.FaceDetectorYN.create(
                "onnx", dane, np.empty(0, np.uint8), (640, 480),
                self.prog_yunet, 0.3, 5000,
            )
        except (cv2.error, ValueError) as blad:
            print(
                f"[uwaga] Nie udalo sie wczytac modelu YuNet z {SCIEZKA_YUNET} "
                f"({blad}) - przechodze na kaskade Haara."
            )
            return None

    def _wykryj_na_obrazie(self, obraz_szary: np.ndarray, min_sasiadow: int) -> np.ndarray:
        """Uruchamia surowa detekcje kaskady Haara na juz przygotowanym
        (jednokanalowym) obrazie i zwraca tablice wykryc w formacie OpenCV
        - lista czwórek (x, y, w, h). Wydzielone do osobnej metody, zeby nie
        powtarzac tych samych parametrow w kazdym miejscu, gdzie wywolujemy
        detekcje (obraz na wprost oraz kazda z obroconych kopii). Przyjmuje
        min_sasiadow jako parametr (a nie od razu self.min_sasiadow), bo
        proby awaryjne na obroconych kopiach obrazu uzywaja innego,
        nizszego progu - patrz min_sasiadow_awaryjny w __init__."""
        # Skanujemy klatke w pelnej rozdzielczosci; niska granica minSize
        # pozwala znalezc twarze w oddali.
        krotszy_bok = min(obraz_szary.shape[:2])
        min_rozmiar_px = max(
            self.MIN_ROZMIAR_DETEKCJI, int(krotszy_bok * self.min_rozmiar_wzgledny)
        )
        return self.kaskada.detectMultiScale(
            obraz_szary,
            scaleFactor=self.wspolczynnik_skali,
            minNeighbors=min_sasiadow,
            minSize=(min_rozmiar_px, min_rozmiar_px),
        )

    def _potwierdz_wykrycia(
        self, obraz_szary: np.ndarray, wykrycia: np.ndarray
    ) -> np.ndarray:
        """Odrzuca kandydatow, ktorych nie potwierdza druga kaskada Haara."""
        zatwierdzone = []
        for x, y, w, h in wykrycia:
            margines = max(12, int(max(w, h) * 0.5))
            x1 = max(0, x - margines)
            y1 = max(0, y - margines)
            x2 = min(obraz_szary.shape[1], x + w + margines)
            y2 = min(obraz_szary.shape[0], y + h + margines)
            roi = obraz_szary[y1:y2, x1:x2]
            potwierdzenia = self.kaskada_potwierdzajaca.detectMultiScale(
                roi,
                scaleFactor=self.wspolczynnik_skali,
                minNeighbors=self.min_sasiadow_awaryjny,
                minSize=(self.MIN_ROZMIAR_DETEKCJI, self.MIN_ROZMIAR_DETEKCJI),
            )
            for px, py, pw, ph in potwierdzenia:
                px += x1
                py += y1
                lewo = max(x, px)
                gora = max(y, py)
                prawo = min(x + w, px + pw)
                dol = min(y + h, py + ph)
                pole_wspolne = max(0, prawo - lewo) * max(0, dol - gora)
                pole_unii = w * h + pw * ph - pole_wspolne
                if pole_unii > 0 and pole_wspolne / pole_unii >= self.MIN_IOU_POTWIERDZENIA:
                    zatwierdzone.append((x, y, w, h))
                    break
        return np.asarray(zatwierdzone, dtype=np.int32).reshape(-1, 4)

    def _wykryj_yunet(self, klatka: np.ndarray) -> list[WykrytaTwarz]:
        """Szybka detekcja siecia YuNet (DNN) - odporna na poze i swiatlo,
        prawie bez falszywych wykryc zegarow/kafelkow. Obraz skalujemy do
        szerokosci DETEKCJA_SZEROKOSC tylko na czas detekcji."""
        wysokosc, szerokosc = klatka.shape[:2]
        skala = min(1.0, self.DETEKCJA_SZEROKOSC / szerokosc)
        if skala < 1.0:
            obraz = cv2.resize(
                klatka,
                (int(round(szerokosc * skala)), int(round(wysokosc * skala))),
                interpolation=cv2.INTER_AREA,
            )
        else:
            obraz = klatka
        rozmiar = (obraz.shape[1], obraz.shape[0])
        if self._rozmiar_yunet != rozmiar:
            self.yunet.setInputSize(rozmiar)
            self._rozmiar_yunet = rozmiar
        _, wyniki = self.yunet.detect(obraz)
        if wyniki is None:
            return []

        twarze = []
        for wynik in wyniki:
            x, y, w, h = (float(v) / skala for v in wynik[:4])
            # Prostokat YuNet jest waski; zamieniamy go na kwadrat o boku
            # rownym wysokosci, wspolsrodkowy - tak jak ramki Haara, na
            # ktorych wytrenowano wzorce.
            bok = max(h, w)
            srodek_x = x + w / 2
            srodek_y = y + h / 2
            bok_px = int(round(bok))
            # Przesuwamy kwadrat (zamiast go obcinac) tak, aby w calosci
            # miescil sie w klatce - inaczej przy twarzy blisko krawedzi
            # otrzymalibysmy nieproporcjonalnie maly wycinek.
            x1 = int(round(srodek_x - bok_px / 2))
            y1 = int(round(srodek_y - bok_px / 2))
            x1 = max(0, min(x1, szerokosc - bok_px))
            y1 = max(0, min(y1, wysokosc - bok_px))
            bok_px = min(bok_px, szerokosc, wysokosc)
            if bok_px >= self.MIN_ROZMIAR_DETEKCJI:
                pewnosc = float(wynik[14]) if len(wynik) > 14 else 1.0
                twarze.append(
                    WykrytaTwarz(x1, y1, bok_px, bok_px, pewnosc=pewnosc)
                )
        return twarze

    def wykryj(self, klatka: np.ndarray) -> list[WykrytaTwarz]:
        """Zwraca liste wszystkich twarzy wykrytych na podanej klatce (obrazie
        BGR, tak jak dostarcza go OpenCV z kamery)."""
        if self.yunet is not None:
            return self._wykryj_yunet(klatka)

        # Kaskada Haara dziala na obrazach w skali szarosci; detekcja
        # zachowuje pelna rozdzielczosc obrazu otrzymanego z kamery.
        szarosc = cv2.cvtColor(klatka, cv2.COLOR_BGR2GRAY)
        # Wyrownanie histogramu poprawia kontrast, co pomaga w slabym/nierownym
        # oswietleniu (typowa sytuacja w sali lekcyjnej).
        szarosc = cv2.equalizeHist(szarosc)

        wykrycia = self._potwierdz_wykrycia(
            szarosc, self._wykryj_na_obrazie(szarosc, self.min_sasiadow)
        )
        if len(wykrycia) > 0:
            return [WykrytaTwarz(int(x), int(y), int(w), int(h)) for x, y, w, h in wykrycia]

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
            wykrycia_obrocone = self._potwierdz_wykrycia(
                obrocony_obraz,
                self._wykryj_na_obrazie(obrocony_obraz, self.min_sasiadow_awaryjny),
            )
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
                wyniki.append(
                    WykrytaTwarz(
                        nowy_x,
                        nowy_y,
                        int(w),
                        int(h),
                    )
                )
            return wyniki

        return []
