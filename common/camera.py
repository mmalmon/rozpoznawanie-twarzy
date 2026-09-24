"""
Modul odpowiedzialny za obsluge kamery internetowej (np. RoWave RC16) na Windows.

DLA UCZNIOW: kamera USB (typu UVC - "USB Video Class") moze pracowac w kilku
trybach transmisji obrazu (tzw. FOURCC - czterolitrowy kod kompresji). Dwa
najwazniejsze to:
  - MJPG - kazda klatka skompresowana osobno jak zdjecie JPEG. Male dane,
    duzo klatek na sekunde (FPS) nawet przy wysokiej rozdzielczosci.
  - YUY2/RAW - klatki nieskompresowane. Przy rozdzielczosci 4K generuja tak
    duzo danych, ze magistrala USB nie nadaza je przeslac, a FPS spada
    czasem do kilku klatek na sekunde.

Dlatego w tym module ZAWSZE wymuszamy tryb MJPG przed ustawieniem
rozdzielczosci kamery.
"""

from __future__ import annotations

# "from __future__ import annotations" to specjalny import techniczny (nie
# zwykla biblioteka) - mowi Pythonowi, zeby traktowal adnotacje typow (np.
# "-> list[WykrytaKamera]") jako zwykly tekst, a nie wyliczal ich od razu.
# Dzieki temu mozemy pisac nowoczesne, czytelniejsze adnotacje typow (np.
# "list[int]" zamiast starszego "List[int]") nawet na starszych wersjach
# Pythona. To nie zmienia dzialania programu - tylko ulatwia czytanie kodu
# i podpowiedzi w edytorze.

# dataclass to "dekorator" (funkcja ozdabiajaca inna funkcje/klase) z
# biblioteki standardowej Pythona. Automatycznie generuje dla naszej klasy
# metody takie jak __init__ (konstruktor) czy __repr__ (wyswietlanie),
# na podstawie samej listy pol - bez pisania tego recznie. Uzywamy jej
# ponizej do zdefiniowania WykrytaKamera.
from dataclasses import dataclass

# cv2 to nazwa modulu biblioteki OpenCV (Open Source Computer Vision) po
# zainstalowaniu pakietu "opencv-python". To glowna biblioteka tego
# projektu - obsluguje kamere, przetwarzanie obrazu i (w common/face_detector.py)
# detekcje twarzy.
import cv2

# Docelowa rozdzielczosc probna uzywana przy wykrywaniu mozliwosci kamery -
# jesli kamera potrafi jej dostarczyc, prawdopodobnie jest to prawdziwa
# kamera 4K (np. RoWave RC16), a nie zwykla kamera wbudowana w laptopa.
PROBNA_SZEROKOSC_4K = 3840
PROBNA_WYSOKOSC_4K = 2160


@dataclass
class WykrytaKamera:
    """Opisuje jedna kamere znaleziona w systemie: jej indeks i maksymalna
    rozdzielczosc, jaka udalo sie z niej odczytac podczas testu."""

    indeks: int
    maks_szerokosc: int
    maks_wysokosc: int

    @property
    def czy_prawdopodobnie_4k(self) -> bool:
        """Zwraca True, jesli kamera osiagnela pelna rozdzielczosc 4K.

        To tylko heurystyka pomagajaca uczniom odgadnac, ktora kamera w
        systemie to RoWave RC16 (prawdziwe 4K), a ktora to np. kamera
        wbudowana w ekran laptopa (zazwyczaj nizsza rozdzielczosc).
        """
        return (self.maks_szerokosc, self.maks_wysokosc) == (
            PROBNA_SZEROKOSC_4K,
            PROBNA_WYSOKOSC_4K,
        )


def wykryj_dostepne_kamery(maks_indeks: int = 5) -> list[WykrytaKamera]:
    """Sprawdza kolejne indeksy kamer (0, 1, 2, ...) i zwraca liste tych,
    ktore udalo sie otworzyc.

    Dla kazdej znalezionej kamery probujemy ustawic rozdzielczosc 4K i
    sprawdzamy, jaka rozdzielczosc faktycznie zostala przyznana - system
    operacyjny sam "przytnie" ja do maksimum, jakie dana kamera obsluguje.
    Dzieki temu mozemy odroznic kamery od siebie bez pytania uzytkownika
    o nazwe sterownika.
    """
    znalezione_kamery: list[WykrytaKamera] = []

    # Petla "for" po kolejnych indeksach - system operacyjny numeruje kamery
    # od 0 wzwyz w kolejnosci, w jakiej je wykryl (kolejnosc ta moze sie
    # zmienic np. po ponownym podlaczeniu kamery USB, dlatego zawsze warto
    # sprawdzic to na nowo zamiast zakladac "na sztywno", ze RC16 to zawsze
    # indeks 1).
    for indeks in range(maks_indeks):
        # cv2.VideoCapture(...) probuje otworzyc polaczenie z kamera o danym
        # indeksie. CAP_DSHOW to nazwa konkretnego "backendu" (sterownika
        # posredniczacego) systemu Windows do obslugi kamer USB (UVC).
        # Bez niego OpenCV czasem ignoruje nasze zadania zmiany FOURCC/rozdzielczosci.
        kamera = cv2.VideoCapture(indeks, cv2.CAP_DSHOW)
        if not kamera.isOpened():
            # isOpened() zwraca False, gdy pod danym indeksem nie ma zadnej
            # kamery (np. sprawdzamy indeks 3, a w systemie sa tylko 2
            # kamery) - wtedy po prostu przechodzimy do kolejnego indeksu.
            kamera.release()
            continue

        # kamera.set(...) prosi kamere o zmiane jej ustawien. Zwraca True/False,
        # ale NIE gwarantuje, ze zadanie zostalo spelnione w 100% - dlatego
        # kilka linijek nizej odczytujemy, co faktycznie zostalo ustawione.
        kamera.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))
        kamera.set(cv2.CAP_PROP_FRAME_WIDTH, PROBNA_SZEROKOSC_4K)
        kamera.set(cv2.CAP_PROP_FRAME_HEIGHT, PROBNA_WYSOKOSC_4K)

        # kamera.read() pobiera jedna klatke obrazu. Pierwsza wartosc zwrocona
        # (tutaj: udalo_sie_odczytac) mowi, czy odczyt sie powiodl - druga
        # wartosc to sama klatka (obraz), ktorej tutaj nie potrzebujemy,
        # wiec zapisujemy ja do zmiennej "_" (konwencja Pythona oznaczajaca
        # "ta wartosc celowo ignorujemy").
        udalo_sie_odczytac, _ = kamera.read()
        # kamera.get(...) odczytuje AKTUALNA wartosc danego ustawienia -
        # czyli to, co kamera naprawde przyznala, a nie to, o co prosilismy.
        szerokosc = int(kamera.get(cv2.CAP_PROP_FRAME_WIDTH))
        wysokosc = int(kamera.get(cv2.CAP_PROP_FRAME_HEIGHT))
        # release() zwalnia dostep do kamery - bardzo wazne, zeby inne
        # aplikacje (albo kolejne uruchomienie tej petli) mogly z niej
        # korzystac. Bez tego kamera "zawiesilaby sie" jako zajeta.
        kamera.release()

        if udalo_sie_odczytac:
            znalezione_kamery.append(WykrytaKamera(indeks, szerokosc, wysokosc))

    return znalezione_kamery


def wybierz_kamere_interaktywnie(maks_indeks: int = 5) -> int:
    """Wykrywa dostepne kamery i pyta uzytkownika w konsoli, ktorej ma uzyc.

    Jesli w systemie jest tylko jedna kamera, program wybiera ja automatycznie
    bez zadawania pytania. Ta funkcja jest wywolywana, gdy uczen uruchomi
    skrypt BEZ podania argumentu --kamera.
    """
    print("[kamera] Szukam dostepnych kamer...")
    kamery = wykryj_dostepne_kamery(maks_indeks)

    if not kamery:
        raise RuntimeError(
            "Nie znaleziono zadnej kamery. Sprawdz, czy RoWave RC16 (lub inna "
            "kamera) jest podlaczona przez USB i czy nie jest uzywana przez "
            "inna aplikacje."
        )

    if len(kamery) == 1:
        # if/else po dlugosci listy: gdy jest tylko jedna kamera, nie ma
        # sensu pytac uczniow o wybor - po prostu jej uzywamy.
        kamera = kamery[0]
        print(
            f"[kamera] Znaleziono jedna kamere: --kamera {kamera.indeks} "
            f"({kamera.maks_szerokosc}x{kamera.maks_wysokosc})"
        )
        return kamera.indeks

    print("\nZnaleziono kilka kamer:")
    for kamera in kamery:
        podpowiedz = (
            "  <- prawdopodobnie RoWave RC16 (prawdziwe 4K)"
            if kamera.czy_prawdopodobnie_4k
            else ""
        )
        print(
            f"  [{kamera.indeks}] maks. rozdzielczosc "
            f"{kamera.maks_szerokosc}x{kamera.maks_wysokosc}{podpowiedz}"
        )

    # Zbior (set) poprawnych indeksow - uzywamy zbioru zamiast listy, bo
    # sprawdzenie "czy liczba nalezy do zbioru" (operator "in") jest bardzo
    # szybkie, niezaleznie od tego, ile jest kamer.
    poprawne_indeksy = {kamera.indeks for kamera in kamery}
    while True:
        # input() wstrzymuje program i czeka, az uczen wpisze cos w konsoli
        # i nacisnie Enter. .strip() usuwa przypadkowe spacje/entery na
        # poczatku i koncu wpisanego tekstu.
        wybor = input(f"\nWybierz numer kamery ({sorted(poprawne_indeksy)}): ").strip()
        # isdigit() sprawdza, czy tekst sklada sie WYLACZNIE z cyfr - dzieki
        # temu unikamy bledu, gdyby uczen wpisal np. litery zamiast liczby.
        if wybor.isdigit() and int(wybor) in poprawne_indeksy:
            return int(wybor)
        print("Niepoprawny wybor, sprobuj ponownie.")


def otworz_kamere(
    indeks: int = 0,
    szerokosc: int = 1920,
    wysokosc: int = 1080,
    fps: int = 30,
) -> cv2.VideoCapture:
    """Otwiera kamere o danym indeksie w zadanej rozdzielczosci i zwraca
    obiekt cv2.VideoCapture gotowy do odczytu kolejnych klatek metoda .read().

    DLA UCZNIOW: domyslnie uzywamy 1920x1080 (Full HD), a nie pelnego 4K
    (3840x2160) - do detekcji i rozpoznawania twarzy pelna rozdzielczosc
    kamery nie jest potrzebna, a mniejsze klatki oznaczaja wyzsze FPS i
    mniejsze obciazenie procesora/karty graficznej. Jesli chcecie zrobic
    bardzo ostre zdjecia treningowe w pelnym 4K, podajcie
    szerokosc=3840, wysokosc=2160.
    """
    # CAP_DSHOW jest wymagany na Windows, aby poprawnie ustawic FOURCC/rozdzielczosc
    # dla kamer UVC - domyslny backend MSMF czasem ignoruje te ustawienia.
    kamera = cv2.VideoCapture(indeks, cv2.CAP_DSHOW)

    if not kamera.isOpened():
        raise RuntimeError(
            f"Nie udalo sie otworzyc kamery o indeksie {indeks}. "
            "Sprawdz, czy kamera jest podlaczona i czy nie jest "
            "uzywana przez inna aplikacje (np. Kamera Windows, Teams, Zoom)."
        )

    kamera.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))
    # cv2.VideoWriter_fourcc(*"MJPG") zamienia napis "MJPG" na specjalny
    # 4-bajtowy kod liczbowy (FOURCC), ktorego oczekuje kamera/sterownik -
    # gwiazdka (*) przed "MJPG" rozpakowuje napis na 4 pojedyncze znaki
    # jako oddzielne argumenty tej funkcji.
    kamera.set(cv2.CAP_PROP_FRAME_WIDTH, szerokosc)
    kamera.set(cv2.CAP_PROP_FRAME_HEIGHT, wysokosc)
    kamera.set(cv2.CAP_PROP_FPS, fps)

    # Kamera moze zignorowac nasze zadanie i ustawic np. najblizsza obslugiwana
    # rozdzielczosc - dlatego odczytujemy, co faktycznie zostalo ustawione.
    rzeczywista_szerokosc = int(kamera.get(cv2.CAP_PROP_FRAME_WIDTH))
    rzeczywista_wysokosc = int(kamera.get(cv2.CAP_PROP_FRAME_HEIGHT))
    rzeczywiste_fps = kamera.get(cv2.CAP_PROP_FPS)
    print(
        f"[kamera] Kamera otwarta: {rzeczywista_szerokosc}x{rzeczywista_wysokosc} "
        f"@ {rzeczywiste_fps:.1f} FPS (zadano {szerokosc}x{wysokosc} @ {fps} FPS)"
    )

    return kamera

