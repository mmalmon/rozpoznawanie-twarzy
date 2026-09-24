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
  - rozne katy glowy (lekko w lewo/prawo/gore/dol), rozne wyrazy twarzy,
  - rozne oswietlenie (dzien/wieczor, swiatlo z roznych stron),
  - z okularami i bez, jesli osoba je czasem nosi,
  - staraj sie NIE zbierac serii identycznych, nieruchomych klatek - lepiej
    ruszac lekko glowa miedzy kolejnymi zdjeciami.
"""

from __future__ import annotations

# argparse - obsluga argumentow uruchomieniowych skryptu (--osoba, --kamera itd.).
import argparse
# time - do mierzenia uplywu czasu (np. czy minelo juz 0.3s od ostatniego
# automatycznego zapisu zdjecia).
import time
# Path - obiektowa reprezentacja sciezek plikow/folderow.
from pathlib import Path

# cv2 - OpenCV: obsluga kamery, rysowanie, skalowanie obrazow.
import cv2

# Wlasny kod z folderu common/.
from common.camera import otworz_kamere, wybierz_kamere_interaktywnie
from common.face_detector import DetektorTwarzy
from common.imgio import zapisz_obraz

FOLDER_DANYCH = Path(__file__).parent / "data" / "raw"
DOCELOWY_ROZMIAR = (224, 224)


def parsuj_argumenty() -> argparse.Namespace:
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
    argumenty = parsuj_argumenty()

    # mkdir(parents=True, exist_ok=True) tworzy folder danej osoby (razem z
    # ewentualnymi brakujacymi folderami nadrzednymi) - "exist_ok=True"
    # oznacza, ze NIE zglosi bledu, jesli folder juz istnieje (np. gdy
    # dogrywamy kolejne zdjecia do wczesniej rozpoczetego zbioru).
    folder_osoby = FOLDER_DANYCH / argumenty.osoba
    folder_osoby.mkdir(parents=True, exist_ok=True)
    # glob("*.jpg") wyszukuje wszystkie pliki o podanym wzorcu nazwy (tu:
    # dowolna nazwa konczaca sie na ".jpg") w danym folderze - liczymy je,
    # zeby wiedziec, ile zdjec mamy juz zebranych z poprzednich uruchomien.
    liczba_juz_zebranych = len(list(folder_osoby.glob("*.jpg")))

    indeks_kamery = (
        argumenty.kamera if argumenty.kamera is not None else wybierz_kamere_interaktywnie()
    )
    kamera = otworz_kamere(indeks_kamery, szerokosc=argumenty.szerokosc, wysokosc=argumenty.wysokosc)
    detektor = DetektorTwarzy()

    liczba_zapisanych = liczba_juz_zebranych
    auto_zapis_wlaczony = False
    czas_ostatniego_auto_zapisu = 0.0

    print(f"[info] Zapisuje do: {folder_osoby}")
    print(f"[info] Juz zebranych zdjec: {liczba_juz_zebranych}. Cel: {argumenty.cel}.")
    print("[info] SPACJA=zapisz  a=auto-zapis  q/ESC=koniec")

    try:
        while True:
            czy_odczytano, klatka = kamera.read()
            if not czy_odczytano:
                print("[blad] Nie udalo sie odczytac klatki z kamery.")
                break

            wykryte_twarze = detektor.wykryj(klatka)
            # .copy() tworzy niezalezna kopie klatki do rysowania podgladu -
            # dzieki temu oryginalna "klatka" pozostaje czysta (bez
            # narysowanych ramek/napisow) i to wlasnie z niej wycinamy
            # zdjecia zapisywane na dysk.
            podglad = klatka.copy()

            # Ramka wokol wykrytej twarzy - tu juz jest zielona, bo to skrypt
            # do ZBIERANIA danych treningowych, wiec przypomina uczniowi,
            # ktora twarz zostanie zapisana po nacisnieciu spacji.
            for twarz in wykryte_twarze:
                cv2.rectangle(
                    podglad,
                    (twarz.x, twarz.y),
                    (twarz.x + twarz.w, twarz.y + twarz.h),
                    (0, 200, 0),
                    2,
                )

            pasek_stanu = (
                f"Zebrano: {liczba_zapisanych}/{argumenty.cel}  "
                f"auto={'ON' if auto_zapis_wlaczony else 'OFF'}"
            )
            cv2.putText(
                podglad, pasek_stanu, (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 200, 0), 2
            )
            cv2.imshow("Zbieranie danych (q=koniec)", podglad)

            klawisz = cv2.waitKey(1) & 0xFF
            # cv2.waitKey(1) czeka maksymalnie 1 milisekunde na nacisniecie
            # klawisza i zwraca jego kod - operacja "& 0xFF" (bitowe AND)
            # jest tu potrzebna technicznie, zeby dzialalo to poprawnie na
            # wszystkich systemach operacyjnych. 32 to kod klawisza SPACJA
            # w standardzie ASCII, a ord("s") zamienia litere "s" na jej
            # kod liczbowy.
            czy_zapisac_teraz = klawisz in (32, ord("s"))  # 32 = kod klawisza SPACJA

            if klawisz == ord("a"):
                # Operator "not" odwraca wartosc logiczna - kazde
                # nacisniecie "a" przelacza tryb auto-zapisu wlaczony/wylaczony.
                auto_zapis_wlaczony = not auto_zapis_wlaczony
            if (
                auto_zapis_wlaczony
                and wykryte_twarze
                and (time.time() - czas_ostatniego_auto_zapisu) > 0.3
            ):
                czy_zapisac_teraz = True

            if czy_zapisac_teraz and wykryte_twarze:
                # Jesli w kadrze jest kilka twarzy (np. ktos przechodzi w
                # tle), bierzemy najwieksza - jest najblizej kamery, czyli
                # najprawdopodobniej to osoba, ktora aktualnie pozuje do zdjec.
                # key=lambda t: t.w * t.h mowi funkcji max(), zeby porownywala
                # twarze wedlug ich pola powierzchni (szerokosc razy wysokosc),
                # a nie np. wedlug kolejnosci wykrycia.
                twarz = max(wykryte_twarze, key=lambda t: t.w * t.h)
                wycinek = twarz.wytnij(klatka)
                wycinek = cv2.resize(wycinek, DOCELOWY_ROZMIAR)

                sciezka_pliku = folder_osoby / f"{argumenty.osoba}_{liczba_zapisanych:04d}.jpg"
                zapisz_obraz(sciezka_pliku, wycinek)
                liczba_zapisanych += 1
                czas_ostatniego_auto_zapisu = time.time()
                print(f"[zapisano] {sciezka_pliku.name} ({liczba_zapisanych}/{argumenty.cel})")

            if liczba_zapisanych >= argumenty.cel:
                print("[info] Osiagnieto docelowa liczbe zdjec.")
                break
            if klawisz in (ord("q"), 27):
                break
    finally:
        kamera.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
