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

from common.camera import otworz_kamere, wybierz_kamere_interaktywnie
from common.face_detector import DetektorTwarzy


def parsuj_argumenty() -> argparse.Namespace:
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
    argumenty = parsuj_argumenty()
    indeks_kamery = (
        argumenty.kamera if argumenty.kamera is not None else wybierz_kamere_interaktywnie()
    )

    kamera = otworz_kamere(indeks_kamery, szerokosc=argumenty.szerokosc, wysokosc=argumenty.wysokosc)
    detektor = DetektorTwarzy()

    print("[info] Nacisnij 'q' lub ESC, aby zakonczyc.")
    poprzedni_czas = time.time()

    try:
        while True:
            czy_odczytano, klatka = kamera.read()
            if not czy_odczytano:
                print("[blad] Nie udalo sie odczytac klatki z kamery.")
                break

            wykryte_twarze = detektor.wykryj(klatka)

            # Rysujemy ramke wokol kazdej wykrytej twarzy. Na tym etapie NIE
            # wiemy jeszcze, kto to jest - dlatego ramka jest zolta i bez
            # zadnego imienia (to bedzie krok 4, po wytrenowaniu modelu).
            for twarz in wykryte_twarze:
                cv2.rectangle(
                    klatka,
                    (twarz.x, twarz.y),
                    (twarz.x + twarz.w, twarz.y + twarz.h),
                    (0, 220, 220),  # kolor w formacie BGR (zolty)
                    2,
                )

            teraz = time.time()
            # FPS (klatki na sekunde) liczymy jako odwrotnosc czasu miedzy
            # kolejnymi klatkami - to standardowy sposob mierzenia plynnosci
            # obrazu w aplikacjach czasu rzeczywistego.
            fps = 1.0 / max(1e-6, teraz - poprzedni_czas)
            poprzedni_czas = teraz

            cv2.putText(
                klatka,
                f"Wykryto twarzy: {len(wykryte_twarze)}  FPS: {fps:.1f}",
                (20, 40),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                (0, 220, 220),
                2,
            )

            cv2.imshow("Test detekcji twarzy (q=koniec)", klatka)

            klawisz = cv2.waitKey(1) & 0xFF
            if klawisz in (ord("q"), 27):  # 27 = kod klawisza ESC
                break
    finally:
        # Zawsze zwalniamy kamere i zamykamy okna, nawet jesli program
        # przerwal petle przez blad - inaczej kamera moglaby pozostac
        # "zajeta" i kolejne uruchomienie skryptu by sie nie powiodlo.
        kamera.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
