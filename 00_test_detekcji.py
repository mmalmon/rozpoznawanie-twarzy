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

# argparse to modul standardowej biblioteki Pythona sluzacy do obslugi
# argumentow uruchomieniowych skryptu (to, co wpisujemy po nazwie skryptu w
# konsoli, np. "--kamera 1") - dzieki niemu nie musimy recznie parsowac
# tekstu wpisanego przez uzytkownika.
import argparse
# time udostepnia funkcje zwiazane z czasem, tutaj uzywana do zmierzenia,
# ile sekund minelo miedzy kolejnymi klatkami obrazu (do wyliczenia FPS).
import time

# cv2 - biblioteka OpenCV odpowiedzialna za obsluge kamery, rysowanie
# ksztaltow (prostokatow, tekstu) na obrazie i wyswietlanie okna podgladu.
import cv2

# Importujemy wlasne funkcje z folderu common/ (nasz "wspolny kod"
# wykorzystywany przez kilka skryptow projektu, zeby nie powielac go w
# kazdym pliku osobno).
from common.camera import dopasuj_do_ekranu, otworz_kamere, wybierz_kamere_interaktywnie
from common.face_detector import DetektorTwarzy


def parsuj_argumenty() -> argparse.Namespace:
    """Definiuje i odczytuje argumenty uruchomieniowe skryptu (to, co
    wpisujemy po nazwie pliku w konsoli, np. "python 00_test_detekcji.py
    --kamera 1")."""
    parser = argparse.ArgumentParser(description="Test detekcji twarzy na zywo.")
    parser.add_argument(
        "--kamera",
        type=int,
        default=None,
        help="Indeks kamery. Jesli pominiety, program zapyta interaktywnie.",
    )
    parser.add_argument("--szerokosc", type=int, default=1920)
    parser.add_argument("--wysokosc", type=int, default=1080)
    # parse_args() faktycznie odczytuje to, co uzytkownik wpisal w konsoli
    # (sys.argv) i zwraca obiekt, z ktorego mozemy odczytac kazda wartosc
    # jako pole, np. argumenty.kamera.
    return parser.parse_args()


def main() -> None:
    """Glowna funkcja skryptu - uruchamiana tylko wtedy, gdy plik jest
    odpalony bezposrednio (patrz blok if __name__ == "__main__" na koncu
    pliku), a nie gdy jest importowany przez inny skrypt."""
    argumenty = parsuj_argumenty()
    # Operator warunkowy "x if warunek else y" w jednej linii: jesli uczen
    # podal --kamera, uzywamy jej indeksu; jesli nie podal (wartosc None),
    # pytamy interaktywnie, ktorej kamery uzyc.
    indeks_kamery = (
        argumenty.kamera if argumenty.kamera is not None else wybierz_kamere_interaktywnie()
    )

    kamera = otworz_kamere(indeks_kamery, szerokosc=argumenty.szerokosc, wysokosc=argumenty.wysokosc)
    detektor = DetektorTwarzy()

    # Patrz wyjasnienie flagi WINDOW_NORMAL w 04_rozpoznawanie_na_zywo.py -
    # dzieki niej mozna recznie zmienic rozmiar okna, a obraz sam sie
    # przeskaluje, bez przycinania.
    nazwa_okna = "Test detekcji twarzy (q=koniec)"
    cv2.namedWindow(nazwa_okna, cv2.WINDOW_NORMAL)

    print("[info] Nacisnij 'q' lub ESC, aby zakonczyc.")
    poprzedni_czas = time.time()

    try:
        # Petla "while True" dziala w kolko, klatka po klatce, az uczen
        # nacisnie 'q'/ESC albo wystapi blad odczytu z kamery (wtedy petla
        # jest przerywana instrukcja "break").
        while True:
            # kamera.read() zwraca dwie wartosci na raz: flage powodzenia
            # (czy_odczytano) oraz sama klatke obrazu jako tablice numpy.
            czy_odczytano, klatka = kamera.read()
            if not czy_odczytano:
                print("[blad] Nie udalo sie odczytac klatki z kamery.")
                break

            # cv2.flip(klatka, 1) odbija obraz w poziomie (parametr 1 =
            # odbicie wzgledem osi pionowej). Kamera pokazuje domyslnie
            # obraz "jak dla widza z zewnatrz" - gdy poruszamy glowa w lewo,
            # na podgladzie widac ruch w prawo. Ludzie sa przyzwyczajeni do
            # obrazu "jak w lusterku" (tak jak w aplikacjach do wideorozmow
            # czy w aparacie w telefonie), dlatego odbijamy klatke, zanim
            # cokolwiek na niej narysujemy lub zapiszemy.
            klatka = cv2.flip(klatka, 1)

            wykryte_twarze = detektor.wykryj(klatka)

            # Rysujemy ramke wokol kazdej wykrytej twarzy. Na tym etapie NIE
            # wiemy jeszcze, kto to jest - dlatego ramka jest zolta i bez
            # zadnego imienia (to bedzie krok 4, po wytrenowaniu modelu).
            for twarz in wykryte_twarze:
                # cv2.rectangle rysuje prostokat na obrazie "klatka" (modyfikuje
                # go bezposrednio - "w miejscu"): pierwszy punkt to lewy-gorny
                # rog, drugi to prawy-dolny rog, nastepnie kolor (BGR) i
                # grubosc linii w pikselach.
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

            # cv2.putText rysuje napis na obrazie: tekst, wspolrzedne lewego-
            # dolnego rogu tekstu, czcionke, mnoznik jej rozmiaru, kolor
            # (BGR) i grubosc linii.
            cv2.putText(
                klatka,
                f"Wykryto twarzy: {len(wykryte_twarze)}  FPS: {fps:.1f}",
                (20, 40),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                (0, 220, 220),
                2,
            )

            # Skalujemy TYLKO obraz wyswietlany w oknie (nie wplywa to na
            # detekcje powyzej) - patrz wyjasnienie w common/camera.py.
            cv2.imshow(nazwa_okna, dopasuj_do_ekranu(klatka))

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
