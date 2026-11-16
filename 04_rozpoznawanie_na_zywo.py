"""
Krok 4: Rozpoznawanie twarzy na zywo z kamery.

Wymaga wczesniejszego zbudowania wzorcow tozsamosci (03_trenowanie_modelu.py).

Uzycie:
    python 04_rozpoznawanie_na_zywo.py
    python 04_rozpoznawanie_na_zywo.py --korekta-progu 0.05

DLA UCZNIOW - jak dziala rozpoznawanie w tej wersji projektu?

W odroznieniu od "klasycznej" klasyfikacji (siec zawsze wybiera
"najbardziej prawdopodobna" osobe sposrod znanych jej klas, nawet gdy w
kadrze jest ktos zupelnie inny), tutaj uzywamy podejscia zwanego
"rozpoznawaniem przez podobienstwo" (verification):
  1. Kazda wykryta twarz jest zamieniana na embedding - wektor kilkuset
     liczb opisujacy jej wyglad (uzywajac tej samej, zamrozonej sieci
     ImageNet, co przy budowaniu wzorcow w skrypcie 03).
  2. Dwa kadry tej samej twarzy sa porownywane z embeddingami zdjec kazdej
     znanej osoby; wygrywa najlepsze dopasowanie z models/wzorce_osob.json.
  3. Jesli NAJWYZSZE znalezione podobienstwo przekracza wczesniej
     skalibrowany prog danej osoby - twarz zostaje rozpoznana jako ta
     osoba. Jesli nie przekracza progu ZADNEJ znanej osoby - etykieta
     "Nieznana osoba".

Kolory ramek:
  - ZIELONA + imie   -> podobienstwo do wzorca >= skalibrowanego progu,
  - CZERWONA + "Nieznana osoba" -> podobienstwo ponizej progu kazdej osoby.
"""

from __future__ import annotations

# Ta linijka MUSI byc pierwszym kodem wykonywalnym w tym pliku - patrz
# szczegolowe wyjasnienie w common/uruchom_w_venv.py (automatyczne
# przelaczenie na Python z .venv, jesli uruchomiono innym interpreterem).
from common.uruchom_w_venv import przelacz_na_venv_jesli_trzeba

przelacz_na_venv_jesli_trzeba()

# argparse - obsluga argumentow uruchomieniowych (--kamera, --korekta-progu itd.).
import argparse
from concurrent.futures import ThreadPoolExecutor
# sys - sprawdzenie, czy program dziala w interaktywnym terminalu (przed
# pytaniem o usuniecie danych osoby po zakonczeniu podgladu).
import sys
# time - do mierzenia FPS (klatek na sekunde).
import time

# cv2 - OpenCV: obsluga kamery, rysowanie ramek/tekstu, konwersja kolorow.
import cv2
# torch - glowny silnik PyTorch, tu uzywany do uruchomienia zamrozonego
# ekstraktora cech na obrazie z kamery (tzw. inferencja).
import torch
from common.camera import dopasuj_do_ekranu, otworz_kamere, wybierz_kamere_interaktywnie
from common.dane_osob import (
    lista_osob,
    przetrenuj_bez_usunietych,
    usun_dane_osoby,
    usun_stare_pliki_modelu,
)
from common.emotions import RozpoznawaczEmocji
from common.face_detector import DetektorTwarzy, WykrytaTwarz
from common.model import pobierz_urzadzenie, zbuduj_ekstraktor_cech
# Wspolna logika rozpoznawania (ta sama, ktorej uzywa panel GUI):
# wczytanie wzorcow, dopasowanie do osob i analiza pojedynczej klatki.
from common.rozpoznawanie import (
    BrakWzorcowError,
    NiepoprawneWzorceError,
    analizuj_klatke,
    przygotuj_obrazy_kamery,
    wczytaj_wzorce as wczytaj_wzorce_z_pliku,
)

# Kolory ramek w formacie BGR (tak przechowuje kolory OpenCV - odwrotnie niz RGB).
KOLOR_ROZPOZNANY = (0, 200, 0)  # zielony
KOLOR_NIEZNANY = (0, 0, 220)  # czerwony
LICZBA_WATKOW_OPENCV = 8


def parsuj_argumenty() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Rozpoznawanie twarzy na zywo.")
    parser.add_argument(
        "--kamera",
        type=int,
        default=None,
        help="Indeks kamery. Jesli pominiety, program zapyta interaktywnie.",
    )
    parser.add_argument("--szerokosc", type=int, default=3840)
    parser.add_argument("--wysokosc", type=int, default=2160)
    parser.add_argument(
        "--emocje",
        action="store_true",
        help=(
            "wlacza rozpoznawanie emocji (domyslnie WYLACZONE). Tylko do nauki "
            "poza szkola/na wlasnej twarzy - w szkole zabronione przez AI Act "
            "(art. 5 ust. 1 lit. f)."
        ),
    )
    parser.add_argument(
        "--korekta-progu",
        type=float,
        default=0.0,
        help=(
            "Dodatkowa korekta (+/-) do progu podobienstwa skalibrowanego w "
            "skrypcie 03 - przydatne do szybkiego strojenia bez ponownego "
            "liczenia wzorcow (dodatnia wartosc = trudniej o rozpoznanie)."
        ),
    )
    return parser.parse_args()


def wczytaj_wzorce(urzadzenie: torch.device) -> dict[str, dict]:
    """Wczytuje wzorce tozsamosci i zamienia bledy na czytelny komunikat.

    Wlasciwa praca dzieje sie w common/rozpoznawaniu.py - tutaj tylko
    tlumaczymy wyjatki na SystemExit, bo skrypt konsolowy konczy sie w ten
    sposob (a panel GUI obsluguje te same wyjatki samodzielnie).
    """
    try:
        return wczytaj_wzorce_z_pliku(urzadzenie=urzadzenie)
    except (BrakWzorcowError, NiepoprawneWzorceError) as blad:
        raise SystemExit(str(blad)) from blad


def main() -> None:
    argumenty = parsuj_argumenty()
    cv2.setNumThreads(LICZBA_WATKOW_OPENCV)
    urzadzenie = pobierz_urzadzenie()

    wzorce = wczytaj_wzorce(urzadzenie)
    if not wzorce:
        raise SystemExit(
            "Plik models/wzorce_osob.json nie zawiera zadnej osoby. "
            "Uruchom 03_trenowanie_modelu.py po zebraniu danych."
        )
    print(f"[info] Wczytano wzorce dla {len(wzorce)} osob: {list(wzorce.keys())}")

    ekstraktor = zbuduj_ekstraktor_cech().to(urzadzenie)
    detektor = DetektorTwarzy()
    emocje = None
    if argumenty.emocje:
        if RozpoznawaczEmocji.dostepny():
            emocje = RozpoznawaczEmocji()
            print("[info] Rozpoznawanie emocji WLACZONE (--emocje).")
        else:
            print("[info] Brak modelu emocji w common/modele - pomijam emocje.")

    # Kadry korzystaja z tego samego rozmiaru i normalizacji co wzorce;
    # wejscie z OpenCV jest przed nimi konwertowane do PIL.
    przygotuj_obrazy = przygotuj_obrazy_kamery()

    print(
        f"[info] Zadana rozdzielczosc kamery: "
        f"{argumenty.szerokosc}x{argumenty.wysokosc} "
        "(domyslnie 3840x2160; mozna zmienic przez --szerokosc i --wysokosc)."
    )
    indeks_kamery = (
        argumenty.kamera if argumenty.kamera is not None else wybierz_kamere_interaktywnie()
    )
    try:
        kamera = otworz_kamere(
            indeks_kamery,
            szerokosc=argumenty.szerokosc,
            wysokosc=argumenty.wysokosc,
        )
    except RuntimeError as blad:
        raise SystemExit(f"[blad] {blad}") from blad

    # cv2.WINDOW_NORMAL (zamiast domyslnego cv2.WINDOW_AUTOSIZE) tworzy okno,
    # ktore uczen moze recznie zmniejszyc/powiekszyc myszka (przeciagajac
    # jego krawedz) - obraz z kamery automatycznie SKALUJE SIE do aktualnego
    # rozmiaru okna, wiec nic nie jest przycinane. Bez tej flagi OpenCV
    # tworzy okno o rozmiarze DOKLADNIE rownym rozmiarowi klatki, ktorego
    # nie da sie zmniejszyc - a jesli klatka jest wieksza niz ekran, taki
    # sztywny rozmiar okna wygladal jak przypadkowe przyciecie obrazu.
    nazwa_okna = "Rozpoznawanie twarzy (q=koniec)"
    cv2.namedWindow(nazwa_okna, cv2.WINDOW_NORMAL)

    print("[info] Nacisnij 'q' lub ESC, aby zakonczyc.")
    poprzedni_czas = time.time()
    ostatnie_wyniki: list[tuple[WykrytaTwarz, str, float, str, float]] = []

    with ThreadPoolExecutor(
        max_workers=1, thread_name_prefix="analiza-twarzy"
    ) as executor:
        przyszla_analiza = None
        czas_zlecenia_analizy = 0.0
        fps_analizy = 0.0
        analiza_wylaczona = False
        try:
            while True:
                czy_odczytano, klatka = kamera.read()
                if not czy_odczytano:
                    print("[blad] Nie udalo sie odczytac klatki z kamery.")
                    break

                klatka = cv2.flip(klatka, 1)
                if przyszla_analiza is not None and przyszla_analiza.done():
                    try:
                        ostatnie_wyniki = przyszla_analiza.result()
                    except Exception as blad:  # noqa: BLE001 - nie przerywamy podgladu
                        print(
                            f"[blad] Analiza klatki nie powiodla sie: {blad}. "
                            "Wylaczam rozpoznawanie, podglad z kamery nadal dziala."
                        )
                        analiza_wylaczona = True
                        ostatnie_wyniki = []
                    else:
                        fps_analizy = 1.0 / max(
                            1e-6, time.perf_counter() - czas_zlecenia_analizy
                        )
                    przyszla_analiza = None

                if przyszla_analiza is None and not analiza_wylaczona:
                    czas_zlecenia_analizy = time.perf_counter()
                    przyszla_analiza = executor.submit(
                        analizuj_klatke,
                        klatka.copy(),
                        detektor,
                        ekstraktor,
                        przygotuj_obrazy,
                        wzorce,
                        urzadzenie,
                        argumenty.korekta_progu,
                        emocje,
                    )

                # Rysujemy na kopii ZMniejszonej do rozmiaru okna, a nie na
                # pelnej klatce 4K. Dzieki temu tekst i ramki maja czytelny
                # rozmiar na ekranie (na 4K bylyby 4x za male), a pelna klatka
                # pozostaje nietknieta na potrzeby analizy.
                podglad = dopasuj_do_ekranu(klatka)
                skala_x = podglad.shape[1] / klatka.shape[1]
                skala_y = podglad.shape[0] / klatka.shape[0]
                skala_tekstu = max(0.6, podglad.shape[1] / 1600)

                for twarz, etykieta_osoby, podobienstwo, emocja, pewnosc_emocji in ostatnie_wyniki:
                    etykieta = f"{etykieta_osoby} ({podobienstwo * 100:.0f}%)"
                    if emocja:
                        etykieta += f" | {emocja} {pewnosc_emocji * 100:.0f}%"
                    kolor_ramki = (
                        KOLOR_NIEZNANY
                        if etykieta_osoby == "Nieznana osoba"
                        else KOLOR_ROZPOZNANY
                    )
                    x1 = int(twarz.x * skala_x)
                    y1 = int(twarz.y * skala_y)
                    x2 = int((twarz.x + twarz.w) * skala_x)
                    y2 = int((twarz.y + twarz.h) * skala_y)
                    cv2.rectangle(podglad, (x1, y1), (x2, y2), kolor_ramki, 2)
                    cv2.putText(
                        podglad,
                        etykieta,
                        (x1, max(0, y1 - int(8 * skala_tekstu))),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.7 * skala_tekstu,
                        kolor_ramki,
                        max(2, int(2 * skala_tekstu)),
                    )

                teraz = time.time()
                fps = 1.0 / max(1e-6, teraz - poprzedni_czas)
                poprzedni_czas = teraz
                cv2.putText(
                    podglad,
                    f"{klatka.shape[1]}x{klatka.shape[0]}  "
                    f"Podglad: {fps:.1f} FPS  Analiza: {fps_analizy:.1f} FPS",
                    (20, int(40 * skala_tekstu)),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.8 * skala_tekstu,
                    (255, 255, 0),
                    max(2, int(2 * skala_tekstu)),
                )

                # Podglad nie czeka na kosztowne wykrywanie i rozpoznawanie.
                cv2.imshow(nazwa_okna, podglad)

                klawisz = cv2.waitKey(1) & 0xFF
                if klawisz in (ord("q"), 27):
                    break
        finally:
            kamera.release()
            cv2.destroyAllWindows()


def zapytaj_o_usuniecie_danych() -> None:
    """Po zakonczeniu pyta, czy usunac dane osob i przetrenowac wzorce od nowa."""
    if not sys.stdin or not sys.stdin.isatty():
        return
    osoby = lista_osob()
    if not osoby:
        return
    print(f"\nOsoby w bazie: {', '.join(osoby)}")
    odpowiedz = input("Czy usunac dane jakichs osob? [t/N]: ").strip().lower()
    if odpowiedz not in ("t", "tak", "y", "yes"):
        return
    wybor = input(
        "Podaj nazwy osob po przecinku lub 'wszystkie' (puste = anuluj): "
    ).strip()
    if not wybor:
        return
    if wybor.lower() == "wszystkie":
        wybrane = osoby
    else:
        wybrane = [n.strip() for n in wybor.split(",") if n.strip()]
        nieznane = [n for n in wybrane if n not in osoby]
        if nieznane:
            print(f"[blad] Nie ma takich osob: {', '.join(nieznane)}. Nic nie usunieto.")
            return
    potwierdzenie = input(
        f"POTWIERDZ: trwale usunac dane osob ({', '.join(wybrane)}) i przetrenowac "
        "wzorce od nowa? Wpisz 'tak': "
    ).strip().lower()
    if potwierdzenie != "tak":
        print("Anulowano - nic nie usunieto.")
        return
    for osoba in wybrane:
        usun_dane_osoby(osoba)
    usun_stare_pliki_modelu()
    if przetrenuj_bez_usunietych():
        print("Gotowe: dane usuniete, wzorce przeliczone bez tych osob.")


if __name__ == "__main__":
    main()
    zapytaj_o_usuniecie_danych()
