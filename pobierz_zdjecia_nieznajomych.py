"""
Skrypt pomocniczy: pobiera zdjecia twarzy wielu roznych osob i zapisuje je
jako dane treningowe klasy "nieznajomy".

DLA UCZNIOW - problem "otwartego zbioru" (open-set recognition):

Jesli w bazie mamy tylko JEDNA osobe (np. tylko Ciebie), to siec podczas
klasyfikacji ma do wyboru tylko jedna etykiete - w efekcie funkcja softmax
ZAWSZE zwroci ~100% pewnosci dla tej jedynej znanej klasy, niezaleznie od
tego, czyja twarz naprawde jest w kadrze! Model nigdy nie widzial, jak
wyglada "ktos inny", wiec nie potrafi tego rozpoznac.

Rozwiazaniem jest dodanie SZTUCZNEJ, dodatkowej klasy "nieznajomy" -
zbioru zdjec wielu roznych, przypadkowych twarzy. Dzieki temu model uczy
sie odrozniac "Twoja twarz" od "jakiejkolwiek innej twarzy", a nie tylko
"zgadywac" jedyna klase, jaka zna.

Skad biora sie te zdjecia? Korzystamy z serwisu randomuser.me - darmowego
API zwracajacego przykladowe, gotowe portrety (zdjecia z profesjonalnej
sesji fotograficznej), ktorych autorzy WPROST udostepnili je do uzytku w
takich wlasnie celach testowych/edukacyjnych (nie sa to zdjecia uczniow
ani przypadkowych osob z internetu bez ich zgody).

WAZNE: folder data/ jest w .gitignore - te zdjecia (podobnie jak Twoje
wlasne) NIGDY nie trafiaja do publicznego repozytorium GitHub, zostaja
tylko lokalnie na tym komputerze.

Uzycie:
    python pobierz_zdjecia_nieznajomych.py
    python pobierz_zdjecia_nieznajomych.py --cel 80
"""

from __future__ import annotations

# argparse - obsluga argumentow uruchomieniowych (--cel, --nazwa-klasy).
import argparse
# random - do wymieszania kolejnosci pobieranych portretow, zeby przy
# malej liczbie --cel nie zawsze brac tych samych pierwszych zdjec.
import random
# time - do odczekania chwili miedzy kolejnymi pobraniami (uprzejmosc wobec
# darmowego serwisu - nie chcemy zasypac go zbyt szybkimi zadaniami).
import time
# urllib.request to modul standardowej biblioteki Pythona do pobierania
# danych z internetu (adresow URL) - uzywamy go zamiast dodatkowej
# biblioteki (np. requests), zeby projekt mial jak najmniej zaleznosci.
import urllib.request
from pathlib import Path

import cv2
import numpy as np

from common.face_detector import DetektorTwarzy
from common.imgio import zapisz_obraz

FOLDER_DANYCH = Path(__file__).parent / "data" / "raw"
DOCELOWY_ROZMIAR = (224, 224)
# randomuser.me udostepnia stala pule 100 portretow mezczyzn i 100 portretow
# kobiet pod przewidywalnymi adresami - to wystarczy na potrzeby tego
# projektu edukacyjnego (typowo zbieramy 60-100 zdjec klasy "nieznajomy").
ADRESY_PORTRETOW = [
    f"https://randomuser.me/api/portraits/{plec}/{numer}.jpg"
    for plec in ("men", "women")
    for numer in range(100)
]
# Rozmiar, do ktorego skalujemy pobrany portret PRZED detekcja twarzy.
# Oryginalne portrety maja tylko 128x128 pikseli - to za malo, zeby
# kaskada Haara (z naszym progiem minSize=80x80) niezawodnie wykryla
# twarz. Powiekszenie obrazu (bez utraty informacji, tylko interpolacja)
# znaczaco poprawia skutecznosc wykrywania.
ROZMIAR_DO_DETEKCJI = (384, 384)


def parsuj_argumenty() -> argparse.Namespace:
    """Definiuje i odczytuje argumenty uruchomieniowe skryptu."""
    parser = argparse.ArgumentParser(
        description="Pobiera zdjecia wielu osob do klasy 'nieznajomy'."
    )
    parser.add_argument(
        "--cel", type=int, default=60, help="Docelowa liczba zdjec do pobrania"
    )
    parser.add_argument(
        "--nazwa-klasy",
        type=str,
        default="nieznajomy",
        help="Nazwa folderu/klasy, do ktorej trafia zdjecia (domyslnie 'nieznajomy')",
    )
    parser.add_argument(
        "--opoznienie",
        type=float,
        default=0.3,
        help="Ile sekund odczekac miedzy kolejnymi pobraniami (uprzejmosc wobec serwisu)",
    )
    return parser.parse_args()


def pobierz_jedno_zdjecie(url: str) -> np.ndarray | None:
    """Pobiera jeden portret spod podanego adresu URL i zwraca go jako
    obraz OpenCV (tablice numpy), albo None w razie bledu."""
    try:
        # urllib.request.urlopen wysyla zadanie HTTP GET pod wskazany adres
        # i zwraca obiekt, z ktorego mozemy odczytac odpowiedz serwera.
        # Naglowek "User-Agent" udajemy przegladarke - niektore serwery
        # odrzucaja zadania bez tego naglowka, traktujac je jak boty.
        zadanie = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
        with urllib.request.urlopen(zadanie, timeout=15) as odpowiedz:
            surowe_bajty = odpowiedz.read()
    except Exception as blad:  # celowo szeroki wyjatek - to skrypt pomocniczy,
        # a nie krytyczna czesc systemu; jeden nieudany request nie powinien
        # przerywac calego pobierania.
        print(f"[uwaga] Nie udalo sie pobrac zdjecia: {blad}")
        return None

    # np.frombuffer zamienia surowe bajty odpowiedzi HTTP (skompresowany
    # plik JPEG) na tablice liczb typu uint8 - to samo, co np.fromfile w
    # common/imgio.py, tylko zrodlem danych jest tu pamiec, a nie dysk.
    dane = np.frombuffer(surowe_bajty, dtype=np.uint8)
    return cv2.imdecode(dane, cv2.IMREAD_COLOR)


def main() -> None:
    argumenty = parsuj_argumenty()

    folder_docelowy = FOLDER_DANYCH / argumenty.nazwa_klasy
    folder_docelowy.mkdir(parents=True, exist_ok=True)
    liczba_juz_zebranych = len(list(folder_docelowy.glob("*.jpg")))

    detektor = DetektorTwarzy()

    # Tasujemy kolejnosc adresow, zeby przy malych wartosciach --cel zbior
    # skladal sie z losowo wybranych osob (a nie zawsze tych samych,
    # pierwszych w kolejnosci "men/0, men/1, men/2...").
    adresy = ADRESY_PORTRETOW.copy()
    random.shuffle(adresy)

    print(f"[info] Zapisuje do: {folder_docelowy}")
    print(f"[info] Juz zebranych zdjec: {liczba_juz_zebranych}. Cel: {argumenty.cel}.")
    print("[info] Zrodlo: randomuser.me (gotowe portrety udostepnione do celow testowych)")

    liczba_zapisanych = liczba_juz_zebranych
    # enumerate(adresy, start=1) daje nam zarowno kolejny adres, jak i
    # numer proby (przydatny do czytelnych komunikatow w konsoli).
    for numer_proby, url in enumerate(adresy, start=1):
        if liczba_zapisanych >= argumenty.cel:
            break

        obraz = pobierz_jedno_zdjecie(url)
        if obraz is None:
            time.sleep(argumenty.opoznienie)
            continue

        # Powiekszamy mały portret przed detekcja - patrz wyjasnienie przy
        # stalej ROZMIAR_DO_DETEKCJI powyzej.
        obraz_do_detekcji = cv2.resize(obraz, ROZMIAR_DO_DETEKCJI, interpolation=cv2.INTER_CUBIC)
        wykryte_twarze = detektor.wykryj(obraz_do_detekcji)
        if not wykryte_twarze:
            print(f"[uwaga] Proba {numer_proby}: nie wykryto twarzy, pomijam ({url}).")
            time.sleep(argumenty.opoznienie)
            continue

        # Bierzemy najwieksza wykryta twarz (w tych portretach zazwyczaj
        # i tak jest tylko jedna, wyrazna twarz na srodku kadru).
        twarz = max(wykryte_twarze, key=lambda t: t.w * t.h)
        wycinek = twarz.wytnij(obraz_do_detekcji, margines=0.2)
        wycinek = cv2.resize(wycinek, DOCELOWY_ROZMIAR)

        sciezka_pliku = folder_docelowy / f"{argumenty.nazwa_klasy}_{liczba_zapisanych:04d}.jpg"
        zapisz_obraz(sciezka_pliku, wycinek)
        liczba_zapisanych += 1
        print(f"[zapisano] {sciezka_pliku.name} ({liczba_zapisanych}/{argumenty.cel})")

        # Krotka przerwa miedzy zadaniami - dobra praktyka przy korzystaniu
        # z darmowych, publicznych serwisow internetowych.
        time.sleep(argumenty.opoznienie)

    if liczba_zapisanych < argumenty.cel:
        print(
            f"\n[uwaga] Zabraklo dostepnych portretow - zebrano {liczba_zapisanych}"
            f"/{argumenty.cel}. Mozesz uruchomic skrypt ponownie, zwiekszy to "
            "pule prob (chociaz pula portretow jest ograniczona do ok. 200)."
        )
    print(f"\n[gotowe] Zebrano {liczba_zapisanych} zdjec klasy '{argumenty.nazwa_klasy}'.")


if __name__ == "__main__":
    main()
