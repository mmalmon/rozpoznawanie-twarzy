"""
Krok 2: Podzial zebranych danych na zbior treningowy i walidacyjny.

DLA UCZNIOW - dlaczego dzielimy dane na dwa zbiory?

Gdybysmy trenowali model na WSZYSTKICH zdjeciach i tez na nich sprawdzali
jego dokladnosc, model moglby po prostu "zapamietac" kazde zdjecie z osobna
(tzw. przeuczenie / overfitting), zamiast nauczyc sie ogolnych cech twarzy.
Wygladalby wtedy na bardzo dokladny, ale zawodzilby na nowych zdjeciach
(np. z innego dnia, innego oswietlenia).

Dlatego dzielimy dane na:
  - zbior TRENINGOWY (train) - na nim model sie uczy (dostosowuje swoje wagi),
  - zbior WALIDACYJNY (val) - model NIGDY nie widzi tych zdjec podczas
    uczenia; sluzy on wylacznie do sprawdzania, czy model radzi sobie z
    danymi, ktorych "nie zna na pamiec".

Wczytuje zdjecia z data/raw/<osoba>/*.jpg i kopiuje je do:
    data/processed/train/<osoba>/...
    data/processed/val/<osoba>/...

wg proporcji ustawionej parametrem --val-split (domyslnie 20% do walidacji).

Uzycie:
    python 02_podzial_danych.py
    python 02_podzial_danych.py --val-split 0.15
"""

from __future__ import annotations

import argparse
import random
import shutil
from pathlib import Path

FOLDER_SUROWYCH_DANYCH = Path(__file__).parent / "data" / "raw"
FOLDER_PRZETWORZONYCH_DANYCH = Path(__file__).parent / "data" / "processed"


def parsuj_argumenty() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Podzial danych na zbior treningowy i walidacyjny.")
    parser.add_argument(
        "--val-split",
        type=float,
        default=0.2,
        help="Ulamek danych przeznaczony na zbior walidacyjny (np. 0.2 = 20%)",
    )
    parser.add_argument("--seed", type=int, default=42, help="Ziarno losowosci podzialu (dla powtarzalnosci)")
    return parser.parse_args()


def main() -> None:
    argumenty = parsuj_argumenty()
    # Ustawienie "ziarna" (seed) generatora liczb losowych sprawia, ze
    # losowy podzial danych bedzie zawsze taki sam przy tym samym ziarnie -
    # to wazne dla powtarzalnosci eksperymentow (np. porownywania wynikow
    # kilku uczniow miedzy soba).
    random.seed(argumenty.seed)

    foldery_osob = [folder for folder in FOLDER_SUROWYCH_DANYCH.iterdir() if folder.is_dir()]
    if not foldery_osob:
        raise SystemExit(
            f"Brak danych w {FOLDER_SUROWYCH_DANYCH}. "
            "Najpierw uruchom 01_zbieranie_danych.py dla kazdej osoby."
        )

    # Czyscimy poprzedni podzial, zeby uniknac mieszania starych i nowych danych.
    if FOLDER_PRZETWORZONYCH_DANYCH.exists():
        shutil.rmtree(FOLDER_PRZETWORZONYCH_DANYCH)

    podsumowanie = []
    for folder_osoby in sorted(foldery_osob):
        zdjecia = sorted(folder_osoby.glob("*.jpg"))
        if len(zdjecia) < 10:
            print(
                f"[uwaga] Osoba '{folder_osoby.name}' ma tylko {len(zdjecia)} zdjec - "
                "to bardzo malo, zalecane min. 100-150."
            )

        # Losowo tasujemy kolejnosc zdjec przed podzialem, zeby zbior
        # walidacyjny nie skladal sie np. wylacznie z ostatnio zrobionych
        # zdjec (ktore moga byc do siebie bardzo podobne, jesli byly zrobione
        # jedno po drugim).
        random.shuffle(zdjecia)
        liczba_walidacyjnych = max(1, int(len(zdjecia) * argumenty.val_split))
        zdjecia_walidacyjne = zdjecia[:liczba_walidacyjnych]
        zdjecia_treningowe = zdjecia[liczba_walidacyjnych:]

        for nazwa_zbioru, zdjecia_zbioru in (
            ("train", zdjecia_treningowe),
            ("val", zdjecia_walidacyjne),
        ):
            folder_wyjsciowy = FOLDER_PRZETWORZONYCH_DANYCH / nazwa_zbioru / folder_osoby.name
            folder_wyjsciowy.mkdir(parents=True, exist_ok=True)
            for sciezka_zdjecia in zdjecia_zbioru:
                shutil.copy2(sciezka_zdjecia, folder_wyjsciowy / sciezka_zdjecia.name)

        podsumowanie.append((folder_osoby.name, len(zdjecia_treningowe), len(zdjecia_walidacyjne)))

    print("\nPodsumowanie podzialu danych:")
    print(f"{'Osoba':<25}{'Trening':>8}{'Walidacja':>11}")
    for nazwa_osoby, liczba_treningowych, liczba_walidacyjnych in podsumowanie:
        print(f"{nazwa_osoby:<25}{liczba_treningowych:>8}{liczba_walidacyjnych:>11}")

    print(f"\nDane gotowe w: {FOLDER_PRZETWORZONYCH_DANYCH}")


if __name__ == "__main__":
    main()
