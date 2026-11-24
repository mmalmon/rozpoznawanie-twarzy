"""
Krok 2: Podzial zebranych danych na zbior treningowy, walidacyjny i testowy.

DLA UCZNIOW - dlaczego dzielimy dane na TRZY zbiory?

Gdybysmy trenowali model na WSZYSTKICH zdjeciach i tez na nich sprawdzali
jego dokladnosc, model moglby po prostu "zapamietac" kazde zdjecie z osobna
(tzw. przeuczenie / overfitting), zamiast nauczyc sie ogolnych cech twarzy.
Wygladalby wtedy na bardzo dokladny, ale zawodzilby na nowych zdjeciach
(np. z innego dnia, innego oswietlenia).

Dlatego dzielimy dane na:
  - zbior TRENINGOWY (train) - na nim model buduje wzorce,
  - zbior WALIDACYJNY (val) - na nim kalibrujemy prog decyzyjny; model NIGDY
    nie uczy sie na tych zdjeciach,
  - zbior TESTOWY (test) - lezy w szufladzie do samego konca. Siegamy po
    niego dopiero w skryptach 07 i 08, zeby zmierzyc ostateczna jakosc.

Po co az trzy? Bo jesli prog ustawimy na val i potem na tym samym val
raportujemy wynik, to wynik jest optymistyczny - prog "dopasowal sie" do
tych zdjec. Test musi byc nietkniety az do finalnego pomiaru.

Wczytuje zdjecia z data/raw/<osoba>/*.jpg i kopiuje je do:
    data/processed/train/<osoba>/...
    data/processed/val/<osoba>/...
    data/processed/test/<osoba>/...

wg proporcji ustawionych parametrami --val-split i --test-split
(domyslnie po 15% na walidacje i test, czyli 70% na trening).

Uzycie:
    python 02_podzial_danych.py
    python 02_podzial_danych.py --val-split 0.15 --test-split 0.15
"""

from __future__ import annotations

# Ta linijka MUSI byc pierwszym kodem wykonywalnym w tym pliku - patrz
# szczegolowe wyjasnienie w common/uruchom_w_venv.py (automatyczne
# przelaczenie na Python z .venv, jesli uruchomiono innym interpreterem).
from common.uruchom_w_venv import przelacz_na_venv_jesli_trzeba

przelacz_na_venv_jesli_trzeba()

# argparse - obsluga argumentow uruchomieniowych (--val-split, --seed).
import argparse
# random - generator liczb (i wyborow) pseudolosowych z biblioteki
# standardowej Pythona; uzywamy go do losowego, ale powtarzalnego
# przemieszania zdjec przed podzialem na train/val/test.
import random
# shutil - operacje na plikach wyzszego poziomu: kopiowanie (copy2) i
# usuwanie calych folderow z zawartoscia (rmtree).
import shutil
# Path - obiektowa reprezentacja sciezek plikow/folderow.
from pathlib import Path

FOLDER_SUROWYCH_DANYCH = Path(__file__).parent / "data" / "raw"
FOLDER_PRZETWORZONYCH_DANYCH = Path(__file__).parent / "data" / "processed"


def parsuj_argumenty() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Podzial danych na zbior treningowy, walidacyjny i testowy."
    )
    parser.add_argument(
        "--val-split",
        type=float,
        default=0.15,
        help="Ulamek danych przeznaczony na zbior walidacyjny (np. 0.15 = 15%)",
    )
    parser.add_argument(
        "--test-split",
        type=float,
        default=0.15,
        help="Ulamek danych przeznaczony na zbior testowy (np. 0.15 = 15%)",
    )
    parser.add_argument("--seed", type=int, default=42, help="Ziarno losowosci podzialu (dla powtarzalnosci)")
    return parser.parse_args()


def main() -> None:
    argumenty = parsuj_argumenty()
    if not 0 < argumenty.val_split < 1:
        raise SystemExit(
            "--val-split musi byc wiekszy od 0 i mniejszy od 1 "
            f"(otrzymano {argumenty.val_split})."
        )
    if not 0 < argumenty.test_split < 1:
        raise SystemExit(
            "--test-split musi byc wiekszy od 0 i mniejszy od 1 "
            f"(otrzymano {argumenty.test_split})."
        )
    if argumenty.val_split + argumenty.test_split >= 1:
        raise SystemExit(
            "--val-split + --test-split musi byc mniejsze od 1, "
            "zeby zostalo chociaz jedno zdjecie na trening "
            f"(otrzymano {argumenty.val_split} + {argumenty.test_split})."
        )
    if not FOLDER_SUROWYCH_DANYCH.exists():
        raise SystemExit(
            f"Brak folderu {FOLDER_SUROWYCH_DANYCH}. "
            "Najpierw uruchom 01_zbieranie_danych.py dla kazdej osoby."
        )

    # Ustawienie "ziarna" (seed) generatora liczb losowych sprawia, ze
    # losowy podzial danych bedzie zawsze taki sam przy tym samym ziarnie -
    # to wazne dla powtarzalnosci eksperymentow (np. porownywania wynikow
    # kilku uczniow miedzy soba).
    random.seed(argumenty.seed)

    foldery_osob = sorted(
        folder for folder in FOLDER_SUROWYCH_DANYCH.iterdir() if folder.is_dir()
    )
    if not foldery_osob:
        raise SystemExit(
            f"Brak danych w {FOLDER_SUROWYCH_DANYCH}. "
            "Najpierw uruchom 01_zbieranie_danych.py dla kazdej osoby."
        )

    # Najpierw sprawdzamy, ktore osoby maja dosc zdjec. Osobe z jednym lub
    # dwoma zdjeciami trzeba pominac: po podziale na trzy zbiory nie mialaby
    # kompletnie nic w ktoryms z nich, a torchvision.ImageFolder wymaga
    # niepustych folderow klas.
    osoby_do_podzialu: list[tuple[Path, list[Path]]] = []
    for folder_osoby in foldery_osob:
        zdjecia = sorted(folder_osoby.glob("*.jpg"))
        if len(zdjecia) < 3:
            print(
                f"[uwaga] Osoba '{folder_osoby.name}' ma tylko {len(zdjecia)} "
                "zdjec - pomijam (potrzeba min. 3, zalecane 100-150)."
            )
            continue
        osoby_do_podzialu.append((folder_osoby, zdjecia))

    if not osoby_do_podzialu:
        raise SystemExit(
            "Zadna osoba nie ma wystarczajacej liczby zdjec (min. 3). "
            "Zbierz wiecej danych skryptem 01_zbieranie_danych.py."
        )

    # Czyscimy poprzedni podzial, zeby uniknac mieszania starych i nowych danych.
    if FOLDER_PRZETWORZONYCH_DANYCH.exists():
        shutil.rmtree(FOLDER_PRZETWORZONYCH_DANYCH)

    # Katalog data/processed jest w .gitignore (bo zawiera duzo zdjec) - w
    # repozytorium trzymamy jedynie pusty plik .gitkeep, zeby sam PUSTY
    # folder byl widoczny w git (Git nie sledzi pustych folderow). rmtree
    # powyzej usuwa go razem z reszta, wiec odtwarzamy go od nowa, aby
    # przypadkowo nie zniknal z repozytorium przy kolejnym "git add".
    FOLDER_PRZETWORZONYCH_DANYCH.mkdir(parents=True, exist_ok=True)
    (FOLDER_PRZETWORZONYCH_DANYCH / ".gitkeep").touch()

    podsumowanie = []
    for folder_osoby, zdjecia in osoby_do_podzialu:
        # Losowo tasujemy kolejnosc zdjec przed podzialem, zeby zbior
        # walidacyjny i testowy nie skladaly sie np. wylacznie z ostatnio
        # zrobionych zdjec (ktore moga byc do siebie bardzo podobne, jesli
        # byly zrobione jedno po drugim).
        random.shuffle(zdjecia)
        # int(...) obcina liczbe do czesci calkowitej (np. 0.15 * 157 = 23.5
        # -> 23 zdjecia). max(1, ...) gwarantuje, ze nawet przy bardzo malej
        # liczbie zdjec w kazdym zbiorze znajdzie sie co najmniej jedno.
        # min(..., n-2) i min(..., n-val-1) zostawiaja przynajmniej jedno
        # zdjecie na trening.
        liczba_walidacyjnych = min(
            max(1, int(len(zdjecia) * argumenty.val_split)), len(zdjecia) - 2
        )
        liczba_testowych = min(
            max(1, int(len(zdjecia) * argumenty.test_split)),
            len(zdjecia) - liczba_walidacyjnych - 1,
        )
        # Wycinanie listy (tzw. "slicing"): [:N] bierze pierwsze N elementow,
        # [N:M] elementy od N do M-1, a [M:] wszystkie POZOSTALE - dzieki temu
        # kazde zdjecie trafia dokladnie do jednego zbioru, nigdy do dwoch.
        zdjecia_walidacyjne = zdjecia[:liczba_walidacyjnych]
        zdjecia_testowe = zdjecia[
            liczba_walidacyjnych : liczba_walidacyjnych + liczba_testowych
        ]
        zdjecia_treningowe = zdjecia[liczba_walidacyjnych + liczba_testowych :]

        # Petla po trzech parach (nazwa_zbioru, lista_zdjec) - zamiast pisac
        # niemal identyczny kod kopiowania trzy razy, robimy to raz w petli.
        for nazwa_zbioru, zdjecia_zbioru in (
            ("train", zdjecia_treningowe),
            ("val", zdjecia_walidacyjne),
            ("test", zdjecia_testowe),
        ):
            folder_wyjsciowy = FOLDER_PRZETWORZONYCH_DANYCH / nazwa_zbioru / folder_osoby.name
            folder_wyjsciowy.mkdir(parents=True, exist_ok=True)
            for sciezka_zdjecia in zdjecia_zbioru:
                # shutil.copy2 kopiuje plik razem z jego metadanymi (np.
                # data modyfikacji) - uzywamy kopiowania, a nie przenoszenia,
                # zeby oryginalne zdjecia w data/raw/ zawsze zostaly
                # nietkniete (mozna bezpiecznie uruchomic ten skrypt ponownie).
                shutil.copy2(sciezka_zdjecia, folder_wyjsciowy / sciezka_zdjecia.name)

        podsumowanie.append(
            (
                folder_osoby.name,
                len(zdjecia_treningowe),
                len(zdjecia_walidacyjne),
                len(zdjecia_testowe),
            )
        )

    print("\nPodsumowanie podzialu danych:")
    print(f"{'Osoba':<25}{'Trening':>8}{'Walidacja':>11}{'Test':>7}")
    for (
        nazwa_osoby,
        liczba_treningowych,
        liczba_walidacyjnych,
        liczba_testowych,
    ) in podsumowanie:
        print(
            f"{nazwa_osoby:<25}{liczba_treningowych:>8}"
            f"{liczba_walidacyjnych:>11}{liczba_testowych:>7}"
        )

    print(f"\nDane gotowe w: {FOLDER_PRZETWORZONYCH_DANYCH}")


if __name__ == "__main__":
    main()
