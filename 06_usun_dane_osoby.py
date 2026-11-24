"""Usuwa dane wybranych osob (np. po lekcji z pokazem) i przelicza wzorce.

Kasuje: data/raw/<osoba>, data/processed/{train,val,test}/<osoba>, nieuzywane
stare pliki modelu, a potem uruchamia ponownie 02_podzial_danych.py i
03_trenowanie_modelu.py na pozostalych danych (wzorce nie zawieraja juz
usunietych osob). Domyslnie tylko POKAZUJE, co zostanie usuniete;
usuwa dopiero z --tak.

Uruchomienie: python 06_usun_dane_osoby.py NAZWA [NAZWA ...] [--tak]
"""

from __future__ import annotations

# Ta linijka MUSI byc pierwszym kodem wykonywalnym w tym pliku - patrz
# szczegolowe wyjasnienie w common/uruchom_w_venv.py (automatyczne
# przelaczenie na Python z .venv, jesli uruchomiono innym interpreterem).
from common.uruchom_w_venv import przelacz_na_venv_jesli_trzeba

przelacz_na_venv_jesli_trzeba()

import argparse

from common.dane_osob import (
    poprawna_nazwa,
    przetrenuj_bez_usunietych,
    sciezki_do_usuniecia,
    usun_dane_osoby,
    usun_stare_pliki_modelu,
)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("osoby", nargs="+")
    parser.add_argument("--tak", action="store_true", help="faktycznie usun dane")
    argumenty = parser.parse_args()

    for osoba in argumenty.osoby:
        if not poprawna_nazwa(osoba):
            raise SystemExit(f"Niepoprawna nazwa: {osoba!r} (tylko litery, cyfry, _ i -).")

    do_usuniecia = {o: sciezki_do_usuniecia(o) for o in argumenty.osoby}
    for osoba, sciezki in do_usuniecia.items():
        for sciezka in sciezki:
            print(f"[do usuniecia] {sciezka}")
    if not any(do_usuniecia.values()):
        print("Nic do usuniecia.")
        return
    if not argumenty.tak:
        print("To tylko podglad. Dodaj --tak, aby usunac i przeliczyc wzorce.")
        return

    for osoba in argumenty.osoby:
        usun_dane_osoby(osoba)
    usun_stare_pliki_modelu()
    if przetrenuj_bez_usunietych():
        print("Gotowe: dane usuniete, wzorce przeliczone bez tych osob.")
    else:
        raise SystemExit(
            "Dane usunieto, ale przeliczenie wzorcow nie powiodlo sie - "
            "sprawdz komunikaty powyzej."
        )


if __name__ == "__main__":
    main()
