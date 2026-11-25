"""Rejestr rozpoznan i frekwencji - podglad, eksport i porzadki z konsoli.

DLA UCZNIOW: panel GUI robi to samo klikaniem, ale ten skrypt pokazuje, ze
rejestr to zwykla lokalna baza SQLite. Mozesz ja otworzyc np. programem
DB Browser for SQLite i zobaczyc tabele `rozpoznania`.

Przyklady:
    python 10_rejestr_obecnosci.py
    python 10_rejestr_obecnosci.py --data 2026-11-24
    python 10_rejestr_obecnosci.py --ostatnie 20
    python 10_rejestr_obecnosci.py --eksport frekwencja.csv --data 2026-11-24
    python 10_rejestr_obecnosci.py --wyczysc-dzien 2026-11-24 --tak
    python 10_rejestr_obecnosci.py --wyczysc-wszystko --tak
    python 10_rejestr_obecnosci.py --retencja 90 --tak

Bez `--tak` skrypt tylko pokazuje, co by zrobil - nie usuwa danych.
"""

from __future__ import annotations

# Ta linijka MUSI byc pierwszym kodem wykonywalnym w tym pliku - patrz
# szczegolowe wyjasnienie w common/uruchom_w_venv.py.
from common.uruchom_w_venv import przelacz_na_venv_jesli_trzeba

przelacz_na_venv_jesli_trzeba()

import argparse
from pathlib import Path

from common.rejestr import RejestrObecnosci, dzisiaj, poprawna_data


def _wypisz_dzien(dane: dict) -> None:
    print(f"Frekwencja: {dane['data']}")
    print(f"Obecni: {len(dane['obecni'])}/{len(dane['osoby'])}")
    if dane["obecni"]:
        print(f"  {'OSOBA':<16} {'PIERWSZE':<10} {'OSTATNIE':<10} {'LICZBA':>6}")
        for wpis in dane["obecni"]:
            pierwsze = wpis["pierwsze"][11:19]
            ostatnie = wpis["ostatnie"][11:19]
            print(
                f"  {wpis['osoba']:<16} {pierwsze:<10} {ostatnie:<10} "
                f"{wpis['liczba']:>6}"
            )
    else:
        print("  (brak rozpoznan tego dnia)")

    if dane["nieobecni"]:
        print("Nieobecni: " + ", ".join(dane["nieobecni"]))
    else:
        print("Nieobecni: brak")
    print(f"Razem rozpoznan: {dane['liczba_rozpoznan']}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Rejestr rozpoznan i frekwencji")
    parser.add_argument(
        "--data",
        default=None,
        help="Dzien w formacie RRRR-MM-DD (domyslnie dzisiaj)",
    )
    parser.add_argument(
        "--ostatnie",
        type=int,
        default=None,
        metavar="N",
        help="Pokaz N ostatnich rozpoznan",
    )
    parser.add_argument(
        "--eksport",
        default=None,
        metavar="PLIK",
        help="Zapisz CSV z frekwencja wybranego dnia",
    )
    parser.add_argument(
        "--wyczysc-dzien",
        action="store_true",
        help="Usun wpisy wybranego dnia (wymaga --tak)",
    )
    parser.add_argument(
        "--wyczysc-wszystko",
        action="store_true",
        help="Usun caly rejestr (wymaga --tak)",
    )
    parser.add_argument(
        "--retencja",
        type=int,
        default=None,
        metavar="DNI",
        help="Usun wpisy starsze niz DNI (wymaga --tak)",
    )
    parser.add_argument(
        "--baza",
        default=None,
        metavar="PLIK",
        help="Inna sciezka bazy SQLite (np. do testow)",
    )
    parser.add_argument(
        "--tak",
        action="store_true",
        help="Potwierdzenie operacji usuwajacej dane",
    )
    argumenty = parser.parse_args()

    rejestr = RejestrObecnosci(
        sciezka=Path(argumenty.baza) if argumenty.baza else None
    )
    status = rejestr.status()
    if status["blad"]:
        raise SystemExit(f"[blad] {status['blad']}")

    if argumenty.ostatnie is not None:
        wpisy = rejestr.ostatnie(argumenty.ostatnie)
        if not wpisy:
            print("Rejestr jest pusty.")
            return
        print(f"Ostatnie {len(wpisy)} rozpoznan:")
        print(f"  {'CZAS':<20} {'OSOBA':<16} {'PODOBIENSTWO':>12} {'ZRODLO':<10}")
        for wpis in wpisy:
            podobienstwo = wpis["podobienstwo"]
            tekst = "-" if podobienstwo is None else f"{podobienstwo * 100:.1f}%"
            print(
                f"  {wpis['czas']:<20} {wpis['osoba']:<16} {tekst:>12} "
                f"{wpis['zrodlo']:<10}"
            )
        return

    data = poprawna_data(argumenty.data)
    if data is None:
        raise SystemExit("[blad] Data musi miec format RRRR-MM-DD.")

    if argumenty.wyczysc_wszystko:
        if not argumenty.tak:
            print("[podglad] Usunieto by caly rejestr. Dodaj --tak, zeby wykonac.")
            return
        liczba = rejestr.wyczysc()
        print(f"[gotowe] Usunieto wpisow: {liczba}")
        return

    if argumenty.wyczysc_dzien:
        if not argumenty.tak:
            print(
                f"[podglad] Usunieto by wpisy z dnia {data}. "
                "Dodaj --tak, zeby wykonac."
            )
            return
        liczba = rejestr.usun_dzien(data)
        print(f"[gotowe] Usunieto wpisow z dnia {data}: {liczba}")
        return

    if argumenty.retencja is not None:
        if not argumenty.tak:
            print(
                f"[podglad] Usunieto by wpisy starsze niz {argumenty.retencja} dni. "
                "Dodaj --tak, zeby wykonac."
            )
            return
        liczba = rejestr.usun_starsze(argumenty.retencja)
        print(
            f"[gotowe] Usunieto wpisow starszych niz {argumenty.retencja} dni: "
            f"{liczba}"
        )
        return

    if argumenty.eksport:
        sciezka = Path(argumenty.eksport)
        sciezka.parent.mkdir(parents=True, exist_ok=True)
        sciezka.write_text(rejestr.eksport_csv(data), encoding="utf-8")
        print(f"[gotowe] Zapisano CSV: {sciezka}")
        return

    if argumenty.data is None:
        data = dzisiaj()
    _wypisz_dzien(rejestr.dzien(data))


if __name__ == "__main__":
    main()
