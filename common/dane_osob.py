"""Usuwanie danych osob i ponowne przeliczenie wzorcow bez nich."""

from __future__ import annotations

import json
import re
import shutil
import subprocess
import sys
from pathlib import Path

KORZEN = Path(__file__).resolve().parent.parent
NAZWA_KLASY_NIEZNAJOMY = "nieznajomy"
FOLDER_RAW = KORZEN / "data" / "raw"
FOLDER_RAW_TEST = KORZEN / "data" / "raw_test"
FOLDER_PROCESSED = KORZEN / "data" / "processed"
PLIK_WZORCOW = KORZEN / "models" / "wzorce_osob.json"

# Nieuzywane przez program pliki po starszym treningu - zawieraja dane osob.
STARE_PLIKI_MODELU = ("model_twarzy.pt", "klasy.json", "krzywe_uczenia.png")


def poprawna_nazwa(osoba: str) -> bool:
    return re.fullmatch(r"[\w\-]+", osoba) is not None


def lista_osob() -> list[str]:
    """Osoby, ktorych dane sa na dysku (bez klasy 'nieznajomy')."""
    nazwy = set()
    if FOLDER_RAW.exists():
        nazwy |= {f.name for f in FOLDER_RAW.iterdir() if f.is_dir()}
    if FOLDER_RAW_TEST.exists():
        nazwy |= {f.name for f in FOLDER_RAW_TEST.iterdir() if f.is_dir()}
    if PLIK_WZORCOW.exists():
        try:
            dane = json.loads(PLIK_WZORCOW.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as blad:
            print(f"[uwaga] Nie mozna odczytac {PLIK_WZORCOW}: {blad}")
        else:
            nazwy |= set(dane.get("osoby", {}))
    return sorted(n for n in nazwy if n != NAZWA_KLASY_NIEZNAJOMY and poprawna_nazwa(n))


def nastepny_numer_zdjecia(folder: Path, prefiks: str) -> int:
    """Zwraca pierwszy wolny numer pliku, np. dla zdjec 0000-0004 i 0009
    zwroci 10. Dzieki temu dogrywanie danych po recznym usunieciu czesci
    zdjec NIE nadpisuje istniejacych plikow."""
    wzorzec = re.compile(rf"^{re.escape(prefiks)}_(\d+)\.jpg$", re.IGNORECASE)
    numery = [
        int(dopasowanie.group(1))
        for plik in folder.glob(f"{prefiks}_*.jpg")
        if (dopasowanie := wzorzec.match(plik.name)) is not None
    ]
    return max(numery, default=-1) + 1


def sciezki_do_usuniecia(osoba: str) -> list[Path]:
    kandydaci = [
        FOLDER_RAW / osoba,
        FOLDER_RAW_TEST / osoba,
        FOLDER_PROCESSED / "train" / osoba,
        FOLDER_PROCESSED / "val" / osoba,
        FOLDER_PROCESSED / "test" / osoba,
    ]
    return [s for s in kandydaci if s.exists()]


def usun_dane_osoby(osoba: str) -> None:
    if not poprawna_nazwa(osoba):
        raise ValueError(f"Niepoprawna nazwa osoby: {osoba!r}")
    for sciezka in sciezki_do_usuniecia(osoba):
        shutil.rmtree(sciezka)


def usun_stare_pliki_modelu() -> list[Path]:
    usuniete = []
    for nazwa in STARE_PLIKI_MODELU:
        sciezka = KORZEN / "models" / nazwa
        if sciezka.exists():
            sciezka.unlink()
            usuniete.append(sciezka)
    return usuniete


def przetrenuj_bez_usunietych() -> bool:
    """Przebudowuje podzial danych i wzorce (02 + 03) z pozostalych danych.
    Gdy nie zostala zadna osoba, kasuje wzorce. Zwraca True przy sukcesie."""
    zostali = [
        f for f in FOLDER_RAW.iterdir() if f.is_dir() and f.name != NAZWA_KLASY_NIEZNAJOMY
    ] if FOLDER_RAW.exists() else []
    if not zostali:
        PLIK_WZORCOW.unlink(missing_ok=True)
        shutil.rmtree(FOLDER_PROCESSED, ignore_errors=True)
        print("[info] Nie zostala zadna osoba - usunieto wzorce i podzial danych.")
        return True

    for skrypt in ("02_podzial_danych.py", "03_trenowanie_modelu.py"):
        print(f"[info] Uruchamiam {skrypt}...")
        wynik = subprocess.run([sys.executable, str(KORZEN / skrypt)], cwd=KORZEN)
        if wynik.returncode != 0:
            print(f"[blad] {skrypt} zakonczyl sie bledem ({wynik.returncode}).")
            return False
    return True
