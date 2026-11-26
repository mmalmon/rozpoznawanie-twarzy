"""Krok 9: Wyjasnianie decyzji modelu (XAI) dla zapisanego zdjecia.

DLA UCZNIOW: siec neuronowa nie podaje gotowego powodu decyzji. Zamiast tego
robimy dwa eksperymenty, ktore mozna zobaczyc i zmierzyc:

1. **Najblizsze wzorce** - szukamy zdjec treningowych, ktorych embedding jest
   najbardziej podobny do embeddingu badanej twarzy. To odpowiedz na pytanie
   "na co model patrzyl?".
2. **Mapa okluzji** - zaslaniamy kolejne fragmenty twarzy szarym kwadratem i
   mierzymy, o ile spada podobienstwo do wybranej osoby. Fragmenty, ktorych
   zasloniecie najbardziej psuje wynik, sa dla modelu najwazniejsze.

Ten skrypt to konsolowy odpowiednik karty "Wyjasnienie decyzji (XAI)" w panelu
GUI. Zapisuje raport JSON i wizualizacje PNG (oryginal + mapa cieplna).

Uzycie:
    python 11_wyjasnij.py --zdjecie data/processed/train/gracjan/gracjan_0000.jpg
    python 11_wyjasnij.py --zdjecie moje_zdjecie.jpg --top 8 --siatka 8
    python 11_wyjasnij.py --zdjecie moje_zdjecie.jpg --bez-zapisu

Wymaga wczesniejszego zbudowania wzorcow (03_trenowanie_modelu.py). Galeria
"najblizszych wzorcow" pojawi sie dopiero po ponownym treningu, ktory zapisuje
metadane `zrodla` w models/wzorce_osob.json.
"""

from __future__ import annotations

# Ta linijka MUSI byc pierwszym kodem wykonywalnym w tym pliku - patrz
# szczegolowe wyjasnienie w common/uruchom_w_venv.py.
from common.uruchom_w_venv import przelacz_na_venv_jesli_trzeba

przelacz_na_venv_jesli_trzeba()

import argparse
import json
from datetime import datetime
from pathlib import Path

from common.face_detector import DetektorTwarzy
from common.imgio import wczytaj_obraz
from common.model import pobierz_urzadzenie, zbuduj_ekstraktor_cech
from common.rozpoznawanie import (
    BrakWzorcowError,
    NiepoprawneWzorceError,
    przygotuj_obrazy_kamery,
    wczytaj_wzorce,
)
from common.xai import FOLDER_XAI, wczytaj_zrodla_wzorcow, wyjasnij_twarz

NAZWA_RAPORTU = "raport_xai.json"


def parsuj_argumenty() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Wyjasnianie decyzji modelu rozpoznawania twarzy (XAI)."
    )
    parser.add_argument(
        "--zdjecie",
        required=True,
        metavar="PLIK",
        help="Zdjecie z twarza (JPG/PNG).",
    )
    parser.add_argument(
        "--top",
        type=int,
        default=5,
        metavar="N",
        help="Ile najblizszych wzorcow pokazac (1-20, domyslnie 5).",
    )
    parser.add_argument(
        "--siatka",
        type=int,
        default=6,
        metavar="N",
        help="Siatka mapy okluzji, np. 6 = 6x6 fragmentow (2-10, domyslnie 6).",
    )
    parser.add_argument(
        "--korekta-progu",
        type=float,
        default=0.0,
        help="Taka sama korekta progu jak w panelu (-0.5 do 0.5).",
    )
    parser.add_argument(
        "--bez-zapisu",
        action="store_true",
        help="Nie zapisuj wizualizacji PNG (raport JSON nadal powstanie).",
    )
    argumenty = parser.parse_args()

    if not 1 <= argumenty.top <= 20:
        parser.error("--top musi byc w zakresie 1-20.")
    if not 2 <= argumenty.siatka <= 10:
        parser.error("--siatka musi byc w zakresie 2-10.")
    if not -0.5 <= argumenty.korekta_progu <= 0.5:
        parser.error("--korekta-progu musi byc w zakresie -0.5 do 0.5.")
    return argumenty


def _wybierz_twarz(obraz, detektor: DetektorTwarzy):
    """Zwraca prostokat najwiekszej twarzy albo None (cale zdjecie)."""
    twarze = detektor.wykryj(obraz)
    if not twarze:
        return None, 0
    # Najwieksza twarz = osoba najblizej kamery = ta, ktora warto wyjasnic.
    twarz = max(twarze, key=lambda wykryta: wykryta.w * wykryta.h)
    prostokat = (int(twarz.x), int(twarz.y), int(twarz.w), int(twarz.h))
    return prostokat, len(twarze)


def _wypisz_wynik(wynik: dict) -> None:
    rozpoznanie = "rozpoznana" if wynik["rozpoznana"] else "PONIZEJ PROGU"
    prog = wynik.get("prog")
    prog_tekst = "-" if prog is None else f"{prog * 100:.1f}%"

    print()
    print(f"[wynik] Osoba: {wynik['osoba']} ({rozpoznanie})")
    print(
        f"[wynik] Podobienstwo: {wynik['podobienstwo'] * 100:.1f}% "
        f"(prog {prog_tekst})"
    )
    print(f"[wynik] Prototypow w modelu: {wynik['liczba_wzorcow']}")
    print(f"[wynik] Mapa okluzji: {wynik['opis']}")
    print(
        "[wynik] Najwiekszy spadek podobienstwa: "
        f"{wynik['maks_spadek'] * 100:.1f}%"
    )
    print(f"[wynik] Czas obliczen: {wynik['czas_ms']} ms")

    print("[wynik] Ranking kandydatow:")
    for kandydat in wynik.get("kandydaci", []):
        znacznik = "OK" if kandydat["akceptacja"] else "--"
        print(
            f"  {znacznik} {kandydat['osoba']:<16} "
            f"{kandydat['podobienstwo'] * 100:5.1f}% / "
            f"prog {kandydat['prog'] * 100:5.1f}%"
        )

    print("[wynik] Najblizsze wzorce treningowe:")
    wzorce = wynik.get("wzorce") or []
    if not wzorce:
        print(
            "  (brak metadanych `zrodla` - uruchom ponownie "
            "03_trenowanie_modelu.py)"
        )
        return
    for numer, wzorzec in enumerate(wzorce, start=1):
        plik = wzorzec.get("plik") or "(brak sciezki)"
        jasnosc = wzorzec.get("jasnosc")
        jasnosc_tekst = "-" if jasnosc is None else f"{jasnosc:.2f}"
        print(
            f"  {numer}. {wzorzec['osoba']:<16} "
            f"{wzorzec['podobienstwo'] * 100:5.1f}%  "
            f"jasnosc {jasnosc_tekst}  {plik}"
        )


def _zapisz_raport(
    zdjecie: Path, wynik: dict, argumenty: argparse.Namespace
) -> Path:
    FOLDER_XAI.mkdir(parents=True, exist_ok=True)
    raport = {
        "zdjecie": str(zdjecie.resolve()),
        "data": datetime.now().isoformat(timespec="seconds"),
        "parametry": {
            "top": argumenty.top,
            "siatka": argumenty.siatka,
            "korekta_progu": argumenty.korekta_progu,
        },
        "wynik": wynik,
    }
    sciezka = FOLDER_XAI / NAZWA_RAPORTU
    sciezka.write_text(
        json.dumps(raport, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    return sciezka


def main() -> None:
    argumenty = parsuj_argumenty()

    zdjecie = Path(argumenty.zdjecie)
    obraz = wczytaj_obraz(zdjecie)
    if obraz is None:
        raise SystemExit(f"[blad] Nie mozna wczytac zdjecia: {zdjecie}")

    urzadzenie = pobierz_urzadzenie()
    print(f"[info] Zdjecie: {zdjecie}")
    print(f"[info] Urzadzenie: {urzadzenie}")

    try:
        wzorce = wczytaj_wzorce(urzadzenie=urzadzenie)
    except (BrakWzorcowError, NiepoprawneWzorceError) as blad:
        raise SystemExit(f"[blad] {blad}") from blad

    zrodla = wczytaj_zrodla_wzorcow()
    if not zrodla:
        print(
            "[uwaga] Plik wzorcow nie ma metadanych `zrodla` (stary trening). "
            "Mapa okluzji zadziala, ale galeria wzorcow bedzie pusta."
        )

    ekstraktor = zbuduj_ekstraktor_cech().to(urzadzenie)
    przygotuj_obrazy = przygotuj_obrazy_kamery()

    detektor = DetektorTwarzy()
    prostokat, liczba_twarzy = _wybierz_twarz(obraz, detektor)
    if prostokat is None:
        print(
            "[uwaga] Nie wykryto twarzy - traktuje cale zdjecie jako wycinek "
            "twarzy (przydatne dla gotowych kadrow 224x224)."
        )
    else:
        print(
            f"[info] Wykryto twarzy: {liczba_twarzy}. Wyjasniam najwieksza: "
            f"x={prostokat[0]}, y={prostokat[1]}, "
            f"w={prostokat[2]}, h={prostokat[3]}."
        )

    try:
        wynik = wyjasnij_twarz(
            obraz,
            ekstraktor,
            przygotuj_obrazy,
            wzorce,
            urzadzenie,
            zrodla=zrodla,
            korekta_progu=argumenty.korekta_progu,
            prostokat=prostokat,
            top=argumenty.top,
            siatka=argumenty.siatka,
            zapisz=not argumenty.bez_zapisu,
        )
    except (ValueError, RuntimeError) as blad:
        raise SystemExit(f"[blad] {blad}") from blad

    _wypisz_wynik(wynik)

    raport = _zapisz_raport(zdjecie, wynik, argumenty)
    print(f"[gotowe] Raport JSON: {raport}")

    obraz_url = wynik.get("obraz_url") or ""
    if obraz_url:
        nazwa = obraz_url.rsplit("/", 1)[-1]
        print(f"[gotowe] Wizualizacja: {FOLDER_XAI / nazwa}")


if __name__ == "__main__":
    main()
