"""
Test trudny (09_test_trudny.py): czy model dziala poza "latwym" zbiorem testowym?

DLA UCZNIOW - najwazniejsze pytanie weryfikacji modelu.

W skrypcie 07_ewaluacja.py zmierzylismy model na zbiorze testowym, ktory
powstal z tych samych sesji zdjec co trening: ten sam dzien, to samo swiatlo,
ta sama kamera. Wynik 100% mowi wiec tylko tyle, ze "ten konkretny test nie
pokazal bledu". Nie mowi, co sie stanie jutro, w innym pomieszczeniu,
przy gorszym swietle albo z innym urzadzeniem.

Ten skrypt ocenia model na CELOWO TRUDNIEJSZYCH zdjeciach:
  - zebranych w innym dniu niz trening,
  - przy innym oswietleniu,
  - pod innym katem,
  - ewentualnie inna kamera.

Takie zdjecia zbierasz skryptem 01 z parametrem --folder:

    python 01_zbieranie_danych.py \\\
        --osoba gracjan --folder data/raw_test --cel 40

albo w panelu GUI (zakladka "Dodaj twarz" -> "Zapisz jako test trudny").
Folder data/raw_test NIE jest mieszany z data/raw, wiec te zdjecia nigdy
nie trafiaja do treningu ani do kalibracji progu. To warunek uczciwego
pomiaru.

Co porownujemy?
  - accuracy (zamknieta i otwarta),
  - FAR / FRR / EER / AUC,
  - przedzialy ufnosci 95% (Wilson dla odsetkow, bootstrap dla EER/AUC),
  - wynik na latwym tescie z raportu 07 (jesli istnieje) i roznice.

Wazna lekcja: jesli wynik na trudnym tescie jest wyraznie gorszy, to nie
jest "awaria" - to prawdziwa informacja o modelu. Wlasnie po to zbiera sie
trudne dane. Model, ktory dziala tylko na latwych zdjeciach, nie jest
gotowy do uzycia.

Uzycie:
    python 09_test_trudny.py
    python 09_test_trudny.py --postep
    python 09_test_trudny.py --folder data/raw_test --n-bootstrap 600

Wynik:
    models/test_trudny/raport_test_trudny.json
    models/test_trudny/wykresy/*.png
"""

from __future__ import annotations

# Ta linijka MUSI byc pierwszym kodem wykonywalnym w tym pliku - patrz
# szczegolowe wyjasnienie w common/uruchom_w_venv.py.
from common.uruchom_w_venv import przelacz_na_venv_jesli_trzeba

przelacz_na_venv_jesli_trzeba()

import argparse
import json
import time
from datetime import datetime
from pathlib import Path

import numpy as np
import torch

from common.embeddingi import policz_embeddingi_folderu, policz_obrazy, transformacja
from common.metryki import (
    accuracy_zamknieta,
    auc,
    decyzje_prototypy,
    krzywa_i_eer,
    metryki_openset,
    podobienstwa_do_osob,
    przedzialy_ufnosci,
)
from common.model import pobierz_urzadzenie, zbuduj_ekstraktor_cech
from common.rozpoznawanie import NAZWA_NIEZNANEJ, wczytaj_wzorce

KORZEN_PROJEKTU = Path(__file__).resolve().parent
FOLDER_TESTOWY = KORZEN_PROJEKTU / "data" / "raw_test"
FOLDER_MODELI = KORZEN_PROJEKTU / "models"
PLIK_WZORCOW = FOLDER_MODELI / "wzorce_osob.json"
PLIK_EWALUACJI = FOLDER_MODELI / "ewaluacja" / "raport_ewaluacji.json"
FOLDER_RAPORTU = FOLDER_MODELI / "test_trudny"
FOLDER_WYKRESOW = FOLDER_RAPORTU / "wykresy"
PLIK_RAPORTU = FOLDER_RAPORTU / "raport_test_trudny.json"

NAZWA_OBCEGO = "obcy (nieznajomy)"
WERSJA_RAPORTU = 1


# =========================================================================
# Postep (te same maszynowe linie co w 03, 07 i 08 - panel GUI je rozumie)
# =========================================================================
class Postep:
    """Wypisuje linie "[postep-start] faza total" i "[postep] faza zrobione"."""

    def __init__(self, wlaczony: bool) -> None:
        self.wlaczony = wlaczony
        self.faza = ""
        self.zrobione = 0
        self.total = 0

    def start(self, faza: str, total: int) -> None:
        self.faza = faza
        self.zrobione = 0
        self.total = max(0, int(total))
        if self.wlaczony:
            print(f"[postep-start] {faza} {self.total}", flush=True)

    def dodaj(self, ile: int) -> None:
        if not self.wlaczony:
            return
        self.zrobione += max(0, int(ile))
        print(f"[postep] {self.faza} {self.zrobione}", flush=True)


def parsuj_argumenty() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Test trudny: ocena modelu na zdjeciach z innych warunkow "
            "(data/raw_test), z porownaniem do latwego testu z kroku 7."
        )
    )
    parser.add_argument(
        "--folder",
        default="data/raw_test",
        help="Folder ze zdjeciami trudnymi (podfoldery = osoby, np. nieznajomy)",
    )
    parser.add_argument(
        "--batch-size", type=int, default=32, help="Ile zdjec naraz trafia do sieci"
    )
    parser.add_argument(
        "--n-bootstrap",
        type=int,
        default=400,
        help="Ile razy losowac probe w przedzialach ufnosci EER/AUC",
    )
    parser.add_argument("--seed", type=int, default=42, help="Ziarno losowosci")
    parser.add_argument(
        "--postep",
        action="store_true",
        help="wypisuje maszynowe linie postepu dla panelu GUI",
    )
    return parser.parse_args()


# =========================================================================
# Pomocnicze
# =========================================================================
def _format_ci(przedzial: list[float] | None, procent: bool = True) -> str:
    if not przedzial:
        return "brak przedzialu"
    dolny, gorny = przedzial
    if procent:
        return f"95% CI {dolny:.1%}-{gorny:.1%}"
    return f"95% CI {dolny:.3f}-{gorny:.3f}"


def _macierz_pomylek(
    prawdziwe: list[str], przewidziane: list[str], osoby: list[str]
) -> dict:
    """Macierz: wiersze = prawda, kolumny = decyzja modelu.

    Wiersze to osoby z modelu plus wiersz "obcy" (zdjecia folderow spoza
    bazy). Kolumny to osoby plus "nieznana osoba" - model nie ma etykiety
    "obcy", wiec kazde odrzucenie obcego laduje w kolumnie "nieznana".
    """
    wiersze = osoby + [NAZWA_OBCEGO]
    kolumny = osoby + [NAZWA_NIEZNANEJ]
    indeks_wierszy = {nazwa: numer for numer, nazwa in enumerate(wiersze)}
    indeks_kolumn = {nazwa: numer for numer, nazwa in enumerate(kolumny)}
    macierz = [[0 for _ in kolumny] for _ in wiersze]

    for prawdziwa, przewidziana in zip(prawdziwe, przewidziane):
        if prawdziwa not in indeks_wierszy:
            continue
        kolumna = przewidziana if przewidziana in indeks_kolumn else NAZWA_NIEZNANEJ
        macierz[indeks_wierszy[prawdziwa]][indeks_kolumn[kolumna]] += 1

    return {"wiersze": wiersze, "kolumny": kolumny, "wartosci": macierz}


def _wczytaj_latwy_test() -> dict | None:
    """Bierze wynik prototypow z raportu 07, jesli jest dostepny."""
    if not PLIK_EWALUACJI.is_file():
        return None
    try:
        raport = json.loads(PLIK_EWALUACJI.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    prototypy = raport.get("prototypy")
    dane = raport.get("dane") or {}
    test = dane.get("test") or {}
    if not isinstance(prototypy, dict) or "metryki" not in prototypy:
        return None
    return {
        "zbior": "data/processed/test",
        "zdjecia_znanych": test.get("zdjecia_znanych"),
        "zdjecia_obcych": (dane.get("obcy") or {}).get("zdjecia"),
        "metryki": prototypy.get("metryki"),
        "accuracy_zamknieta": prototypy.get("accuracy_zamknieta"),
        "eer": prototypy.get("eer"),
        "auc": prototypy.get("auc"),
        "przedzialy_ufnosci": prototypy.get("przedzialy_ufnosci"),
    }


def _roznice(latwy: dict | None, trudny: dict) -> dict | None:
    """Roznica trudny - latwy dla wspolnych metryk (ujemna = gorzej)."""
    if not latwy:
        return None
    wynik: dict[str, float] = {}
    for klucz in ("accuracy", "far", "frr"):
        stara = (latwy.get("metryki") or {}).get(klucz)
        nowa = (trudny.get("metryki") or {}).get(klucz)
        if stara is not None and nowa is not None:
            wynik[klucz] = round(float(nowa) - float(stara), 4)
    for klucz in ("eer", "auc"):
        stara = latwy.get(klucz)
        nowa = trudny.get(klucz)
        if stara is not None and nowa is not None:
            wynik[klucz] = round(float(nowa) - float(stara), 4)
    return wynik or None


# =========================================================================
# Wykresy
# =========================================================================
def _matplotlib():
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    return plt


def _wykres_podsumowania(
    metryki: dict, ci: dict, eer: float | None, sciezka: Path
) -> None:
    plt = _matplotlib()
    etykiety = ["Accuracy", "FAR", "FRR", "EER"]
    wartosci = [
        metryki["accuracy"],
        metryki["far"] if metryki["far"] is not None else 0.0,
        metryki["frr"],
        eer if eer is not None else 0.0,
    ]
    przedzialy = [ci["metryki"].get(klucz) for klucz in ("accuracy", "far", "frr", "eer")]
    dolne = [max(0.0, w - (p[0] if p else w)) for w, p in zip(wartosci, przedzialy)]
    gorne = [max(0.0, (p[1] if p else w) - w) for w, p in zip(wartosci, przedzialy)]

    rysunek, os = plt.subplots(figsize=(8.5, 4.5))
    os.bar(etykiety, wartosci, color="#0d6efd", width=0.55)
    os.errorbar(
        range(len(etykiety)),
        wartosci,
        yerr=np.vstack([dolne, gorne]),
        fmt="none",
        ecolor="#212529",
        capsize=5,
        linewidth=1.2,
    )
    os.set_ylim(0, 1)
    os.set_ylabel("wartosc")
    os.set_title("Test trudny - wynik z 95% przedzialem ufnosci")
    os.grid(axis="y", alpha=0.25)
    for numer, wartosc in enumerate(wartosci):
        os.text(
            numer,
            min(0.97, wartosc + 0.04),
            f"{wartosc:.1%}",
            ha="center",
            fontsize=9,
        )
    rysunek.tight_layout()
    rysunek.savefig(sciezka, dpi=140)
    plt.close(rysunek)


def _wykres_porownania(
    latwy: dict | None, trudny: dict, sciezka: Path
) -> None:
    plt = _matplotlib()
    grupy = ["Accuracy", "FAR", "EER"]
    klucze = ("accuracy", "far", "eer")

    def _wartosc(zrodlo: dict | None, klucz: str) -> float:
        if not zrodlo:
            return 0.0
        wartosc = (zrodlo.get("metryki") or {}).get(klucz)
        if wartosc is None:
            # EER i AUC sa w raporcie na poziomie glownym, nie w "metryki".
            wartosc = zrodlo.get(klucz)
        return float(wartosc) if wartosc is not None else 0.0

    latwe = [_wartosc(latwy, klucz) for klucz in klucze]
    trudne = [_wartosc(trudny, klucz) for klucz in klucze]
    x = np.arange(len(grupy))
    szerokosc = 0.36

    rysunek, os = plt.subplots(figsize=(8.5, 4.5))
    os.bar(x - szerokosc / 2, latwe, szerokosc, label="test latwy (07)", color="#adb5bd")
    os.bar(x + szerokosc / 2, trudne, szerokosc, label="test trudny (09)", color="#dc3545")
    os.set_xticks(x, grupy)
    os.set_ylim(0, 1.05)
    os.set_ylabel("wartosc")
    os.set_title("Ten sam model, inne warunki")
    os.legend()
    os.grid(axis="y", alpha=0.25)
    for numer, wartosc in enumerate(latwe):
        os.text(numer - szerokosc / 2, wartosc + 0.02, f"{wartosc:.0%}", ha="center", fontsize=8)
    for numer, wartosc in enumerate(trudne):
        os.text(numer + szerokosc / 2, wartosc + 0.02, f"{wartosc:.0%}", ha="center", fontsize=8)
    rysunek.tight_layout()
    rysunek.savefig(sciezka, dpi=140)
    plt.close(rysunek)


def _wykres_rozkladu(
    wlasne: np.ndarray, obce: np.ndarray, prog: float | None, sciezka: Path
) -> None:
    plt = _matplotlib()
    rysunek, os = plt.subplots(figsize=(8.5, 4.2))
    zakres = (0.0, 1.0)
    if len(wlasne):
        os.hist(wlasne, bins=20, range=zakres, alpha=0.6, label="swoi (podobienstwo do wlasnej osoby)", color="#0d6efd")
    if len(obce):
        os.hist(obce, bins=20, range=zakres, alpha=0.6, label="obcy (najwyzsze podobienstwo)", color="#dc3545")
    if prog is not None:
        os.axvline(prog, color="#198754", linestyle="--", label=f"prog EER = {prog:.2f}")
    os.set_xlabel("podobienstwo kosinusowe")
    os.set_ylabel("liczba zdjec")
    os.set_title("Test trudny - rozklad podobienstw")
    os.legend(fontsize=8)
    rysunek.tight_layout()
    rysunek.savefig(sciezka, dpi=140)
    plt.close(rysunek)


def _wykres_macierzy(macierz: dict, sciezka: Path) -> None:
    plt = _matplotlib()
    wartosci = np.array(macierz["wartosci"], dtype=int)
    wiersze = macierz["wiersze"]
    kolumny = macierz["kolumny"]

    rysunek, os = plt.subplots(
        figsize=(max(5.0, 1.1 * len(kolumny) + 2), max(4.0, 1.0 * len(wiersze) + 1.5))
    )
    obraz = os.imshow(wartosci, cmap="Blues", vmin=0)
    rysunek.colorbar(obraz, ax=os, fraction=0.046, pad=0.04)
    os.set_xticks(range(len(kolumny)), kolumny, rotation=30, ha="right")
    os.set_yticks(range(len(wiersze)), wiersze)
    os.set_xlabel("decyzja modelu")
    os.set_ylabel("prawdziwa osoba")
    os.set_title("Test trudny - macierz pomylek")

    maksimum = max(1, int(wartosci.max()))
    for wiersz in range(len(wiersze)):
        for kolumna in range(len(kolumny)):
            wartosc = int(wartosci[wiersz, kolumna])
            kolor = "white" if wartosc > maksimum / 2 else "#1f2937"
            os.text(kolumna, wiersz, str(wartosc), ha="center", va="center", color=kolor)

    rysunek.tight_layout()
    rysunek.savefig(sciezka, dpi=140)
    plt.close(rysunek)


# =========================================================================
# Glowna procedura
# =========================================================================
def main() -> None:
    argumenty = parsuj_argumenty()
    if argumenty.batch_size <= 0:
        raise SystemExit("--batch-size musi byc dodatni.")
    if argumenty.n_bootstrap <= 0:
        raise SystemExit("--n-bootstrap musi byc dodatnie.")

    folder_testu = Path(argumenty.folder)
    if not folder_testu.is_absolute():
        folder_testu = KORZEN_PROJEKTU / folder_testu
    if not folder_testu.is_dir():
        raise SystemExit(
            f"Brak folderu {folder_testu}. Zbierz zdjecia trudne skryptem 01 "
            "z opcja --folder data/raw_test (albo w panelu: Dodaj twarz -> "
            "test trudny)."
        )
    podfoldery = [
        folder
        for folder in sorted(folder_testu.iterdir())
        if folder.is_dir() and policz_obrazy(folder) > 0
    ]
    if not podfoldery:
        raise SystemExit(
            f"Folder {folder_testu} nie zawiera zadnych zdjec w podfolderach. "
            "Podfoldery to nazwy osob (np. gracjan, nieznajomy)."
        )
    if not PLIK_WZORCOW.exists():
        raise SystemExit(
            f"Brak pliku {PLIK_WZORCOW}. Najpierw uruchom 03_trenowanie_modelu.py."
        )

    FOLDER_RAPORTU.mkdir(parents=True, exist_ok=True)
    FOLDER_WYKRESOW.mkdir(parents=True, exist_ok=True)
    start = time.perf_counter()

    torch.manual_seed(argumenty.seed)
    np.random.seed(argumenty.seed)

    urzadzenie = pobierz_urzadzenie()
    ekstraktor = zbuduj_ekstraktor_cech().to(urzadzenie)
    wzorce = wczytaj_wzorce(urzadzenie=torch.device("cpu"))
    osoby = sorted(wzorce)
    if not osoby:
        raise SystemExit("Plik wzorcow nie zawiera zadnej osoby.")
    progi = {osoba: float(wzorce[osoba]["prog"]) for osoba in osoby}

    print(f"[info] Model zawiera osoby: {osoby}")
    print(f"[info] Test trudny: {folder_testu}")
    print(f"[info] Urzadzenie: {urzadzenie}")

    liczba_zdjec = policz_obrazy(folder_testu)
    postep = Postep(argumenty.postep)
    postep.start("embeddingi", liczba_zdjec)

    print("[info] Liczenie embeddingow testu trudnego...")
    embeddingi, etykiety, klasy, sciezki = policz_embeddingi_folderu(
        ekstraktor,
        folder_testu,
        urzadzenie,
        argumenty.batch_size,
        transformacja(1.0),
        postep,
    )
    prawdziwe = [klasy[etykieta] for etykieta in etykiety]
    indeksy_znane = [
        numer for numer, prawdziwa in enumerate(prawdziwe) if prawdziwa in osoby
    ]
    indeksy_obce = [
        numer for numer, prawdziwa in enumerate(prawdziwe) if prawdziwa not in osoby
    ]
    if not indeksy_znane:
        raise SystemExit(
            "W tescie trudnym nie ma zadnej osoby z pliku wzorcow. "
            f"Foldery w zbiorze: {klasy}. Nazwy musza byc takie same jak w bazie."
        )

    brakujace = [osoba for osoba in osoby if osoba not in klasy]
    znane_bez_modelu = [nazwa for nazwa in klasy if nazwa not in osoby]
    print(
        f"[info] Zdjecia znanych osob: {len(indeksy_znane)}, "
        f"zdjecia obcych: {len(indeksy_obce)}."
    )
    if brakujace:
        print(f"[uwaga] Brak zdjec w tescie dla osob: {brakujace}")
    if znane_bez_modelu:
        print(
            f"[info] Foldery spoza modelu traktuje jako obcych: {znane_bez_modelu}"
        )
    if not indeksy_obce:
        print(
            "[uwaga] Brak zdjec obcych - nie policze FAR, EER ani AUC. "
            "Dodaj folder 'nieznajomy' z nowymi osobami."
        )

    embeddingi_znane = embeddingi[indeksy_znane]
    prawdziwe_znane = [prawdziwe[numer] for numer in indeksy_znane]
    indeksy_prawdziwe = [osoby.index(osoba) for osoba in prawdziwe_znane]
    embeddingi_obce = (
        embeddingi[indeksy_obce]
        if indeksy_obce
        else torch.empty((0, embeddingi.shape[1]))
    )

    # ---------------------------------------------------------- metryki
    postep.start("metryki", 4)
    print("[info] Prototypy z pliku wzorcow + progi kalibrowane w 03...")
    podobienstwa_znane = podobienstwa_do_osob(embeddingi_znane, wzorce, osoby)
    podobienstwa_obce = podobienstwa_do_osob(embeddingi_obce, wzorce, osoby)
    przewidziane_znane, _ = decyzje_prototypy(podobienstwa_znane, osoby, progi)
    przewidziane_obce, _ = decyzje_prototypy(podobienstwa_obce, osoby, progi)
    metryki = metryki_openset(prawdziwe_znane, przewidziane_znane, przewidziane_obce)

    indeksy = np.arange(len(embeddingi_znane))
    wlasne = podobienstwa_znane[indeksy, indeksy_prawdziwe]
    obce = (
        podobienstwa_obce.max(axis=1)
        if len(podobienstwa_obce)
        else np.zeros(0, dtype=float)
    )
    krzywa = krzywa_i_eer(wlasne, obce)
    auc_testu = auc(wlasne, obce)
    accuracy_zamknieta_testu = accuracy_zamknieta(podobienstwa_znane, indeksy_prawdziwe)

    trafienia = np.array(
        [
            float(prawdziwa == przewidziana)
            for prawdziwa, przewidziana in zip(prawdziwe_znane, przewidziane_znane)
        ],
        dtype=float,
    )
    akceptacje_obcych = np.array(
        [float(przewidziana != NAZWA_NIEZNANEJ) for przewidziana in przewidziane_obce],
        dtype=float,
    )
    odrzucenia_znanych = np.array(
        [float(przewidziana == NAZWA_NIEZNANEJ) for przewidziana in przewidziane_znane],
        dtype=float,
    )
    trafienia_zamknieta = np.array(
        [
            float(int(podobienstwa_znane[numer].argmax()) == prawdziwy)
            for numer, prawdziwy in enumerate(indeksy_prawdziwe)
        ],
        dtype=float,
    )
    ci = przedzialy_ufnosci(
        trafienia,
        akceptacje_obcych,
        odrzucenia_znanych,
        wlasne,
        obce,
        n_bootstrap=argumenty.n_bootstrap,
        seed=argumenty.seed,
        trafienia_zamknieta=trafienia_zamknieta,
    )
    postep.dodaj(1)

    na_osobe: dict[str, dict] = {}
    for osoba in osoby:
        pozycje = [
            numer
            for numer, prawdziwa in enumerate(prawdziwe_znane)
            if prawdziwa == osoba
        ]
        if not pozycje:
            continue
        poprawne = sum(
            przewidziane_znane[numer] == osoba for numer in pozycje
        )
        na_osobe[osoba] = {
            "zdjecia": len(pozycje),
            "poprawne": poprawne,
            "recall": round(poprawne / len(pozycje), 4),
        }

    prawdziwe_wszystkie = prawdziwe_znane + [NAZWA_OBCEGO] * len(przewidziane_obce)
    przewidziane_wszystkie = przewidziane_znane + przewidziane_obce
    macierz = _macierz_pomylek(prawdziwe_wszystkie, przewidziane_wszystkie, osoby)

    # ------------------------------------------------ porownanie z 07
    latwy = _wczytaj_latwy_test()
    trudny = {
        "zbior": str(folder_testu.relative_to(KORZEN_PROJEKTU))
        if folder_testu.is_relative_to(KORZEN_PROJEKTU)
        else str(folder_testu),
        "zdjecia_znanych": len(prawdziwe_znane),
        "zdjecia_obcych": len(przewidziane_obce),
        "metryki": metryki,
        "accuracy_zamknieta": round(accuracy_zamknieta_testu, 4),
        "eer": krzywa["eer"] if krzywa else None,
        "prog_eer": krzywa["prog"] if krzywa else None,
        "auc": auc_testu,
        "przedzialy_ufnosci": ci,
    }

    # ---------------------------------------------------------- wykresy
    wykresy: dict[str, str] = {}
    try:
        wykresy["podsumowanie"] = "podsumowanie_testu_trudnego.png"
        _wykres_podsumowania(
            metryki,
            ci,
            krzywa["eer"] if krzywa else None,
            FOLDER_WYKRESOW / wykresy["podsumowanie"],
        )
        if latwy:
            wykresy["porownanie"] = "porownanie_latwy_trudny.png"
            _wykres_porownania(latwy, trudny, FOLDER_WYKRESOW / wykresy["porownanie"])
        wykresy["rozklad_podobienstw"] = "rozklad_podobienstw.png"
        _wykres_rozkladu(
            wlasne,
            obce,
            krzywa["prog"] if krzywa else None,
            FOLDER_WYKRESOW / wykresy["rozklad_podobienstw"],
        )
        wykresy["macierz_pomylek"] = "macierz_pomylek.png"
        _wykres_macierzy(macierz, FOLDER_WYKRESOW / wykresy["macierz_pomylek"])
    except ImportError:
        print("[uwaga] matplotlib niedostepny - pomijam wykresy.")
    postep.dodaj(3)

    # ---------------------------------------------------------- raport
    czas = time.perf_counter() - start
    ostrzezenia = [
        "Test trudny sluzy WYLACZNIE do pomiaru - te zdjecia nie braly "
        "udzialu w treningu ani w kalibracji progow. Nie dodawaj ich do "
        "data/raw, bo wtedy przestana byc trudnym testem.",
        "Trudny test jest zwykle mniejszy od latwego, wiec przedzialy "
        "ufnosci beda szersze. Jeden blad moze zmienic wynik o kilka "
        "punktow procentowych.",
    ]
    if not latwy:
        ostrzezenia.append(
            "Brak raportu z kroku 7 - nie ma z czym porownac wyniku. "
            "Uruchom 07_ewaluacja.py, zeby zobaczyc roznice."
        )
    elif trudny["metryki"]["accuracy"] is not None and latwy["metryki"].get("accuracy") is not None:
        roznica = (
            trudny["metryki"]["accuracy"] - latwy["metryki"]["accuracy"]
        )
        if roznica < -0.05:
            ostrzezenia.append(
                f"Accuracy spadla o {abs(roznica) * 100:.1f} pkt proc. "
                "wobec latwego testu. To oczekiwane na trudniejszych "
                "zdjeciach - pokazuje, ile wart byl wynik z kroku 7."
            )
    if len(prawdziwe_znane) < 30:
        ostrzezenia.append(
            f"W tescie trudnym jest tylko {len(prawdziwe_znane)} zdjec "
            "znanych osob. Przy tak malej probie nawet duza roznica moze "
            "byc przypadkiem - zbierz wiecej zdjec."
        )
    if not indeksy_obce:
        ostrzezenia.append(
            "Brak zdjec obcych w tescie trudnym - FAR, EER i AUC sa "
            "niepoliczone. Dodaj folder 'nieznajomy' z osobami spoza bazy."
        )
    if brakujace:
        ostrzezenia.append(
            f"Brak zdjec w tescie dla osob: {brakujace}. Ich wyniki "
            "nie wchodza do accuracy, wiec porownanie z krokiem 7 jest "
            "niepelne."
        )

    raport = {
        "wersja_raportu": WERSJA_RAPORTU,
        "wygenerowano": datetime.now().isoformat(timespec="seconds"),
        "czas_s": round(czas, 1),
        "urzadzenie": str(urzadzenie),
        "dane": {
            "folder": trudny["zbior"],
            "osoby_w_modelu": osoby,
            "klasy_w_folderze": klasy,
            "brakujace_osoby": brakujace,
            "zdjecia_znanych": len(prawdziwe_znane),
            "zdjecia_obcych": len(przewidziane_obce),
            "rola": (
                "test trudny: zdjecia z innych warunkow niz trening, "
                "bez udzialu w uczeniu i kalibracji"
            ),
            "ostrzezenia": ostrzezenia,
        },
        "prototypy": {
            "opis": "Prototypy i progi z pliku wzorcow (tak jak w 04).",
            "progi_osob": {osoba: round(prog, 4) for osoba, prog in progi.items()},
            "metryki": metryki,
            "accuracy_zamknieta": round(accuracy_zamknieta_testu, 4),
            "eer": krzywa["eer"] if krzywa else None,
            "prog_eer": krzywa["prog"] if krzywa else None,
            "auc": auc_testu,
            "przedzialy_ufnosci": ci,
            "na_osobe": na_osobe,
            "macierz_pomylek": macierz,
        },
        "porownanie_latwy_trudny": {
            "latwy": latwy,
            "trudny": trudny,
            "roznica": _roznice(latwy, trudny),
        },
        "wykresy": wykresy,
    }

    sciezka_tymczasowa = PLIK_RAPORTU.with_suffix(".json.tmp")
    with open(sciezka_tymczasowa, "w", encoding="utf-8") as plik:
        json.dump(raport, plik, ensure_ascii=False, indent=2)
    sciezka_tymczasowa.replace(PLIK_RAPORTU)

    # ------------------------------------------------------ podsumowanie
    print("\n" + "=" * 68)
    print("PODSUMOWANIE TESTU TRUDNEGO (prototypy, progi z 03)")
    print("=" * 68)
    print(
        f"  Accuracy: {metryki['accuracy']:.1%}  "
        f"({metryki['poprawne']}/{metryki['liczba_znanych']})"
    )
    print(f"             {_format_ci(ci['metryki']['accuracy'])}")
    if metryki["far"] is not None:
        print(f"  FAR:      {metryki['far']:.1%}")
        print(f"             {_format_ci(ci['metryki']['far'])}")
    else:
        print("  FAR:      brak zdjec obcych")
    print(f"  FRR:      {metryki['frr']:.1%}  ({metryki['odrzucone_znane']} odrzuconych)")
    print(f"             {_format_ci(ci['metryki']['frr'])}")
    if krzywa:
        print(f"  EER:      {krzywa['eer']:.1%} przy progu {krzywa['prog']:.2f}")
        print(f"             {_format_ci(ci['metryki']['eer'])}")
    if auc_testu is not None:
        print(f"  AUC:      {auc_testu:.3f}")
        print(f"             {_format_ci(ci['metryki']['auc'], procent=False)}")
    print("-" * 68)
    if latwy:
        print("  Porownanie z latwym testem (07):")
        for klucz, nazwa in (
            ("accuracy", "accuracy"),
            ("far", "FAR"),
            ("frr", "FRR"),
            ("eer", "EER"),
            ("auc", "AUC"),
        ):
            if klucz in ("eer", "auc"):
                stara = latwy.get(klucz)
                nowa = trudny.get(klucz)
            else:
                stara = (latwy.get("metryki") or {}).get(klucz)
                nowa = (trudny.get("metryki") or {}).get(klucz)
            if stara is None or nowa is None:
                continue
            if klucz == "auc":
                print(f"    {nazwa:<8} {stara:.3f} -> {nowa:.3f} ({nowa - stara:+.3f})")
            else:
                print(
                    f"    {nazwa:<8} {stara:.1%} -> {nowa:.1%} "
                    f"({(nowa - stara) * 100:+.1f} pkt proc.)"
                )
    else:
        print("  Brak raportu z 07 - porownanie niedostepne.")
    print("=" * 68)
    print(f"[gotowe] Raport: {PLIK_RAPORTU}")
    print(f"[gotowe] Wykresy: {FOLDER_WYKRESOW}")
    print(
        f"[info] Test trudny: {len(prawdziwe_znane)} zdjec znanych osob + "
        f"{len(przewidziane_obce)} zdjec obcych."
    )
    print(
        "[uwaga] To osobny zbior - nie sluzyl do treningu ani kalibracji. "
        "Porownaj go z krokiem 7, zeby zobaczyc, ile wart byl latwy wynik."
    )


if __name__ == "__main__":
    main()
