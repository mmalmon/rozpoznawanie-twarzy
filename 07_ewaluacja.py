"""
Krok 7: Ewaluacja modelu rozpoznawania twarzy.

DLA UCZNIOW - po co osobny skrypt do oceny?

Do tej pory pytalismy "czy program dziala?". Ten skrypt odpowiada na znacznie
wazniejsze pytanie: "JAK DOBRZE dziala i gdzie sie myli?". To jest roznica
miedzy zrobieniem modelu a prowadzeniem eksperymentu naukowego.

Co liczymy?
  1. ACCURACY (dokladnosc) - jaka czesc znanych twarzy zostala rozpoznana
     poprawnie. Uwaga: sama dokladnosc nie wystarcza, bo nie mowi nic o tym,
     jak czesto model akceptuje OBCE osoby.
  2. PRECISION / RECALL na osobe - czy model nie myli danej osoby z innymi
     i czy nie odrzuca jej zbyt czesto.
  3. MACIERZ POMYLEK - tabela "kogo z kim myli". Od razu widac, ktore osoby
     sa dla modelu podobne.
  4. FAR (False Acceptance Rate) - odsetek OBCYCH twarzy blednie
     zaakceptowanych jako znana osoba. W szkole to najwazniejszy blad:
     system wpuszcza kogos, kogo nie powinien.
  5. FRR (False Rejection Rate) - odsetek ZNANYCH twarzy odrzuconych jako
     "Nieznana osoba". To blad odwrotny: swoj czlowiek nie zostaje wpuszczony.
  6. EER (Equal Error Rate) - punkt, w ktorym FAR = FRR. Im nizszy, tym
     lepszy model. To jedna liczba do porownywania metod.
  7. AUC - jak dobrze sam ranking podobienstw rozdziela "swoich" od "obcych"
     (1.0 = idealnie, 0.5 = losowo).
  8. PRZEDZIALY UFNOSCI - jak pewny jest wynik. Dla odsetkow (accuracy,
     FAR, FRR) liczymy je metoda Wilsona, dla EER i AUC bootstrapem.
     Przy malym tescie 100% accuracy nie znaczy "model jest idealny",
     tylko "ten test nie pokazal bledu". Przedzial mowi, o ile wynik
     moglby sie zmienic przy innych zdjeciach testowych.

Co porownujemy?
  - PROTOTYPY - obecna metoda projektu: najblizszy zapisany wzorzec osoby.
  - k-NN - glosowanie kilku najblizszych zdjec treningowych.
  - REGRESJA LOGISTYCZNA - klasyczny klasyfikator na tych samych embeddingach.
To swietna lekcja: te same cechy (embeddingi) + rozne sposoby decyzji =
rozne wyniki. Deep learning nie zawsze musi byc ostatnim slowem.

WYMAGANIA:
  - data/processed/train, data/processed/val i data/processed/test (uruchom 02),
  - models/wzorce_osob.json (uruchom 03).

JAK DZIELIMY ROLE ZBIOROW (to jest sedno metodologii):
  - train - z niego skrypt 03 buduje wzorce,
  - val   - na nim skrypt 03 kalibruje progi, a ten skrypt wybiera prog EER
            dla k-NN i regresji logistycznej,
  - test  - NIGDY nie byl uzyty do uczenia ani do kalibracji; tylko na nim
            liczymy ostateczne metryki.

Dzieki temu wynik nie jest "dopasowany" do danych, na ktorych wybieralismy
prog. To roznica miedzy eksperymentem a chwaleniem sie wynikiem.

UWAGA: test jest maly (zwykle kilkadziesiat do kilkuset zdjec), wiec jedna
pomylka potrafi zmienic wynik o kilka punktow procentowych. To normalne -
dlatego w raporcie podajemy tez surowe wartosci liczbowe, a nie tylko procenty.

Uzycie:
    python 07_ewaluacja.py
    python 07_ewaluacja.py --postep
    python 07_ewaluacja.py --k-sasiadow 7 --maks-pomylek 30

Wynik:
    models/ewaluacja/raport_ewaluacji.json
    models/ewaluacja/wykresy/*.png
    models/ewaluacja/pomylki/*.jpg
"""

from __future__ import annotations

# Ta linijka MUSI byc pierwszym kodem wykonywalnym w tym pliku - patrz
# szczegolowe wyjasnienie w common/uruchom_w_venv.py.
from common.uruchom_w_venv import przelacz_na_venv_jesli_trzeba

przelacz_na_venv_jesli_trzeba()

import argparse
import json
import re
import shutil
import time
from datetime import datetime
from pathlib import Path

import numpy as np
import torch

from common.embeddingi import (
    WSPOLCZYNNIKI_JASNOSCI,
    policz_embeddingi_folderu,
    policz_obrazy,
    transformacja,
)
from common.metryki import (
    accuracy_zamknieta,
    auc,
    decyzje_prototypy,
    decyzje_z_progiem,
    krzywa_i_eer,
    metryki_openset,
    podobienstwa_do_osob,
    podobienstwa_knn,
    przedzialy_ufnosci,
)
from common.model import pobierz_urzadzenie, zbuduj_ekstraktor_cech
from common.rozpoznawanie import NAZWA_NIEZNANEJ, wczytaj_wzorce

KORZEN_PROJEKTU = Path(__file__).resolve().parent
FOLDER_PRZETWORZONYCH_DANYCH = KORZEN_PROJEKTU / "data" / "processed"
FOLDER_MODELI = KORZEN_PROJEKTU / "models"
FOLDER_EWALUACJI = FOLDER_MODELI / "ewaluacja"
FOLDER_WYKRESOW = FOLDER_EWALUACJI / "wykresy"
FOLDER_POMYLEK = FOLDER_EWALUACJI / "pomylki"
PLIK_WZORCOW = FOLDER_MODELI / "wzorce_osob.json"
PLIK_RAPORTU = FOLDER_EWALUACJI / "raport_ewaluacji.json"

NAZWA_KLASY_NIEZNAJOMY = "nieznajomy"
NAZWA_OBCEGO = "obcy (nieznajomy)"
WERSJA_RAPORTU = 3


# =========================================================================
# Postep (te same maszynowe linie co w skrypcie 03 - panel GUI je rozumie)
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
        description="Ewaluacja modelu: accuracy, FAR/FRR, EER, macierz pomylek, porownanie metod."
    )
    parser.add_argument("--batch-size", type=int, default=32, help="Ile zdjec naraz trafia do sieci")
    parser.add_argument("--k-sasiadow", type=int, default=5, help="Liczba sasiadow w klasyfikatorze k-NN")
    parser.add_argument(
        "--maks-pomylek",
        type=int,
        default=24,
        help="Ile zdjec z bledami zapisac do galerii pomylek (0 = nie zapisuj)",
    )
    parser.add_argument("--seed", type=int, default=42, help="Ziarno losowosci (powtarzalnosc)")
    parser.add_argument(
        "--n-bootstrap",
        type=int,
        default=400,
        help="Ile razy losowac probe w przedzialach ufnosci (wiecej = wolniej, dokladniej)",
    )
    parser.add_argument(
        "--postep",
        action="store_true",
        help="wypisuje maszynowe linie postepu dla panelu GUI",
    )
    return parser.parse_args()


# =========================================================================
# Metryki
# =========================================================================
def _metryki_na_osobe(
    prawdziwe: list[str], przewidziane: list[str], osoby: list[str]
) -> dict:
    """Precision / recall / F1 dla kazdej osoby (z uwzglednieniem obcych)."""
    wynik: dict[str, dict] = {}
    for osoba in osoby:
        tp = fp = fn = 0
        for prawdziwa, przewidziana in zip(prawdziwe, przewidziane):
            if przewidziana == osoba and prawdziwa == osoba:
                tp += 1
            elif przewidziana == osoba and prawdziwa != osoba:
                fp += 1
            elif prawdziwa == osoba and przewidziana != osoba:
                fn += 1
        precision = tp / (tp + fp) if tp + fp else 0.0
        recall = tp / (tp + fn) if tp + fn else 0.0
        f1 = (
            2 * precision * recall / (precision + recall)
            if precision + recall
            else 0.0
        )
        wynik[osoba] = {
            "zdjecia": tp + fn,
            "tp": tp,
            "fp": fp,
            "fn": fn,
            "precision": round(precision, 4),
            "recall": round(recall, 4),
            "f1": round(f1, 4),
        }
    return wynik


def _macierz_pomylek(
    prawdziwe: list[str], przewidziane: list[str], osoby: list[str]
) -> dict:
    """Macierz: wiersze = prawda, kolumny = decyzja modelu."""
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


def _histogram(wartosci: np.ndarray, zakres: tuple[float, float], bins: int = 25) -> dict:
    liczby, krawedzie = np.histogram(wartosci, bins=bins, range=zakres)
    return {
        "granice": [round(float(x), 4) for x in krawedzie],
        "liczby": [int(x) for x in liczby],
    }


# =========================================================================
# Porownywane metody: k-NN (ze wspolnego modulu) i regresja logistyczna
# =========================================================================
def _wyniki_logistycznej(
    trening: torch.Tensor,
    etykiety_trening: list[int],
    klasy_trening: list[str],
    zapytania: torch.Tensor,
    osoby: list[str],
    seed: int,
) -> np.ndarray | None:
    """Trenuje regresje logistyczna na embeddingach i zwraca prawdopodobienstwa."""
    try:
        from sklearn.linear_model import LogisticRegression
    except ImportError:
        print("[uwaga] Brak scikit-learn - pomijam regresje logistyczna.")
        return None

    indeksy = [
        numer
        for numer, etykieta in enumerate(etykiety_trening)
        if klasy_trening[etykieta] in osoby
    ]
    if len(set(etykiety_trening[numer] for numer in indeksy)) < 2:
        print("[uwaga] Za malo klas do regresji logistycznej - pomijam.")
        return None

    X = trening[indeksy].numpy()
    y = np.array([etykiety_trening[numer] for numer in indeksy])
    model = LogisticRegression(
        max_iter=3000,
        class_weight="balanced",
        random_state=seed,
    )
    model.fit(X, y)

    prawdopodobienstwa = model.predict_proba(zapytania.numpy())
    nazwy_klas = [klasy_trening[int(klasa)] for klasa in model.classes_]
    wynik = np.zeros((len(zapytania), len(osoby)), dtype=float)
    for numer, nazwa in enumerate(nazwy_klas):
        if nazwa in osoby:
            wynik[:, osoby.index(nazwa)] = prawdopodobienstwa[:, numer]
    return wynik


# =========================================================================
# Wykresy i galeria pomylek
# =========================================================================
def _matplotlib():
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    return plt


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
    os.set_title("Macierz pomylek - prototypy")

    maksimum = max(1, int(wartosci.max()))
    for wiersz in range(len(wiersze)):
        for kolumna in range(len(kolumny)):
            wartosc = int(wartosci[wiersz, kolumna])
            kolor = "white" if wartosc > maksimum / 2 else "#1f2937"
            os.text(kolumna, wiersz, str(wartosc), ha="center", va="center", color=kolor)

    rysunek.tight_layout()
    rysunek.savefig(sciezka, dpi=140)
    plt.close(rysunek)


def _wykres_rozkladu(
    wlasne: np.ndarray, obce: np.ndarray, prog_eer: float | None, sciezka: Path
) -> None:
    plt = _matplotlib()
    rysunek, os = plt.subplots(figsize=(8, 4.5))
    zakres = (0.0, 1.0)
    os.hist(wlasne, bins=25, range=zakres, alpha=0.65, label="znane twarze (podobienstwo do siebie)")
    os.hist(obce, bins=25, range=zakres, alpha=0.65, label="obce twarze (najlepsze podobienstwo)")
    if prog_eer is not None:
        os.axvline(
            prog_eer,
            color="#dc3545",
            linestyle="--",
            label=f"prog EER = {prog_eer:.2f}",
        )
    os.set_xlabel("podobienstwo kosinusowe")
    os.set_ylabel("liczba zdjec")
    os.set_title("Rozklad podobienstw: swoi vs obcy")
    os.legend()
    rysunek.tight_layout()
    rysunek.savefig(sciezka, dpi=140)
    plt.close(rysunek)


def _wykres_krzywej(krzywa: dict | None, sciezka: Path) -> None:
    plt = _matplotlib()
    rysunek, os = plt.subplots(figsize=(8, 4.5))
    if krzywa is None:
        os.text(0.5, 0.5, "Brak zdjec obcych - nie mozna policzyc FAR/FRR", ha="center")
        os.axis("off")
    else:
        os.plot(krzywa["progi"], krzywa["far"], label="FAR (obcy zaakceptowani)", color="#dc3545")
        os.plot(krzywa["progi"], krzywa["frr"], label="FRR (swoi odrzuceni)", color="#0d6efd")
        os.axvline(
            krzywa["prog"],
            color="#198754",
            linestyle="--",
            label=f"EER = {krzywa['eer']:.1%} przy progu {krzywa['prog']:.2f}",
        )
        os.set_xlabel("prog podobienstwa")
        os.set_ylabel("odsetek bledow")
        os.set_title("Kompromis FAR / FRR")
        os.legend()
    rysunek.tight_layout()
    rysunek.savefig(sciezka, dpi=140)
    plt.close(rysunek)


def _wykres_porownania(porownanie: list[dict], sciezka: Path) -> None:
    plt = _matplotlib()
    metody = [wpis["metoda"] for wpis in porownanie]
    accuracy = [wpis.get("accuracy_zamknieta") or 0.0 for wpis in porownanie]
    eer = [wpis.get("eer") or 0.0 for wpis in porownanie]

    x = np.arange(len(metody))
    szerokosc = 0.36
    rysunek, os = plt.subplots(figsize=(9, 4.5))
    os.bar(x - szerokosc / 2, accuracy, szerokosc, label="accuracy (zamknieta)", color="#0d6efd")
    os.bar(x + szerokosc / 2, eer, szerokosc, label="EER (mniej = lepiej)", color="#fd7e14")
    os.set_xticks(x, metody)
    os.set_ylim(0, 1)
    os.set_ylabel("wartosc")
    os.set_title("Porownanie metod na tych samych embeddingach")
    os.legend()
    for numer, wartosc in enumerate(accuracy):
        os.text(numer - szerokosc / 2, wartosc + 0.02, f"{wartosc:.0%}", ha="center", fontsize=8)
    for numer, wartosc in enumerate(eer):
        os.text(numer + szerokosc / 2, wartosc + 0.02, f"{wartosc:.0%}", ha="center", fontsize=8)
    rysunek.tight_layout()
    rysunek.savefig(sciezka, dpi=140)
    plt.close(rysunek)


def _tablice_bootstrapu(
    prawdziwe: list[str],
    przewidziane_znane: list[str],
    przewidziane_obce: list[str],
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Zamienia decyzje na tablice 0/1 potrzebne do przedzialow ufnosci."""
    trafienia = np.array(
        [
            float(prawdziwa == przewidziana)
            for prawdziwa, przewidziana in zip(prawdziwe, przewidziane_znane)
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
    return trafienia, akceptacje_obcych, odrzucenia_znanych


def _format_ci(przedzial: list[float] | None, procent: bool = True) -> str:
    """Krotki opis przedzialu ufnosci, np. '95% CI 97.2-100.0%'."""
    if not przedzial:
        return "brak przedzialu"
    dolny, gorny = przedzial
    if procent:
        return f"95% CI {dolny:.1%}-{gorny:.1%}"
    return f"95% CI {dolny:.3f}-{gorny:.3f}"


def _wykres_przedzialow(dane: list[dict], sciezka: Path) -> None:
    """Dwa panele: accuracy i EER z przedzialami ufnosci (bootstrap)."""
    plt = _matplotlib()
    etykiety = [wpis["etykieta"] for wpis in dane]
    x = np.arange(len(etykiety))

    def _slupki(os, wartosci, przedzialy, kolor, tytul, procent):
        dolne = np.array(
            [max(0.0, wartosc - (ci[0] if ci else wartosc)) for wartosc, ci in zip(wartosci, przedzialy)]
        )
        gorne = np.array(
            [max(0.0, (ci[1] if ci else wartosc) - wartosc) for wartosc, ci in zip(wartosci, przedzialy)]
        )
        os.bar(x, wartosci, color=kolor, width=0.55)
        os.errorbar(
            x,
            wartosci,
            yerr=np.vstack([dolne, gorne]),
            fmt="none",
            ecolor="#212529",
            capsize=5,
            linewidth=1.2,
        )
        os.set_xticks(x, etykiety, fontsize=8)
        os.set_ylim(0, 1)
        os.set_title(tytul)
        os.grid(axis="y", alpha=0.25)
        for numer, wartosc in enumerate(wartosci):
            os.text(
                numer,
                min(0.97, wartosc + 0.04),
                f"{wartosc:.1%}" if procent else f"{wartosc:.3f}",
                ha="center",
                fontsize=8,
            )

    rysunek, osie = plt.subplots(1, 2, figsize=(11, 4.4))
    _slupki(
        osie[0],
        [wpis["accuracy"] for wpis in dane],
        [wpis["accuracy_ci"] for wpis in dane],
        "#0d6efd",
        "Accuracy (otwarta) z 95% przedzialem ufnosci",
        True,
    )
    _slupki(
        osie[1],
        [wpis["eer"] for wpis in dane],
        [wpis["eer_ci"] for wpis in dane],
        "#fd7e14",
        "EER z 95% przedzialem ufnosci (mniej = lepiej)",
        True,
    )
    rysunek.suptitle("Ile warty jest ten wynik? Przedzialy ufnosci bootstrap")
    rysunek.tight_layout()
    rysunek.savefig(sciezka, dpi=140)
    plt.close(rysunek)


def _wykres_pca(
    embeddingi_znane: torch.Tensor,
    osoby_znane: list[str],
    embeddingi_obce: torch.Tensor,
    osoby: list[str],
    seed: int,
    sciezka: Path,
) -> bool:
    """Rzutuje embeddingi na 2 wymiary, zeby pokazac klastry osob."""
    try:
        from sklearn.decomposition import PCA
    except ImportError:
        print("[uwaga] Brak scikit-learn - pomijam wykres PCA.")
        return False

    fragmenty = [embeddingi_znane]
    etykiety = list(osoby_znane)
    if len(embeddingi_obce):
        fragmenty.append(embeddingi_obce)
        etykiety.extend([NAZWA_OBCEGO] * len(embeddingi_obce))

    X = torch.cat(fragmenty, dim=0).numpy()
    if len(X) < 3:
        return False

    # Za duzo punktow na czytelny wykres - probkujemy powtarzalnie.
    maks_punktow = 1200
    if len(X) > maks_punktow:
        generator = np.random.default_rng(seed)
        wybrane = np.sort(generator.choice(len(X), maks_punktow, replace=False))
        X = X[wybrane]
        etykiety = [etykiety[numer] for numer in wybrane]

    wspolrzedne = PCA(n_components=2, random_state=seed).fit_transform(X)
    plt = _matplotlib()
    rysunek, os = plt.subplots(figsize=(8, 6))
    kolory = plt.get_cmap("tab10")

    for numer, osoba in enumerate(osoby):
        maska = np.array([etykieta == osoba for etykieta in etykiety])
        if maska.any():
            os.scatter(
                wspolrzedne[maska, 0],
                wspolrzedne[maska, 1],
                s=22,
                alpha=0.75,
                color=kolory(numer % 10),
                label=osoba,
            )

    maska_obcych = np.array([etykieta == NAZWA_OBCEGO for etykieta in etykiety])
    if maska_obcych.any():
        os.scatter(
            wspolrzedne[maska_obcych, 0],
            wspolrzedne[maska_obcych, 1],
            s=16,
            alpha=0.45,
            color="#6c757d",
            marker="x",
            label="obcy",
        )

    os.set_title("Embeddingi w 2D (PCA) - czy osoby tworza klastry?")
    os.set_xlabel("skladowa PCA 1")
    os.set_ylabel("skladowa PCA 2")
    os.legend(fontsize=8)
    rysunek.tight_layout()
    rysunek.savefig(sciezka, dpi=140)
    plt.close(rysunek)
    return True


def _bezpieczna_nazwa(tekst: str) -> str:
    return re.sub(r"[^A-Za-z0-9_-]+", "_", tekst).strip("_") or "x"


def _zapisz_pomylki(pomylki: list[dict], maks: int) -> list[dict]:
    """Kopiuje zdjecia bledow do osobnego folderu i zwraca ich opisy."""
    if FOLDER_POMYLEK.exists():
        shutil.rmtree(FOLDER_POMYLEK)
    FOLDER_POMYLEK.mkdir(parents=True, exist_ok=True)
    if maks <= 0:
        return []

    # Lista jest juz posortowana od najbardziej "pewnych" bledow, wiec bierzemy
    # pierwsze N. Dzieki temu galeria pokazuje najciekawsze przypadki.
    wybrane = pomylki[:maks]

    wynik: list[dict] = []
    for indeks, wpis in enumerate(wybrane[:maks], start=1):
        zrodlo = Path(wpis["oryginal"])
        if not zrodlo.is_file():
            continue
        rozszerzenie = zrodlo.suffix.lower() or ".jpg"
        nazwa = (
            f"{indeks:03d}_{_bezpieczna_nazwa(wpis['prawdziwa'])}"
            f"__{_bezpieczna_nazwa(wpis['przewidziana'])}{rozszerzenie}"
        )
        shutil.copy2(zrodlo, FOLDER_POMYLEK / nazwa)
        wynik.append({**wpis, "plik": nazwa})
    return wynik


# =========================================================================
# Glowna procedura
# =========================================================================
def main() -> None:
    argumenty = parsuj_argumenty()
    if argumenty.batch_size <= 0:
        raise SystemExit("--batch-size musi byc dodatni.")
    if argumenty.k_sasiadow <= 0:
        raise SystemExit("--k-sasiadow musi byc dodatni.")
    if argumenty.maks_pomylek < 0:
        raise SystemExit("--maks-pomylek nie moze byc ujemny.")
    if argumenty.n_bootstrap <= 0:
        raise SystemExit("--n-bootstrap musi byc dodatnie.")

    folder_train = FOLDER_PRZETWORZONYCH_DANYCH / "train"
    folder_val = FOLDER_PRZETWORZONYCH_DANYCH / "val"
    folder_test = FOLDER_PRZETWORZONYCH_DANYCH / "test"

    if not folder_train.is_dir() or not folder_val.is_dir() or not folder_test.is_dir():
        raise SystemExit(
            f"Brak kompletu zbiorow w {FOLDER_PRZETWORZONYCH_DANYCH} "
            "(potrzebne train, val i test). Uruchom ponownie 02_podzial_danych.py."
        )
    if not PLIK_WZORCOW.exists():
        raise SystemExit(
            f"Brak pliku {PLIK_WZORCOW}. Najpierw uruchom 03_trenowanie_modelu.py."
        )

    FOLDER_EWALUACJI.mkdir(parents=True, exist_ok=True)
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

    print(f"[info] Osoby w modelu ({len(osoby)}): {osoby}")
    print(f"[info] Urzadzenie: {urzadzenie}")

    liczba_train = policz_obrazy(folder_train)
    liczba_val = policz_obrazy(folder_val)
    liczba_test = policz_obrazy(folder_test)
    postep = Postep(argumenty.postep)
    postep.start(
        "embeddingi",
        liczba_train * len(WSPOLCZYNNIKI_JASNOSCI) + liczba_val + liczba_test,
    )

    # ---------------------------------------------------------- trening
    print("[info] Liczenie embeddingow zbioru treningowego (3 warianty jasnosci)...")
    embeddingi_warianty: list[torch.Tensor] = []
    etykiety_trening: list[int] = []
    klasy_trening: list[str] = []
    sciezki_trening_oryginal: list[Path] = []
    for jasnosc in WSPOLCZYNNIKI_JASNOSCI:
        embeddingi, etykiety, klasy, sciezki = policz_embeddingi_folderu(
            ekstraktor,
            folder_train,
            urzadzenie,
            argumenty.batch_size,
            transformacja(jasnosc),
            postep,
        )
        if not klasy_trening:
            klasy_trening = klasy
            etykiety_trening = etykiety
        elif klasy != klasy_trening:
            raise SystemExit("Lista klas zmienila sie podczas czytania danych treningowych.")
        if jasnosc == 1.0:
            sciezki_trening_oryginal = sciezki
        embeddingi_warianty.append(embeddingi)

    indeks_oryginalu = WSPOLCZYNNIKI_JASNOSCI.index(1.0)
    embeddingi_trening_oryginal = embeddingi_warianty[indeks_oryginalu]
    embeddingi_trening = torch.cat(embeddingi_warianty, dim=0)
    etykiety_trening_wszystkie = etykiety_trening * len(WSPOLCZYNNIKI_JASNOSCI)
    print(
        f"[info] Trening: {len(etykiety_trening)} zdjec x "
        f"{len(WSPOLCZYNNIKI_JASNOSCI)} warianty = {len(etykiety_trening_wszystkie)} embeddingow."
    )

    # -------------------------------------------------------- walidacja
    # Val sluzy TYLKO do wyboru progu (03 robi to dla prototypow, a ten
    # skrypt dla k-NN i regresji). Nie raportujemy na nim metryk.
    print("[info] Liczenie embeddingow walidacji (wybor progow)...")
    embeddingi_val, etykiety_val, klasy_val, _ = policz_embeddingi_folderu(
        ekstraktor,
        folder_val,
        urzadzenie,
        argumenty.batch_size,
        transformacja(1.0),
        postep,
    )
    prawdziwe_val = [klasy_val[etykieta] for etykieta in etykiety_val]
    indeksy_znane_val = [
        numer for numer, prawdziwa in enumerate(prawdziwe_val) if prawdziwa in osoby
    ]
    indeksy_obce_val = [
        numer
        for numer, prawdziwa in enumerate(prawdziwe_val)
        if prawdziwa == NAZWA_KLASY_NIEZNAJOMY
    ]
    if not indeksy_znane_val:
        raise SystemExit(
            "W zbiorze walidacyjnym nie ma zadnej osoby z pliku wzorcow. "
            "Uruchom ponownie 02 i 03."
        )
    embeddingi_val_znane = embeddingi_val[indeksy_znane_val]
    indeksy_prawdziwe_val = [
        osoby.index(prawdziwe_val[numer]) for numer in indeksy_znane_val
    ]
    embeddingi_val_obce = embeddingi_val[indeksy_obce_val]
    print(
        f"[info] Walidacja: {len(indeksy_znane_val)} zdjec znanych osob, "
        f"{len(indeksy_obce_val)} zdjec klasy 'nieznajomy'."
    )

    # ------------------------------------------------------------- test
    print("[info] Liczenie embeddingow zbioru testowego...")
    embeddingi_test, etykiety_test, klasy_test, sciezki_test = policz_embeddingi_folderu(
        ekstraktor,
        folder_test,
        urzadzenie,
        argumenty.batch_size,
        transformacja(1.0),
        postep,
    )
    prawdziwe_test = [klasy_test[etykieta] for etykieta in etykiety_test]
    indeksy_znane_test = [
        numer for numer, prawdziwa in enumerate(prawdziwe_test) if prawdziwa in osoby
    ]
    indeksy_obce_test = [
        numer
        for numer, prawdziwa in enumerate(prawdziwe_test)
        if prawdziwa == NAZWA_KLASY_NIEZNAJOMY
    ]
    if not indeksy_znane_test:
        raise SystemExit(
            "W zbiorze testowym nie ma zadnej osoby z pliku wzorcow. "
            "Uruchom ponownie 02 i 03."
        )

    embeddingi_znane = embeddingi_test[indeksy_znane_test]
    prawdziwe_znane = [prawdziwe_test[numer] for numer in indeksy_znane_test]
    sciezki_znane = [sciezki_test[numer] for numer in indeksy_znane_test]
    indeksy_prawdziwe = [osoby.index(osoba) for osoba in prawdziwe_znane]
    print(
        f"[info] Test: {len(prawdziwe_znane)} zdjec znanych osob, "
        f"{len(indeksy_obce_test)} zdjec klasy 'nieznajomy'."
    )

    # ---------------------------------------------------------- obcy
    # Zdjecia "nieznajomy" ze zbioru treningowego i testowego. Val jest
    # celowo pominieta, bo jej zdjecia posluzyly do kalibracji progow -
    # wliczenie ich tutaj zawyzyloby FAR w optymistyczna strone.
    maska_obcych_trening = torch.tensor(
        [
            klasy_trening[etykieta] == NAZWA_KLASY_NIEZNAJOMY
            for etykieta in etykiety_trening
        ],
        dtype=torch.bool,
    )
    czesci_obce: list[torch.Tensor] = []
    sciezki_obce: list[Path] = []
    if bool(maska_obcych_trening.any()):
        czesci_obce.append(embeddingi_trening_oryginal[maska_obcych_trening])
        sciezki_obce.extend(
            sciezka
            for sciezka, czy_obcy in zip(
                sciezki_trening_oryginal, maska_obcych_trening.tolist()
            )
            if czy_obcy
        )
    if indeksy_obce_test:
        czesci_obce.append(embeddingi_test[indeksy_obce_test])
        sciezki_obce.extend(sciezki_test[numer] for numer in indeksy_obce_test)

    if czesci_obce:
        embeddingi_obce = torch.cat(czesci_obce, dim=0)
    else:
        print(
            "[uwaga] Brak zdjec klasy 'nieznajomy' w train/test - "
            "nie policze FAR. Uruchom pobierz_zdjecia_nieznajomych.py i 02."
        )
        embeddingi_obce = torch.empty((0, embeddingi_znane.shape[1]))
    print(f"[info] Obcy: {len(embeddingi_obce)} zdjec (train + test, bez val).")

    # ------------------------------------------------------ prototypy
    postep.start("metryki", 4)
    print("\n[info] Metoda 1/3: prototypy (obecny model z pliku wzorcow)...")
    progi = {osoba: float(wzorce[osoba]["prog"]) for osoba in osoby}
    podobienstwa_znane = podobienstwa_do_osob(embeddingi_znane, wzorce, osoby)
    podobienstwa_obce = podobienstwa_do_osob(embeddingi_obce, wzorce, osoby)
    przewidziane_znane_proto, _ = decyzje_prototypy(
        podobienstwa_znane, osoby, progi
    )
    przewidziane_obce_proto, _ = decyzje_prototypy(
        podobienstwa_obce, osoby, progi
    )
    metryki_proto = metryki_openset(
        prawdziwe_znane, przewidziane_znane_proto, przewidziane_obce_proto
    )

    indeksy = np.arange(len(embeddingi_znane))
    wlasne_proto = podobienstwa_znane[indeksy, indeksy_prawdziwe]
    obce_proto = (
        podobienstwa_obce.max(axis=1)
        if len(podobienstwa_obce)
        else np.zeros(0, dtype=float)
    )
    krzywa_proto = krzywa_i_eer(wlasne_proto, obce_proto)
    auc_proto = auc(wlasne_proto, obce_proto)
    trafienia_proto, akceptacje_obce_proto, odrzucenia_znane_proto = _tablice_bootstrapu(
        prawdziwe_znane, przewidziane_znane_proto, przewidziane_obce_proto
    )
    ci_proto = przedzialy_ufnosci(
        trafienia_proto,
        akceptacje_obce_proto,
        odrzucenia_znane_proto,
        wlasne_proto,
        obce_proto,
        n_bootstrap=argumenty.n_bootstrap,
        seed=argumenty.seed,
        trafienia_zamknieta=np.array(
            [
                float(int(podobienstwa_znane[numer].argmax()) == prawdziwy)
                for numer, prawdziwy in enumerate(indeksy_prawdziwe)
            ],
            dtype=float,
        ),
    )
    postep.dodaj(1)

    # ---------------------------------------------------------- k-NN
    print(f"[info] Metoda 2/3: k-NN (k={argumenty.k_sasiadow})...")
    # Prog wybieramy na val (punkt EER), a metryki liczymy na test.
    zapytania_val = torch.cat([embeddingi_val_znane, embeddingi_val_obce], dim=0)
    pod_knn_val = podobienstwa_knn(
        zapytania_val,
        embeddingi_trening,
        etykiety_trening_wszystkie,
        klasy_trening,
        osoby,
        argumenty.k_sasiadow,
    )
    pod_knn_val_znane = pod_knn_val[: len(embeddingi_val_znane)]
    pod_knn_val_obce = pod_knn_val[len(embeddingi_val_znane) :]
    indeksy_val = np.arange(len(embeddingi_val_znane))
    wlasne_knn_val = pod_knn_val_znane[indeksy_val, indeksy_prawdziwe_val]
    obce_knn_val = (
        pod_knn_val_obce.max(axis=1)
        if len(pod_knn_val_obce)
        else np.zeros(0, dtype=float)
    )
    krzywa_knn_val = krzywa_i_eer(wlasne_knn_val, obce_knn_val)
    prog_knn = krzywa_knn_val["prog"] if krzywa_knn_val else 0.5

    zapytania_test = torch.cat([embeddingi_znane, embeddingi_obce], dim=0)
    pod_knn_test = podobienstwa_knn(
        zapytania_test,
        embeddingi_trening,
        etykiety_trening_wszystkie,
        klasy_trening,
        osoby,
        argumenty.k_sasiadow,
    )
    pod_knn_test_znane = pod_knn_test[: len(embeddingi_znane)]
    pod_knn_test_obce = pod_knn_test[len(embeddingi_znane) :]
    wlasne_knn = pod_knn_test_znane[indeksy, indeksy_prawdziwe]
    obce_knn = (
        pod_knn_test_obce.max(axis=1)
        if len(pod_knn_test_obce)
        else np.zeros(0, dtype=float)
    )
    krzywa_knn = krzywa_i_eer(wlasne_knn, obce_knn)
    przewidziane_znane_knn, _ = decyzje_z_progiem(
        pod_knn_test_znane, osoby, prog_knn
    )
    przewidziane_obce_knn, _ = decyzje_z_progiem(
        pod_knn_test_obce, osoby, prog_knn
    )
    metryki_knn = metryki_openset(
        prawdziwe_znane, przewidziane_znane_knn, przewidziane_obce_knn
    )
    auc_knn = auc(wlasne_knn, obce_knn)
    trafienia_knn, akceptacje_obce_knn, odrzucenia_znane_knn = _tablice_bootstrapu(
        prawdziwe_znane, przewidziane_znane_knn, przewidziane_obce_knn
    )
    ci_knn = przedzialy_ufnosci(
        trafienia_knn,
        akceptacje_obce_knn,
        odrzucenia_znane_knn,
        wlasne_knn,
        obce_knn,
        n_bootstrap=argumenty.n_bootstrap,
        seed=argumenty.seed,
        trafienia_zamknieta=np.array(
            [
                float(int(pod_knn_test_znane[numer].argmax()) == prawdziwy)
                for numer, prawdziwy in enumerate(indeksy_prawdziwe)
            ],
            dtype=float,
        ),
    )
    postep.dodaj(1)

    # ---------------------------------------------- regresja logistyczna
    print("[info] Metoda 3/3: regresja logistyczna na embeddingach...")
    zapytania_log = torch.cat([zapytania_val, zapytania_test], dim=0)
    podobienstwa_log = _wyniki_logistycznej(
        embeddingi_trening_oryginal,
        etykiety_trening,
        klasy_trening,
        zapytania_log,
        osoby,
        argumenty.seed,
    )
    if podobienstwa_log is not None:
        pod_log_val = podobienstwa_log[: len(zapytania_val)]
        pod_log_test = podobienstwa_log[len(zapytania_val) :]

        pod_log_val_znane = pod_log_val[: len(embeddingi_val_znane)]
        pod_log_val_obce = pod_log_val[len(embeddingi_val_znane) :]
        wlasne_log_val = pod_log_val_znane[indeksy_val, indeksy_prawdziwe_val]
        obce_log_val = (
            pod_log_val_obce.max(axis=1)
            if len(pod_log_val_obce)
            else np.zeros(0, dtype=float)
        )
        krzywa_log_val = krzywa_i_eer(wlasne_log_val, obce_log_val)
        prog_log = krzywa_log_val["prog"] if krzywa_log_val else 0.5

        pod_log_test_znane = pod_log_test[: len(embeddingi_znane)]
        pod_log_test_obce = pod_log_test[len(embeddingi_znane) :]
        wlasne_log = pod_log_test_znane[indeksy, indeksy_prawdziwe]
        obce_log = (
            pod_log_test_obce.max(axis=1)
            if len(pod_log_test_obce)
            else np.zeros(0, dtype=float)
        )
        krzywa_log = krzywa_i_eer(wlasne_log, obce_log)
        przewidziane_znane_log, _ = decyzje_z_progiem(
            pod_log_test_znane, osoby, prog_log
        )
        przewidziane_obce_log, _ = decyzje_z_progiem(
            pod_log_test_obce, osoby, prog_log
        )
        metryki_log = metryki_openset(
            prawdziwe_znane, przewidziane_znane_log, przewidziane_obce_log
        )
        auc_log = auc(wlasne_log, obce_log)
        accuracy_log = accuracy_zamknieta(pod_log_test_znane, indeksy_prawdziwe)
        trafienia_log, akceptacje_obce_log, odrzucenia_znane_log = _tablice_bootstrapu(
            prawdziwe_znane, przewidziane_znane_log, przewidziane_obce_log
        )
        ci_log = przedzialy_ufnosci(
            trafienia_log,
            akceptacje_obce_log,
            odrzucenia_znane_log,
            wlasne_log,
            obce_log,
            n_bootstrap=argumenty.n_bootstrap,
            seed=argumenty.seed,
            trafienia_zamknieta=np.array(
                [
                    float(int(pod_log_test_znane[numer].argmax()) == prawdziwy)
                    for numer, prawdziwy in enumerate(indeksy_prawdziwe)
                ],
                dtype=float,
            ),
        )
    else:
        krzywa_log = None
        prog_log = None
        metryki_log = None
        auc_log = None
        accuracy_log = None
        ci_log = None
    postep.dodaj(1)

    # ------------------------------------------------ wspolne podsumowania
    prawdziwe_wszystkie = prawdziwe_znane + [NAZWA_OBCEGO] * len(przewidziane_obce_proto)
    przewidziane_wszystkie = przewidziane_znane_proto + przewidziane_obce_proto
    macierz = _macierz_pomylek(prawdziwe_wszystkie, przewidziane_wszystkie, osoby)
    na_osobe = _metryki_na_osobe(
        prawdziwe_wszystkie, przewidziane_wszystkie, osoby
    )
    accuracy_zamknieta_proto = accuracy_zamknieta(
        podobienstwa_znane, indeksy_prawdziwe
    )

    porownanie = [
        {
            "metoda": "Prototypy (progi z 03)",
            "accuracy_zamknieta": round(accuracy_zamknieta_proto, 4),
            "accuracy": metryki_proto["accuracy"],
            "far": metryki_proto["far"],
            "frr": metryki_proto["frr"],
            "eer": krzywa_proto["eer"] if krzywa_proto else None,
            "auc": auc_proto,
        },
        {
            "metoda": f"k-NN (k={argumenty.k_sasiadow})",
            "accuracy_zamknieta": round(
                accuracy_zamknieta(pod_knn_test_znane, indeksy_prawdziwe), 4
            ),
            "accuracy": metryki_knn["accuracy"],
            "far": metryki_knn["far"],
            "frr": metryki_knn["frr"],
            "eer": krzywa_knn["eer"] if krzywa_knn else None,
            "auc": auc_knn,
        },
    ]
    if metryki_log is not None:
        porownanie.append(
            {
                "metoda": "Regresja logistyczna",
                "accuracy_zamknieta": round(accuracy_log or 0.0, 4),
                "accuracy": metryki_log["accuracy"],
                "far": metryki_log["far"],
                "frr": metryki_log["frr"],
                "eer": krzywa_log["eer"] if krzywa_log else None,
                "auc": auc_log,
            }
        )

    # ------------------------------------------------- przedzialy ufnosci
    dane_przedzialow = [
        {
            "etykieta": "Prototypy",
            "accuracy": metryki_proto["accuracy"],
            "accuracy_ci": ci_proto["metryki"]["accuracy"],
            "eer": krzywa_proto["eer"] if krzywa_proto else 0.0,
            "eer_ci": ci_proto["metryki"]["eer"],
        },
        {
            "etykieta": f"k-NN (k={argumenty.k_sasiadow})",
            "accuracy": metryki_knn["accuracy"],
            "accuracy_ci": ci_knn["metryki"]["accuracy"],
            "eer": krzywa_knn["eer"] if krzywa_knn else 0.0,
            "eer_ci": ci_knn["metryki"]["eer"],
        },
    ]
    if metryki_log is not None and ci_log is not None:
        dane_przedzialow.append(
            {
                "etykieta": "Regresja",
                "accuracy": metryki_log["accuracy"],
                "accuracy_ci": ci_log["metryki"]["accuracy"],
                "eer": krzywa_log["eer"] if krzywa_log else 0.0,
                "eer_ci": ci_log["metryki"]["eer"],
            }
        )

    # ------------------------------------------------------- pomylki
    pomylki: list[dict] = []
    for numer, (prawdziwa, przewidziana) in enumerate(
        zip(prawdziwe_znane, przewidziane_znane_proto)
    ):
        if prawdziwa == przewidziana:
            continue
        pomylki.append(
            {
                "typ": "znana osoba",
                "prawdziwa": prawdziwa,
                "przewidziana": przewidziana,
                "podobienstwo": round(float(podobienstwa_znane[numer].max()), 4),
                "oryginal": str(sciezki_znane[numer]),
            }
        )
    for numer, przewidziana in enumerate(przewidziane_obce_proto):
        if przewidziana == NAZWA_NIEZNANEJ:
            continue
        pomylki.append(
            {
                "typ": "obcy zaakceptowany",
                "prawdziwa": NAZWA_OBCEGO,
                "przewidziana": przewidziana,
                "podobienstwo": round(float(podobienstwa_obce[numer].max()), 4),
                "oryginal": str(sciezki_obce[numer]) if numer < len(sciezki_obce) else "",
            }
        )
    pomylki.sort(key=lambda wpis: wpis["podobienstwo"], reverse=True)
    pomylki = _zapisz_pomylki(pomylki, argumenty.maks_pomylek)

    # --------------------------------------------------------- wykresy
    wykresy = {
        "macierz_pomylek": "macierz_pomylek.png",
        "rozklad_podobienstw": "rozklad_podobienstw.png",
        "krzywa_progu": "krzywa_progu.png",
        "porownanie_metod": "porownanie_metod.png",
        "przedzialy_ufnosci": "przedzialy_ufnosci.png",
    }
    try:
        _wykres_macierzy(macierz, FOLDER_WYKRESOW / wykresy["macierz_pomylek"])
        _wykres_rozkladu(
            wlasne_proto,
            obce_proto,
            krzywa_proto["prog"] if krzywa_proto else None,
            FOLDER_WYKRESOW / wykresy["rozklad_podobienstw"],
        )
        _wykres_krzywej(krzywa_proto, FOLDER_WYKRESOW / wykresy["krzywa_progu"])
        _wykres_porownania(porownanie, FOLDER_WYKRESOW / wykresy["porownanie_metod"])
        _wykres_przedzialow(
            dane_przedzialow, FOLDER_WYKRESOW / wykresy["przedzialy_ufnosci"]
        )
        if _wykres_pca(
            embeddingi_znane,
            prawdziwe_znane,
            embeddingi_obce,
            osoby,
            argumenty.seed,
            FOLDER_WYKRESOW / "pca_embeddingi.png",
        ):
            wykresy["pca_embeddingi"] = "pca_embeddingi.png"
    except ImportError:
        print("[uwaga] matplotlib niedostepny - pomijam wykresy.")
    postep.dodaj(1)

    # ---------------------------------------------------------- raport
    czas = time.perf_counter() - start
    ostrzezenia = [
        "Metryki policzono na zbiorze TESTOWYM, ktorego nie uzyto ani do "
        "budowy wzorcow, ani do kalibracji progow. To najbardziej rzetelny "
        "wynik, jaki potrafi dac ten projekt.",
        "Progi prototypow pochodza z 03 i byly kalibrowane na zbiorze "
        "walidacyjnym. Prog EER dla k-NN i regresji logistycznej wybrano na "
        "walidacji i zastosowano bez zmian do testu - dlatego te metody sa "
        "uczciwie porownywalne z prototypami.",
        "Zbior testowy jest maly (kilkadziesiat zdjec), wiec pojedyncza "
        "pomylka zmienia wynik o kilka punktow procentowych. Nie traktuj "
        "roznicy 1-2 punktow jako dowodu, ze jedna metoda jest lepsza.",
        "Przedzialy ufnosci policzono metoda Wilsona (odsetki) i bootstrapem "
        "(EER, AUC). Pokazuja one tylko niepewnosc losowania probki - nie "
        "poprawiaja tego, ze test moze byc zbyt latwy ani ze pochodzi z "
        "jednego dnia i warunkow. To dwie rozne rzeczy. Zdegenerowany "
        "przedzial EER/AUC (np. 0.0-0.0) oznacza, ze w tescie nie ma "
        "zadnego trudnego przypadku - nie ze model jest doskonaly.",
    ]
    if len(embeddingi_obce):
        ostrzezenia.append(
            "Zdjecia obcych pochodza z klas 'nieznajomy' w zbiorach train i "
            "test (celowo bez val, ktory sluzyl do kalibracji progow)."
        )
    else:
        ostrzezenia.append(
            "Brak zdjec klasy 'nieznajomy' w train/test - nie policzono FAR, "
            "EER ani AUC. Uruchom pobierz_zdjecia_nieznajomych.py i 02."
        )

    raport = {
        "wersja_raportu": WERSJA_RAPORTU,
        "wygenerowano": datetime.now().isoformat(timespec="seconds"),
        "czas_s": round(czas, 1),
        "urzadzenie": str(urzadzenie),
        "dane": {
            "osoby": osoby,
            "trening": {
                "zdjecia": len(etykiety_trening),
                "embeddingi": len(etykiety_trening_wszystkie),
                "warianty_jasnosci": list(WSPOLCZYNNIKI_JASNOSCI),
                "rola": "budowa wzorcow (03)",
            },
            "walidacja": {
                "zdjecia_znanych": len(indeksy_znane_val),
                "zdjecia_nieznajomych": len(indeksy_obce_val),
                "rola": "kalibracja progow (03) i wybor progu EER dla k-NN oraz regresji",
            },
            "test": {
                "zdjecia_znanych": len(prawdziwe_znane),
                "zdjecia_nieznajomych": len(indeksy_obce_test),
                "rola": "ostateczny pomiar metryk (bez udzialu w uczeniu i kalibracji)",
            },
            "obcy": {
                "zdjecia": len(embeddingi_obce),
                "zrodlo": "data/processed/{train,test}/nieznajomy",
            },
            "wymiar_cech": int(embeddingi_znane.shape[1]),
            "ostrzezenia": ostrzezenia,
        },
        "prototypy": {
            "opis": "Najlepszy prototyp kazdej osoby + jej wlasny prog z pliku wzorcow (tak jak w 04).",
            "progi_osob": {osoba: round(prog, 4) for osoba, prog in progi.items()},
            "metryki": metryki_proto,
            "accuracy_zamknieta": round(accuracy_zamknieta_proto, 4),
            "na_osobe": na_osobe,
            "eer": krzywa_proto["eer"] if krzywa_proto else None,
            "prog_eer": krzywa_proto["prog"] if krzywa_proto else None,
            "auc": auc_proto,
            "przedzialy_ufnosci": ci_proto,
            "krzywa_progu": krzywa_proto,
            "rozklad": {
                "wlasne": _histogram(wlasne_proto, (0.0, 1.0)),
                "obce": _histogram(obce_proto, (0.0, 1.0)),
            },
            "macierz_pomylek": macierz,
        },
        "knn": {
            "opis": f"Srednia z {argumenty.k_sasiadow} najblizszych zdjec treningowych kazdej osoby.",
            "k": argumenty.k_sasiadow,
            "prog": prog_knn,
            "metryki": metryki_knn,
            "accuracy_zamknieta": round(
                accuracy_zamknieta(pod_knn_test_znane, indeksy_prawdziwe), 4
            ),
            "eer": krzywa_knn["eer"] if krzywa_knn else None,
            "auc": auc_knn,
            "przedzialy_ufnosci": ci_knn,
        },
        "regresja_logistyczna": (
            {
                "opis": "Klasyfikator liniowy trenowany na embeddingach (scikit-learn).",
                "prog": prog_log,
                "metryki": metryki_log,
                "accuracy_zamknieta": round(accuracy_log or 0.0, 4),
                "eer": krzywa_log["eer"] if krzywa_log else None,
                "auc": auc_log,
                "przedzialy_ufnosci": ci_log,
            }
            if metryki_log is not None
            else None
        ),
        "porownanie": porownanie,
        "pomylki": pomylki,
        "wykresy": wykresy,
    }

    sciezka_tymczasowa = PLIK_RAPORTU.with_suffix(".json.tmp")
    with open(sciezka_tymczasowa, "w", encoding="utf-8") as plik:
        json.dump(raport, plik, ensure_ascii=False, indent=2)
    sciezka_tymczasowa.replace(PLIK_RAPORTU)

    # ------------------------------------------------------- podsumowanie
    print("\n" + "=" * 68)
    print("PODSUMOWANIE EWALUACJI (zbior testowy, prototypy z progami z 03)")
    print("=" * 68)
    print(f"  Accuracy: {metryki_proto['accuracy']:.1%}  ({metryki_proto['poprawne']}/{metryki_proto['liczba_znanych']})")
    print(f"             {_format_ci(ci_proto['metryki']['accuracy'])}")
    print(f"  FAR:      {metryki_proto['far']:.1%}" if metryki_proto["far"] is not None else "  FAR:      brak danych")
    if metryki_proto["far"] is not None:
        print(f"             {_format_ci(ci_proto['metryki']['far'])}")
    print(f"  FRR:      {metryki_proto['frr']:.1%}  ({metryki_proto['odrzucone_znane']} odrzuconych)")
    print(f"             {_format_ci(ci_proto['metryki']['frr'])}")
    if krzywa_proto:
        print(f"  EER:      {krzywa_proto['eer']:.1%} przy progu {krzywa_proto['prog']:.2f}")
        print(f"             {_format_ci(ci_proto['metryki']['eer'])}")
    if auc_proto is not None:
        print(f"  AUC:      {auc_proto:.3f}")
        print(f"             {_format_ci(ci_proto['metryki']['auc'], procent=False)}")
    print("-" * 68)
    for wpis in porownanie:
        print(
            f"  {wpis['metoda']:<28} accuracy(zamkn.)={wpis['accuracy_zamknieta']:.1%}"
            f"  EER={wpis['eer']:.1%}" if wpis["eer"] is not None else
            f"  {wpis['metoda']:<28} accuracy(zamkn.)={wpis['accuracy_zamknieta']:.1%}"
        )
    print("=" * 68)
    print(f"[gotowe] Raport: {PLIK_RAPORTU}")
    print(f"[gotowe] Wykresy: {FOLDER_WYKRESOW}")
    print(f"[gotowe] Galeria pomylek: {FOLDER_POMYLEK} ({len(pomylki)} zdjec)")
    print(
        f"[info] Test: {len(prawdziwe_znane)} zdjec znanych osob + "
        f"{len(przewidziane_obce_proto)} zdjec obcych."
    )
    print(
        "[uwaga] Metryki pochodza ze zbioru testowego, ktorego nie uzyto do "
        "uczenia ani kalibracji progow - to wynik uczciwy, ale obarczony "
        "niepewnoscia malej proby."
    )


if __name__ == "__main__":
    main()
