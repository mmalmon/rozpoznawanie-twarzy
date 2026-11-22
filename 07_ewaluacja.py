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

Co porownujemy?
  - PROTOTYPY - obecna metoda projektu: najblizszy zapisany wzorzec osoby.
  - k-NN - glosowanie kilku najblizszych zdjec treningowych.
  - REGRESJA LOGISTYCZNA - klasyczny klasyfikator na tych samych embeddingach.
To swietna lekcja: te same cechy (embeddingi) + rozne sposoby decyzji =
rozne wyniki. Deep learning nie zawsze musi byc ostatnim slowem.

WYMAGANIA:
  - data/processed/train i data/processed/val (uruchom 02),
  - models/wzorce_osob.json (uruchom 03),
  - najlepiej data/raw/nieznajomy (zdjecia obcych osob do FAR).

UWAGA METODOLOGICZNA (bardzo wazna):
  Zbior walidacyjny (val) byl juz uzyty w skrypcie 03 do KALIBRACJI PROGOW.
  Oceniamy wiec na danych, ktore model "widzial" posrednio - wyniki sa
  OPTYMISTYCZNE. Do rzetelnej oceny trzeba osobnego zbioru TESTOWEGO, ktorego
  nie uzyto ani do treningu, ani do kalibracji. Skrypt wypisuje to ostrzezenie
  takze w raporcie JSON.

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
import math
import re
import shutil
import time
from datetime import datetime
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision import datasets, transforms

from common.model import (
    oblicz_znormalizowane_cechy,
    pobierz_urzadzenie,
    policz_podobienstwa_wzorca,
    przygotuj_przeksztalcenie,
    zbuduj_ekstraktor_cech,
)
from common.rozpoznawanie import NAZWA_NIEZNANEJ, wczytaj_wzorce

KORZEN_PROJEKTU = Path(__file__).resolve().parent
FOLDER_PRZETWORZONYCH_DANYCH = KORZEN_PROJEKTU / "data" / "processed"
FOLDER_RAW = KORZEN_PROJEKTU / "data" / "raw"
FOLDER_MODELI = KORZEN_PROJEKTU / "models"
FOLDER_EWALUACJI = FOLDER_MODELI / "ewaluacja"
FOLDER_WYKRESOW = FOLDER_EWALUACJI / "wykresy"
FOLDER_POMYLEK = FOLDER_EWALUACJI / "pomylki"
PLIK_WZORCOW = FOLDER_MODELI / "wzorce_osob.json"
PLIK_RAPORTU = FOLDER_EWALUACJI / "raport_ewaluacji.json"

NAZWA_KLASY_NIEZNAJOMY = "nieznajomy"
NAZWA_OBCEGO = "obcy (nieznajomy)"
WSPOLCZYNNIKI_JASNOSCI = (0.65, 1.0, 1.4)
ROZSZERZENIA_OBRAZOW = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
WERSJA_RAPORTU = 1


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
        "--postep",
        action="store_true",
        help="wypisuje maszynowe linie postepu dla panelu GUI",
    )
    return parser.parse_args()


# =========================================================================
# Wczytywanie zdjec i liczenie embeddingow
# =========================================================================
def policz_obrazy(folder: Path) -> int:
    """Liczy pliki-obrazki w folderze (do mianownika paska postepu)."""
    if not folder.exists():
        return 0
    return sum(
        1
        for plik in folder.rglob("*")
        if plik.is_file() and plik.suffix.lower() in ROZSZERZENIA_OBRAZOW
    )


class ZbiorPlikow(Dataset):
    """Prosty zbior zdjec z jednego, plaskiego folderu (np. data/raw/nieznajomy).

    ImageFolder wymaga podfolderow-klas. Tutaj wszystkie zdjecia to jedna
    klasa "obcy", wiec wystarczy lista plikow.
    """

    def __init__(self, pliki: list[Path], transform) -> None:
        self.pliki = pliki
        self.transform = transform

    def __len__(self) -> int:
        return len(self.pliki)

    def __getitem__(self, indeks: int):
        with Image.open(self.pliki[indeks]) as obraz:
            obraz = obraz.convert("RGB")
            return self.transform(obraz), 0


def _transformacja(jasnosc: float = 1.0) -> transforms.Compose:
    """Dokladnie ten sam sposob przygotowania obrazu co w skrypcie 03."""
    kroki = []
    if jasnosc != 1.0:
        kroki.append(transforms.ColorJitter(brightness=(jasnosc, jasnosc)))
    kroki.append(przygotuj_przeksztalcenie())
    return transforms.Compose(kroki)


def _ladowarka(zbior: Dataset, rozmiar_batcha: int) -> DataLoader:
    return DataLoader(
        zbior,
        batch_size=rozmiar_batcha,
        shuffle=False,
        num_workers=2 if len(zbior) > 50 else 0,
    )


def policz_embeddingi_folderu(
    ekstraktor: torch.nn.Module,
    folder: Path,
    urzadzenie: torch.device,
    rozmiar_batcha: int,
    transform,
    postep: Postep,
) -> tuple[torch.Tensor, list[int], list[str], list[Path]]:
    """Liczy embeddingi zdjec z folderu z podfolderami-klasami (train/val).

    Zwraca: macierz [N, D], etykiety liczbowe, nazwy klas i sciezki plikow
    w tej samej kolejnosci co wiersze macierzy.
    """
    zbior = datasets.ImageFolder(folder, transform=transform)
    if len(zbior) == 0:
        raise SystemExit(f"Brak zdjec w {folder}.")

    ladowarka = _ladowarka(zbior, rozmiar_batcha)
    embeddingi: list[torch.Tensor] = []
    etykiety: list[int] = []
    for obrazy, etykiety_batch in ladowarka:
        embeddingi.append(
            oblicz_znormalizowane_cechy(ekstraktor, obrazy.to(urzadzenie)).cpu()
        )
        etykiety.extend(etykiety_batch.tolist())
        postep.dodaj(obrazy.shape[0])

    return (
        torch.cat(embeddingi, dim=0),
        etykiety,
        list(zbior.classes),
        [Path(plik) for plik, _ in zbior.samples],
    )


def policz_embeddingi_plaskie(
    ekstraktor: torch.nn.Module,
    folder: Path,
    urzadzenie: torch.device,
    rozmiar_batcha: int,
    transform,
    postep: Postep,
) -> tuple[torch.Tensor, list[Path]]:
    """Liczy embeddingi wszystkich zdjec z plaskiego folderu (bez podfolderow)."""
    pliki = sorted(
        plik
        for plik in folder.iterdir()
        if plik.is_file() and plik.suffix.lower() in ROZSZERZENIA_OBRAZOW
    )
    if not pliki:
        return torch.empty((0, 0)), []

    zbior = ZbiorPlikow(pliki, transform)
    ladowarka = _ladowarka(zbior, rozmiar_batcha)
    embeddingi: list[torch.Tensor] = []
    for obrazy, _ in ladowarka:
        embeddingi.append(
            oblicz_znormalizowane_cechy(ekstraktor, obrazy.to(urzadzenie)).cpu()
        )
        postep.dodaj(obrazy.shape[0])
    return torch.cat(embeddingi, dim=0), pliki


# =========================================================================
# Metryki
# =========================================================================
def _podobienstwa_do_osob(
    embeddingi: torch.Tensor, wzorce: dict[str, dict], osoby: list[str]
) -> np.ndarray:
    """Macierz [liczba_zdjec, liczba_osob] - podobienstwo do najlepszego prototypu."""
    if len(embeddingi) == 0:
        return np.zeros((0, len(osoby)), dtype=float)
    kolumny = [
        policz_podobienstwa_wzorca(embeddingi, wzorce[osoba]["prototypy"]).numpy()
        for osoba in osoby
    ]
    return np.stack(kolumny, axis=1)


def _decyzje_prototypy(
    podobienstwa: np.ndarray, osoby: list[str], progi: dict[str, float]
) -> tuple[list[str], np.ndarray]:
    """Decyzja dokladnie taka jak w 04: najpierw wlasny prog, potem max."""
    etykiety: list[str] = []
    wyniki = np.zeros(len(podobienstwa), dtype=float)
    for indeks, wiersz in enumerate(podobienstwa):
        najlepsza = NAZWA_NIEZNANEJ
        najlepsze = -1.0
        for numer, osoba in enumerate(osoby):
            if wiersz[numer] >= progi[osoba] and wiersz[numer] > najlepsze:
                najlepsza = osoba
                najlepsze = float(wiersz[numer])
        etykiety.append(najlepsza)
        wyniki[indeks] = najlepsze if najlepsza != NAZWA_NIEZNANEJ else float(wiersz.max())
    return etykiety, wyniki


def _decyzje_z_progiem(
    podobienstwa: np.ndarray, osoby: list[str], prog: float
) -> tuple[list[str], np.ndarray]:
    """Wspolny prog dla wszystkich osob (k-NN, regresja logistyczna)."""
    if len(podobienstwa) == 0:
        return [], np.zeros(0, dtype=float)
    indeksy = podobienstwa.argmax(axis=1)
    maksima = podobienstwa.max(axis=1)
    etykiety = [
        osoby[int(indeks)] if maksima[numer] >= prog else NAZWA_NIEZNANEJ
        for numer, indeks in enumerate(indeksy)
    ]
    return etykiety, maksima


def _accuracy_zamknieta(
    podobienstwa: np.ndarray, indeksy_prawdziwe: list[int]
) -> float:
    """Dokladnosc, gdy NIE ma opcji "nieznana" - czyste kogo-kim."""
    if len(podobienstwa) == 0:
        return 0.0
    trafienia = sum(
        int(podobienstwa[numer].argmax()) == prawdziwy
        for numer, prawdziwy in enumerate(indeksy_prawdziwe)
    )
    return trafienia / len(podobienstwa)


def _metryki_openset(
    prawdziwe_znane: list[str],
    przewidziane_znane: list[str],
    przewidziane_obce: list[str],
) -> dict:
    """Accuracy, FAR i FRR dla wybranego progu decyzyjnego."""
    liczba_znanych = len(prawdziwe_znane)
    poprawne = sum(
        prawdziwa == przewidziana
        for prawdziwa, przewidziana in zip(prawdziwe_znane, przewidziane_znane)
    )
    odrzucone = sum(przewidziana == NAZWA_NIEZNANEJ for przewidziana in przewidziane_znane)
    zaakceptowani_obcy = sum(
        przewidziana != NAZWA_NIEZNANEJ for przewidziana in przewidziane_obce
    )
    return {
        "accuracy": round(poprawne / liczba_znanych, 4) if liczba_znanych else 0.0,
        "poprawne": poprawne,
        "liczba_znanych": liczba_znanych,
        "far": round(zaakceptowani_obcy / len(przewidziane_obce), 4)
        if przewidziane_obce
        else None,
        "zaakceptowani_obcy": zaakceptowani_obcy,
        "liczba_obcych": len(przewidziane_obce),
        "frr": round(odrzucone / liczba_znanych, 4) if liczba_znanych else 0.0,
        "odrzucone_znane": odrzucone,
    }


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


def _krzywa_i_eer(
    wlasne: np.ndarray, obce: np.ndarray, liczba_progow: int = 201
) -> dict | None:
    """Liczy FAR/FRR dla wielu progow i znajduje EER (FAR = FRR)."""
    if len(wlasne) == 0 or len(obce) == 0:
        return None

    progi = np.linspace(0.0, 1.0, liczba_progow)
    far = np.array([float(np.mean(obce >= prog)) for prog in progi])
    frr = np.array([float(np.mean(wlasne < prog)) for prog in progi])
    indeks = int(np.argmin(np.abs(far - frr)))
    return {
        "prog": round(float(progi[indeks]), 4),
        "eer": round(float((far[indeks] + frr[indeks]) / 2), 4),
        "progi": [round(float(x), 4) for x in progi],
        "far": [round(float(x), 4) for x in far],
        "frr": [round(float(x), 4) for x in frr],
    }


def _auc(wlasne: np.ndarray, obce: np.ndarray) -> float | None:
    """AUC policzone z rang - bez dodatkowych bibliotek i z obsluga remisow."""
    if len(wlasne) == 0 or len(obce) == 0:
        return None

    oceny = np.concatenate([wlasne, obce])
    etykiety = np.concatenate(
        [np.ones(len(wlasne), dtype=int), np.zeros(len(obce), dtype=int)]
    )
    kolejnosc = np.argsort(oceny, kind="mergesort")
    posortowane = oceny[kolejnosc]
    rangi = np.empty(len(oceny), dtype=float)

    poczatek = 0
    while poczatek < len(oceny):
        koniec = poczatek
        while koniec + 1 < len(oceny) and posortowane[koniec + 1] == posortowane[poczatek]:
            koniec += 1
        ranga = (poczatek + 1 + koniec + 1) / 2
        rangi[kolejnosc[poczatek : koniec + 1]] = ranga
        poczatek = koniec + 1

    liczba_pozytywnych = int(etykiety.sum())
    liczba_negatywnych = len(etykiety) - liczba_pozytywnych
    suma_rang = float(rangi[etykiety == 1].sum())
    auc = (
        suma_rang - liczba_pozytywnych * (liczba_pozytywnych + 1) / 2
    ) / (liczba_pozytywnych * liczba_negatywnych)
    return round(float(auc), 4)


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
# Porownywane metody: k-NN i regresja logistyczna
# =========================================================================
def _podobienstwa_knn(
    zapytania: torch.Tensor,
    trening: torch.Tensor,
    etykiety_trening: list[int],
    klasy_trening: list[str],
    osoby: list[str],
    k: int,
) -> np.ndarray:
    """Dla kazdej osoby srednia z k najblizszych jej zdjec treningowych."""
    kolumny: list[np.ndarray] = []
    for osoba in osoby:
        indeks_klasy = klasy_trening.index(osoba)
        maska = torch.tensor(
            [etykieta == indeks_klasy for etykieta in etykiety_trening],
            dtype=torch.bool,
        )
        embeddingi_osoby = trening[maska]
        if embeddingi_osoby.shape[0] == 0:
            kolumny.append(np.zeros(len(zapytania), dtype=float))
            continue
        podobienstwa = zapytania @ embeddingi_osoby.T
        k_skuteczne = min(max(1, k), podobienstwa.shape[1])
        najlepsze = podobienstwa.topk(k_skuteczne, dim=1).values.mean(dim=1)
        kolumny.append(najlepsze.numpy())
    return np.stack(kolumny, axis=1)


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

    folder_train = FOLDER_PRZETWORZONYCH_DANYCH / "train"
    folder_val = FOLDER_PRZETWORZONYCH_DANYCH / "val"
    folder_obcych = FOLDER_RAW / NAZWA_KLASY_NIEZNAJOMY

    if not folder_train.is_dir() or not folder_val.is_dir():
        raise SystemExit(
            f"Brak danych w {FOLDER_PRZETWORZONYCH_DANYCH}. "
            "Najpierw uruchom 02_podzial_danych.py."
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
    liczba_obcych = policz_obrazy(folder_obcych)
    postep = Postep(argumenty.postep)
    postep.start(
        "embeddingi",
        liczba_train * len(WSPOLCZYNNIKI_JASNOSCI) + liczba_val + liczba_obcych,
    )

    # ---------------------------------------------------------- trening
    print("[info] Liczenie embeddingow zbioru treningowego (3 warianty jasnosci)...")
    embeddingi_warianty: list[torch.Tensor] = []
    etykiety_trening: list[int] = []
    klasy_trening: list[str] = []
    for jasnosc in WSPOLCZYNNIKI_JASNOSCI:
        embeddingi, etykiety, klasy, _ = policz_embeddingi_folderu(
            ekstraktor,
            folder_train,
            urzadzenie,
            argumenty.batch_size,
            _transformacja(jasnosc),
            postep,
        )
        if not klasy_trening:
            klasy_trening = klasy
            etykiety_trening = etykiety
        elif klasy != klasy_trening:
            raise SystemExit("Lista klas zmienila sie podczas czytania danych treningowych.")
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
    print("[info] Liczenie embeddingow zbioru walidacyjnego...")
    embeddingi_val, etykiety_val, klasy_val, sciezki_val = policz_embeddingi_folderu(
        ekstraktor,
        folder_val,
        urzadzenie,
        argumenty.batch_size,
        _transformacja(1.0),
        postep,
    )
    prawdziwe_val = [klasy_val[etykieta] for etykieta in etykiety_val]
    indeksy_znane = [
        numer for numer, prawdziwa in enumerate(prawdziwe_val) if prawdziwa in osoby
    ]
    indeksy_obce_val = [
        numer
        for numer, prawdziwa in enumerate(prawdziwe_val)
        if prawdziwa == NAZWA_KLASY_NIEZNAJOMY
    ]
    if not indeksy_znane:
        raise SystemExit(
            "W zbiorze walidacyjnym nie ma zadnej osoby z pliku wzorcow. "
            "Uruchom ponownie 02 i 03."
        )

    embeddingi_znane = embeddingi_val[indeksy_znane]
    prawdziwe_znane = [prawdziwe_val[numer] for numer in indeksy_znane]
    sciezki_znane = [sciezki_val[numer] for numer in indeksy_znane]
    indeksy_prawdziwe = [osoby.index(osoba) for osoba in prawdziwe_znane]
    print(
        f"[info] Walidacja: {len(prawdziwe_znane)} zdjec znanych osob, "
        f"{len(indeksy_obce_val)} zdjec klasy 'nieznajomy'."
    )

    # ---------------------------------------------------------- obcy
    if folder_obcych.is_dir():
        print("[info] Liczenie embeddingow obcych twarzy (data/raw/nieznajomy)...")
        embeddingi_obce, sciezki_obce = policz_embeddingi_plaskie(
            ekstraktor,
            folder_obcych,
            urzadzenie,
            argumenty.batch_size,
            _transformacja(1.0),
            postep,
        )
    else:
        print("[uwaga] Brak folderu data/raw/nieznajomy - nie policze FAR.")
        embeddingi_obce = torch.empty((0, embeddingi_znane.shape[1]))
        sciezki_obce = []
    print(f"[info] Obcy: {len(embeddingi_obce)} zdjec.")

    # ------------------------------------------------------ prototypy
    postep.start("metryki", 4)
    print("\n[info] Metoda 1/3: prototypy (obecny model z pliku wzorcow)...")
    progi = {osoba: float(wzorce[osoba]["prog"]) for osoba in osoby}
    podobienstwa_znane = _podobienstwa_do_osob(embeddingi_znane, wzorce, osoby)
    podobienstwa_obce = _podobienstwa_do_osob(embeddingi_obce, wzorce, osoby)
    przewidziane_znane_proto, _ = _decyzje_prototypy(
        podobienstwa_znane, osoby, progi
    )
    przewidziane_obce_proto, _ = _decyzje_prototypy(
        podobienstwa_obce, osoby, progi
    )
    metryki_proto = _metryki_openset(
        prawdziwe_znane, przewidziane_znane_proto, przewidziane_obce_proto
    )

    indeksy = np.arange(len(embeddingi_znane))
    wlasne_proto = podobienstwa_znane[indeksy, indeksy_prawdziwe]
    obce_proto = (
        podobienstwa_obce.max(axis=1)
        if len(podobienstwa_obce)
        else np.zeros(0, dtype=float)
    )
    krzywa_proto = _krzywa_i_eer(wlasne_proto, obce_proto)
    auc_proto = _auc(wlasne_proto, obce_proto)
    postep.dodaj(1)

    # ---------------------------------------------------------- k-NN
    print(f"[info] Metoda 2/3: k-NN (k={argumenty.k_sasiadow})...")
    zapytania = torch.cat([embeddingi_znane, embeddingi_obce], dim=0)
    podobienstwa_knn = _podobienstwa_knn(
        zapytania,
        embeddingi_trening,
        etykiety_trening_wszystkie,
        klasy_trening,
        osoby,
        argumenty.k_sasiadow,
    )
    podobienstwa_knn_znane = podobienstwa_knn[: len(embeddingi_znane)]
    podobienstwa_knn_obce = podobienstwa_knn[len(embeddingi_znane) :]
    wlasne_knn = podobienstwa_knn_znane[indeksy, indeksy_prawdziwe]
    obce_knn = (
        podobienstwa_knn_obce.max(axis=1)
        if len(podobienstwa_knn_obce)
        else np.zeros(0, dtype=float)
    )
    krzywa_knn = _krzywa_i_eer(wlasne_knn, obce_knn)
    prog_knn = krzywa_knn["prog"] if krzywa_knn else 0.5
    przewidziane_znane_knn, _ = _decyzje_z_progiem(
        podobienstwa_knn_znane, osoby, prog_knn
    )
    przewidziane_obce_knn, _ = _decyzje_z_progiem(
        podobienstwa_knn_obce, osoby, prog_knn
    )
    metryki_knn = _metryki_openset(
        prawdziwe_znane, przewidziane_znane_knn, przewidziane_obce_knn
    )
    auc_knn = _auc(wlasne_knn, obce_knn)
    postep.dodaj(1)

    # ---------------------------------------------- regresja logistyczna
    print("[info] Metoda 3/3: regresja logistyczna na embeddingach...")
    podobienstwa_log = _wyniki_logistycznej(
        embeddingi_trening_oryginal,
        etykiety_trening,
        klasy_trening,
        zapytania,
        osoby,
        argumenty.seed,
    )
    if podobienstwa_log is not None:
        podobienstwa_log_znane = podobienstwa_log[: len(embeddingi_znane)]
        podobienstwa_log_obce = podobienstwa_log[len(embeddingi_znane) :]
        wlasne_log = podobienstwa_log_znane[indeksy, indeksy_prawdziwe]
        obce_log = (
            podobienstwa_log_obce.max(axis=1)
            if len(podobienstwa_log_obce)
            else np.zeros(0, dtype=float)
        )
        krzywa_log = _krzywa_i_eer(wlasne_log, obce_log)
        prog_log = krzywa_log["prog"] if krzywa_log else 0.5
        przewidziane_znane_log, _ = _decyzje_z_progiem(
            podobienstwa_log_znane, osoby, prog_log
        )
        przewidziane_obce_log, _ = _decyzje_z_progiem(
            podobienstwa_log_obce, osoby, prog_log
        )
        metryki_log = _metryki_openset(
            prawdziwe_znane, przewidziane_znane_log, przewidziane_obce_log
        )
        auc_log = _auc(wlasne_log, obce_log)
        accuracy_log = _accuracy_zamknieta(podobienstwa_log_znane, indeksy_prawdziwe)
    else:
        krzywa_log = None
        prog_log = None
        metryki_log = None
        auc_log = None
        accuracy_log = None
    postep.dodaj(1)

    # ------------------------------------------------ wspolne podsumowania
    prawdziwe_wszystkie = prawdziwe_znane + [NAZWA_OBCEGO] * len(przewidziane_obce_proto)
    przewidziane_wszystkie = przewidziane_znane_proto + przewidziane_obce_proto
    macierz = _macierz_pomylek(prawdziwe_wszystkie, przewidziane_wszystkie, osoby)
    na_osobe = _metryki_na_osobe(
        prawdziwe_wszystkie, przewidziane_wszystkie, osoby
    )
    accuracy_zamknieta_proto = _accuracy_zamknieta(
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
                _accuracy_zamknieta(podobienstwa_knn_znane, indeksy_prawdziwe), 4
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
        "Zbior walidacyjny (val) byl uzyty w skrypcie 03 do kalibracji progow, "
        "wiec metryki sa OPTYMISTYCZNE. Do rzetelnej oceny potrzebny jest osobny "
        "zbior testowy, ktorego nie uzyto ani do treningu, ani do kalibracji.",
        "Prog decyzyjny dla k-NN i regresji logistycznej zostal wybrany na tym "
        "samym zbiorze walidacyjnym (punkt EER), dlatego ich FAR/FRR rowniez sa "
        "optymistyczne i nie sa w pelni porownywalne z progami z 03.",
    ]
    if folder_obcych.is_dir():
        ostrzezenia.append(
            "Zdjecia obcych pochodza z data/raw/nieznajomy. Czesc z nich mogla "
            "byc uzyta w kalibracji progow (val), dlatego FAR jest zawyzony "
            "w optymistyczna strone."
        )
    else:
        ostrzezenia.append(
            "Brak folderu data/raw/nieznajomy - nie policzono FAR, EER ani AUC. "
            "Uruchom pobierz_zdjecia_nieznajomych.py."
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
            },
            "walidacja": {
                "zdjecia_znanych": len(prawdziwe_znane),
                "zdjecia_nieznajomych_w_val": len(indeksy_obce_val),
            },
            "obcy": {
                "zdjecia": len(embeddingi_obce),
                "zrodlo": "data/raw/nieznajomy",
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
                _accuracy_zamknieta(podobienstwa_knn_znane, indeksy_prawdziwe), 4
            ),
            "eer": krzywa_knn["eer"] if krzywa_knn else None,
            "auc": auc_knn,
        },
        "regresja_logistyczna": (
            {
                "opis": "Klasyfikator liniowy trenowany na embeddingach (scikit-learn).",
                "prog": prog_log,
                "metryki": metryki_log,
                "accuracy_zamknieta": round(accuracy_log or 0.0, 4),
                "eer": krzywa_log["eer"] if krzywa_log else None,
                "auc": auc_log,
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
    print("PODSUMOWANIE EWALUACJI (prototypy, progi z 03)")
    print("=" * 68)
    print(f"  Accuracy: {metryki_proto['accuracy']:.1%}  ({metryki_proto['poprawne']}/{metryki_proto['liczba_znanych']})")
    print(f"  FAR:      {metryki_proto['far']:.1%}" if metryki_proto["far"] is not None else "  FAR:      brak danych")
    print(f"  FRR:      {metryki_proto['frr']:.1%}  ({metryki_proto['odrzucone_znane']} odrzuconych)")
    if krzywa_proto:
        print(f"  EER:      {krzywa_proto['eer']:.1%} przy progu {krzywa_proto['prog']:.2f}")
    if auc_proto is not None:
        print(f"  AUC:      {auc_proto:.3f}")
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
    print("[uwaga] Metryki sa optymistyczne - val byl uzyty do kalibracji progow.")


if __name__ == "__main__":
    main()
