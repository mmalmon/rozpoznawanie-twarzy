"""Wspólne miary jakości rozpoznawania - używane przez 07 i 08.

DLA UCZNIÓW: dlaczego metryki są w osobnym pliku?

Ewaluacja i eksperymenty muszą liczyć dokładnie te same wskaźniki. Gdyby
każdy skrypt miał własną kopię wzorów, po zmianie jednego z nich wyniki
przestałyby być porównywalne. Tutaj znajdują się wszystkie definicje:

  - accuracy (zamknięta i otwarta),
  - FAR / FRR,
  - EER (punkt, w którym FAR = FRR),
  - AUC (jak dobrze ranking podobieństw rozdziela swoich od obcych),
  - podobieństwa do prototypów oraz k-NN.

Uwaga: samo "accuracy" nie wystarcza. Model, który zawsze odpowiada
"Nieznana osoba", ma 0% FAR, ale odrzuca wszystkich swoich. Dlatego
patrzymy na kilka miar jednocześnie.
"""

from __future__ import annotations

import numpy as np
import torch

from common.model import policz_podobienstwa_wzorca
from common.rozpoznawanie import NAZWA_NIEZNANEJ


def podobienstwa_do_osob(
    embeddingi: torch.Tensor, wzorce: dict[str, dict], osoby: list[str]
) -> np.ndarray:
    """Macierz [liczba_zdjec, liczba_osob] - podobieństwo do najlepszego prototypu."""
    if len(embeddingi) == 0:
        return np.zeros((0, len(osoby)), dtype=float)
    kolumny = [
        policz_podobienstwa_wzorca(embeddingi, wzorce[osoba]["prototypy"]).numpy()
        for osoba in osoby
    ]
    return np.stack(kolumny, axis=1)


def decyzje_prototypy(
    podobienstwa: np.ndarray, osoby: list[str], progi: dict[str, float]
) -> tuple[list[str], np.ndarray]:
    """Decyzja dokładnie taka jak w 04: najpierw własny próg, potem max."""
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


def decyzje_z_progiem(
    podobienstwa: np.ndarray, osoby: list[str], prog: float
) -> tuple[list[str], np.ndarray]:
    """Wspólny próg dla wszystkich osób (k-NN, regresja logistyczna)."""
    if len(podobienstwa) == 0:
        return [], np.zeros(0, dtype=float)
    indeksy = podobienstwa.argmax(axis=1)
    maksima = podobienstwa.max(axis=1)
    etykiety = [
        osoby[int(indeks)] if maksima[numer] >= prog else NAZWA_NIEZNANEJ
        for numer, indeks in enumerate(indeksy)
    ]
    return etykiety, maksima


def accuracy_zamknieta(
    podobienstwa: np.ndarray, indeksy_prawdziwe: list[int]
) -> float:
    """Dokładność, gdy NIE ma opcji "nieznana" - czyste kogo-kim."""
    if len(podobienstwa) == 0:
        return 0.0
    trafienia = sum(
        int(podobienstwa[numer].argmax()) == prawdziwy
        for numer, prawdziwy in enumerate(indeksy_prawdziwe)
    )
    return trafienia / len(podobienstwa)


def metryki_openset(
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
    odrzucone = sum(
        przewidziana == NAZWA_NIEZNANEJ for przewidziana in przewidziane_znane
    )
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


def krzywa_i_eer(
    wlasne: np.ndarray, obce: np.ndarray, liczba_progow: int = 201
) -> dict | None:
    """Liczy FAR/FRR dla wielu progów i znajduje EER (FAR = FRR)."""
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


def auc(wlasne: np.ndarray, obce: np.ndarray) -> float | None:
    """AUC policzone z rang - bez dodatkowych bibliotek i z obsługą remisów."""
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
    wynik = (
        suma_rang - liczba_pozytywnych * (liczba_pozytywnych + 1) / 2
    ) / (liczba_pozytywnych * liczba_negatywnych)
    return round(float(wynik), 4)


def podobienstwa_knn(
    zapytania: torch.Tensor,
    trening: torch.Tensor,
    etykiety_trening: list[int],
    klasy_trening: list[str],
    osoby: list[str],
    k: int,
) -> np.ndarray:
    """Dla każdej osoby średnia z k najbliższych jej zdjęć treningowych."""
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
