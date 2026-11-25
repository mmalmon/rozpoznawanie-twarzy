"""Wspólne miary jakości rozpoznawania - używane przez 07 i 08.

DLA UCZNIÓW: dlaczego metryki są w osobnym pliku?

Ewaluacja i eksperymenty muszą liczyć dokładnie te same wskaźniki. Gdyby
każdy skrypt miał własną kopię wzorów, po zmianie jednego z nich wyniki
przestałyby być porównywalne. Tutaj znajdują się wszystkie definicje:

  - accuracy (zamknięta i otwarta),
  - FAR / FRR,
  - EER (punkt, w którym FAR = FRR),
  - AUC (jak dobrze ranking podobieństw rozdziela swoich od obcych),
  - przedziały ufności (bootstrap) - jak pewny jest pomiar,
  - podobieństwa do prototypów oraz k-NN.

Uwaga: samo "accuracy" nie wystarcza. Model, który zawsze odpowiada
"Nieznana osoba", ma 0% FAR, ale odrzuca wszystkich swoich. Dlatego
patrzymy na kilka miar jednocześnie.
"""

from __future__ import annotations

from statistics import NormalDist

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


def przedzial_ufnosci_proporcji(
    sukcesy: int, proby: int, poziom: float = 0.95
) -> list[float] | None:
    """Przedział ufności Wilsona dla odsetka (np. accuracy, FAR, FRR).

    DLA UCZNIÓW: dlaczego nie zwykły bootstrap? Bo przy wyniku 0 błędów
    na 128 próbach bootstrap daje przedział 0%-0%. To sugerowałoby
    stuprocentową pewność, której nie ma - przecież 129. zdjęcie mogłoby
    przynieść błąd. Przedział Wilsona radzi sobie z takimi skrajnymi
    przypadkami: dla 0/128 daje mniej więcej 0%-2,9%, a dla 130/130
    około 97,1%-100%.

    Wzór pochodzi z klasycznej statystyki i jest używany m.in. w ankietach
    ("poparcie 45% +/- 3 pkt proc.").
    """
    if proby <= 0:
        return None
    sukcesy = max(0, min(int(sukcesy), int(proby)))
    z = NormalDist().inv_cdf(1.0 - (1.0 - poziom) / 2.0)
    p = sukcesy / proby
    mianownik = 1.0 + z * z / proby
    srodek = (p + z * z / (2.0 * proby)) / mianownik
    polowa = (
        z
        * np.sqrt(p * (1.0 - p) / proby + z * z / (4.0 * proby * proby))
        / mianownik
    )
    dolny = max(0.0, srodek - polowa)
    gorny = min(1.0, srodek + polowa)
    return [round(float(dolny), 4), round(float(gorny), 4)]


def przedzial_ufnosci_bootstrap(
    wartosci: np.ndarray,
    poziom: float = 0.95,
    n_bootstrap: int = 400,
    seed: int = 42,
) -> list[float] | None:
    """Przedział ufności dla średniej metodą bootstrap (percentylowy).

    DLA UCZNIÓW: pojedyncza liczba, np. "accuracy 100%", nie mówi, jak
    pewny jest pomiar. Bootstrap odpowiada na pytanie: "gdybyśmy mieli
    trochę inne zdjęcia testowe, jaki wynik moglibyśmy dostać?". Losujemy
    ze zwracaniem tyle samo próbek, ile mamy, liczymy średnią i powtarzamy
    to wiele razy. Potem odcinamy skrajne 2,5% wyników z każdej strony.

    To NIE jest gwarancja, że model zawsze będzie tak dobry. To tylko
    miara szumu losowania zdjęć - przy 20 zdjęciach przedział będzie
    szeroki, przy 2000 wąski.
    """
    if len(wartosci) == 0:
        return None
    rng = np.random.default_rng(seed)
    indeksy = rng.integers(0, len(wartosci), size=(n_bootstrap, len(wartosci)))
    srednie = wartosci[indeksy].mean(axis=1)
    alfa = (1.0 - poziom) / 2.0
    dolny = float(np.quantile(srednie, alfa))
    gorny = float(np.quantile(srednie, 1.0 - alfa))
    return [round(dolny, 4), round(gorny, 4)]


def przedzial_ufnosci_eer(
    wlasne: np.ndarray,
    obce: np.ndarray,
    poziom: float = 0.95,
    n_bootstrap: int = 400,
    seed: int = 42,
) -> list[float] | None:
    """Przedział ufności dla EER (osobno losujemy swoich i obcych).

    EER zależy od dwóch rozkładów podobieństw, dlatego losujemy każdy
    z nich osobno, żeby nie zgubić proporcji między znanymi i obcymi.
    """
    if len(wlasne) == 0 or len(obce) == 0:
        return None
    rng = np.random.default_rng(seed)
    wartosci: list[float] = []
    for _ in range(n_bootstrap):
        indeksy_wlasne = rng.integers(0, len(wlasne), size=len(wlasne))
        indeksy_obce = rng.integers(0, len(obce), size=len(obce))
        krzywa = krzywa_i_eer(
            wlasne[indeksy_wlasne],
            obce[indeksy_obce],
            liczba_progow=101,
        )
        if krzywa is not None:
            wartosci.append(float(krzywa["eer"]))
    if not wartosci:
        return None
    alfa = (1.0 - poziom) / 2.0
    return [
        round(float(np.quantile(wartosci, alfa)), 4),
        round(float(np.quantile(wartosci, 1.0 - alfa)), 4),
    ]


def przedzial_ufnosci_auc(
    wlasne: np.ndarray,
    obce: np.ndarray,
    poziom: float = 0.95,
    n_bootstrap: int = 400,
    seed: int = 42,
) -> list[float] | None:
    """Przedział ufności dla AUC - ten sam pomysł co przy EER."""
    if len(wlasne) == 0 or len(obce) == 0:
        return None
    rng = np.random.default_rng(seed)
    wartosci: list[float] = []
    for _ in range(n_bootstrap):
        indeksy_wlasne = rng.integers(0, len(wlasne), size=len(wlasne))
        indeksy_obce = rng.integers(0, len(obce), size=len(obce))
        wynik = auc(wlasne[indeksy_wlasne], obce[indeksy_obce])
        if wynik is not None:
            wartosci.append(float(wynik))
    if not wartosci:
        return None
    alfa = (1.0 - poziom) / 2.0
    return [
        round(float(np.quantile(wartosci, alfa)), 4),
        round(float(np.quantile(wartosci, 1.0 - alfa)), 4),
    ]


def przedzialy_ufnosci(
    trafienia: np.ndarray,
    akceptacje_obcych: np.ndarray,
    odrzucenia_znanych: np.ndarray,
    wlasne: np.ndarray,
    obce: np.ndarray,
    poziom: float = 0.95,
    n_bootstrap: int = 400,
    seed: int = 42,
    pelne: bool = True,
    trafienia_zamknieta: np.ndarray | None = None,
) -> dict:
    """Zestaw przedziałów ufności dla wszystkich metryk jednej metody.

    Wejście to tablice 0/1 dla pojedynczych zdjęć (trafienie, błędna
    akceptacja obcego, błędne odrzucenie swojego) oraz dwa rankingi
    podobieństw potrzebne do EER i AUC.

    Odsetki (accuracy, FAR, FRR) liczymy przedziałem Wilsona, bo przy
    wyniku 0 lub 100% zwykły bootstrap daje fałszywie wąski przedział.
    EER i AUC wymagają dwóch rozkładów podobieństw, więc tam używamy
    bootstrapu. ``pelne=False`` pomija kosztowny bootstrap (przydatne,
    gdy EER i AUC nie zmieniają się między wariantami).
    """
    metryki = {
        "accuracy": przedzial_ufnosci_proporcji(
            int(np.sum(trafienia)), len(trafienia), poziom
        ),
        "far": przedzial_ufnosci_proporcji(
            int(np.sum(akceptacje_obcych)), len(akceptacje_obcych), poziom
        ),
        "frr": przedzial_ufnosci_proporcji(
            int(np.sum(odrzucenia_znanych)), len(odrzucenia_znanych), poziom
        ),
        "eer": (
            przedzial_ufnosci_eer(wlasne, obce, poziom, n_bootstrap, seed + 3)
            if pelne
            else None
        ),
        "auc": (
            przedzial_ufnosci_auc(wlasne, obce, poziom, n_bootstrap, seed + 4)
            if pelne
            else None
        ),
    }
    if trafienia_zamknieta is not None:
        metryki["accuracy_zamknieta"] = przedzial_ufnosci_proporcji(
            int(np.sum(trafienia_zamknieta)), len(trafienia_zamknieta), poziom
        )
    return {
        "poziom": poziom,
        "n_bootstrap": n_bootstrap if pelne else 0,
        "metoda": "Wilson (odsetki) + bootstrap (EER, AUC)",
        "metryki": metryki,
    }


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
