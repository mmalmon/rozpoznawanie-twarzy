"""
Krok 8: Eksperymenty - co zmienia wynik modelu?

DLA UCZNIOW - po co eksperymenty?

W kroku 7 zmierzylismy model. Teraz zadajemy pytanie, ktore jest sedno
nauki: CO SIE STANIE, JESLI ZMIENIMY JEDNA RZECZ? To metoda naukowa
w praktyce:

  1. wybieramy jedna zmienna (np. liczbe zdjec treningowych),
  2. zmieniamy TYLKO ja, reszte zostawiamy bez zmian,
  3. mierzymy wynik,
  4. porownujemy i wyciagamy wniosek.

Jesli zmienimy wszystko naraz, nie bedziemy wiedziec, co naprawde
wplynelo na wynik. To najczestszy blad w projektach AI.

Eksperymenty w tym skrypcie:

  A. LICZBA ZDJEC - 50, 100, 200 lub wszystkie zdjecia na osobe.
     Pytanie: czy wiecej danych zawsze oznacza lepszy model?
     Uwaga: skrypt 03 ogranicza liczbe prototypow do 128 na wariant
     jasnosci, wiec powyzej 128 zdjec model przestaje sie zmieniac.
     To celowo pokazuje, ze "wiecej danych" dziala tylko do granicy
     ustawionej przez projekt modelu.

  B. JASNOSC - tylko oryginalne zdjecia vs trzy warianty jasnosci
     (0.65, 1.0, 1.4), tak jak w treningu 03.
     Pytanie: czy augmentacja jasnosci naprawde pomaga?

  C. k W k-NN - 1, 3, 5, 9, 15 sasiadow.
     Pytanie: czy model powinien patrzec tylko na najblizsze zdjecie,
     czy na kilka i usredniac?

  D. PROG DECYZYJNY - od 0.50 do 0.95 co 0.05.
     Pytanie: gdzie jest granica miedzy "wpuszczam swoich" a
     "nie wpuszczam obcych"? Podniesienie progu zmniejsza FAR,
     ale zwieksza FRR.

Jak liczymy wynik?

  - accuracy zamknieta: kogo-kim bez opcji "nieznana osoba",
  - accuracy otwarta: z odrzucaniem obcych, przy progu EER,
  - FAR / FRR / EER / AUC: jak w kroku 7.

UWAGA METODOLOGICZNA (to jest sedno tego skryptu):
  Prog decyzyjny (EER) wybieramy na zbiorze WALIDACYJNYM, a metryki
  liczymy na zbiorze TESTOWYM, ktorego nie uzyto ani do budowy prototypow,
  ani do wyboru progu. Dzieki temu wynik nie jest "dopasowany" do danych,
  na ktorych ustawialismy prog.

  Zbior walidacyjny byl wczesniej uzyty w 03 do kalibracji progow, wiec
  nie raportujemy na nim metryk. Test jest maly, dlatego roznice 1-2
  punktow procentowych traktujemy jako szum, a nie dowod.

Uzycie:
    python 08_eksperymenty.py
    python 08_eksperymenty.py --postep
    python 08_eksperymenty.py --batch-size 16

Wynik:
    models/eksperymenty/raport_eksperymentow.json
    models/eksperymenty/wyniki.csv
    models/eksperymenty/wykresy/*.png
"""

from __future__ import annotations

# Ta linijka MUSI byc pierwszym kodem wykonywalnym w tym pliku - patrz
# szczegolowe wyjasnienie w common/uruchom_w_venv.py.
from common.uruchom_w_venv import przelacz_na_venv_jesli_trzeba

przelacz_na_venv_jesli_trzeba()

import argparse
import csv
import json
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
    decyzje_z_progiem,
    krzywa_i_eer,
    metryki_openset,
    podobienstwa_do_osob,
    podobienstwa_knn,
)
from common.model import pobierz_urzadzenie, zbuduj_ekstraktor_cech
from common.rozpoznawanie import NAZWA_NIEZNANEJ

KORZEN_PROJEKTU = Path(__file__).resolve().parent
FOLDER_PRZETWORZONYCH_DANYCH = KORZEN_PROJEKTU / "data" / "processed"
FOLDER_EKSPERYMENTOW = KORZEN_PROJEKTU / "models" / "eksperymenty"
FOLDER_WYKRESOW = FOLDER_EKSPERYMENTOW / "wykresy"
PLIK_RAPORTU = FOLDER_EKSPERYMENTOW / "raport_eksperymentow.json"
PLIK_CSV = FOLDER_EKSPERYMENTOW / "wyniki.csv"

NAZWA_KLASY_NIEZNAJOMY = "nieznajomy"
NAZWA_OBCEGO = "obcy (nieznajomy)"
WERSJA_RAPORTU = 2

# Rozmiary zbioru podajemy jako liczbe zdjec NA OSOBE. Dzieki temu wynik
# nie zalezy od tego, ile osob jest w bazie. "None" oznacza "wszystkie".
ROZMIARY_ZBIORU: tuple[int | None, ...] = (50, 100, 200, None)
WARTOSCI_K = (1, 3, 5, 9, 15)
PROGI_DECYZYJNE = tuple(round(0.50 + 0.05 * numer, 2) for numer in range(10))
MAKS_WZORCOW_NA_JASNOSC = 128
INDEKS_ORYGINALU = WSPOLCZYNNIKI_JASNOSCI.index(1.0)


# =========================================================================
# Postep (te same maszynowe linie co w 03 i 07 - panel GUI je rozumie)
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
        description="Eksperymenty: wplyw liczby zdjec, jasnosci, k w k-NN i progu decyzyjnego."
    )
    parser.add_argument("--batch-size", type=int, default=32, help="Ile zdjec naraz trafia do sieci")
    parser.add_argument("--seed", type=int, default=42, help="Ziarno losowosci (powtarzalnosc)")
    parser.add_argument(
        "--postep",
        action="store_true",
        help="wypisuje maszynowe linie postepu dla panelu GUI",
    )
    return parser.parse_args()


# =========================================================================
# Dane i prototypy
# =========================================================================
def zbuduj_prototypy(
    trening: torch.Tensor,
    etykiety: list[int],
    osoby: list[str],
    liczba_wariantow: int,
    maks_wzorcow: int = MAKS_WZORCOW_NA_JASNOSC,
) -> dict[str, dict]:
    """Buduje wzorce dokladnie tak jak skrypt 03.

    Dla kazdej osoby bierzemy rownomiernie rozlozone zdjecia z kazdego
    wariantu jasnosci (maks. 128 na wariant) - to zapobiega sytuacji,
    w ktorej jedna seria zdjec zdominuje wzorce.
    """
    wzorce: dict[str, dict] = {}
    for indeks_osoby, osoba in enumerate(osoby):
        maska = torch.tensor(
            [etykieta == indeks_osoby for etykieta in etykiety], dtype=torch.bool
        )
        embeddingi_osoby = trening[maska]
        if embeddingi_osoby.shape[0] == 0:
            continue

        prototypy: list[torch.Tensor] = []
        for blok in embeddingi_osoby.chunk(liczba_wariantow):
            liczba = min(maks_wzorcow, blok.shape[0])
            indeksy = torch.linspace(0, blok.shape[0] - 1, liczba).round().long()
            prototypy.append(blok[indeksy])
        wzorce[osoba] = {"prototypy": torch.cat(prototypy, dim=0)}
    return wzorce


def wybierz_zdjecia(
    klasy: list[str], osoby: list[str], limit_na_osobe: int | None
) -> tuple[list[int], dict[str, int]]:
    """Wybiera zdjecia do eksperymentu - rownomiernie z calej serii osoby.

    Zwraca pozycje w tablicy zdjec oraz faktyczna liczbe zdjec na osobe
    (moze byc mniejsza od limitu, jesli osoba ma ich za malo).
    """
    wybrane: list[int] = []
    faktyczne: dict[str, int] = {}
    for osoba in osoby:
        pozycje = [numer for numer, nazwa in enumerate(klasy) if nazwa == osoba]
        if limit_na_osobe is not None and len(pozycje) > limit_na_osobe:
            indeksy = np.linspace(
                0, len(pozycje) - 1, limit_na_osobe
            ).round().astype(int)
            pozycje = [pozycje[int(indeks)] for indeks in indeksy]
        faktyczne[osoba] = len(pozycje)
        wybrane.extend(pozycje)
    return sorted(set(wybrane)), faktyczne


def dane_treningu(
    warianty: list[torch.Tensor],
    etykiety: list[int],
    wybrane: list[int],
    liczba_wariantow: int,
) -> tuple[torch.Tensor, list[int]]:
    """Skleja wybrane zdjecia z wybranych wariantow jasnosci."""
    fragmenty = [wariant[wybrane] for wariant in warianty[:liczba_wariantow]]
    etykiety_wybrane = [etykiety[numer] for numer in wybrane]
    return torch.cat(fragmenty, dim=0), etykiety_wybrane * liczba_wariantow


# =========================================================================
# Metryki eksperymentu
# =========================================================================
def _wlasne_i_obce(
    podobienstwa_znane: np.ndarray,
    podobienstwa_obce: np.ndarray,
    indeksy_prawdziwe: list[int],
) -> tuple[np.ndarray, np.ndarray]:
    """Zwraca dwa rankingi: podobienstwo do wlasnej osoby i do najlepszego obcego."""
    indeksy = np.arange(len(podobienstwa_znane))
    wlasne = podobienstwa_znane[indeksy, indeksy_prawdziwe]
    obce = (
        podobienstwa_obce.max(axis=1)
        if len(podobienstwa_obce)
        else np.zeros(0, dtype=float)
    )
    return wlasne, obce


def wybierz_prog(
    podobienstwa_znane: np.ndarray,
    podobienstwa_obce: np.ndarray,
    indeksy_prawdziwe: list[int],
) -> float:
    """Wyznacza prog decyzyjny w punkcie EER na zbiorze WALIDACYJNYM.

    Ten prog zostanie pozniej zastosowany bez zmian do zbioru testowego.
    Dzieki temu test nie wplywa na wybor progu - inaczej wynik bylby
    optymistyczny.
    """
    wlasne, obce = _wlasne_i_obce(
        podobienstwa_znane, podobienstwa_obce, indeksy_prawdziwe
    )
    krzywa = krzywa_i_eer(wlasne, obce)
    return float(krzywa["prog"]) if krzywa else 0.5


def ocen_podobienstwa(
    podobienstwa_znane: np.ndarray,
    podobienstwa_obce: np.ndarray,
    prawdziwe_znane: list[str],
    indeksy_prawdziwe: list[int],
    osoby: list[str],
    prog: float,
) -> dict:
    """Liczy metryki na zbiorze TESTOWYM przy progu wybranym na walidacji.

    EER i AUC nie zaleza od wybranego progu - opisuja sam ranking
    podobienstw - dlatego liczymy je bezposrednio na tescie.
    """
    wlasne, obce = _wlasne_i_obce(
        podobienstwa_znane, podobienstwa_obce, indeksy_prawdziwe
    )
    krzywa = krzywa_i_eer(wlasne, obce)

    przewidziane_znane, _ = decyzje_z_progiem(podobienstwa_znane, osoby, prog)
    przewidziane_obce, _ = decyzje_z_progiem(podobienstwa_obce, osoby, prog)
    metryki = metryki_openset(
        prawdziwe_znane, przewidziane_znane, przewidziane_obce
    )

    return {
        "accuracy_zamknieta": round(
            accuracy_zamknieta(podobienstwa_znane, indeksy_prawdziwe), 4
        ),
        "accuracy_otwarta": metryki["accuracy"],
        "far": metryki["far"],
        "frr": metryki["frr"],
        "eer": krzywa["eer"] if krzywa else None,
        "auc": auc(wlasne, obce),
        "prog_eer": round(prog, 4),
        "prog_zrodlo": "walidacja",
        "zaakceptowani_obcy": metryki["zaakceptowani_obcy"],
        "odrzucone_znane": metryki["odrzucone_znane"],
    }


def ocen_prototypy(
    trening: torch.Tensor,
    etykiety_trening: list[int],
    osoby: list[str],
    liczba_wariantow: int,
    walidacja_znane: torch.Tensor,
    walidacja_obce: torch.Tensor,
    indeksy_prawdziwe_val: list[int],
    test_znane: torch.Tensor,
    test_obce: torch.Tensor,
    prawdziwe_test: list[str],
    indeksy_prawdziwe_test: list[int],
) -> tuple[dict, int]:
    """Buduje prototypy, wybiera prog na walidacji i ocenia na tescie.

    Zwraca metryki testowe oraz liczbe uzytych prototypow.
    """
    wzorce = zbuduj_prototypy(
        trening, etykiety_trening, osoby, liczba_wariantow
    )
    pod_val_znane = podobienstwa_do_osob(walidacja_znane, wzorce, osoby)
    pod_val_obce = podobienstwa_do_osob(walidacja_obce, wzorce, osoby)
    prog = wybierz_prog(pod_val_znane, pod_val_obce, indeksy_prawdziwe_val)

    pod_test_znane = podobienstwa_do_osob(test_znane, wzorce, osoby)
    pod_test_obce = podobienstwa_do_osob(test_obce, wzorce, osoby)
    metryki = ocen_podobienstwa(
        pod_test_znane,
        pod_test_obce,
        prawdziwe_test,
        indeksy_prawdziwe_test,
        osoby,
        prog,
    )
    liczba_prototypow = sum(
        int(wpis["prototypy"].shape[0]) for wpis in wzorce.values()
    )
    return metryki, liczba_prototypow


# =========================================================================
# Wykresy
# =========================================================================
def _matplotlib():
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    return plt


def _wykres_liniowy(
    etykiety_x: list[str],
    serie: list[tuple[str, list[float | None], str, str]],
    tytul: str,
    opis_x: str,
    sciezka: Path,
) -> None:
    """Wspolny wykres liniowy: kilka serii, brak danych oznaczony przerwa."""
    plt = _matplotlib()
    rysunek, os = plt.subplots(figsize=(8.5, 4.8))
    for nazwa, wartosci, kolor, znacznik in serie:
        punkty = [
            (numer, wartosc)
            for numer, wartosc in enumerate(wartosci)
            if wartosc is not None
        ]
        if not punkty:
            continue
        os.plot(
            [punkt[0] for punkt in punkty],
            [punkt[1] for punkt in punkty],
            marker=znacznik,
            linewidth=2,
            label=nazwa,
            color=kolor,
        )
    os.set_xticks(range(len(etykiety_x)), etykiety_x)
    os.set_ylim(-0.02, 1.02)
    os.set_xlabel(opis_x)
    os.set_ylabel("wartosc (0-1)")
    os.set_title(tytul)
    os.grid(alpha=0.25)
    os.legend(fontsize=8)
    rysunek.tight_layout()
    rysunek.savefig(sciezka, dpi=140)
    plt.close(rysunek)


def _wykres_rozmiaru(rozmiary: list[dict], sciezka: Path) -> None:
    _wykres_liniowy(
        [wpis["etykieta"] for wpis in rozmiary],
        [
            (
                "accuracy zamknieta",
                [wpis["metryki"]["accuracy_zamknieta"] for wpis in rozmiary],
                "#0d6efd",
                "o",
            ),
            (
                "accuracy otwarta",
                [wpis["metryki"]["accuracy_otwarta"] for wpis in rozmiary],
                "#198754",
                "s",
            ),
            (
                "EER (mniej = lepiej)",
                [wpis["metryki"]["eer"] for wpis in rozmiary],
                "#fd7e14",
                "^",
            ),
        ],
        "Wplyw liczby zdjec treningowych na jakosc modelu",
        "zdjecia na osobe",
        sciezka,
    )


def _wykres_jasnosci(jasnosci: list[dict], sciezka: Path) -> None:
    plt = _matplotlib()
    grupy = [wpis["etykieta"] for wpis in jasnosci]
    metryki = [
        ("accuracy zamknieta", "accuracy_zamknieta", "#0d6efd"),
        ("accuracy otwarta", "accuracy_otwarta", "#198754"),
        ("EER", "eer", "#fd7e14"),
    ]
    x = np.arange(len(grupy))
    szerokosc = 0.8 / len(metryki)

    rysunek, os = plt.subplots(figsize=(8.5, 4.8))
    for numer, (nazwa, klucz, kolor) in enumerate(metryki):
        wartosci = [wpis["metryki"][klucz] or 0.0 for wpis in jasnosci]
        przesuniecie = (numer - (len(metryki) - 1) / 2) * szerokosc
        os.bar(x + przesuniecie, wartosci, szerokosc, label=nazwa, color=kolor)
        for indeks, wartosc in enumerate(wartosci):
            os.text(
                indeks + przesuniecie,
                wartosc + 0.02,
                f"{wartosc:.0%}",
                ha="center",
                fontsize=7,
            )

    os.set_xticks(x, grupy)
    os.set_ylim(0, 1.08)
    os.set_ylabel("wartosc (0-1)")
    os.set_title("Wplyw augmentacji jasnosci na jakosc modelu")
    os.legend(fontsize=8)
    os.grid(axis="y", alpha=0.25)
    rysunek.tight_layout()
    rysunek.savefig(sciezka, dpi=140)
    plt.close(rysunek)


def _wykres_k(knn: list[dict], sciezka: Path) -> None:
    _wykres_liniowy(
        [f"k={wpis['k']}" for wpis in knn],
        [
            (
                "accuracy zamknieta",
                [wpis["metryki"]["accuracy_zamknieta"] for wpis in knn],
                "#0d6efd",
                "o",
            ),
            (
                "accuracy otwarta",
                [wpis["metryki"]["accuracy_otwarta"] for wpis in knn],
                "#198754",
                "s",
            ),
            (
                "EER (mniej = lepiej)",
                [wpis["metryki"]["eer"] for wpis in knn],
                "#fd7e14",
                "^",
            ),
        ],
        "Wplyw liczby sasiadow k w k-NN",
        "liczba sasiadow k",
        sciezka,
    )


def _wykres_progu(progi: list[dict], sciezka: Path) -> None:
    _wykres_liniowy(
        [f"{wpis['prog']:.2f}" for wpis in progi],
        [
            ("accuracy otwarta", [wpis["accuracy"] for wpis in progi], "#0d6efd", "o"),
            ("FAR (obcy wpuszczeni)", [wpis["far"] for wpis in progi], "#dc3545", "s"),
            ("FRR (swoi odrzuceni)", [wpis["frr"] for wpis in progi], "#fd7e14", "^"),
        ],
        "Kompromis przy roznych progach decyzyjnych",
        "prog podobienstwa",
        sciezka,
    )


# =========================================================================
# Wnioski
# =========================================================================
def zbuduj_wnioski(
    rozmiary: list[dict], jasnosci: list[dict], knn: list[dict], progi: list[dict]
) -> list[str]:
    """Zamienia tabele wynikow na krotkie, ostrozne wnioski."""
    wnioski: list[str] = []
    if rozmiary:
        najmniejszy = rozmiary[0]
        wszystkie = rozmiary[-1]
        roznica = (
            wszystkie["metryki"]["accuracy_zamknieta"]
            - najmniejszy["metryki"]["accuracy_zamknieta"]
        )
        kierunek = "wzrosla" if roznica > 0 else ("spadla" if roznica < 0 else "nie zmienila sie")
        if abs(roznica) < 1e-9:
            opis_rozmiaru = (
                f"Przy {najmniejszy['etykieta']} zdjeciach accuracy zamknieta "
                f"wyniosla {najmniejszy['metryki']['accuracy_zamknieta']:.1%} - "
                "dokladnie tyle samo co przy wszystkich zdjeciach. Na tym zbiorze "
                "juz mala proba wystarcza, wiec nie widac korzysci z dokladania "
                "danych. To nie znaczy, ze na trudniejszych zdjeciach byloby tak samo."
            )
            wnioski.append(opis_rozmiaru)
        else:
            wnioski.append(
                f"Przy {najmniejszy['etykieta']} zdjeciach accuracy zamknieta wyniosla "
                f"{najmniejszy['metryki']['accuracy_zamknieta']:.1%}, a przy wszystkich "
                f"zdjeciach {wszystkie['metryki']['accuracy_zamknieta']:.1%} - {kierunek} "
                f"o {abs(roznica) * 100:.1f} pkt proc. To jeden pomiar, nie dowod: "
                "wynik zalezy tez od tego, ktore zdjecia wybralismy."
            )

        eery_rozmiarow = [
            wpis["metryki"]["eer"]
            for wpis in rozmiary
            if wpis["metryki"]["eer"] is not None
        ]
        if eery_rozmiarow:
            najnizszy_eer = min(eery_rozmiarow)
            najlepsze = [
                f"'{wpis['etykieta']}'"
                for wpis in rozmiary
                if wpis["metryki"]["eer"] is not None
                and abs(wpis["metryki"]["eer"] - najnizszy_eer) < 1e-9
            ]
            wnioski.append(
                f"Najnizszy EER w eksperymencie z liczba zdjec "
                f"({najnizszy_eer:.1%}) osiagnely warianty: "
                f"{', '.join(najlepsze)}. Przy mniejszej liczbie zdjec prototypy "
                "sa mniej roznorodne, wiec model moze latwiej mylic osoby podobne "
                "- ale przy malej probie wynik moze byc tez przypadkiem."
            )

        liczba_prototypow = {
            wpis["liczba_prototypow"] for wpis in rozmiary
        }
        if len(liczba_prototypow) < len(rozmiary):
            wnioski.append(
                "Uwaga: produkcja (skrypt 03) ogranicza liczbe prototypow do "
                f"{MAKS_WZORCOW_NA_JASNOSC} na wariant jasnosci. Dlatego warianty "
                "'200' i 'wszystkie' maja tyle samo prototypow i daja identyczny "
                "wynik - dodatkowe zdjecia nie zmieniaja decyzji, dopoki nie "
                "podniesiemy tego limitu."
            )

    if len(jasnosci) == 2:
        oryginal, warianty = jasnosci[0], jasnosci[1]
        wnioski.append(
            "Trzy warianty jasnosci: accuracy zamknieta "
            f"{warianty['metryki']['accuracy_zamknieta']:.1%} vs "
            f"{oryginal['metryki']['accuracy_zamknieta']:.1%} dla samego oryginalu, "
            f"EER {warianty['metryki']['eer']:.1%} vs "
            f"{oryginal['metryki']['eer']:.1%}. Augmentacja pomaga tylko wtedy, "
            "gdy realne zdjecia roznia sie oswietleniem - trzeba to sprawdzic "
            "na wlasnych danych, a nie zakladac z gory."
        )

    if knn:
        wartosci_eer = [
            wpis["metryki"]["eer"]
            for wpis in knn
            if wpis["metryki"]["eer"] is not None
        ]
        if len({round(wartosc, 4) for wartosc in wartosci_eer}) <= 1:
            lista_k = ", ".join(str(wpis["k"]) for wpis in knn)
            wnioski.append(
                f"Wszystkie badane wartosci k ({lista_k}) daja taki sam wynik "
                "EER. To znak, ze przy tych danych ranking podobienstw jest juz "
                "bardzo wyrazny - liczba sasiadow nie zmienia decyzji. Na "
                "trudniejszym zbiorze roznice miedzy k moga sie pojawic."
            )
        else:
            najlepszy_k = min(
                knn,
                key=lambda wpis: (
                    wpis["metryki"]["eer"]
                    if wpis["metryki"]["eer"] is not None
                    else 1.0
                ),
            )
            najgorszy_k = max(
                knn,
                key=lambda wpis: (
                    wpis["metryki"]["eer"]
                    if wpis["metryki"]["eer"] is not None
                    else 0.0
                ),
            )
            wnioski.append(
                f"Najlepsze EER dla k-NN: k={najlepszy_k['k']} "
                f"({najlepszy_k['metryki']['eer']:.1%}), najgorsze: "
                f"k={najgorszy_k['k']} ({najgorszy_k['metryki']['eer']:.1%}). "
                "Male k jest czule na pojedyncze zdjecie (szum), duze k wygladza "
                "decyzje, ale moze mieszac osoby podobne."
            )

    if progi:
        najostrzejszy = progi[-1]
        najlagszy = progi[0]
        wnioski.append(
            f"Przy progu {najlagszy['prog']:.2f} FAR={najlagszy['far']:.1%}, "
            f"FRR={najlagszy['frr']:.1%}; przy progu {najostrzejszy['prog']:.2f} "
            f"FAR={najostrzejszy['far']:.1%}, FRR={najostrzejszy['frr']:.1%}. "
            "Nie ma progu idealnego - wybor zalezy od tego, co jest gorsze "
            "w danym zastosowaniu."
        )

    wnioski.append(
        "Wszystkie metryki powyzej pochodza ze zbioru testowego, ktorego nie "
        "uzyto do budowy prototypow ani do wyboru progu. Prog decyzyjny (EER) "
        "wybrano na walidacji i zastosowano bez zmian."
    )

    return wnioski


# =========================================================================
# CSV
# =========================================================================
def zapisz_csv(
    rozmiary: list[dict], jasnosci: list[dict], knn: list[dict], progi: list[dict]
) -> None:
    """Zapisuje wszystkie wyniki w jednym pliku - latwym do Excela."""
    kolumny = [
        "eksperyment",
        "parametr",
        "liczba_zdjec_treningowych",
        "liczba_prototypow",
        "accuracy_zamknieta",
        "accuracy_otwarta",
        "far",
        "frr",
        "eer",
        "auc",
        "prog_eer",
    ]

    with open(PLIK_CSV, "w", encoding="utf-8", newline="") as plik:
        pisarz = csv.writer(plik, delimiter=";", lineterminator="\n")
        pisarz.writerow(kolumny)

        for wpis in rozmiary:
            pisarz.writerow(
                [
                    "rozmiar_zbioru",
                    wpis["etykieta"],
                    wpis["liczba_zdjec_treningowych"],
                    wpis["liczba_prototypow"],
                    wpis["metryki"]["accuracy_zamknieta"],
                    wpis["metryki"]["accuracy_otwarta"],
                    wpis["metryki"]["far"],
                    wpis["metryki"]["frr"],
                    wpis["metryki"]["eer"],
                    wpis["metryki"]["auc"],
                    wpis["metryki"]["prog_eer"],
                ]
            )

        for wpis in jasnosci:
            pisarz.writerow(
                [
                    "jasnosc",
                    wpis["etykieta"],
                    wpis["liczba_zdjec_treningowych"],
                    wpis["liczba_prototypow"],
                    wpis["metryki"]["accuracy_zamknieta"],
                    wpis["metryki"]["accuracy_otwarta"],
                    wpis["metryki"]["far"],
                    wpis["metryki"]["frr"],
                    wpis["metryki"]["eer"],
                    wpis["metryki"]["auc"],
                    wpis["metryki"]["prog_eer"],
                ]
            )

        for wpis in knn:
            pisarz.writerow(
                [
                    "knn",
                    f"k={wpis['k']}",
                    wpis["liczba_zdjec_treningowych"],
                    "",
                    wpis["metryki"]["accuracy_zamknieta"],
                    wpis["metryki"]["accuracy_otwarta"],
                    wpis["metryki"]["far"],
                    wpis["metryki"]["frr"],
                    wpis["metryki"]["eer"],
                    wpis["metryki"]["auc"],
                    wpis["metryki"]["prog_eer"],
                ]
            )

        for wpis in progi:
            pisarz.writerow(
                [
                    "prog_decyzyjny",
                    f"prog={wpis['prog']:.2f}",
                    wpis["liczba_zdjec_treningowych"],
                    wpis["liczba_prototypow"],
                    wpis["accuracy_zamknieta"],
                    wpis["accuracy"],
                    wpis["far"],
                    wpis["frr"],
                    "",
                    "",
                    "",
                ]
            )


# =========================================================================
# Glowna procedura
# =========================================================================
def main() -> None:
    argumenty = parsuj_argumenty()
    if argumenty.batch_size <= 0:
        raise SystemExit("--batch-size musi byc dodatni.")

    folder_train = FOLDER_PRZETWORZONYCH_DANYCH / "train"
    folder_val = FOLDER_PRZETWORZONYCH_DANYCH / "val"
    folder_test = FOLDER_PRZETWORZONYCH_DANYCH / "test"

    if (
        not folder_train.is_dir()
        or not folder_val.is_dir()
        or not folder_test.is_dir()
    ):
        raise SystemExit(
            f"Brak kompletu zbiorow w {FOLDER_PRZETWORZONYCH_DANYCH} "
            "(potrzebne train, val i test). Uruchom ponownie 02_podzial_danych.py."
        )

    FOLDER_EKSPERYMENTOW.mkdir(parents=True, exist_ok=True)
    FOLDER_WYKRESOW.mkdir(parents=True, exist_ok=True)
    start = time.perf_counter()

    torch.manual_seed(argumenty.seed)
    np.random.seed(argumenty.seed)

    urzadzenie = pobierz_urzadzenie()
    ekstraktor = zbuduj_ekstraktor_cech().to(urzadzenie)

    liczba_train = policz_obrazy(folder_train)
    liczba_val = policz_obrazy(folder_val)
    liczba_test = policz_obrazy(folder_test)
    postep = Postep(argumenty.postep)
    postep.start(
        "embeddingi",
        liczba_train * len(WSPOLCZYNNIKI_JASNOSCI) + liczba_val + liczba_test,
    )

    # ---------------------------------------------------------- trening
    print("[info] Liczenie embeddingow treningu (3 warianty jasnosci)...")
    warianty_wszystkie: list[torch.Tensor] = []
    etykiety_wszystkie: list[int] = []
    klasy_wszystkie: list[str] = []
    for jasnosc in WSPOLCZYNNIKI_JASNOSCI:
        embeddingi, etykiety, klasy, _ = policz_embeddingi_folderu(
            ekstraktor,
            folder_train,
            urzadzenie,
            argumenty.batch_size,
            transformacja(jasnosc),
            postep,
        )
        if not klasy_wszystkie:
            klasy_wszystkie = klasy
            etykiety_wszystkie = etykiety
        elif klasy != klasy_wszystkie or etykiety != etykiety_wszystkie:
            raise SystemExit("Kolejnosc zdjec zmienila sie miedzy wariantami jasnosci.")
        warianty_wszystkie.append(embeddingi)

    # Osoby bierzemy z folderu treningowego, ale pomijamy klase "nieznajomy".
    osoby = [nazwa for nazwa in klasy_wszystkie if nazwa != NAZWA_KLASY_NIEZNAJOMY]
    if not osoby:
        raise SystemExit("Brak znanych osob w zbiorze treningowym.")

    pozycje_znane = [
        numer
        for numer, etykieta in enumerate(etykiety_wszystkie)
        if klasy_wszystkie[etykieta] in osoby
    ]
    klasy_znane = [klasy_wszystkie[etykiety_wszystkie[numer]] for numer in pozycje_znane]
    etykiety_znane = [osoby.index(nazwa) for nazwa in klasy_znane]
    warianty_znane = [wariant[pozycje_znane] for wariant in warianty_wszystkie]
    dostepnosc = {
        osoba: sum(1 for nazwa in klasy_znane if nazwa == osoba) for osoba in osoby
    }
    # "Obcy" do FAR ze zbioru treningowego: klasa "nieznajomy" w oryginalnej
    # jasnosci. Nie sluzy ona do budowy prototypow (te powstaja tylko dla
    # znanych osob), wiec mozna jej uzyc jako przykladu obcej twarzy.
    maska_obcych_trening = np.array(
        [
            klasy_wszystkie[etykieta] == NAZWA_KLASY_NIEZNAJOMY
            for etykieta in etykiety_wszystkie
        ],
        dtype=bool,
    )
    obce_trening = warianty_wszystkie[INDEKS_ORYGINALU][
        torch.tensor(maska_obcych_trening)
    ]

    print(
        f"[info] Osoby ({len(osoby)}): {osoby} | dostepne zdjecia na osobe: {dostepnosc}"
    )
    print(f"[info] Urzadzenie: {urzadzenie}")

    # -------------------------------------------------------- walidacja
    # Val sluzy TYLKO do wyboru progu decyzyjnego. Metryki liczymy na tescie.
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
        numer for numer, nazwa in enumerate(prawdziwe_val) if nazwa in osoby
    ]
    indeksy_obce_val = [
        numer
        for numer, nazwa in enumerate(prawdziwe_val)
        if nazwa == NAZWA_KLASY_NIEZNAJOMY
    ]
    if not indeksy_znane_val:
        raise SystemExit(
            "W zbiorze walidacyjnym nie ma zadnej osoby z treningu. "
            "Uruchom ponownie 02 i 03."
        )

    embeddingi_val_znane = embeddingi_val[indeksy_znane_val]
    indeksy_prawdziwe_val = [
        osoby.index(prawdziwe_val[numer]) for numer in indeksy_znane_val
    ]
    embeddingi_val_obce = embeddingi_val[indeksy_obce_val]
    print(
        f"[info] Walidacja: {len(embeddingi_val_znane)} zdjec znanych osob, "
        f"{len(embeddingi_val_obce)} zdjec klasy 'nieznajomy'."
    )

    # ------------------------------------------------------------- test
    # Test jest nietkniety az do finalnego pomiaru - tu tylko go wczytujemy.
    print("[info] Liczenie embeddingow zbioru testowego...")
    embeddingi_test, etykiety_test, klasy_test, _ = policz_embeddingi_folderu(
        ekstraktor,
        folder_test,
        urzadzenie,
        argumenty.batch_size,
        transformacja(1.0),
        postep,
    )
    prawdziwe_test = [klasy_test[etykieta] for etykieta in etykiety_test]
    indeksy_znane_test = [
        numer for numer, nazwa in enumerate(prawdziwe_test) if nazwa in osoby
    ]
    indeksy_obce_test = [
        numer
        for numer, nazwa in enumerate(prawdziwe_test)
        if nazwa == NAZWA_KLASY_NIEZNAJOMY
    ]
    if not indeksy_znane_test:
        raise SystemExit(
            "W zbiorze testowym nie ma zadnej osoby z treningu. "
            "Uruchom ponownie 02 i 03."
        )

    embeddingi_test_znane = embeddingi_test[indeksy_znane_test]
    prawdziwe_test_znane = [prawdziwe_test[numer] for numer in indeksy_znane_test]
    indeksy_prawdziwe_test = [
        osoby.index(nazwa) for nazwa in prawdziwe_test_znane
    ]
    print(
        f"[info] Test: {len(embeddingi_test_znane)} zdjec znanych osob, "
        f"{len(indeksy_obce_test)} zdjec klasy 'nieznajomy'."
    )

    # ---------------------------------------------------------- obcy
    # Obcy = klasa "nieznajomy" z treningu i testu, celowo BEZ val, ktora
    # posluzyla do kalibracji progow. Inaczej FAR bylby zawyzony.
    czesci_obce = [obce_trening]
    if indeksy_obce_test:
        czesci_obce.append(embeddingi_test[indeksy_obce_test])
    if obce_trening.shape[0] or indeksy_obce_test:
        embeddingi_obce = torch.cat(czesci_obce, dim=0)
    else:
        print("[uwaga] Brak zdjec klasy 'nieznajomy' - nie policze FAR.")
        embeddingi_obce = torch.empty((0, embeddingi_test_znane.shape[1]))
    print(f"[info] Obcy: {len(embeddingi_obce)} zdjec (train + test, bez val).")

    # ------------------------------------------------------ eksperymenty
    liczba_eksperymentow = (
        len(ROZMIARY_ZBIORU)
        + 2
        + len(WARTOSCI_K)
        + len(PROGI_DECYZYJNE)
        + 4
    )
    postep.start("eksperymenty", liczba_eksperymentow)

    # A. liczba zdjec
    rozmiary: list[dict] = []
    for limit in ROZMIARY_ZBIORU:
        wybrane, faktyczne = wybierz_zdjecia(klasy_znane, osoby, limit)
        trening, etykiety = dane_treningu(
            warianty_znane, etykiety_znane, wybrane, len(WSPOLCZYNNIKI_JASNOSCI)
        )
        metryki, liczba_prototypow = ocen_prototypy(
            trening,
            etykiety,
            osoby,
            len(WSPOLCZYNNIKI_JASNOSCI),
            embeddingi_val_znane,
            embeddingi_val_obce,
            indeksy_prawdziwe_val,
            embeddingi_test_znane,
            embeddingi_obce,
            prawdziwe_test_znane,
            indeksy_prawdziwe_test,
        )
        etykieta = "wszystkie" if limit is None else str(limit)
        rozmiary.append(
            {
                "etykieta": etykieta,
                "limit_na_osobe": limit,
                "liczba_zdjec_na_osobe": faktyczne,
                "liczba_zdjec_treningowych": len(wybrane)
                * len(WSPOLCZYNNIKI_JASNOSCI),
                "liczba_prototypow": liczba_prototypow,
                "metryki": metryki,
            }
        )
        if metryki["eer"] is not None:
            print(
                f"[info] Rozmiar {etykieta}: accuracy zamknieta "
                f"{metryki['accuracy_zamknieta']:.1%}, EER {metryki['eer']:.1%}"
            )
        else:
            print(
                f"[info] Rozmiar {etykieta}: accuracy zamknieta "
                f"{metryki['accuracy_zamknieta']:.1%}, EER brak danych"
            )
        postep.dodaj(1)

    # B. jasnosc
    jasnosci: list[dict] = []
    warianty_badane = [
        ("tylko oryginal", [INDEKS_ORYGINALU], 1),
        (
            "3 warianty jasnosci",
            list(range(len(WSPOLCZYNNIKI_JASNOSCI))),
            len(WSPOLCZYNNIKI_JASNOSCI),
        ),
    ]
    for etykieta, indeksy_wariantow, liczba_wariantow in warianty_badane:
        wybrane, faktyczne = wybierz_zdjecia(klasy_znane, osoby, None)
        fragmenty = [warianty_znane[indeks][wybrane] for indeks in indeksy_wariantow]
        trening = torch.cat(fragmenty, dim=0)
        etykiety = [etykiety_znane[numer] for numer in wybrane] * liczba_wariantow
        metryki, liczba_prototypow = ocen_prototypy(
            trening,
            etykiety,
            osoby,
            liczba_wariantow,
            embeddingi_val_znane,
            embeddingi_val_obce,
            indeksy_prawdziwe_val,
            embeddingi_test_znane,
            embeddingi_obce,
            prawdziwe_test_znane,
            indeksy_prawdziwe_test,
        )
        jasnosci.append(
            {
                "etykieta": etykieta,
                "liczba_wariantow": liczba_wariantow,
                "liczba_zdjec_na_osobe": faktyczne,
                "liczba_zdjec_treningowych": len(wybrane) * liczba_wariantow,
                "liczba_prototypow": liczba_prototypow,
                "metryki": metryki,
            }
        )
        if metryki["eer"] is not None:
            print(
                f"[info] Jasnosc '{etykieta}': accuracy zamknieta "
                f"{metryki['accuracy_zamknieta']:.1%}, EER {metryki['eer']:.1%}"
            )
        else:
            print(
                f"[info] Jasnosc '{etykieta}': accuracy zamknieta "
                f"{metryki['accuracy_zamknieta']:.1%}, EER brak danych"
            )
        postep.dodaj(1)

    # C. k-NN
    wybrane, faktyczne = wybierz_zdjecia(klasy_znane, osoby, None)
    trening_knn = torch.cat([wariant[wybrane] for wariant in warianty_znane], dim=0)
    etykiety_knn = [etykiety_znane[numer] for numer in wybrane] * len(
        WSPOLCZYNNIKI_JASNOSCI
    )
    zapytania_val = torch.cat([embeddingi_val_znane, embeddingi_val_obce], dim=0)
    zapytania_test = torch.cat([embeddingi_test_znane, embeddingi_obce], dim=0)
    knn: list[dict] = []
    for k in WARTOSCI_K:
        # Prog wybieramy na walidacji, metryki liczymy na tescie.
        pod_val = podobienstwa_knn(
            zapytania_val, trening_knn, etykiety_knn, osoby, osoby, k
        )
        prog = wybierz_prog(
            pod_val[: len(embeddingi_val_znane)],
            pod_val[len(embeddingi_val_znane) :],
            indeksy_prawdziwe_val,
        )

        pod_test = podobienstwa_knn(
            zapytania_test, trening_knn, etykiety_knn, osoby, osoby, k
        )
        metryki = ocen_podobienstwa(
            pod_test[: len(embeddingi_test_znane)],
            pod_test[len(embeddingi_test_znane) :],
            prawdziwe_test_znane,
            indeksy_prawdziwe_test,
            osoby,
            prog,
        )
        knn.append(
            {
                "k": k,
                "liczba_zdjec_treningowych": len(etykiety_knn),
                "metryki": metryki,
            }
        )
        if metryki["eer"] is not None:
            print(
                f"[info] k-NN k={k}: accuracy zamknieta "
                f"{metryki['accuracy_zamknieta']:.1%}, EER {metryki['eer']:.1%}"
            )
        else:
            print(
                f"[info] k-NN k={k}: accuracy zamknieta "
                f"{metryki['accuracy_zamknieta']:.1%}, EER brak danych"
            )
        postep.dodaj(1)

    # D. prog decyzyjny (na prototypach ze wszystkich zdjec)
    wybrane, faktyczne = wybierz_zdjecia(klasy_znane, osoby, None)
    trening, etykiety = dane_treningu(
        warianty_znane, etykiety_znane, wybrane, len(WSPOLCZYNNIKI_JASNOSCI)
    )
    wzorce = zbuduj_prototypy(
        trening, etykiety, osoby, len(WSPOLCZYNNIKI_JASNOSCI)
    )
    pod_val_znane = podobienstwa_do_osob(embeddingi_val_znane, wzorce, osoby)
    pod_val_obce = podobienstwa_do_osob(embeddingi_val_obce, wzorce, osoby)
    prog_wybrany = wybierz_prog(pod_val_znane, pod_val_obce, indeksy_prawdziwe_val)
    podobienstwa_znane = podobienstwa_do_osob(embeddingi_test_znane, wzorce, osoby)
    podobienstwa_obce = podobienstwa_do_osob(embeddingi_obce, wzorce, osoby)
    liczba_prototypow = sum(
        int(wpis["prototypy"].shape[0]) for wpis in wzorce.values()
    )
    progi: list[dict] = []
    for prog in PROGI_DECYZYJNE:
        metryki = ocen_podobienstwa(
            podobienstwa_znane,
            podobienstwa_obce,
            prawdziwe_test_znane,
            indeksy_prawdziwe_test,
            osoby,
            prog,
        )
        progi.append(
            {
                "prog": prog,
                "liczba_zdjec_treningowych": len(wybrane)
                * len(WSPOLCZYNNIKI_JASNOSCI),
                "liczba_prototypow": liczba_prototypow,
                "accuracy_zamknieta": metryki["accuracy_zamknieta"],
                "accuracy": metryki["accuracy_otwarta"],
                "far": metryki["far"],
                "frr": metryki["frr"],
                "zaakceptowani_obcy": metryki["zaakceptowani_obcy"],
                "odrzucone_znane": metryki["odrzucone_znane"],
            }
        )
        postep.dodaj(1)
    print(
        f"[info] Tabela progow: {len(progi)} wartosci od 0.50 do 0.95 "
        f"(prog EER z walidacji: {prog_wybrany:.2f})."
    )

    # ---------------------------------------------------------- wykresy
    wykresy = {
        "wplyw_liczby_zdjec": "wplyw_liczby_zdjec.png",
        "wplyw_jasnosci": "wplyw_jasnosci.png",
        "wplyw_k": "wplyw_k.png",
        "prog_decyzyjny": "prog_decyzyjny.png",
    }
    try:
        _wykres_rozmiaru(rozmiary, FOLDER_WYKRESOW / wykresy["wplyw_liczby_zdjec"])
        postep.dodaj(1)
        _wykres_jasnosci(jasnosci, FOLDER_WYKRESOW / wykresy["wplyw_jasnosci"])
        postep.dodaj(1)
        _wykres_k(knn, FOLDER_WYKRESOW / wykresy["wplyw_k"])
        postep.dodaj(1)
        _wykres_progu(progi, FOLDER_WYKRESOW / wykresy["prog_decyzyjny"])
        postep.dodaj(1)
    except ImportError:
        print("[uwaga] matplotlib niedostepny - pomijam wykresy.")
        wykresy = {}

    # ------------------------------------------------------------ CSV
    try:
        zapisz_csv(rozmiary, jasnosci, knn, progi)
    except OSError as blad:
        print(f"[uwaga] Nie udalo sie zapisac CSV: {blad}")

    # ---------------------------------------------------------- raport
    czas = time.perf_counter() - start
    ostrzezenia = [
        "Metryki wszystkich eksperymentow liczymy na zbiorze TESTOWYM, ktorego "
        "nie uzyto do budowy prototypow ani do wyboru progu. To uczciwy wynik, "
        "a nie dopasowany do danych kalibracyjnych.",
        "Prog decyzyjny (EER) wybieramy na zbiorze walidacyjnym i stosujemy bez "
        "zmian do testu. Walidacja byla tez uzyta w 03 do kalibracji progow - "
        "dlatego nie raportujemy na niej metryk.",
        "Test jest maly (kilkadziesiat zdjec), wiec pojedyncza pomylka zmienia "
        "wynik o kilka punktow procentowych. Roznice 1-2 punktow nie sa dowodem "
        "na wyzszosc jednej konfiguracji.",
    ]
    if len(embeddingi_obce):
        ostrzezenia.append(
            "Zdjecia obcych pochodza z klas 'nieznajomy' w zbiorach train i test "
            "(celowo bez val, ktory sluzyl do kalibracji progow)."
        )
    else:
        ostrzezenia.append(
            "Brak zdjec klasy 'nieznajomy' w train/test - nie policzono FAR, "
            "EER ani AUC."
        )
    ostrzezenia.append(
        "Eksperymenty pokazuja zaleznosc na JEDNYM zbiorze danych i jednym "
        "podziale train/val/test. Inny podzial moze dac inne wyniki - to normalne."
    )

    raport = {
        "wersja_raportu": WERSJA_RAPORTU,
        "wygenerowano": datetime.now().isoformat(timespec="seconds"),
        "czas_s": round(czas, 1),
        "urzadzenie": str(urzadzenie),
        "dane": {
            "osoby": osoby,
            "zdjecia_treningowe_na_osobe": dostepnosc,
            "zdjecia_treningowe_razem": len(klasy_znane),
            "walidacja": {
                "zdjecia_znanych": len(embeddingi_val_znane),
                "zdjecia_nieznajomych": len(indeksy_obce_val),
                "rola": "wybor progu decyzyjnego (EER)",
            },
            "test": {
                "zdjecia_znanych": len(embeddingi_test_znane),
                "zdjecia_nieznajomych": len(indeksy_obce_test),
                "rola": "ostateczny pomiar metryk",
            },
            "obcy": {
                "zdjecia": len(embeddingi_obce),
                "zrodlo": "data/processed/{train,test}/nieznajomy",
            },
            "warianty_jasnosci": list(WSPOLCZYNNIKI_JASNOSCI),
            "wymiar_cech": int(embeddingi_test_znane.shape[1]),
            "ostrzezenia": ostrzezenia,
        },
        "eksperymenty": {
            "rozmiar_zbioru": rozmiary,
            "jasnosc": jasnosci,
            "knn": knn,
            "prog_decyzyjny": progi,
        },
        "wnioski": zbuduj_wnioski(rozmiary, jasnosci, knn, progi),
        "wykresy": wykresy,
    }

    sciezka_tymczasowa = PLIK_RAPORTU.with_suffix(".json.tmp")
    with open(sciezka_tymczasowa, "w", encoding="utf-8") as plik:
        json.dump(raport, plik, ensure_ascii=False, indent=2)
    sciezka_tymczasowa.replace(PLIK_RAPORTU)

    # ------------------------------------------------------ podsumowanie
    print("\n" + "=" * 72)
    print("PODSUMOWANIE EKSPERYMENTOW (metryki na tescie, prog EER z walidacji)")
    print("=" * 72)
    for wpis in rozmiary:
        metryki = wpis["metryki"]
        eer = f"{metryki['eer']:.1%}" if metryki["eer"] is not None else "brak"
        print(
            f"  rozmiar {wpis['etykieta']:>10}: "
            f"accuracy(zamkn.)={metryki['accuracy_zamknieta']:.1%}  EER={eer}"
        )
    for wpis in jasnosci:
        metryki = wpis["metryki"]
        eer = f"{metryki['eer']:.1%}" if metryki["eer"] is not None else "brak"
        print(
            f"  jasnosc {wpis['etykieta']:>18}: "
            f"accuracy(zamkn.)={metryki['accuracy_zamknieta']:.1%}  EER={eer}"
        )
    for wpis in knn:
        metryki = wpis["metryki"]
        eer = f"{metryki['eer']:.1%}" if metryki["eer"] is not None else "brak"
        print(
            f"  k-NN k={wpis['k']:>2}: "
            f"accuracy(zamkn.)={metryki['accuracy_zamknieta']:.1%}  EER={eer}"
        )
    print("-" * 72)
    print("  prog   accuracy(otw.)     FAR       FRR")
    for wpis in progi:
        print(
            f"  {wpis['prog']:.2f}   {wpis['accuracy']:.1%}          "
            f"{wpis['far']:.1%}     {wpis['frr']:.1%}"
        )
    print("=" * 72)
    for wniosek in raport["wnioski"]:
        print(f"  - {wniosek}")
    print("=" * 72)
    print(f"[gotowe] Raport: {PLIK_RAPORTU}")
    print(f"[gotowe] CSV:    {PLIK_CSV}")
    print(f"[gotowe] Wykresy: {FOLDER_WYKRESOW}")
    print(
        "[uwaga] Metryki pochodza ze zbioru testowego, a prog EER wybrano na "
        "walidacji - wynik jest uczciwy, ale obarczony niepewnoscia malej proby."
    )


if __name__ == "__main__":
    main()
