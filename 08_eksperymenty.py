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

UWAGA METODOLOGICZNA:
  Prog EER wybieramy na tym samym zbiorze walidacyjnym, na ktorym
  liczymy metryki. To znaczy, ze wyniki sa OPTYMISTYCZNE. W prawdziwym
  projekcie prog ustawia sie na osobnym zbiorze walidacyjnym, a wynik
  raportuje na zupelnie nowym zbiorze testowym. Ten skrypt wypisuje
  ostrzezenie takze w raporcie JSON - nie ukrywamy tego.

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
    policz_embeddingi_plaskie,
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
FOLDER_RAW = KORZEN_PROJEKTU / "data" / "raw"
FOLDER_EKSPERYMENTOW = KORZEN_PROJEKTU / "models" / "eksperymenty"
FOLDER_WYKRESOW = FOLDER_EKSPERYMENTOW / "wykresy"
PLIK_RAPORTU = FOLDER_EKSPERYMENTOW / "raport_eksperymentow.json"
PLIK_CSV = FOLDER_EKSPERYMENTOW / "wyniki.csv"

NAZWA_KLASY_NIEZNAJOMY = "nieznajomy"
NAZWA_OBCEGO = "obcy (nieznajomy)"
WERSJA_RAPORTU = 1

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
def ocen_podobienstwa(
    podobienstwa_znane: np.ndarray,
    podobienstwa_obce: np.ndarray,
    prawdziwe_znane: list[str],
    indeksy_prawdziwe: list[int],
    osoby: list[str],
) -> dict:
    """Liczy wszystkie metryki dla jednej macierzy podobienstw.

    Prog decyzyjny wybieramy w punkcie EER (FAR = FRR), zeby porownywac
    metody przy podobnym kompromisie bledow. To wlasnie ten prog jest
    pozniej uzywany do accuracy otwartej, FAR i FRR.
    """
    indeksy = np.arange(len(podobienstwa_znane))
    wlasne = podobienstwa_znane[indeksy, indeksy_prawdziwe]
    obce = (
        podobienstwa_obce.max(axis=1)
        if len(podobienstwa_obce)
        else np.zeros(0, dtype=float)
    )

    krzywa = krzywa_i_eer(wlasne, obce)
    prog = krzywa["prog"] if krzywa else 0.5

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
        "prog_eer": krzywa["prog"] if krzywa else None,
        "zaakceptowani_obcy": metryki["zaakceptowani_obcy"],
        "odrzucone_znane": metryki["odrzucone_znane"],
    }


def ocen_prototypy(
    trening: torch.Tensor,
    etykiety_trening: list[int],
    osoby: list[str],
    liczba_wariantow: int,
    embeddingi_znane: torch.Tensor,
    embeddingi_obce: torch.Tensor,
    prawdziwe_znane: list[str],
    indeksy_prawdziwe: list[int],
) -> tuple[dict, int]:
    """Buduje prototypy i zwraca metryki oraz liczbe uzytych prototypow."""
    wzorce = zbuduj_prototypy(
        trening, etykiety_trening, osoby, liczba_wariantow
    )
    podobienstwa_znane = podobienstwa_do_osob(embeddingi_znane, wzorce, osoby)
    podobienstwa_obce = podobienstwa_do_osob(embeddingi_obce, wzorce, osoby)
    metryki = ocen_podobienstwa(
        podobienstwa_znane,
        podobienstwa_obce,
        prawdziwe_znane,
        indeksy_prawdziwe,
        osoby,
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
        wnioski.append(
            f"Przy {najmniejszy['etykieta']} zdjeciach accuracy zamknieta wyniosla "
            f"{najmniejszy['metryki']['accuracy_zamknieta']:.1%}, a przy wszystkich "
            f"zdjeciach {wszystkie['metryki']['accuracy_zamknieta']:.1%} - {kierunek} "
            f"o {abs(roznica) * 100:.1f} pkt proc. To jeden pomiar, nie dowod: "
            "wynik zalezy tez od tego, ktore zdjecia wybralismy."
        )

        najlepszy = min(
            rozmiary,
            key=lambda wpis: (
                wpis["metryki"]["eer"] if wpis["metryki"]["eer"] is not None else 1.0
            ),
        )
        wnioski.append(
            f"Najnizszy EER w eksperymencie z liczba zdjec mial wariant "
            f"'{najlepszy['etykieta']}' ({najlepszy['metryki']['eer']:.1%}). "
            "Przy malym zbiorze prototypy sa mniej roznorodne, wiec model moze "
            "mylc osoby podobne."
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
        najlepszy_k = min(
            knn,
            key=lambda wpis: (
                wpis["metryki"]["eer"] if wpis["metryki"]["eer"] is not None else 1.0
            ),
        )
        najgorszy_k = max(
            knn,
            key=lambda wpis: (
                wpis["metryki"]["eer"] if wpis["metryki"]["eer"] is not None else 0.0
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
    folder_obcych = FOLDER_RAW / NAZWA_KLASY_NIEZNAJOMY

    if not folder_train.is_dir() or not folder_val.is_dir():
        raise SystemExit(
            f"Brak danych w {FOLDER_PRZETWORZONYCH_DANYCH}. "
            "Najpierw uruchom 02_podzial_danych.py."
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
    liczba_obcych = policz_obrazy(folder_obcych)
    postep = Postep(argumenty.postep)
    postep.start(
        "embeddingi",
        liczba_train * len(WSPOLCZYNNIKI_JASNOSCI) + liczba_val + liczba_obcych,
    )

    # ---------------------------------------------------------- trening
    print("[info] Liczenie embeddingow treningu (3 warianty jasnosci)...")
    warianty_wszystkie: list[torch.Tensor] = []
    etykiety_wszystkie: list[int] = []
    klasy_wszystkie: list[str] = []
    sciezki_wszystkie: list[Path] = []
    for jasnosc in WSPOLCZYNNIKI_JASNOSCI:
        embeddingi, etykiety, klasy, sciezki = policz_embeddingi_folderu(
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
            sciezki_wszystkie = sciezki
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

    print(
        f"[info] Osoby ({len(osoby)}): {osoby} | dostepne zdjecia na osobe: {dostepnosc}"
    )
    print(f"[info] Urzadzenie: {urzadzenie}")

    # -------------------------------------------------------- walidacja
    print("[info] Liczenie embeddingow walidacji...")
    embeddingi_val, etykiety_val, klasy_val, sciezki_val = policz_embeddingi_folderu(
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

    embeddingi_znane = embeddingi_val[indeksy_znane_val]
    prawdziwe_znane = [prawdziwe_val[numer] for numer in indeksy_znane_val]
    indeksy_prawdziwe = [osoby.index(nazwa) for nazwa in prawdziwe_znane]
    print(
        f"[info] Walidacja: {len(prawdziwe_znane)} zdjec znanych osob, "
        f"{len(indeksy_obce_val)} zdjec klasy 'nieznajomy'."
    )

    # ---------------------------------------------------------- obcy
    if folder_obcych.is_dir():
        print("[info] Liczenie embeddingow obcych twarzy...")
        embeddingi_obce, _ = policz_embeddingi_plaskie(
            ekstraktor,
            folder_obcych,
            urzadzenie,
            argumenty.batch_size,
            transformacja(1.0),
            postep,
        )
    else:
        print("[uwaga] Brak folderu data/raw/nieznajomy - nie policze FAR.")
        embeddingi_obce = torch.empty((0, embeddingi_znane.shape[1]))
    print(f"[info] Obcy: {len(embeddingi_obce)} zdjec.")

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
            embeddingi_znane,
            embeddingi_obce,
            prawdziwe_znane,
            indeksy_prawdziwe,
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
            embeddingi_znane,
            embeddingi_obce,
            prawdziwe_znane,
            indeksy_prawdziwe,
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
    zapytania = torch.cat([embeddingi_znane, embeddingi_obce], dim=0)
    knn: list[dict] = []
    for k in WARTOSCI_K:
        podobienstwa = podobienstwa_knn(
            zapytania, trening_knn, etykiety_knn, osoby, osoby, k
        )
        podobienstwa_znane = podobienstwa[: len(embeddingi_znane)]
        podobienstwa_obce = podobienstwa[len(embeddingi_znane) :]
        metryki = ocen_podobienstwa(
            podobienstwa_znane,
            podobienstwa_obce,
            prawdziwe_znane,
            indeksy_prawdziwe,
            osoby,
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
    podobienstwa_znane = podobienstwa_do_osob(embeddingi_znane, wzorce, osoby)
    podobienstwa_obce = podobienstwa_do_osob(embeddingi_obce, wzorce, osoby)
    liczba_prototypow = sum(
        int(wpis["prototypy"].shape[0]) for wpis in wzorce.values()
    )
    progi: list[dict] = []
    for prog in PROGI_DECYZYJNE:
        przewidziane_znane, _ = decyzje_z_progiem(
            podobienstwa_znane, osoby, prog
        )
        przewidziane_obce, _ = decyzje_z_progiem(podobienstwa_obce, osoby, prog)
        metryki = metryki_openset(
            prawdziwe_znane, przewidziane_znane, przewidziane_obce
        )
        progi.append(
            {
                "prog": prog,
                "liczba_zdjec_treningowych": len(wybrane)
                * len(WSPOLCZYNNIKI_JASNOSCI),
                "liczba_prototypow": liczba_prototypow,
                "accuracy_zamknieta": round(
                    accuracy_zamknieta(podobienstwa_znane, indeksy_prawdziwe), 4
                ),
                "accuracy": metryki["accuracy"],
                "far": metryki["far"],
                "frr": metryki["frr"],
                "zaakceptowani_obcy": metryki["zaakceptowani_obcy"],
                "odrzucone_znane": metryki["odrzucone_znane"],
            }
        )
        postep.dodaj(1)
    print(f"[info] Tabela progow: {len(progi)} wartosci od 0.50 do 0.95.")

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
        "Prog decyzyjny (EER) jest wybierany na tym samym zbiorze walidacyjnym, "
        "na ktorym liczymy metryki. Wyniki sa przez to OPTYMISTYCZNE. "
        "W prawdziwym projekcie prog ustawia sie na jednym zbiorze, a raportuje "
        "na innym, nowym zbiorze testowym.",
        "Zbior walidacyjny (val) byl juz uzyty w skrypcie 03 do kalibracji progow, "
        "wiec takze z tego powodu wyniki sa optymistyczne.",
    ]
    if folder_obcych.is_dir():
        ostrzezenia.append(
            "Zdjecia obcych pochodza z data/raw/nieznajomy. Czesc z nich mogla "
            "byc uzyta w kalibracji progow (val), dlatego FAR jest zawyzony "
            "w optymistyczna strone."
        )
    else:
        ostrzezenia.append(
            "Brak folderu data/raw/nieznajomy - nie policzono FAR, EER ani AUC."
        )
    ostrzezenia.append(
        "Eksperymenty pokazuja zaleznosc na JEDNYM zbiorze danych i jednym "
        "podziale train/val. Inny podzial moze dac inne wyniki - to normalne."
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
                "zdjecia_znanych": len(prawdziwe_znane),
                "zdjecia_nieznajomych_w_val": len(indeksy_obce_val),
            },
            "obcy": {
                "zdjecia": len(embeddingi_obce),
                "zrodlo": "data/raw/nieznajomy",
            },
            "warianty_jasnosci": list(WSPOLCZYNNIKI_JASNOSCI),
            "wymiar_cech": int(embeddingi_znane.shape[1]),
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
    print("PODSUMOWANIE EKSPERYMENTOW (prototypy, prog EER)")
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
    print("[uwaga] Metryki sa optymistyczne - prog EER wybrano na zbiorze walidacyjnym.")


if __name__ == "__main__":
    main()
