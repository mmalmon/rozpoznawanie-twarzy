"""Wspólna logika rozpoznawania twarzy (używana przez skrypt 04 i panel GUI).

Ten moduł wydziela z `04_rozpoznawanie_na_zywo.py` wszystkie funkcje, które
nie zależą od okna OpenCV: wczytanie wzorców, porównanie embeddingu z
wzorcami oraz analizę pojedynczej klatki. Dzięki temu GUI robi dokładnie to
samo, co skrypt konsolowy — bez kopiowania reguł i bez ryzyka, że obie
wersje zaczną się różnić.

DLA UCZNIÓW - jak działa rozpoznawanie przez podobieństwo?
  1. Twarz z kamery zamieniamy na embedding (wektor liczb) tym samym,
     zamrożonym ekstraktorem cech, którego użył skrypt 03.
  2. Liczymy podobieństwo kosinusowe (zwykły iloczyn skalarny wektorów
     jednostkowych) do prototypów każdej znanej osoby.
  3. Osoba zostaje rozpoznana tylko wtedy, gdy podobieństwo przekracza jej
     WŁASNY, skalibrowany próg. W przeciwnym razie to "Nieznana osoba".
"""

from __future__ import annotations

import json
from collections.abc import Callable
from pathlib import Path

import cv2
import numpy as np
import torch

from common.emotions import RozpoznawaczEmocji
from common.face_detector import DetektorTwarzy, WykrytaTwarz
from common.model import (
    WERSJA_PRZYGOTOWANIA_OBRAZU,
    oblicz_znormalizowane_cechy,
    policz_podobienstwa_wzorca,
    przygotuj_przeksztalcenie,
)

KORZEN_PROJEKTU = Path(__file__).resolve().parent.parent
PLIK_WZORCOW = KORZEN_PROJEKTU / "models" / "wzorce_osob.json"

# Etykieta zwracana, gdy żadna osoba nie przekroczyła swojego progu.
NAZWA_NIEZNANEJ = "Nieznana osoba"

# Dwa ciasne kadry tej samej twarzy. Skrypt 03 liczy wzorce z kadrowaniem
# 1.0, a przy rozpoznawaniu warto sprawdzić także wersję "przyciętą" - jedna
# z nich lepiej trafia w dany prototyp (np. przy okularach albo grzywce).
SKALE_KADRU_ROZPOZNAWANIA = (0.82, 0.72)


class BrakWzorcowError(RuntimeError):
    """Plik wzorców nie istnieje - trzeba najpierw uruchomić trening."""


class NiepoprawneWzorceError(RuntimeError):
    """Plik wzorców jest uszkodzony albo powstał w starej wersji programu."""


def wczytaj_wzorce(
    sciezka: Path = PLIK_WZORCOW,
    urzadzenie: torch.device | None = None,
) -> dict[str, dict]:
    """Wczytuje wzorce tożsamości i progi zapisane przez skrypt 03.

    Zwraca słownik {nazwa_osoby: {"prototypy": Tensor, "prog": float}}.
    Embeddingi z JSON-a (listy liczb) zamieniamy z powrotem na tensory
    PyTorch, od razu na wybranym urządzeniu (CPU/GPU).
    """
    if urzadzenie is None:
        urzadzenie = torch.device("cpu")

    if not sciezka.exists():
        raise BrakWzorcowError(
            "Brak wzorcow tozsamosci. Najpierw uruchom 03_trenowanie_modelu.py."
        )

    try:
        with open(sciezka, "r", encoding="utf-8") as plik:
            dane = json.load(plik)
    except (OSError, json.JSONDecodeError) as blad:
        raise NiepoprawneWzorceError(
            f"Plik {sciezka} jest uszkodzony ({blad}). "
            "Uruchom ponownie 03_trenowanie_modelu.py."
        ) from blad

    if dane.get("wersja_przygotowania_obrazu") != WERSJA_PRZYGOTOWANIA_OBRAZU:
        raise NiepoprawneWzorceError(
            "Wzorce twarzy sa nieaktualne i nie pasuja do nowego kadrowania. "
            "Uruchom ponownie 03_trenowanie_modelu.py."
        )

    wzorce: dict[str, dict] = {}
    for nazwa_osoby, wpis in dane.get("osoby", {}).items():
        try:
            prototypy = torch.tensor(wpis["prototypy"], device=urzadzenie)
            prog = float(wpis["prog_podobienstwa"])
        except (KeyError, TypeError, ValueError) as blad:
            raise NiepoprawneWzorceError(
                f"Niepoprawny wpis dla osoby '{nazwa_osoby}' w {sciezka}: "
                f"{blad}. Uruchom ponownie 03_trenowanie_modelu.py."
            ) from blad
        wzorce[nazwa_osoby] = {"prototypy": prototypy, "prog": prog}
    return wzorce


def rozpoznaj_twarz(
    embedding: torch.Tensor,
    wzorce: dict[str, dict],
    korekta_progu: float = 0.0,
) -> tuple[str, float]:
    """Porównuje jeden lub kilka kadrów tej samej twarzy ze wzorcami osób.

    DLA UCZNIÓW: przeglądamy wszystkie znane osoby, liczymy podobieństwo
    kosinusowe do każdej z nich i wybieramy NAJLEPSZE dopasowanie spośród
    kadrów oraz prototypów. Każda osoba ma WŁASNY, skalibrowany próg, dlatego:

      1. najpierw odrzucamy osoby, których próg nie został osiągnięty,
      2. spośród pozostałych wybieramy tę z najwyższym podobieństwem,
      3. jeśli żadna osoba nie przekroczyła swojego progu - "Nieznana osoba".

    To ważne: samo "najwyższe podobieństwo" nie wystarcza, bo osoba z bardzo
    ostrym progiem może mieć surowo wyższy wynik, a mimo to nie zostać
    rozpoznana, podczas gdy inna osoba z niższym progiem przekracza go.
    """
    if not wzorce:
        return NAZWA_NIEZNANEJ, 0.0

    embeddingi = embedding.unsqueeze(0) if embedding.ndim == 1 else embedding
    najlepsza_nazwa = NAZWA_NIEZNANEJ
    najlepsze_podobienstwo = -1.0
    najlepsze_podobienstwo_ogolne = -1.0

    for nazwa_osoby, wpis in wzorce.items():
        podobienstwo = (
            policz_podobienstwa_wzorca(embeddingi, wpis["prototypy"]).max().item()
        )
        najlepsze_podobienstwo_ogolne = max(
            najlepsze_podobienstwo_ogolne, podobienstwo
        )

        # Korekta progu nie może wypchnąć go poza sensowny zakres
        # podobieństwa kosinusowego [0, 1].
        prog = min(1.0, max(0.0, wpis["prog"] + korekta_progu))
        if podobienstwo >= prog and podobienstwo > najlepsze_podobienstwo:
            najlepsza_nazwa = nazwa_osoby
            najlepsze_podobienstwo = podobienstwo

    if najlepsza_nazwa == NAZWA_NIEZNANEJ:
        return najlepsza_nazwa, max(0.0, najlepsze_podobienstwo_ogolne)
    return najlepsza_nazwa, najlepsze_podobienstwo


def przygotuj_obrazy_kamery(
    skale: tuple[float, ...] = SKALE_KADRU_ROZPOZNAWANIA,
) -> list[Callable[[np.ndarray], torch.Tensor]]:
    """Buduje listę transformacji dla klatek z kamery (wejście RGB z OpenCV).

    Każda transformacja to ten sam zestaw operacji, którego użył trening,
    tylko z inną skalą centralnego kadrowania.
    """
    return [
        przygotuj_przeksztalcenie(wejscie_z_kamery=True, skala_kadru=skala)
        for skala in skale
    ]


def analizuj_klatke(
    klatka: np.ndarray,
    detektor: DetektorTwarzy,
    ekstraktor: torch.nn.Module,
    przygotuj_obrazy: list[Callable[[np.ndarray], torch.Tensor]],
    wzorce: dict[str, dict],
    urzadzenie: torch.device,
    korekta_progu: float = 0.0,
    emocje: RozpoznawaczEmocji | None = None,
) -> list[tuple[WykrytaTwarz, str, float, str, float]]:
    """Analizuje jedną klatkę: detekcja, embeddingi, dopasowanie do wzorców.

    Zwraca listę krotek (twarz, osoba, podobieństwo, emocja, pewność_emocji).
    Emocje są opcjonalne - bez modelu zwracamy pustą etykietę i 0.0.
    """
    tensory_wejsciowe: list[torch.Tensor] = []
    twarze: list[WykrytaTwarz] = []
    for twarz in detektor.wykryj(klatka):
        wycinek = twarz.wytnij(klatka, margines=0.2)
        if wycinek.size == 0:
            continue

        wycinek_rgb = cv2.cvtColor(wycinek, cv2.COLOR_BGR2RGB)
        tensory_wejsciowe.extend(
            przygotuj_obraz(wycinek_rgb) for przygotuj_obraz in przygotuj_obrazy
        )
        twarze.append(twarz)

    if not tensory_wejsciowe:
        if emocje is not None:
            emocje.rozpoznaj(klatka, [])
        return []

    wyniki_emocji = (
        emocje.rozpoznaj(klatka, twarze)
        if emocje is not None
        else [("", 0.0)] * len(twarze)
    )

    obrazy = torch.stack(tensory_wejsciowe).to(urzadzenie)
    embeddingi = oblicz_znormalizowane_cechy(ekstraktor, obrazy)
    liczba_kadrow = len(przygotuj_obrazy)
    wyniki = []
    for indeks, twarz in enumerate(twarze):
        zakres = slice(indeks * liczba_kadrow, (indeks + 1) * liczba_kadrow)
        etykieta, podobienstwo = rozpoznaj_twarz(
            embeddingi[zakres], wzorce, korekta_progu
        )
        emocja, pewnosc_emocji = wyniki_emocji[indeks]
        wyniki.append((twarz, etykieta, podobienstwo, emocja, pewnosc_emocji))
    return wyniki
