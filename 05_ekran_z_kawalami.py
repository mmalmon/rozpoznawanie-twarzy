"""Ekran z kawalami na korytarz sterowany gestem (tryb zgodny z AI Act).

Sterowanie: PODNIES REKE nad glowe (jak w Kinekcie), zeby przejsc dalej:
  "Podnies reke" -> pytanie kawalu -> (podnies reke) -> puanta -> poczatek.
Kamera sluzy TYLKO do sprawdzenia, czy ktos podnosi reke. Program:
- nie rozpoznaje osob ani emocji i nie uzywa rozpoznawania twarzy,
- nie zapisuje i nie wyswietla obrazu z kamery,
- nie przechowuje zadnych danych o osobach.
Klawisze awaryjne: SPACJA = dalej, q/ESC = koniec.

Uruchomienie: python 05_ekran_z_kawalami.py [--kamera N] [--bez-kamery]
"""

from __future__ import annotations

# Ta linijka MUSI byc pierwszym kodem wykonywalnym w tym pliku - patrz
# szczegolowe wyjasnienie w common/uruchom_w_venv.py (automatyczne
# przelaczenie na Python z .venv, jesli uruchomiono innym interpreterem).
from common.uruchom_w_venv import przelacz_na_venv_jesli_trzeba

przelacz_na_venv_jesli_trzeba()

import argparse
import random
import time

import cv2
import numpy as np

from common.camera import otworz_kamere, wybierz_kamere_interaktywnie

KAWALY = [
    ("Dlaczego informatyk nosi okulary?", "Bo nie umie C#."),
    ("Co mowi jedynka do zera?", "Fajny masz pierscionek!"),
    ("Dlaczego komputer poszedl do lekarza?", "Bo zlapal wirusa."),
    ("Jak nazywa sie ulubiony ser programisty?", "Cheddar... a nie, Python!"),
    ("Dlaczego nauczyciel matematyki jest smutny?", "Ma za duzo problemow."),
    ("Co robi programista na plazy?", "Szuka bugow w piasku."),
    ("Dlaczego kalendarz sie stresuje?", "Bo ma policzone dni."),
    ("Jaki jest ulubiony napoj programisty?", "Java."),
    ("Co mowi pamiec RAM do dysku?", "Zapomnialem, o czym chcialem pogadac."),
    ("Dlaczego dlugopis poszedl na zwolnienie?", "Bo mu sie skonczyl wklad w prace."),
]

PUANTA_AUTOMATYCZNIE_PO_S = 25.0
KONIEC_PUANTY_PO_S = 8.0
BLOKADA_PO_GESCIE_S = 1.5
SPRAWDZANIE_CO_S = 0.25
WYMAGANE_KOLEJNE_WYKRYCIA = 2


def parsuj_argumenty() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "--kamera",
        type=int,
        default=None,
        help="Indeks kamery. Jesli pominiety, program zapyta interaktywnie.",
    )
    parser.add_argument("--bez-kamery", action="store_true",
                        help="tylko klawisz SPACJA, bez kamery")
    return parser.parse_args()

def narysuj(tekst_gora: str, tekst_dol: str, szerokosc: int = 1920, wysokosc: int = 1080):
    ekran = np.full((wysokosc, szerokosc, 3), (60, 30, 20), np.uint8)
    for tekst, y, skala in ((tekst_gora, wysokosc // 2 - 40, 2.0), (tekst_dol, wysokosc // 2 + 100, 2.4)):
        # Dzielimy na linie, zeby dlugi tekst miescil sie na ekranie.
        slowa, linie, biezaca = tekst.split(), [], ""
        for slowo in slowa:
            proba = f"{biezaca} {slowo}".strip()
            if cv2.getTextSize(proba, cv2.FONT_HERSHEY_SIMPLEX, skala, 4)[0][0] > szerokosc - 160:
                if biezaca:
                    linie.append(biezaca)
                biezaca = slowo
            else:
                biezaca = proba
        if biezaca:
            linie.append(biezaca)
        for i, linia in enumerate(linie):
            rozmiar = cv2.getTextSize(linia, cv2.FONT_HERSHEY_SIMPLEX, skala, 4)[0]
            cv2.putText(ekran, linia, ((szerokosc - rozmiar[0]) // 2, y + i * 90),
                        cv2.FONT_HERSHEY_SIMPLEX, skala, (255, 255, 255), 4, cv2.LINE_AA)
    return ekran


def main() -> None:
    argumenty = parsuj_argumenty()
    kamera = detektor = None
    if not argumenty.bez_kamery:
        try:
            from common.gesture import DetektorGestuReki
        except ImportError as blad:
            print(
                f"[uwaga] Brak biblioteki mediapipe ({blad}) - uruchamiam tryb "
                "klawiatury (SPACJA = dalej, q/ESC = koniec)."
            )
        else:
            if not DetektorGestuReki.dostepny():
                print(
                    "[uwaga] Brak modelu common/modele/pose_landmarker_lite.task - "
                    "uruchamiam tryb klawiatury."
                )
            else:
                try:
                    indeks = (
                        argumenty.kamera
                        if argumenty.kamera is not None
                        else wybierz_kamere_interaktywnie()
                    )
                    kamera = otworz_kamere(indeks, szerokosc=1280, wysokosc=720)
                    detektor = DetektorGestuReki()
                except (RuntimeError, OSError) as blad:
                    print(
                        f"[uwaga] Nie mozna uruchomic kamery ({blad}) - "
                        "przechodze na tryb klawiatury."
                    )
                    if kamera is not None:
                        kamera.release()
                    kamera = detektor = None

    nazwa_okna = "Kawaly (q=koniec)"
    cv2.namedWindow(nazwa_okna, cv2.WINDOW_NORMAL)
    cv2.setWindowProperty(nazwa_okna, cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)

    kolejka: list[int] = []
    stan = "czekanie"
    zmiana_stanu = ostatnie_sprawdzenie = 0.0
    kolejne_wykrycia = 0
    kawal = KAWALY[0]

    def nastepny_kawal():
        if not kolejka:
            kolejka.extend(random.sample(range(len(KAWALY)), len(KAWALY)))
        return KAWALY[kolejka.pop()]

    try:
        while True:
            teraz = time.time()
            gest = False
            if kamera is not None and teraz - ostatnie_sprawdzenie >= SPRAWDZANIE_CO_S:
                ostatnie_sprawdzenie = teraz
                ok, klatka = kamera.read()
                if ok:
                    # Klatka sluzy tylko do wykrycia gestu i jest od razu porzucana.
                    kolejne_wykrycia = (
                        kolejne_wykrycia + 1 if detektor.reka_podniesiona(klatka) else 0
                    )
                    if kolejne_wykrycia >= WYMAGANE_KOLEJNE_WYKRYCIA:
                        gest, kolejne_wykrycia = True, 0
                else:
                    kolejne_wykrycia = 0

            klawisz = cv2.waitKey(30) & 0xFF
            if klawisz in (ord("q"), 27):
                break
            if klawisz == ord(" "):
                gest = True
            if gest and teraz - zmiana_stanu < BLOKADA_PO_GESCIE_S:
                gest = False

            if stan == "czekanie" and gest:
                kawal, stan, zmiana_stanu = nastepny_kawal(), "pytanie", teraz
            elif stan == "pytanie" and (
                gest or teraz - zmiana_stanu > PUANTA_AUTOMATYCZNIE_PO_S
            ):
                stan, zmiana_stanu = "puanta", teraz
            elif stan == "puanta" and (
                gest or teraz - zmiana_stanu > KONIEC_PUANTY_PO_S
            ):
                stan, zmiana_stanu = "czekanie", teraz

            if stan == "czekanie":
                ekran = narysuj("Podnies reke,", "aby uslyszec kawal")
            elif stan == "pytanie":
                ekran = narysuj(kawal[0], "Podnies reke - puanta")
            else:
                ekran = narysuj(kawal[0], kawal[1])
            cv2.imshow(nazwa_okna, ekran)
    finally:
        if kamera is not None:
            kamera.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()