"""Rozpoznawanie emocji na twarzy (gotowy model FER+ w formacie ONNX).

Model dostaje szary wycinek twarzy 64x64 i zwraca wyniki dla 8 emocji.
Dziala na CPU przez cv2.dnn (kilka milisekund na twarz).
"""

from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np

SCIEZKA_MODELU_EMOCJI = Path(__file__).parent / "modele" / "emotion-ferplus-8.onnx"

# Kolejnosc klas zgodna z modelem FER+; nazwy bez polskich znakow, bo
# cv2.putText ich nie rysuje.
NAZWY_EMOCJI = (
    "neutralna",
    "radosc",
    "zdziwienie",
    "smutek",
    "zlosc",
    "wstret",
    "strach",
    "pogarda",
)


class RozpoznawaczEmocji:
    """Klasyfikuje emocje i wygladza wynik w czasie dla sledzonych twarzy."""

    WSPOLCZYNNIK_WYGLADZANIA = 0.5
    MAKS_ODLEGLOSC_SLEDZENIA = 1.0  # w szerokosciach twarzy

    def __init__(self) -> None:
        self.siec = cv2.dnn.readNetFromONNX(str(SCIEZKA_MODELU_EMOCJI))
        # Stan sledzenia: lista (srodek_x, srodek_y, szerokosc, prawdopodobienstwa).
        self._sledzone: list[tuple[float, float, float, np.ndarray]] = []

    @staticmethod
    def dostepny() -> bool:
        return SCIEZKA_MODELU_EMOCJI.exists()

    def _prawdopodobienstwa(self, wycinek_bgr: np.ndarray) -> np.ndarray:
        szary = cv2.cvtColor(wycinek_bgr, cv2.COLOR_BGR2GRAY)
        szary = cv2.resize(szary, (64, 64), interpolation=cv2.INTER_AREA)
        self.siec.setInput(szary.astype(np.float32).reshape(1, 1, 64, 64))
        wyniki = self.siec.forward().reshape(-1)
        wyniki = np.exp(wyniki - wyniki.max())
        return wyniki / wyniki.sum()

    def rozpoznaj(
        self, klatka: np.ndarray, twarze: list
    ) -> list[tuple[str, float]]:
        """Zwraca (nazwa_emocji, pewnosc) dla kazdej twarzy z listy."""
        nowe_slady = []
        wyniki = []
        wolne = list(self._sledzone)
        for twarz in twarze:
            wycinek = twarz.wytnij(klatka, margines=0.1)
            if wycinek.size == 0:
                wyniki.append(("", 0.0))
                continue
            prawdopodobienstwa = self._prawdopodobienstwa(wycinek)
            cx = twarz.x + twarz.w / 2
            cy = twarz.y + twarz.h / 2

            # Wygladzanie: laczymy z najblizsza twarza z poprzedniej analizy.
            najlepszy = None
            najlepszy_dystans = self.MAKS_ODLEGLOSC_SLEDZENIA * twarz.w
            for slad in wolne:
                dystans = float(np.hypot(slad[0] - cx, slad[1] - cy))
                if dystans < najlepszy_dystans:
                    najlepszy, najlepszy_dystans = slad, dystans
            if najlepszy is not None:
                wolne.remove(najlepszy)
                a = self.WSPOLCZYNNIK_WYGLADZANIA
                prawdopodobienstwa = a * prawdopodobienstwa + (1 - a) * najlepszy[3]

            nowe_slady.append((cx, cy, float(twarz.w), prawdopodobienstwa))
            indeks = int(np.argmax(prawdopodobienstwa))
            wyniki.append((NAZWY_EMOCJI[indeks], float(prawdopodobienstwa[indeks])))

        self._sledzone = nowe_slady
        return wyniki
