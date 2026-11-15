"""Wykrywanie gestu "podniesiona reka" (sterowanie bez dotykania ekranu).

Uzywa modelu pozy MediaPipe (common/modele/pose_landmarker_lite.task).
Nie rozpoznaje osob ani emocji - sprawdza tylko, czy nadgarstek jest
wyraznie powyzej nosa. Obraz nie jest zapisywany.
"""

from __future__ import annotations

from pathlib import Path

import cv2
import mediapipe as mp
import numpy as np
from mediapipe.tasks.python import BaseOptions, vision

SCIEZKA_MODELU_POZY = Path(__file__).parent / "modele" / "pose_landmarker_lite.task"

NOS, LEWY_NADGARSTEK, PRAWY_NADGARSTEK = 0, 15, 16
LEWE_RAMIE, PRAWE_RAMIE = 11, 12


class DetektorGestuReki:
    # Nadgarstek musi byc wyzej od nosa o co najmniej tyle szerokosci barkow.
    MARGINES_WZGLEDNY = 0.15

    def __init__(self) -> None:
        opcje = vision.PoseLandmarkerOptions(
            base_options=BaseOptions(model_asset_path=str(SCIEZKA_MODELU_POZY)),
            num_poses=3,
            running_mode=vision.RunningMode.IMAGE,
        )
        self.model = vision.PoseLandmarker.create_from_options(opcje)

    @staticmethod
    def dostepny() -> bool:
        return SCIEZKA_MODELU_POZY.exists()

    def reka_podniesiona(self, klatka_bgr: np.ndarray) -> bool:
        """True, jesli ktorakolwiek widoczna osoba ma podniesiona reke
        powyzej poziomu glowy (nos)."""
        obraz = mp.Image(
            image_format=mp.ImageFormat.SRGB,
            data=np.ascontiguousarray(cv2.cvtColor(klatka_bgr, cv2.COLOR_BGR2RGB)),
        )
        for punkty in self.model.detect(obraz).pose_landmarks:
            nos = punkty[NOS]
            szerokosc_barkow = abs(punkty[LEWE_RAMIE].x - punkty[PRAWE_RAMIE].x)
            margines = max(0.03, szerokosc_barkow * self.MARGINES_WZGLEDNY)
            for indeks in (LEWY_NADGARSTEK, PRAWY_NADGARSTEK):
                nadgarstek = punkty[indeks]
                if nadgarstek.visibility > 0.5 and nadgarstek.y < nos.y - margines:
                    return True
        return False
