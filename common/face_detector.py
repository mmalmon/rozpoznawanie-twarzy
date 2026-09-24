"""
Prosty detektor twarzy oparty o kaskade Haara z OpenCV.

Do celow edukacyjnych wybieramy kaskade Haara zamiast np. MediaPipe/MTCNN,
poniewaz:
  - jest wbudowana w opencv-python (zero dodatkowych pobran modeli),
  - dziala szybko na CPU (GPU zostawiamy dla sieci rozpoznajacej osobe),
  - jej dzialanie latwo wytlumaczyc uczniom (cechy Haara + AdaBoost + okno
    przesuwne), w przeciwienstwie do detektorow opartych o sieci neuronowe.

W realnym / produkcyjnym systemie lepszym wyborem bylby detektor oparty o
sieci neuronowe (np. MediaPipe Face Detector, RetinaFace) - jest to opisane
jako "zadanie dodatkowe" w README.
"""

from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np


@dataclass
class FaceBox:
    x: int
    y: int
    w: int
    h: int

    def crop(self, frame: np.ndarray, margin: float = 0.2) -> np.ndarray:
        """Wycina twarz z klatki, dodajac margines (domyslnie 20%)."""
        frame_h, frame_w = frame.shape[:2]
        mx = int(self.w * margin)
        my = int(self.h * margin)

        x1 = max(0, self.x - mx)
        y1 = max(0, self.y - my)
        x2 = min(frame_w, self.x + self.w + mx)
        y2 = min(frame_h, self.y + self.h + my)

        return frame[y1:y2, x1:x2]


class FaceDetector:
    def __init__(self, scale_factor: float = 1.1, min_neighbors: int = 6):
        cascade_path = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
        self.cascade = cv2.CascadeClassifier(cascade_path)
        if self.cascade.empty():
            raise RuntimeError(f"Nie udalo sie wczytac kaskady Haara z {cascade_path}")

        self.scale_factor = scale_factor
        self.min_neighbors = min_neighbors

    def detect(self, frame: np.ndarray) -> list[FaceBox]:
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        gray = cv2.equalizeHist(gray)

        faces = self.cascade.detectMultiScale(
            gray,
            scaleFactor=self.scale_factor,
            minNeighbors=self.min_neighbors,
            minSize=(80, 80),
        )

        return [FaceBox(x, y, w, h) for (x, y, w, h) in faces]
