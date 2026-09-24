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

import atexit
import shutil
import tempfile
from dataclasses import dataclass
from pathlib import Path

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


def _load_cascade_safely(filename: str) -> cv2.CascadeClassifier:
    """Wczytuje kaskade Haara w sposob odporny na polskie znaki w sciezce.

    cv2.CascadeClassifier na Windows potrafi nie wczytac pliku, jesli jego
    sciezka zawiera znaki spoza ASCII (np. gdy projekt lezy w folderze typu
    "OneDrive - Zespół Szkół..."). Dlatego kopiujemy plik XML do katalogu
    tymczasowego systemu (ktorego sciezka jest zazwyczaj czysto ASCII) i
    wczytujemy kaskade juz stamtad.
    """
    source_path = Path(cv2.data.haarcascades) / filename

    cascade = cv2.CascadeClassifier(str(source_path))
    if not cascade.empty():
        return cascade

    tmp_dir = Path(tempfile.mkdtemp(prefix="haarcascade_"))
    tmp_path = tmp_dir / filename
    shutil.copyfile(source_path, tmp_path)
    atexit.register(shutil.rmtree, tmp_dir, True)

    cascade = cv2.CascadeClassifier(str(tmp_path))
    if cascade.empty():
        raise RuntimeError(
            f"Nie udalo sie wczytac kaskady Haara ani z {source_path}, ani z kopii {tmp_path}"
        )
    return cascade


class FaceDetector:
    def __init__(self, scale_factor: float = 1.1, min_neighbors: int = 6):
        self.cascade = _load_cascade_safely("haarcascade_frontalface_default.xml")
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
