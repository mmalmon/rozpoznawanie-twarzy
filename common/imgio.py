"""
Bezpieczne odczyt/zapis obrazow, odporne na polskie znaki w sciezkach.

Kontekst: biblioteka OpenCV na Windows przy operacjach cv2.imread/cv2.imwrite
korzysta wewnetrznie z funkcji API, ktore przy sciezkach zawierajacych znaki
spoza ASCII (np. polskie "ł", "ó" - jak w folderze uzytkownika typu
"OneDrive - Zespół Szkół") czesto zawodza (zwracaja False / None) bez
rzucania wyjatku. Ponizsze funkcje omijaja ten problem, korzystajac z
cv2.imencode/cv2.imdecode w polaczeniu ze zwyklym, w pelni Unicode-owym
odczytem/zapisem pliku przez Pythona (numpy.ndarray.tofile / np.fromfile).
"""

from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np


def imwrite_unicode(path: str | Path, image: np.ndarray) -> bool:
    """Odpowiednik cv2.imwrite() dzialajacy poprawnie ze sciezkami Unicode."""
    path = Path(path)
    ok, encoded = cv2.imencode(path.suffix or ".jpg", image)
    if not ok:
        return False
    encoded.tofile(str(path))
    return True


def imread_unicode(path: str | Path, flags: int = cv2.IMREAD_COLOR) -> np.ndarray | None:
    """Odpowiednik cv2.imread() dzialajacy poprawnie ze sciezkami Unicode."""
    path = Path(path)
    if not path.exists():
        return None
    data = np.fromfile(str(path), dtype=np.uint8)
    return cv2.imdecode(data, flags)
