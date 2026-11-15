"""
Zgodność wsteczna dla starej ścieżki importu ``detection.detector``.

Właściwa implementacja znajduje się w ``src/detection/detector.py``.
Ten plik pozwala, aby starsze przykłady (``from detection.detector import
detect_face``) nadal działały.
"""

from __future__ import annotations

from src.detection.detector import detect_face

__all__ = ["detect_face"]
