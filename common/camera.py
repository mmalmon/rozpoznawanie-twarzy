"""
Obsluga kamery RoWave RC16 (4K, USB UVC) na Windows.

Kamera potrafi pracowac w rozdzielczosci az do 3840x2160, ale przy tak duzej
rozdzielczosci klatki na sekunde moga byc bardzo niskie, jesli sterownik nie
przelaczy sie w tryb kompresji MJPG. Dlatego wymuszamy FOURCC=MJPG przed
ustawieniem rozdzielczosci - w trybie domyslnym (YUY2/RAW) kamera 4K potrafi
dawac zaledwie kilka klatek na sekunde.
"""

from __future__ import annotations

import cv2


def open_camera(
    index: int = 0,
    width: int = 1920,
    height: int = 1080,
    fps: int = 30,
) -> cv2.VideoCapture:
    """Otwiera kamere RoWave RC16 w zadanej rozdzielczosci.

    Domyslnie uzywamy 1920x1080 zamiast pelnego 4K (3840x2160) - do detekcji
    twarzy i treningu nie potrzebujemy pelnej rozdzielczosci, a mniejsze klatki
    to wyzsze FPS i mniejsze obciazenie CPU/GPU. Jesli chcesz pelne 4K
    (np. do robienia bardzo ostrych zdjec treningowych), podaj width=3840,
    height=2160.
    """
    # CAP_DSHOW jest wymagany na Windows, aby poprawnie ustawic FOURCC/rozdzielczosc
    # dla kamer UVC - domyslny backend MSMF czasem ignoruje te ustawienia.
    cap = cv2.VideoCapture(index, cv2.CAP_DSHOW)

    if not cap.isOpened():
        raise RuntimeError(
            f"Nie udalo sie otworzyc kamery o indeksie {index}. "
            "Sprawdz, czy RoWave RC16 jest podlaczona i czy nie jest "
            "uzywana przez inna aplikacje (np. Kamera Windows, Teams, Zoom)."
        )

    cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
    cap.set(cv2.CAP_PROP_FPS, fps)

    actual_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    actual_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    actual_fps = cap.get(cv2.CAP_PROP_FPS)
    print(
        f"[camera] Kamera otwarta: {actual_w}x{actual_h} @ {actual_fps:.1f} FPS "
        f"(zadano {width}x{height} @ {fps} FPS)"
    )

    return cap
