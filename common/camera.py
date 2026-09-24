"""
Obsluga kamery RoWave RC16 (4K, USB UVC) na Windows.

Kamera potrafi pracowac w rozdzielczosci az do 3840x2160, ale przy tak duzej
rozdzielczosci klatki na sekunde moga byc bardzo niskie, jesli sterownik nie
przelaczy sie w tryb kompresji MJPG. Dlatego wymuszamy FOURCC=MJPG przed
ustawieniem rozdzielczosci - w trybie domyslnym (YUY2/RAW) kamera 4K potrafi
dawac zaledwie kilka klatek na sekunde.
"""

from __future__ import annotations

from dataclasses import dataclass

import cv2


@dataclass
class DetectedCamera:
    index: int
    max_width: int
    max_height: int

    @property
    def is_likely_4k(self) -> bool:
        return (self.max_width, self.max_height) == (3840, 2160)


def list_available_cameras(max_index: int = 5) -> list[DetectedCamera]:
    """Wykrywa dostepne kamery, probujac otworzyc kolejne indeksy 0..max_index-1.

    Dla kazdej znalezionej kamery probuje ustawic 4K (3840x2160) i sprawdza,
    jaka rozdzielczosc faktycznie udalo sie uzyskac - pomaga to odroznic
    prawdziwa kamere 4K (np. RoWave RC16) od kamery wbudowanej w laptopa
    (zazwyczaj max. 720p/1080p/1440p).
    """
    cameras: list[DetectedCamera] = []

    for index in range(max_index):
        cap = cv2.VideoCapture(index, cv2.CAP_DSHOW)
        if not cap.isOpened():
            cap.release()
            continue

        cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 3840)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 2160)

        ok, _ = cap.read()
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        cap.release()

        if ok:
            cameras.append(DetectedCamera(index, width, height))

    return cameras


def choose_camera_interactive(max_index: int = 5) -> int:
    """Wykrywa dostepne kamery i pyta uzytkownika w konsoli, ktorej uzyc.

    Jesli znaleziono tylko jedna kamere, wybiera ja automatycznie bez
    pytania. Uzywane, gdy skrypt zostal uruchomiony bez jawnie podanego
    argumentu --kamera.
    """
    print("[camera] Szukam dostepnych kamer...")
    cameras = list_available_cameras(max_index)

    if not cameras:
        raise RuntimeError(
            "Nie znaleziono zadnej kamery. Sprawdz, czy RoWave RC16 (lub inna "
            "kamera) jest podlaczona przez USB i czy nie jest uzywana przez "
            "inna aplikacje."
        )

    if len(cameras) == 1:
        cam = cameras[0]
        print(f"[camera] Znaleziono jedna kamere: --kamera {cam.index} ({cam.max_width}x{cam.max_height})")
        return cam.index

    print("\nZnaleziono kilka kamer:")
    for cam in cameras:
        hint = "  <- prawdopodobnie RoWave RC16 (prawdziwe 4K)" if cam.is_likely_4k else ""
        print(f"  [{cam.index}] maks. rozdzielczosc {cam.max_width}x{cam.max_height}{hint}")

    valid_indexes = {cam.index for cam in cameras}
    while True:
        choice = input(f"\nWybierz numer kamery ({sorted(valid_indexes)}): ").strip()
        if choice.isdigit() and int(choice) in valid_indexes:
            return int(choice)
        print("Niepoprawny wybor, sprobuj ponownie.")


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
