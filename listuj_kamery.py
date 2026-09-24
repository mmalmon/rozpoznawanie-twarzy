"""
Pomocniczy skrypt: wypisuje dostepne kamery i ich maksymalna wykryta
rozdzielczosc, aby ulatwic wybranie wlasciwego indeksu (--kamera N) dla
kamery RoWave RC16 na danym komputerze.

Na laptopach z wbudowana kamera zazwyczaj beda co najmniej dwa urzadzenia:
  - kamera wbudowana w laptopa (nizsza max. rozdzielczosc, np. 720p/1440p),
  - RoWave RC16 (prawdziwe 4K: 3840x2160).

Uzycie:
    python listuj_kamery.py

Uwaga: skrypty 00/01/04 domyslnie same pytaja o wybor kamery przy starcie
(jesli nie podasz --kamera), wiec ten skrypt jest potrzebny glownie do
podejrzenia sytuacji z wyprzedzeniem albo do debugowania.
"""

from __future__ import annotations

from common.camera import list_available_cameras


def main() -> None:
    print("Szukam dostepnych kamer (indeksy 0-4)...\n")
    cameras = list_available_cameras()

    if not cameras:
        print("Nie znaleziono zadnej kamery. Sprawdz podlaczenie USB.")
        return

    for cam in cameras:
        hint = " <- prawdopodobnie RoWave RC16 (prawdziwe 4K)" if cam.is_likely_4k else ""
        print(f"--kamera {cam.index}: maks. wykryta rozdzielczosc {cam.max_width}x{cam.max_height}{hint}")


if __name__ == "__main__":
    main()
