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

# Ta linijka MUSI byc pierwszym kodem wykonywalnym w tym pliku - patrz
# szczegolowe wyjasnienie w common/uruchom_w_venv.py (automatyczne
# przelaczenie na Python z .venv, jesli uruchomiono innym interpreterem).
from common.uruchom_w_venv import przelacz_na_venv_jesli_trzeba

przelacz_na_venv_jesli_trzeba()

# Importujemy wlasna funkcje z folderu common/ - ten sam kod, ktorego
# uzywaja skrypty 00, 01 i 04 do wykrywania kamer podlaczonych do komputera.
from common.camera import wykryj_dostepne_kamery


def main() -> None:
    print("Szukam dostepnych kamer (indeksy 0-4)...\n")
    kamery = wykryj_dostepne_kamery()

    if not kamery:
        print("Nie znaleziono zadnej kamery. Sprawdz podlaczenie USB.")
        return

    for kamera in kamery:
        podpowiedz = (
            " <- prawdopodobnie RoWave RC16 (prawdziwe 4K)" if kamera.czy_prawdopodobnie_4k else ""
        )
        print(
            f"--kamera {kamera.indeks}: maks. wykryta rozdzielczosc "
            f"{kamera.maks_szerokosc}x{kamera.maks_wysokosc}{podpowiedz}"
        )


if __name__ == "__main__":
    main()
