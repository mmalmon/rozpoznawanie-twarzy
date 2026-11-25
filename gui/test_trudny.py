"""Uruchamianie testu trudnego (skrypt 09) z panelu GUI.

To ta sama maszyneria co ewaluacja: osobny proces, log linia po linii,
maszynowe linie postępu i odczyt gotowego raportu JSON. Różnica jest tylko
w danych i skrypcie — tutaj model jest oceniany na zdjęciach z folderu
`data/raw_test`, których nie użyto do treningu ani kalibracji progu.
"""

from __future__ import annotations

from pathlib import Path

from gui.ewaluacja import PLIK_WZORCOW, MenedzerEwaluacji

KORZEN_PROJEKTU = Path(__file__).resolve().parent.parent
FOLDER_TESTU_TRUDNEGO = KORZEN_PROJEKTU / "data" / "raw_test"
FOLDER_RAPORTU = KORZEN_PROJEKTU / "models" / "test_trudny"
FOLDER_WYKRESOW = FOLDER_RAPORTU / "wykresy"
PLIK_RAPORTU = FOLDER_RAPORTU / "raport_test_trudny.json"

ROZSZERZENIA_ZDJEC = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


class MenedzerTestuTrudnego(MenedzerEwaluacji):
    """Steruje skryptem 09_test_trudny.py i udostępnia jego stan dla API."""

    NAZWA = "Test trudny"
    SKRYPT = KORZEN_PROJEKTU / "09_test_trudny.py"
    OPIS_SKRYPTU = "09_test_trudny.py"
    PLIK_RAPORTU = PLIK_RAPORTU
    ETAP_START = "Przygotowanie (09)"
    DODATKOWE_ARGUMENTY = ("--folder", "data/raw_test")

    # Embeddingi to większość pracy; metryki, wykresy i zapis to końcówka.
    WAGI_FAZ = {
        "embeddingi": (0.0, 80.0),
        "metryki": (80.0, 97.0),
    }

    def _sprawdz_dane(self) -> str | None:
        """Sprawdza, czy jest model i czy test trudny ma jakiekolwiek zdjęcia."""
        if not self.SKRYPT.exists():
            return "Brak pliku 09_test_trudny.py."
        if not PLIK_WZORCOW.exists():
            return (
                "Brak wytrenowanych wzorców (models/wzorce_osob.json). "
                "Najpierw uruchom trening."
            )
        if not FOLDER_TESTU_TRUDNEGO.is_dir():
            return (
                "Brak folderu data/raw_test. Zbierz zdjęcia trudne w zakładce "
                "„Dodaj twarz” (opcja „Zapisz jako test trudny”) albo uruchom "
                "01_zbieranie_danych.py --folder data/raw_test."
            )

        podfoldery = [
            folder
            for folder in FOLDER_TESTU_TRUDNEGO.iterdir()
            if folder.is_dir()
            and any(
                plik.is_file() and plik.suffix.lower() in ROZSZERZENIA_ZDJEC
                for plik in folder.iterdir()
            )
        ]
        if not podfoldery:
            return (
                "Folder data/raw_test nie ma zdjęć w podfolderach osób. "
                "Nazwy podfolderów muszą odpowiadać osobom z bazy "
                "(np. gracjan) albo klasie „nieznajomy”."
            )
        return None
