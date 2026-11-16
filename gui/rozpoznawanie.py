"""Silnik rozpoznawania twarzy dla panelu GUI.

Owija wspólną logikę z `common/rozpoznawanie.py` w obiekt gotowy do użycia
przez wątek kamery:

- modele (ekstraktor cech, detektor, transformacje) ładuje się RAZ i trzyma
  w pamięci — wczytanie sieci trwa kilka sekund, więc nie może się dziać przy
  każdej klatce,
- wzorce można przeładować po treningu bez restartu aplikacji,
- rozpoznawanie emocji jest opcjonalne i domyślnie WYŁĄCZONE (AI Act),
- wszystkie metody są bezpieczne wątkowo: Flask obsługuje żądania w kilku
  wątkach, a wątek kamery liczy rozpoznawanie równolegle.
"""

from __future__ import annotations

import threading

import numpy as np

from common.emotions import RozpoznawaczEmocji
from common.face_detector import DetektorTwarzy
from common.model import pobierz_urzadzenie, zbuduj_ekstraktor_cech
from common.rozpoznawanie import (
    BrakWzorcowError,
    NiepoprawneWzorceError,
    analizuj_klatke,
    przygotuj_obrazy_kamery,
    wczytaj_wzorce,
)


class SilnikRozpoznawania:
    """Trzyma modele i wzorce oraz liczy wyniki dla pojedynczej klatki."""

    def __init__(self) -> None:
        self._zamek = threading.RLock()

        self._urzadzenie = None
        self._ekstraktor = None
        self._detektor: DetektorTwarzy | None = None
        self._przygotuj_obrazy = []
        self._wzorce: dict[str, dict] = {}
        self._emocje: RozpoznawaczEmocji | None = None
        self._emocje_wlaczone = False
        self._korekta_progu = 0.0

        self._gotowy = False
        self._ladowanie = False
        self._blad: str | None = None

    # ------------------------------------------------------------ ladowanie
    def przygotuj(self) -> None:
        """Wczytuje sieci i wzorce (kilka sekund). Wolaj z wątku w tle."""
        with self._zamek:
            if self._gotowy or self._ladowanie:
                return
            self._ladowanie = True
            self._blad = None
        self._wczytaj_modele()

    def przygotuj_w_tle(self) -> bool:
        """Startuje wczytywanie modeli w tle. True = uruchomiono nowy wątek."""
        with self._zamek:
            if self._gotowy or self._ladowanie:
                return False
            self._ladowanie = True
            self._blad = None

        watek = threading.Thread(
            target=self._wczytaj_modele,
            name="ladowanie-modelu",
            daemon=True,
        )
        watek.start()
        return True

    def _wczytaj_modele(self) -> None:
        """Ciężka część wczytywania - bez sprawdzania flag (robi to wołający)."""
        try:
            urzadzenie = pobierz_urzadzenie()
            ekstraktor = zbuduj_ekstraktor_cech().to(urzadzenie)
            detektor = DetektorTwarzy()
            przygotuj_obrazy = przygotuj_obrazy_kamery()
            wzorce = wczytaj_wzorce(urzadzenie=urzadzenie)
        except (BrakWzorcowError, NiepoprawneWzorceError) as blad:
            with self._zamek:
                self._blad = str(blad)
                self._ladowanie = False
            return
        except Exception as blad:  # noqa: BLE001 - komunikat trafia do panelu
            with self._zamek:
                self._blad = f"Nie udało się wczytać modeli: {blad}"
                self._ladowanie = False
            return

        with self._zamek:
            self._urzadzenie = urzadzenie
            self._ekstraktor = ekstraktor
            self._detektor = detektor
            self._przygotuj_obrazy = przygotuj_obrazy
            self._wzorce = wzorce
            self._gotowy = True
            self._ladowanie = False

    def przeladuj_wzorce(self) -> int:
        """Wczytuje ponownie models/wzorce_osob.json (np. po treningu).

        Zwraca liczbę osób w nowym pliku. Gdy plik jest uszkodzony, rzuca
        wyjątek, ale zostawia w pamięci poprzednie, działające wzorce.
        """
        with self._zamek:
            if not self._gotowy or self._urzadzenie is None:
                raise RuntimeError("Silnik rozpoznawania nie jest jeszcze gotowy.")
            urzadzenie = self._urzadzenie

        wzorce = wczytaj_wzorce(urzadzenie=urzadzenie)
        with self._zamek:
            self._wzorce = wzorce
            self._blad = None
        return len(wzorce)

    def ustaw_emocje(self, wlaczone: bool) -> None:
        """Włącza/wyłącza rozpoznawanie emocji (model FER+)."""
        with self._zamek:
            if wlaczone and not RozpoznawaczEmocji.dostepny():
                raise RuntimeError(
                    "Brak modelu emocji w common/modele/emotion-ferplus-8.onnx. "
                    "Pobierz go zgodnie z README, żeby włączyć tę funkcję."
                )
            self._emocje_wlaczone = bool(wlaczone)
            if wlaczone and self._emocje is None:
                self._emocje = RozpoznawaczEmocji()
            if not wlaczone:
                # Czyścimy stan wygładzania, żeby po ponownym włączeniu nie
                # mieszać emocji z odległej przeszłości.
                self._emocje = None

    def ustaw_korekte_progu(self, korekta: float) -> None:
        """Dodatnia wartość = trudniej rozpoznać (ostrzejszy próg)."""
        with self._zamek:
            self._korekta_progu = float(korekta)

    # ------------------------------------------------------------- odczyt
    @property
    def gotowy(self) -> bool:
        with self._zamek:
            return self._gotowy

    def status(self) -> dict:
        with self._zamek:
            return {
                "gotowy": self._gotowy,
                "ladowanie": self._ladowanie,
                "blad": self._blad,
                "osoby": sorted(self._wzorce),
                "liczba_osob": len(self._wzorce),
                "emocje_dostepne": RozpoznawaczEmocji.dostepny(),
                "emocje_wlaczone": self._emocje_wlaczone,
                "urzadzenie": str(self._urzadzenie) if self._urzadzenie else None,
                "korekta_progu": self._korekta_progu,
            }

    def analizuj(self, klatka: np.ndarray) -> list[dict]:
        """Zwraca listę wyników dla wszystkich twarzy na klatce.

        Każdy wynik to słownik gotowy do wysłania jako JSON (współrzędne w
        pikselach oryginalnej klatki + etykieta osoby/podobieństwo/emocja).
        """
        with self._zamek:
            if not self._gotowy or self._detektor is None or self._ekstraktor is None:
                raise RuntimeError("Silnik rozpoznawania nie jest jeszcze gotowy.")
            ekstraktor = self._ekstraktor
            detektor = self._detektor
            przygotuj_obrazy = self._przygotuj_obrazy
            wzorce = self._wzorce
            urzadzenie = self._urzadzenie
            emocje = self._emocje if self._emocje_wlaczone else None
            korekta = self._korekta_progu

        wyniki = analizuj_klatke(
            klatka,
            detektor,
            ekstraktor,
            przygotuj_obrazy,
            wzorce,
            urzadzenie,
            korekta,
            emocje,
        )

        return [
            {
                "x": int(twarz.x),
                "y": int(twarz.y),
                "szerokosc": int(twarz.w),
                "wysokosc": int(twarz.h),
                "pewnosc": float(twarz.pewnosc),
                "osoba": osoba,
                "podobienstwo": round(float(podobienstwo), 4),
                "emocja": emocja,
                "pewnosc_emocji": round(float(pewnosc_emocji), 4),
            }
            for twarz, osoba, podobienstwo, emocja, pewnosc_emocji in wyniki
        ]
