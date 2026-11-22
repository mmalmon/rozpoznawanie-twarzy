"""Uruchamianie ewaluacji (skrypt 07) z panelu GUI.

Panel nie liczy metryk sam — uruchamia dokładnie ten sam skrypt, którego
używa się z konsoli (`07_ewaluacja.py --postep`) i czyta jego log oraz
maszynowe linie postępu. Dzięki temu wynik w panelu i w terminalu jest
identyczny.

DLA UCZNIÓW: dlaczego osobny proces?
Ewaluacja liczy embeddingi dla setek zdjęć i trenuje dodatkowe klasyfikatory
(k-NN, regresja logistyczna). Jeśli zabraknie pamięci GPU albo biblioteki,
bezpieczniej jest, żeby padł osobny proces — panel dalej działa i pokazuje
czytelny komunikat.
"""

from __future__ import annotations

import os
import subprocess
import sys
import threading
import time
from collections import deque
from collections.abc import Callable
from datetime import datetime
from pathlib import Path

KORZEN_PROJEKTU = Path(__file__).resolve().parent.parent
FOLDER_EWALUACJI = KORZEN_PROJEKTU / "models" / "ewaluacja"
FOLDER_WYKRESOW = FOLDER_EWALUACJI / "wykresy"
FOLDER_POMYLEK = FOLDER_EWALUACJI / "pomylki"
PLIK_RAPORTU = FOLDER_EWALUACJI / "raport_ewaluacji.json"

SKRYPT_EWALUACJI = KORZEN_PROJEKTU / "07_ewaluacja.py"
PLIK_WZORCOW = KORZEN_PROJEKTU / "models" / "wzorce_osob.json"
FOLDER_WALIDACJI = KORZEN_PROJEKTU / "data" / "processed" / "val"

STAN_BEZCZYNNY = "bezczynny"
STAN_DZIALA = "dziala"
STAN_GOTOWE = "gotowe"
STAN_BLAD = "blad"
STAN_PRZERWANY = "przerwany"

MAKS_LINII_LOGU = 500

# Udział faz w pasku postępu. Liczenie embeddingów to najdłuższa część,
# a metryki, wykresy i zapis raportu - krótka końcówka.
WAGI_FAZ = {
    "embeddingi": (0.0, 82.0),
    "metryki": (82.0, 97.0),
}


class MenedzerEwaluacji:
    """Steruje procesem ewaluacji i udostępnia jego stan dla API."""

    def __init__(self, po_zakonczeniu: Callable[[bool], None] | None = None) -> None:
        self._zamek = threading.Lock()
        self._proces: subprocess.Popen | None = None
        self._watek: threading.Thread | None = None
        self._po_zakonczeniu = po_zakonczeniu

        self._stan = STAN_BEZCZYNNY
        self._etap = ""
        self._postep = 0.0
        self._blad: str | None = None
        self._log: deque[str] = deque(maxlen=MAKS_LINII_LOGU)
        self._czas_startu: float | None = None
        self._czas_konca: float | None = None

        self._faza = ""
        self._faza_total = 0

    # --------------------------------------------------------------- start
    def start(self) -> None:
        """Uruchamia ewaluację w tle. Rzuca RuntimeError, gdy nie można."""
        with self._zamek:
            if self._stan == STAN_DZIALA:
                raise RuntimeError("Ewaluacja już trwa.")

        brak = self._sprawdz_dane()
        if brak:
            raise RuntimeError(brak)

        with self._zamek:
            self._stan = STAN_DZIALA
            self._etap = "Przygotowanie (07)"
            self._postep = 0.0
            self._blad = None
            self._log.clear()
            self._czas_startu = time.monotonic()
            self._czas_konca = None
            self._faza = ""
            self._faza_total = 0

        self._watek = threading.Thread(
            target=self._uruchom_ewaluacje, name="ewaluacja-gui", daemon=True
        )
        self._watek.start()

    def _sprawdz_dane(self) -> str | None:
        """Wczesna walidacja: lepiej powiedzieć od razu, czego brakuje."""
        if not SKRYPT_EWALUACJI.exists():
            return "Brak pliku 07_ewaluacja.py."
        if not PLIK_WZORCOW.exists():
            return (
                "Brak wytrenowanych wzorców (models/wzorce_osob.json). "
                "Najpierw uruchom trening."
            )
        if not FOLDER_WALIDACJI.is_dir():
            return (
                "Brak zbioru walidacyjnego (data/processed/val). "
                "Najpierw uruchom trening — skrypt 02 podzieli dane."
            )
        if not any(
            folder.is_dir() and any(folder.iterdir())
            for folder in FOLDER_WALIDACJI.iterdir()
        ):
            return "Zbiór walidacyjny jest pusty — uruchom ponownie trening."
        return None

    def _uruchom_ewaluacje(self) -> None:
        sukces = False
        try:
            sukces = self._uruchom_skrypt()
            if sukces:
                self._aktualizuj(postep=100.0, etap="Zakończono")
                with self._zamek:
                    self._stan = STAN_GOTOWE
                    self._czas_konca = time.monotonic()
        finally:
            with self._zamek:
                self._proces = None
            if self._po_zakonczeniu is not None:
                try:
                    self._po_zakonczeniu(sukces)
                except Exception as blad:  # noqa: BLE001 - nie psujemy stanu
                    self._dopisz_log(f"[uwaga] Błąd po ewaluacji: {blad}")

    def _przerwano(self) -> bool:
        with self._zamek:
            return self._stan == STAN_PRZERWANY

    def _uruchom_skrypt(self) -> bool:
        """Uruchamia 07 i czyta jego wyjście linia po linii."""
        if self._przerwano():
            return False

        srodowisko = os.environ.copy()
        srodowisko["PYTHONUNBUFFERED"] = "1"
        srodowisko["PYTHONIOENCODING"] = "utf-8"

        flagi = 0
        if os.name == "nt":
            flagi = getattr(subprocess, "CREATE_NO_WINDOW", 0)

        self._dopisz_log("[panel] Uruchamiam 07_ewaluacja.py --postep")
        try:
            proces = subprocess.Popen(
                [sys.executable, str(SKRYPT_EWALUACJI), "--postep"],
                cwd=str(KORZEN_PROJEKTU),
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                encoding="utf-8",
                errors="replace",
                bufsize=1,
                env=srodowisko,
                creationflags=flagi,
            )
        except OSError as blad:
            self._zakoncz_bledem(f"Nie udało się uruchomić 07_ewaluacja.py: {blad}")
            return False

        with self._zamek:
            self._proces = proces

        assert proces.stdout is not None
        for linia in proces.stdout:
            linia = linia.rstrip()
            if not linia:
                continue
            if self._przerwano():
                continue
            if linia.startswith(("[postep-start] ", "[postep] ")):
                self._przetworz_linie_postepu(linia)
                continue
            self._dopisz_log(linia)

        kod = proces.wait()
        with self._zamek:
            self._proces = None

        if kod != 0:
            with self._zamek:
                przerwano = self._stan == STAN_PRZERWANY
            if przerwano:
                return False
            self._zakoncz_bledem(
                f"07_ewaluacja.py zakończył się kodem {kod}. Szczegóły w logu."
            )
            return False
        return True

    # ------------------------------------------------------------ postep
    def _przetworz_linie_postepu(self, linia: str) -> None:
        """Zamienia maszynowe linie skryptu 07 na procent paska postępu."""
        if linia.startswith("[postep-start] "):
            czesci = linia[len("[postep-start] ") :].split()
            if len(czesci) == 2:
                try:
                    total = int(czesci[1])
                except ValueError:
                    return
                with self._zamek:
                    self._faza = czesci[0]
                    self._faza_total = max(0, total)
                return

        if linia.startswith("[postep] "):
            czesci = linia[len("[postep] ") :].split()
            if len(czesci) != 2:
                return
            try:
                zrobione = int(czesci[1])
            except ValueError:
                return

            with self._zamek:
                faza = self._faza
                total = self._faza_total
                if total <= 0 or faza not in WAGI_FAZ:
                    return
                poczatek, koniec = WAGI_FAZ[faza]
                ulamek = min(1.0, max(0.0, zrobione / total))
                postep = poczatek + ulamek * (koniec - poczatek)
            self._aktualizuj(postep=postep)

    # ----------------------------------------------------------- sterowanie
    def zatrzymaj(self) -> bool:
        """Przerywa ewaluację (zabija proces potomny). True = coś zatrzymano."""
        with self._zamek:
            proces = self._proces
            if self._stan != STAN_DZIALA:
                return False
            # Stan zmieniamy PRZED zabiciem procesu - wątek czytający log
            # przestaje dopisywać linie, a błąd kodu wyjścia nie nadpisze
            # informacji o przerwaniu.
            self._stan = STAN_PRZERWANY
            self._etap = "Przerwano"
            self._blad = "Ewaluacja została przerwana przez użytkownika."
            self._czas_konca = time.monotonic()

        if proces is not None and proces.poll() is None:
            try:
                proces.terminate()
                proces.wait(timeout=5)
            except (OSError, subprocess.TimeoutExpired):
                try:
                    proces.kill()
                except OSError:
                    pass

        self._dopisz_log("[panel] Ewaluacja przerwana przez użytkownika.")

        watek = self._watek
        if (
            watek is not None
            and watek.is_alive()
            and watek is not threading.current_thread()
        ):
            watek.join(timeout=10.0)
        return True

    # ---------------------------------------------------------------- stan
    def status(self) -> dict:
        with self._zamek:
            teraz = time.monotonic()
            if self._czas_startu is None:
                czas = 0.0
            else:
                koniec = self._czas_konca or teraz
                czas = max(0.0, koniec - self._czas_startu)

            raport_dostepny = PLIK_RAPORTU.exists()
            czas_raportu = None
            if raport_dostepny:
                try:
                    czas_raportu = datetime.fromtimestamp(
                        PLIK_RAPORTU.stat().st_mtime
                    ).isoformat(timespec="seconds")
                except OSError:
                    czas_raportu = None

            return {
                "stan": self._stan,
                "etap": self._etap,
                "postep": round(self._postep, 1),
                "blad": self._blad,
                "log": list(self._log)[-180:],
                "czas_s": round(czas, 1),
                "raport_dostepny": raport_dostepny,
                "czas_raportu": czas_raportu,
            }

    # ------------------------------------------------------------ pomocnicze
    def _aktualizuj(self, postep: float | None = None, etap: str | None = None) -> None:
        with self._zamek:
            if postep is not None:
                self._postep = min(100.0, max(0.0, postep))
            if etap is not None:
                self._etap = etap

    def _zakoncz_bledem(self, komunikat: str) -> None:
        with self._zamek:
            self._stan = STAN_BLAD
            self._blad = komunikat
            self._etap = "Błąd"
            self._czas_konca = time.monotonic()
        self._dopisz_log(f"[blad] {komunikat}")

    def _dopisz_log(self, linia: str) -> None:
        with self._zamek:
            self._log.append(linia)
