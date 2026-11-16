"""Uruchamianie treningu (skrypty 02 + 03) z panelu GUI.

Panel nie liczy wzorców sam — uruchamia dokładnie te same skrypty, których
używa się z konsoli. Ten moduł:

1. startuje `02_podzial_danych.py` (podział zdjęć na train/val),
2. potem `03_trenowanie_modelu.py --postep` (embeddingi + kalibracja progów),
3. czyta na bieżąco ich wyjście, zamienia linie `[postep-start]`/`[postep]`
   na procenty i pokazuje log w panelu,
4. po sukcesie woła callback, żeby silnik rozpoznawania przeładował wzorce.

DLA UCZNIÓW: dlaczego osobny proces, a nie zwykłe wywołanie funkcji?
Trening importuje PyTorch i sieć neuronową — jeśli coś w nim padnie (np.
brak pamięci GPU), bezpieczniej jest, żeby padł osobny proces, a panel dalej
działał i pokazał komunikat błędu.
"""

from __future__ import annotations

import os
import subprocess
import sys
import threading
import time
from collections import deque
from collections.abc import Callable
from pathlib import Path

KORZEN_PROJEKTU = Path(__file__).resolve().parent.parent
FOLDER_RAW = KORZEN_PROJEKTU / "data" / "raw"
NAZWA_KLASY_NIEZNAJOMY = "nieznajomy"

SKRYPT_PODZIALU = KORZEN_PROJEKTU / "02_podzial_danych.py"
SKRYPT_TRENINGU = KORZEN_PROJEKTU / "03_trenowanie_modelu.py"

STAN_BEZCZYNNY = "bezczynny"
STAN_DZIALA = "dziala"
STAN_GOTOWE = "gotowe"
STAN_BLAD = "blad"
STAN_PRZERWANY = "przerwany"

MAKS_LINII_LOGU = 400

# Udział etapów w pasku postępu: podział danych to szybka operacja (10%),
# liczenie embeddingów i kalibracja to reszta (85%), a ostatnie 5% to zapis
# plików i wykres kalibracji.
POSTEP_PO_PODZIALE = 10.0
POSTEP_PRZED_ZAPISEM = 95.0


class MenedzerTreningu:
    """Steruje procesem treningu i udostępnia jego stan dla API."""

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

        # Bieżąca faza z linii [postep-start] i jej mianownik.
        self._faza = ""
        self._faza_total = 0

    # --------------------------------------------------------------- start
    def start(self) -> None:
        """Uruchamia trening w tle. Rzuca RuntimeError, gdy nie można."""
        with self._zamek:
            if self._stan == STAN_DZIALA:
                raise RuntimeError("Trening już trwa.")

        brak = self._sprawdz_dane()
        if brak:
            raise RuntimeError(brak)

        with self._zamek:
            self._stan = STAN_DZIALA
            self._etap = "Dzielenie danych (02)"
            self._postep = 0.0
            self._blad = None
            self._log.clear()
            self._czas_startu = time.monotonic()
            self._czas_konca = None
            self._faza = ""
            self._faza_total = 0

        self._watek = threading.Thread(
            target=self._uruchom_trening, name="trening-gui", daemon=True
        )
        self._watek.start()

    def _sprawdz_dane(self) -> str | None:
        """Wczesna walidacja: lepiej powiedzieć od razu, czego brakuje."""
        if not FOLDER_RAW.exists():
            return "Brak folderu data/raw. Najpierw dodaj zdjęcia twarzy."
        osoby = [
            folder
            for folder in FOLDER_RAW.iterdir()
            if folder.is_dir() and folder.name != NAZWA_KLASY_NIEZNAJOMY
        ]
        if not osoby:
            return (
                "Brak zdjęć osób w data/raw. Dodaj co najmniej jedną twarz "
                "w zakładce „Dodaj twarz”."
            )
        if not any(any(folder.glob("*.jpg")) for folder in osoby):
            return "Foldery osób są puste — dodaj zdjęcia twarzy."
        return None

    def _uruchom_trening(self) -> None:
        sukces = False
        try:
            if self._przerwano():
                return
            if not self._uruchom_skrypt(
                SKRYPT_PODZIALU, ["02_podzial_danych.py"]
            ):
                return

            if self._przerwano():
                return
            self._aktualizuj(etap="Liczenie embeddingów (03)", postep=POSTEP_PO_PODZIALE)
            if not self._uruchom_skrypt(
                SKRYPT_TRENINGU, ["03_trenowanie_modelu.py", "--postep"], sledz_postep=True
            ):
                return

            if self._przerwano():
                return
            self._aktualizuj(postep=100.0, etap="Zakończono")
            with self._zamek:
                self._stan = STAN_GOTOWE
                self._czas_konca = time.monotonic()
            sukces = True
        finally:
            with self._zamek:
                self._proces = None
            if self._po_zakonczeniu is not None:
                try:
                    self._po_zakonczeniu(sukces)
                except Exception as blad:  # noqa: BLE001 - nie psujemy stanu treningu
                    self._dopisz_log(f"[uwaga] Nie udało się odświeżyć wzorców: {blad}")

    def _przerwano(self) -> bool:
        with self._zamek:
            return self._stan == STAN_PRZERWANY

    def _uruchom_skrypt(
        self, sciezka: Path, opis: list[str], sledz_postep: bool = False
    ) -> bool:
        """Uruchamia jeden skrypt i czyta jego wyjście linia po linii."""
        if not sciezka.exists():
            self._zakoncz_bledem(f"Brak pliku {sciezka.name}.")
            return False
        # Przerwano między etapami - nie startujemy kolejnego skryptu.
        if self._przerwano():
            return False

        srodowisko = os.environ.copy()
        srodowisko["PYTHONUNBUFFERED"] = "1"
        srodowisko["PYTHONIOENCODING"] = "utf-8"

        # Na Windows nie pokazujemy czarnego okna konsoli dla procesu potomnego.
        flagi = 0
        if os.name == "nt":
            flagi = getattr(subprocess, "CREATE_NO_WINDOW", 0)

        self._dopisz_log(f"[panel] Uruchamiam {' '.join(opis)}")
        try:
            proces = subprocess.Popen(
                [sys.executable, str(sciezka), *opis[1:]],
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
            self._zakoncz_bledem(f"Nie udało się uruchomić {sciezka.name}: {blad}")
            return False

        with self._zamek:
            self._proces = proces

        assert proces.stdout is not None
        for linia in proces.stdout:
            linia = linia.rstrip()
            if not linia:
                continue
            # Po kliknięciu "Przerwij" nie dopisujemy już nic do logu.
            # Zabijany proces potrafi wypisać wtedy traceback z biblioteki
            # multiprocessing, który tylko straszy ucznia.
            if self._przerwano():
                continue
            # Linie [postep] to sterowanie paskiem, a nie treść dla ucznia.
            # Nie wrzucamy ich do logu, bo przy setkach zdjęć zalałyby go
            # i wypchnęły ważne komunikaty (np. ostrzeżenia kalibracji).
            if sledz_postep and linia.startswith(("[postep-start] ", "[postep] ")):
                self._przetworz_linie_postepu(linia)
                continue
            self._dopisz_log(linia)

        kod = proces.wait()
        with self._zamek:
            self._proces = None

        if kod != 0:
            # Gdy użytkownik kliknął "Przerwij", stan jest już ustawiony na
            # "przerwany" - nie nadpisujemy go komunikatem o kodzie wyjścia.
            with self._zamek:
                przerwano = self._stan == STAN_PRZERWANY
            if przerwano:
                return False
            self._zakoncz_bledem(
                f"{sciezka.name} zakończył się kodem {kod}. Szczegóły w logu."
            )
            return False
        return True

    # ------------------------------------------------------------ postep
    def _przetworz_linie_postepu(self, linia: str) -> None:
        """Zamienia maszynowe linie skryptu 03 na procent paska postępu."""
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
                total = self._faza_total
                if total <= 0:
                    return
                ulamek = min(1.0, zrobione / total)
                postep = POSTEP_PO_PODZIALE + ulamek * (
                    POSTEP_PRZED_ZAPISEM - POSTEP_PO_PODZIALE
                )
            self._aktualizuj(postep=postep)

    # ----------------------------------------------------------- sterowanie
    def zatrzymaj(self) -> bool:
        """Przerywa trening (zabija proces potomny). Zwraca True, gdy był coś."""
        with self._zamek:
            proces = self._proces
            if self._stan != STAN_DZIALA:
                return False
            # Stan zmieniamy PRZED zabiciem procesu: wątek czytający log
            # od razu przestaje dopisywać linie (np. traceback z zabijanego
            # procesu), a my nie nadpiszemy "przerwany" błędem kodu wyjścia.
            self._stan = STAN_PRZERWANY
            self._etap = "Przerwano"
            self._blad = "Trening został przerwany przez użytkownika."
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

        self._dopisz_log("[panel] Trening przerwany przez użytkownika.")

        # Czekamy, aż wątek treningu domknie swoją robotę (callback i stan).
        # Bez tego szybkie "Przerwij -> Uruchom" mogłoby pomieszać stany.
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

            return {
                "stan": self._stan,
                "etap": self._etap,
                "postep": round(self._postep, 1),
                "blad": self._blad,
                "log": list(self._log)[-150:],
                "czas_s": round(czas, 1),
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

    def dopisz_log(self, linia: str) -> None:
        """Publiczna wersja dla innych modułów (np. po przeładowaniu wzorców)."""
        self._dopisz_log(linia)
