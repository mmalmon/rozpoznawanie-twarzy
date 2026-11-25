"""Rejestr rozpoznań i frekwencji (lokalna baza SQLite).

DLA UCZNIÓW: po co osobna baza, a nie plik CSV?
CSV świetnie nadaje się do eksportu, ale trudno w nim bezpiecznie dopisywać
kolejne wpisy z dwóch wątków (kamera i panel) i filtrować po dacie. SQLite
jest wbudowany w Pythona, działa lokalnie i pozwala jednym zapytaniem
policzyć, kto był obecny danego dnia. Do Excela i tak robimy eksport CSV.

Co zapisujemy?
  - osobę (etykietę z modelu),
  - czas rozpoznania (czas lokalny),
  - podobieństwo do wzorca,
  - numer kamery i źródło wpisu (kamera / ręcznie).

Czego NIE zapisujemy? Zdjęć, embeddingów ani emocji. Rejestr to tylko tekst
i liczby — mniej danych osobowych, mniej ryzyka. Domyślnie wpisy starsze niż
90 dni są usuwane. Rejestr można w każdej chwili wyczyścić z panelu.

DLA UCZNIÓW: dlaczego nie zapisujemy każdej klatki?
Rozpoznawanie działa kilka razy na sekundę. Bez ograniczenia powstałyby
tysiące identycznych wpisów. Dlatego ta sama osoba jest zapisywana najwyżej
raz na ODSTEP_ZAPISU_S sekund (5 minut). Pierwsze i ostatnie rozpoznanie
w dniu nadal pokazują, kiedy osoba pojawiła się w kadrze.
"""

from __future__ import annotations

import csv
import io
import json
import sqlite3
import threading
import time
from datetime import date, datetime, timedelta
from pathlib import Path

KORZEN = Path(__file__).resolve().parent.parent
FOLDER_REJESTRU = KORZEN / "data" / "rejestr"
PLIK_BAZY = FOLDER_REJESTRU / "rejestr.db"
PLIK_WZORCOW = KORZEN / "models" / "wzorce_osob.json"

RETENCJA_DNI = 90
ODSTEP_ZAPISU_S = 300

ZRODLO_KAMERA = "kamera"
ZRODLO_RECZNE = "recznie"


def dzisiaj() -> str:
    """Dzisiejsza data w formacie YYYY-MM-DD (czas lokalny)."""
    return date.today().isoformat()


def poprawna_data(tekst: str | None) -> str | None:
    """Zamienia tekst na YYYY-MM-DD albo zwraca None, gdy data jest zła."""
    if not tekst:
        return dzisiaj()
    try:
        return datetime.strptime(str(tekst), "%Y-%m-%d").date().isoformat()
    except (TypeError, ValueError):
        return None


def osoby_z_modelu() -> list[str]:
    """Osoby zapisane w models/wzorce_osob.json (bez ładowania sieci).

    Do frekwencji potrzebujemy listy uczniów z modelu, a nie wszystkich
    folderów w data/raw_test — osoby z trudnego testu nie są przecież
    uczniami zapisanymi do rozpoznawania.
    """
    try:
        dane = json.loads(PLIK_WZORCOW.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return []
    return sorted(dane.get("osoby", {}))


class RejestrObecnosci:
    """Zapisuje rozpoznania i udostępnia dzienną frekwencję."""

    def __init__(
        self,
        sciezka: Path | None = None,
        retencja_dni: int = RETENCJA_DNI,
    ) -> None:
        self.sciezka = Path(sciezka) if sciezka is not None else PLIK_BAZY
        self.retencja_dni = int(retencja_dni)
        self._zamek = threading.RLock()
        self._blad: str | None = None
        self._wlaczony = True
        self._ostatni_zapis: dict[str, float] = {}
        self._ostatnie_czyszczenie = 0.0

        try:
            self.sciezka.parent.mkdir(parents=True, exist_ok=True)
            self._utworz_schemat()
            self._wlaczony = self._wczytaj_ustawienie()
            self.usun_starsze()
        except (OSError, sqlite3.Error) as blad:
            self._blad = f"Nie udało się otworzyć rejestru: {blad}"

    # ----------------------------------------------------------- schemat
    def _polacz(self) -> sqlite3.Connection:
        baza = sqlite3.connect(str(self.sciezka), timeout=5.0)
        baza.row_factory = sqlite3.Row
        return baza

    def _utworz_schemat(self) -> None:
        baza = self._polacz()
        try:
            # WAL pozwala czytać panelowi, gdy wątek kamery właśnie zapisuje.
            baza.execute("PRAGMA journal_mode=WAL")
            baza.executescript(
                """
                CREATE TABLE IF NOT EXISTS rozpoznania (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    osoba TEXT NOT NULL,
                    czas TEXT NOT NULL,
                    data TEXT NOT NULL,
                    podobienstwo REAL,
                    kamera INTEGER,
                    zrodlo TEXT NOT NULL DEFAULT 'kamera'
                );
                CREATE INDEX IF NOT EXISTS idx_rozpoznania_data
                    ON rozpoznania(data);
                CREATE INDEX IF NOT EXISTS idx_rozpoznania_osoba_data
                    ON rozpoznania(osoba, data);

                CREATE TABLE IF NOT EXISTS ustawienia (
                    klucz TEXT PRIMARY KEY,
                    wartosc TEXT NOT NULL
                );
                """
            )
            baza.commit()
        finally:
            baza.close()

    def _wczytaj_ustawienie(self) -> bool:
        baza = self._polacz()
        try:
            wiersz = baza.execute(
                "SELECT wartosc FROM ustawienia WHERE klucz = 'zapis_wlaczony'"
            ).fetchone()
        finally:
            baza.close()
        return True if wiersz is None else wiersz["wartosc"] == "1"

    def _zapisz_ustawienie(self, wlaczony: bool) -> None:
        baza = self._polacz()
        try:
            baza.execute(
                """
                INSERT INTO ustawienia (klucz, wartosc)
                VALUES ('zapis_wlaczony', ?)
                ON CONFLICT(klucz) DO UPDATE SET wartosc = excluded.wartosc
                """,
                ("1" if wlaczony else "0",),
            )
            baza.commit()
        finally:
            baza.close()

    # ---------------------------------------------------------- ustawienia
    @property
    def wlaczony(self) -> bool:
        return self._wlaczony

    def ustaw_wlaczony(self, wlaczony: bool) -> bool:
        """Włącza/wyłącza zapis nowych rozpoznań (ustawienie jest trwałe)."""
        with self._zamek:
            self._wlaczony = bool(wlaczony)
            self._zapisz_ustawienie(self._wlaczony)
        return self._wlaczony

    # ------------------------------------------------------------- zapis
    def _wstaw(
        self,
        osoba: str,
        czas: str,
        podobienstwo: float | None,
        kamera: int | None,
        zrodlo: str,
    ) -> None:
        baza = self._polacz()
        try:
            baza.execute(
                """
                INSERT INTO rozpoznania
                    (osoba, czas, data, podobienstwo, kamera, zrodlo)
                VALUES (?, ?, ?, ?, ?, ?)
                """,
                (
                    osoba,
                    czas,
                    czas[:10],
                    None if podobienstwo is None else round(float(podobienstwo), 4),
                    None if kamera is None else int(kamera),
                    zrodlo,
                ),
            )
            baza.commit()
        finally:
            baza.close()

    def zapisz(
        self,
        osoba: str,
        podobienstwo: float | None,
        kamera: int | None = None,
        zrodlo: str = ZRODLO_KAMERA,
        wymus: bool = False,
    ) -> bool:
        """Dopisuje rozpoznanie, jeśli minęło 5 minut od poprzedniego.

        Zwraca True, gdy wpis powstał, a False, gdy został pominięty
        (rejestr wyłączony albo to ten sam człowiek w krótkim odstępie).
        """
        osoba = (osoba or "").strip()
        if not osoba:
            return False
        if not self._wlaczony and not wymus:
            return False

        teraz_monotonic = time.monotonic()
        if not wymus:
            ostatni = self._ostatni_zapis.get(osoba)
            if ostatni is not None and teraz_monotonic - ostatni < ODSTEP_ZAPISU_S:
                return False

        teraz = datetime.now().replace(microsecond=0)
        czas = teraz.isoformat(timespec="seconds")

        with self._zamek:
            self._wstaw(osoba, czas, podobienstwo, kamera, zrodlo)
            self._ostatni_zapis[osoba] = teraz_monotonic
            self._moze_wyczyscic()
        return True

    def dopisz(self, osoba: str, data: str | None = None) -> dict:
        """Ręcznie dopisuje obecność (np. gdy model nie rozpoznał ucznia).

        Dla dzisiejszej daty bierzemy aktualną godzinę. Dla dnia z przeszłości
        wpisujemy godzinę 12:00, żeby nie udawać, że znamy dokładny moment.
        """
        osoba = (osoba or "").strip()
        if not osoba:
            raise ValueError("Podaj nazwę osoby.")
        data = poprawna_data(data)
        if data is None:
            raise ValueError("Data musi mieć format RRRR-MM-DD.")

        if data == dzisiaj():
            czas = datetime.now().replace(microsecond=0).isoformat(timespec="seconds")
        else:
            czas = f"{data}T12:00:00"

        with self._zamek:
            self._wstaw(osoba, czas, None, None, ZRODLO_RECZNE)
        return {
            "osoba": osoba,
            "czas": czas,
            "data": data,
            "podobienstwo": None,
            "kamera": None,
            "zrodlo": ZRODLO_RECZNE,
        }

    def _moze_wyczyscic(self) -> None:
        """Usuwa stare wpisy najwyżej raz na godzinę (nie przy każdym zapisie)."""
        teraz = time.monotonic()
        if teraz - self._ostatnie_czyszczenie < 3600:
            return
        self._ostatnie_czyszczenie = teraz
        try:
            self.usun_starsze()
        except (OSError, sqlite3.Error):
            # Sprzątanie nie może przerwać rozpoznawania.
            pass

    # ------------------------------------------------------------- odczyt
    def dzien(self, data: str | None = None) -> dict:
        """Frekwencja wybranego dnia: obecni, nieobecni i liczba wpisów."""
        data = poprawna_data(data)
        if data is None:
            raise ValueError("Data musi mieć format RRRR-MM-DD.")

        with self._zamek:
            baza = self._polacz()
            try:
                wiersze = baza.execute(
                    """
                    SELECT osoba,
                           MIN(czas) AS pierwsze,
                           MAX(czas) AS ostatnie,
                           COUNT(*) AS liczba
                    FROM rozpoznania
                    WHERE data = ?
                    GROUP BY osoba
                    ORDER BY osoba
                    """,
                    (data,),
                ).fetchall()
            finally:
                baza.close()

        obecni = [
            {
                "osoba": wiersz["osoba"],
                "pierwsze": wiersz["pierwsze"],
                "ostatnie": wiersz["ostatnie"],
                "liczba": int(wiersz["liczba"]),
            }
            for wiersz in wiersze
        ]
        obecne_nazwy = {wpis["osoba"] for wpis in obecni}
        znane = osoby_z_modelu()
        if not znane:
            # Bez modelu pokazujemy przynajmniej osoby, które się pojawiły.
            znane = sorted(obecne_nazwy)
        nieobecni = sorted(nazwa for nazwa in znane if nazwa not in obecne_nazwy)

        return {
            "data": data,
            "obecni": obecni,
            "nieobecni": nieobecni,
            "osoby": sorted(set(znane) | obecne_nazwy),
            "liczba_rozpoznan": sum(wpis["liczba"] for wpis in obecni),
        }

    def ostatnie(self, limit: int = 30) -> list[dict]:
        """Ostatnie wpisy, od najnowszego."""
        limit = max(1, min(int(limit), 200))
        with self._zamek:
            baza = self._polacz()
            try:
                wiersze = baza.execute(
                    """
                    SELECT id, osoba, czas, data, podobienstwo, kamera, zrodlo
                    FROM rozpoznania
                    ORDER BY czas DESC, id DESC
                    LIMIT ?
                    """,
                    (limit,),
                ).fetchall()
            finally:
                baza.close()
        return [dict(wiersz) for wiersz in wiersze]

    def status(self) -> dict:
        """Krótkie podsumowanie dla panelu (bez listy wszystkich wpisów)."""
        blad = self._blad
        try:
            with self._zamek:
                baza = self._polacz()
                try:
                    liczba_wpisow = int(
                        baza.execute("SELECT COUNT(*) FROM rozpoznania").fetchone()[0]
                    )
                    liczba_dzis = int(
                        baza.execute(
                            "SELECT COUNT(*) FROM rozpoznania WHERE data = ?",
                            (dzisiaj(),),
                        ).fetchone()[0]
                    )
                    ostatni = baza.execute(
                        """
                        SELECT id, osoba, czas, data, podobienstwo, kamera, zrodlo
                        FROM rozpoznania
                        ORDER BY czas DESC, id DESC
                        LIMIT 1
                        """
                    ).fetchone()
                finally:
                    baza.close()
        except (OSError, sqlite3.Error) as blad_bazy:
            blad = f"Błąd rejestru: {blad_bazy}"
            liczba_wpisow = 0
            liczba_dzis = 0
            ostatni = None

        try:
            plik = self.sciezka.relative_to(KORZEN).as_posix()
        except ValueError:
            plik = str(self.sciezka)

        return {
            "wlaczony": self._wlaczony,
            "plik": plik,
            "liczba_wpisow": liczba_wpisow,
            "liczba_dzis": liczba_dzis,
            "ostatni": dict(ostatni) if ostatni is not None else None,
            "retencja_dni": self.retencja_dni,
            "odstep_s": ODSTEP_ZAPISU_S,
            "blad": blad,
        }

    # ---------------------------------------------------------- porzadki
    def usun_dzien(self, data: str | None = None) -> int:
        """Usuwa wszystkie wpisy wybranego dnia. Zwraca liczbę usuniętych."""
        data = poprawna_data(data)
        if data is None:
            raise ValueError("Data musi mieć format RRRR-MM-DD.")
        with self._zamek:
            baza = self._polacz()
            try:
                wynik = baza.execute(
                    "DELETE FROM rozpoznania WHERE data = ?", (data,)
                )
                baza.commit()
                return int(wynik.rowcount)
            finally:
                baza.close()

    def wyczysc(self) -> int:
        """Usuwa cały rejestr. Zwraca liczbę usuniętych wpisów."""
        with self._zamek:
            baza = self._polacz()
            try:
                liczba = int(
                    baza.execute("SELECT COUNT(*) FROM rozpoznania").fetchone()[0]
                )
                baza.execute("DELETE FROM rozpoznania")
                baza.commit()
            finally:
                baza.close()
            self._ostatni_zapis.clear()
            return liczba

    def usun_starsze(self, dni: int | None = None) -> int:
        """Usuwa wpisy starsze niż `dni` (domyślnie RETENCJA_DNI)."""
        dni = self.retencja_dni if dni is None else int(dni)
        if dni < 1:
            raise ValueError("Retencja musi wynosić co najmniej 1 dzień.")
        granica = (date.today() - timedelta(days=dni)).isoformat()
        with self._zamek:
            baza = self._polacz()
            try:
                wynik = baza.execute(
                    "DELETE FROM rozpoznania WHERE data < ?", (granica,)
                )
                baza.commit()
                return int(wynik.rowcount)
            finally:
                baza.close()

    # ------------------------------------------------------------- eksport
    def eksport_csv(self, data: str | None = None) -> str:
        """CSV z frekwencją dnia (średnik + BOM, gotowe dla Excela)."""
        dzien = self.dzien(data)
        bufor = io.StringIO(newline="")
        zapis = csv.writer(bufor, delimiter=";", lineterminator="\r\n")
        zapis.writerow(["data", "osoba", "pierwsze", "ostatnie", "liczba", "status"])
        for wpis in dzien["obecni"]:
            zapis.writerow(
                [
                    dzien["data"],
                    wpis["osoba"],
                    wpis["pierwsze"],
                    wpis["ostatnie"],
                    wpis["liczba"],
                    "obecny",
                ]
            )
        for osoba in dzien["nieobecni"]:
            zapis.writerow([dzien["data"], osoba, "", "", 0, "nieobecny"])
        return "\ufeff" + bufor.getvalue()
