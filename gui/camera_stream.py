"""Strumień klatek z kamery dla panelu GUI (obraz MJPEG + statystyki).

Moduł jest niezależny od interfejsu: panel webowy tylko podłącza się do
obiektu `StrumienKamery` i pobiera z niego najnowszą klatkę JPEG.

Obsługiwane tryby pracy:
- `detekcja`     - same ramki wokół twarzy (szybkie, bez modeli AI),
- `rozpoznawanie`- ramki + imię i podobieństwo (korzysta z SilnikRozpoznawania),
- `zbieranie`    - zapisuje wycięte twarze 224x224 do data/raw/<osoba>/,
                    dokładnie tak jak skrypt 01_zbieranie_danych.py.

DLA UCZNIÓW - dlaczego rozpoznawanie działa w osobnym wątku?
Detekcja + policzenie embeddingu zajmuje kilkadziesiąt milisekund. Gdyby
robić to w tym samym wątku co odczyt kamery, podgląd "zamrażałby się" przy
każdej twarzy. Dlatego wątek kamery tylko PODAJE klatkę do analizy, a na
kolejnych klatkach rysuje ostatnie znane wyniki. Podgląd pozostaje płynny,
a etykiety mogą być opóźnione o ułamek sekundy.
"""

from __future__ import annotations

import threading
import time
from collections import deque
from concurrent.futures import ThreadPoolExecutor

import cv2

from common.camera import otworz_kamere, wykryj_dostepne_kamery
from common.dane_osob import (
    FOLDER_RAW,
    FOLDER_RAW_TEST,
    nastepny_numer_zdjecia,
    poprawna_nazwa,
)
from common.face_detector import DetektorTwarzy, WykrytaTwarz
from common.imgio import zapisz_obraz
from common.rejestr import ZRODLO_KAMERA, RejestrObecnosci
from common.rozpoznawanie import NAZWA_NIEZNANEJ
from gui.rozpoznawanie import SilnikRozpoznawania

SZEROKOSC_KAMERY = 1280
WYSOKOSC_KAMERY = 720
JAKOSC_JPEG = 80
MAKS_INDEKS_KAMERY = 5

STAN_ZATRZYMANY = "zatrzymany"
STAN_URUCHAMIANIE = "uruchamianie"
STAN_DZIALA = "dziala"
STAN_BLAD = "blad"

TRYB_DETEKCJA = "detekcja"
TRYB_ROZPOZNAWANIE = "rozpoznawanie"
TRYB_ZBIERANIE = "zbieranie"
TRYBY = (TRYB_DETEKCJA, TRYB_ROZPOZNAWANIE, TRYB_ZBIERANIE)

GRANICA_FPS = 1.0

# Zapis twarzy najwyżej co tyle sekund (jak w skrypcie 01) - inaczej
# powstałoby kilkanaście niemal identycznych zdjęć na sekundę.
ODSTEP_ZAPISU = 0.3
ROZMIAR_ZDJECIA = (224, 224)

# Kolor ramki, gdy twarz jest znana / nieznana (BGR).
KOLOR_ROZPOZNANY = (0, 200, 0)
KOLOR_NIEZNANY = (0, 0, 220)
KOLOR_DETEKCJA = (0, 200, 0)
KOLOR_ZAPIS = (255, 160, 0)

# Jak długo rysujemy ostatnie wyniki rozpoznawania (sekundy). Po tym czasie
# uznajemy je za nieaktualne i chowamy etykiety.
WAZNOSC_WYNIKOW = 1.5


class StrumienKamery:
    """Czyta klatki z kamery w wątku roboczym i udostępnia je jako JPEG.

    Wszystkie pola współdzielone między wątkami są chronione zamkiem, bo
    Flask obsługuje żądania w kilku wątkach naraz.
    """

    def __init__(
        self,
        silnik: SilnikRozpoznawania | None = None,
        rejestr: RejestrObecnosci | None = None,
    ) -> None:
        self._zamek = threading.Lock()
        self._stop = threading.Event()
        self._watek: threading.Thread | None = None
        self._silnik = silnik
        # Rejestr obecności jest opcjonalny: testy mogą go nie podawać.
        self._rejestr = rejestr
        self._blad_rejestru: str | None = None

        self._klatka: bytes | None = None
        self._licznik_klatek = 0
        self._czasy_klatek: deque[float] = deque(maxlen=120)
        self._stan = STAN_ZATRZYMANY
        self._blad: str | None = None
        self._twarze = 0
        self._prog = 0.6
        self._proporcja = 0.5
        self._indeks_kamery = 0

        self._tryb = TRYB_DETEKCJA
        self._emocje = False
        self._osoba = ""
        self._cel = 0
        self._zapisane = 0
        self._ostatni_zapis = 0.0
        # Gdzie zapisywać zdjęcia: zwykły zbiór treningowy albo test trudny.
        self._folder_zapisu = FOLDER_RAW

        self._wyniki: list[dict] = []
        self._czas_wynikow = 0.0
        self._analiza_w_toku = False
        self._executor: ThreadPoolExecutor | None = None
        # Detektor dla trybów detekcja/zbieranie. Tworzymy go przy starcie,
        # bo wczytanie YuNet z dysku przy KAŻDEJ klatce zabijałoby wydajność.
        self._detektor: DetektorTwarzy | None = None

        # Lista kamer jest skanowana w tle, bo otwieranie kolejnych
        # urządzeń potrafi potrwać kilka sekund.
        self._kamery: list[dict] = []
        self._kamery_gotowe = False
        self._kamery_blad: str | None = None
        self._skan_w_toku = False

    # ------------------------------------------------------------- kamery
    def odswiez_kamery_w_tle(self) -> None:
        """Uruchamia skanowanie kamer w osobnym wątku (jedno naraz)."""
        with self._zamek:
            if self._skan_w_toku:
                return
            self._skan_w_toku = True
            self._kamery_gotowe = False
            self._kamery_blad = None

        watek = threading.Thread(
            target=self._skanuj_kamery, name="skan-kamer", daemon=True
        )
        watek.start()

    def _skanuj_kamery(self) -> None:
        try:
            znalezione = wykryj_dostepne_kamery(MAKS_INDEKS_KAMERY)
            lista = [
                {
                    "indeks": kamera.indeks,
                    "szerokosc": kamera.maks_szerokosc,
                    "wysokosc": kamera.maks_wysokosc,
                    "cztery_k": kamera.czy_prawdopodobnie_4k,
                }
                for kamera in znalezione
            ]
        except Exception as blad:  # noqa: BLE001 - błąd sprzętu pokazujemy w GUI
            with self._zamek:
                self._kamery_gotowe = True
                self._kamery_blad = str(blad)
                self._skan_w_toku = False
            return

        with self._zamek:
            self._kamery = lista
            self._kamery_gotowe = True
            self._skan_w_toku = False

    def kamery(self) -> dict:
        with self._zamek:
            return {
                "gotowe": self._kamery_gotowe,
                "blad": self._kamery_blad,
                "lista": list(self._kamery),
            }

    # --------------------------------------------------------- sterowanie
    def start(
        self,
        indeks: int,
        prog: float,
        proporcja: float,
        tryb: str = TRYB_DETEKCJA,
        emocje: bool = False,
        osoba: str = "",
        cel: int = 0,
        test_trudny: bool = False,
    ) -> None:
        """Otwiera kamerę i zaczyna przesyłać klatki.

        Kamerę otwieramy synchronicznie, żeby użytkownik od razu dostał
        informację, gdy urządzenie jest zajęte albo niedostępne.
        """
        if not 0 <= indeks < 32:
            raise ValueError("Indeks kamery musi być liczbą od 0 do 31.")
        if not 0.05 <= prog <= 0.99:
            raise ValueError("Próg pewności musi być w przedziale 0,05-0,99.")
        if not 0.1 <= proporcja <= 1.0:
            raise ValueError("Minimalna proporcja musi być w przedziale 0,1-1,0.")
        if tryb not in TRYBY:
            raise ValueError(f"Nieznany tryb pracy: {tryb}.")

        if tryb == TRYB_ROZPOZNAWANIE:
            if self._silnik is None or not self._silnik.gotowy:
                raise RuntimeError(
                    "Model rozpoznawania jeszcze się wczytuje. Spróbuj za chwilę."
                )
            self._silnik.ustaw_emocje(bool(emocje))

        if tryb == TRYB_ZBIERANIE:
            if not osoba or not poprawna_nazwa(osoba):
                raise ValueError(
                    "Podaj nazwę osoby (litery, cyfry, myślnik lub podkreślenie)."
                )
            if cel < 0:
                raise ValueError("Cel musi być liczbą nieujemną (0 = bez limitu).")

        # Restart: zatrzymujemy poprzedni podgląd, jeśli jeszcze działa.
        self.stop()

        try:
            kamera = otworz_kamere(
                indeks, szerokosc=SZEROKOSC_KAMERY, wysokosc=WYSOKOSC_KAMERY
            )
        except Exception as blad:  # noqa: BLE001 - komunikat trafia do GUI
            with self._zamek:
                self._stan = STAN_BLAD
                self._blad = str(blad)
            raise RuntimeError(str(blad)) from blad

        with self._zamek:
            self._indeks_kamery = indeks
            self._prog = prog
            self._proporcja = proporcja
            self._tryb = tryb
            self._emocje = bool(emocje)
            self._osoba = osoba.strip()
            self._cel = int(cel)
            self._zapisane = 0
            self._ostatni_zapis = 0.0
            self._folder_zapisu = FOLDER_RAW_TEST if test_trudny else FOLDER_RAW
            self._stan = STAN_URUCHAMIANIE
            self._blad = None
            self._klatka = None
            self._twarze = 0
            self._licznik_klatek = 0
            self._czasy_klatek.clear()
            self._wyniki = []
            self._czas_wynikow = 0.0
            self._analiza_w_toku = False
            self._detektor = (
                DetektorTwarzy(prog_yunet=prog)
                if tryb in (TRYB_DETEKCJA, TRYB_ZBIERANIE)
                else None
            )
            self._executor = ThreadPoolExecutor(
                max_workers=1, thread_name_prefix="rozpoznawanie"
            )

        self._stop.clear()
        self._watek = threading.Thread(
            target=self._petla,
            args=(kamera,),
            name="kamera-gui",
            daemon=True,
        )
        self._watek.start()

    # --------------------------------------------------------------- petla
    def _petla(self, kamera) -> None:
        """Czyta klatki, rysuje wyniki i zapisuje gotowy JPEG."""
        try:
            while not self._stop.is_set():
                czy_odczytano, klatka = kamera.read()
                if not czy_odczytano or klatka is None:
                    raise RuntimeError("Nie udało się odczytać klatki z kamery.")

                # Odbicie lustrzane - użytkownik widzi siebie naturalnie.
                klatka = cv2.flip(klatka, 1)

                with self._zamek:
                    tryb = self._tryb

                if tryb == TRYB_ROZPOZNAWANIE:
                    self._obsluz_rozpoznawanie(klatka)
                elif tryb == TRYB_ZBIERANIE:
                    self._obsluz_zbieranie(klatka)
                else:
                    self._obsluz_detekcje(klatka)

                czy_ok, bufor = cv2.imencode(
                    ".jpg", klatka, [int(cv2.IMWRITE_JPEG_QUALITY), JAKOSC_JPEG]
                )
                if not czy_ok:
                    raise RuntimeError("Nie udało się zakodować klatki do JPEG.")

                teraz = time.monotonic()
                with self._zamek:
                    self._klatka = bufor.tobytes()
                    self._licznik_klatek += 1
                    self._czasy_klatek.append(teraz)
                    self._stan = STAN_DZIALA
        except Exception as blad:  # noqa: BLE001 - błąd pokazujemy w panelu
            with self._zamek:
                self._stan = STAN_BLAD
                self._blad = str(blad)
                self._klatka = None
                self._twarze = 0
        finally:
            try:
                kamera.release()
            finally:
                with self._zamek:
                    if self._stan != STAN_BLAD:
                        self._stan = STAN_ZATRZYMANY
                    self._klatka = None
                    self._wyniki = []

    # ------------------------------------------------------------- tryby
    def _obsluz_detekcje(self, klatka) -> None:
        """Klasyczna detekcja: ramka + procent pewności."""
        with self._zamek:
            detektor = self._detektor
            proporcja = self._proporcja

        if detektor is None:
            return
        twarze = self._filtruj_proporcje(detektor.wykryj(klatka), proporcja)

        for twarz in twarze:
            etykieta = f"twarz {twarz.pewnosc * 100:.0f}%"
            self._rysuj_twarz(klatka, twarz, etykieta, KOLOR_DETEKCJA)

        with self._zamek:
            self._twarze = len(twarze)

    def _obsluz_rozpoznawanie(self, klatka) -> None:
        """Podaje klatkę do analizy, rysuje ostatnie gotowe wyniki."""
        with self._zamek:
            zajety = self._analiza_w_toku
            executor = self._executor
            if not zajety and executor is not None:
                self._analiza_w_toku = True
                klatka_do_analizy = klatka.copy()

        if not zajety and executor is not None:
            try:
                executor.submit(self._analizuj_w_tle, klatka_do_analizy)
            except RuntimeError:
                # Executor mógł zostać zamknięty równolegle przez stop().
                with self._zamek:
                    self._analiza_w_toku = False

        with self._zamek:
            wyniki = self._wyniki
            czas_wynikow = self._czas_wynikow
            emocje_wlaczone = self._emocje

        if time.monotonic() - czas_wynikow > WAZNOSC_WYNIKOW:
            return

        for wynik in wyniki:
            osoba = wynik.get("osoba", NAZWA_NIEZNANEJ)
            podobienstwo = float(wynik.get("podobienstwo", 0.0))
            etykieta = f"{osoba} ({podobienstwo * 100:.0f}%)"
            if emocje_wlaczone and wynik.get("emocja"):
                etykieta += f" | {wynik['emocja']} {wynik['pewnosc_emocji'] * 100:.0f}%"

            kolor = (
                KOLOR_NIEZNANY
                if osoba == NAZWA_NIEZNANEJ
                else KOLOR_ROZPOZNANY
            )
            twarz = WykrytaTwarz(
                int(wynik["x"]),
                int(wynik["y"]),
                int(wynik["szerokosc"]),
                int(wynik["wysokosc"]),
            )
            self._rysuj_twarz(klatka, twarz, etykieta, kolor)

    def _analizuj_w_tle(self, klatka) -> None:
        """Wątek analizy: rozpoznaje twarze i zapisuje gotowe wyniki."""
        try:
            if self._silnik is None:
                raise RuntimeError("Brak silnika rozpoznawania.")
            wyniki = self._silnik.analizuj(klatka)
        except Exception as blad:  # noqa: BLE001 - pokazujemy błąd w statusie
            with self._zamek:
                if self._stan != STAN_ZATRZYMANY:
                    self._stan = STAN_BLAD
                    self._blad = f"Błąd rozpoznawania: {blad}"
                self._analiza_w_toku = False
            return

        with self._zamek:
            self._wyniki = wyniki
            self._czas_wynikow = time.monotonic()
            self._twarze = len(wyniki)
            self._analiza_w_toku = False
            tryb = self._tryb
            kamera = self._indeks_kamery
            rejestr = self._rejestr

        # Rejestr prowadzimy TYLKO w trybie rozpoznawania: w detekcji nie znamy
        # nazwisk, a przy zbieraniu zdjęć nie chcemy zaśmiecać frekwencji.
        if tryb == TRYB_ROZPOZNAWANIE and rejestr is not None:
            self._zapisz_do_rejestru(wyniki, kamera, rejestr)

    def _zapisz_do_rejestru(
        self, wyniki: list[dict], kamera: int, rejestr: RejestrObecnosci
    ) -> None:
        """Dopisuje rozpoznane osoby do rejestru obecności.

        Rejestr zapisuje najwyżej jedno rozpoznanie osoby na 5 minut, więc
        wołanie tej metody przy każdej klatce jest tanie. Błąd zapisu (np.
        zablokowany plik bazy) trafia do statusu i NIE przerywa rozpoznawania.
        """
        for wynik in wyniki:
            osoba = str(wynik.get("osoba") or "")
            if not osoba or osoba == NAZWA_NIEZNANEJ:
                continue
            try:
                rejestr.zapisz(
                    osoba,
                    wynik.get("podobienstwo"),
                    kamera=kamera,
                    zrodlo=ZRODLO_KAMERA,
                )
            except Exception as blad:  # noqa: BLE001 - rejestr nie może zabić kamery
                with self._zamek:
                    self._blad_rejestru = f"Nie udało się zapisać do rejestru: {blad}"
                return

        with self._zamek:
            self._blad_rejestru = None

    def _obsluz_zbieranie(self, klatka) -> None:
        """Wykrywa twarze i cyklicznie zapisuje największą z nich."""
        with self._zamek:
            osoba = self._osoba
            cel = self._cel
            zapisane = self._zapisane
            ostatni_zapis = self._ostatni_zapis
            detektor = self._detektor
            folder_bazowy = self._folder_zapisu

        if not osoba or detektor is None:
            return

        twarze = detektor.wykryj(klatka)

        if not twarze:
            with self._zamek:
                self._twarze = 0
            return

        # Największa twarz = najbliżej kamery = osoba pozująca do zdjęć.
        twarz = max(twarze, key=lambda t: t.w * t.h)

        teraz = time.monotonic()
        if teraz - ostatni_zapis >= ODSTEP_ZAPISU:
            # WAŻNE: wycinamy twarz z CZYSTEJ klatki, zanim narysujemy
            # ramki. Inaczej do zbioru treningowego trafiłyby zdjęcia
            # z zielonymi/niebieskimi prostokątami i podpisem "zapis: ...",
            # a sieć uczyłaby się tych narysowanych linii.
            wycinek = twarz.wytnij(klatka)
            if wycinek.size > 0:
                wycinek = cv2.resize(wycinek, ROZMIAR_ZDJECIA)

                folder = folder_bazowy / osoba
                folder.mkdir(parents=True, exist_ok=True)
                numer = nastepny_numer_zdjecia(folder, osoba)
                sciezka = folder / f"{osoba}_{numer:04d}.jpg"
                if zapisz_obraz(sciezka, wycinek):
                    with self._zamek:
                        self._zapisane += 1
                        self._ostatni_zapis = teraz
                        zapisane = self._zapisane
                    if cel and zapisane >= cel:
                        # Osiągnięto cel - kończymy zbieranie (stop() woła GUI,
                        # więc tutaj tylko zatrzymujemy pętlę).
                        self._stop.set()

        # Ramki rysujemy dopiero po zapisaniu zdjęcia - są tylko dla oka.
        for inna in twarze:
            self._rysuj_twarz(klatka, inna, "", KOLOR_DETEKCJA, grubosc=1)
        self._rysuj_twarz(
            klatka,
            twarz,
            f"zapis: {zapisane}/{cel}" if cel else f"zapisano: {zapisane}",
            KOLOR_ZAPIS,
        )

        with self._zamek:
            self._twarze = len(twarze)

    # ------------------------------------------------------------ pomocnicze
    @staticmethod
    def _filtruj_proporcje(
        twarze: list[WykrytaTwarz], proporcja: float
    ) -> list[WykrytaTwarz]:
        """Odrzuca twarze o nierealnych proporcjach (jak w detektorze GUI)."""
        wynik = []
        for twarz in twarze:
            if twarz.h <= 0:
                continue
            stosunek = twarz.w / twarz.h
            if proporcja <= stosunek <= 1.0 / proporcja:
                wynik.append(twarz)
        return wynik

    @staticmethod
    def _rysuj_twarz(
        klatka,
        twarz: WykrytaTwarz,
        etykieta: str,
        kolor: tuple[int, int, int],
        grubosc: int = 2,
    ) -> None:
        x, y, szerokosc, wysokosc = twarz.x, twarz.y, twarz.w, twarz.h
        cv2.rectangle(
            klatka, (x, y), (x + szerokosc, y + wysokosc), kolor, grubosc
        )
        if not etykieta:
            return
        cv2.putText(
            klatka,
            etykieta,
            (x, max(18, y - 8)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            kolor,
            2,
            cv2.LINE_AA,
        )

    def ustaw_emocje(self, wlaczone: bool) -> None:
        """Włącza/wyłącza etykiety emocji na podglądzie (model w silniku)."""
        with self._zamek:
            self._emocje = bool(wlaczone)

    def ustaw_rejestr(self, rejestr: RejestrObecnosci | None) -> None:
        """Podłącza albo odpina rejestr obecności używany przy rozpoznawaniu."""
        with self._zamek:
            self._rejestr = rejestr
            self._blad_rejestru = None

    def stop(self) -> None:
        """Zatrzymuje wątek kamery i zwalnia urządzenie."""
        self._stop.set()
        watek = self._watek
        if watek is not None and watek.is_alive():
            watek.join(timeout=5.0)

        with self._zamek:
            executor = self._executor
            self._executor = None
            self._analiza_w_toku = False

        if executor is not None:
            # wait=False: nie blokujemy Stop, gdy trwa pojedyncza analiza.
            executor.shutdown(wait=False, cancel_futures=True)

        with self._zamek:
            self._watek = None
            if self._stan != STAN_BLAD:
                self._stan = STAN_ZATRZYMANY
            self._klatka = None
            self._twarze = 0
            self._wyniki = []
            self._detektor = None
            self._czasy_klatek.clear()

    # ---------------------------------------------------------- odczyt
    def klatka(self) -> bytes | None:
        with self._zamek:
            return self._klatka

    def status(self) -> dict:
        with self._zamek:
            teraz = time.monotonic()
            if self._stan in (STAN_ZATRZYMANY, STAN_BLAD):
                fps = 0.0
            else:
                ostatnie = [t for t in self._czasy_klatek if teraz - t <= GRANICA_FPS]
                fps = len(ostatnie) / GRANICA_FPS if ostatnie else 0.0

            return {
                "stan": self._stan,
                "blad": self._blad,
                "dziala": self._stan in (STAN_URUCHAMIANIE, STAN_DZIALA),
                "fps": round(fps, 1),
                "twarze": self._twarze,
                "kamera": self._indeks_kamery,
                "prog": self._prog,
                "proporcja": self._proporcja,
                "klatki": self._licznik_klatek,
                "tryb": self._tryb,
                "emocje": self._emocje,
                "osoba": self._osoba,
                "cel": self._cel,
                "zapisane": self._zapisane,
                "test_trudny": self._folder_zapisu == FOLDER_RAW_TEST,
                "rejestr": self._rejestr is not None,
                "rejestr_wlaczony": bool(
                    self._rejestr is not None and self._rejestr.wlaczony
                ),
                "rejestr_blad": self._blad_rejestru,
                "wyniki": list(self._wyniki),
                "wyniki_aktualne": bool(self._wyniki)
                and (teraz - self._czas_wynikow <= WAZNOSC_WYNIKOW),
            }

    def strumien(self):
        """Generator klatek w formacie MJPEG (multipart/x-mixed-replace)."""
        ostatni_licznik = -1
        while True:
            if self._stop.is_set():
                return

            with self._zamek:
                klatka = self._klatka
                licznik = self._licznik_klatek
                stan = self._stan

            if klatka is None:
                if stan in (STAN_BLAD, STAN_ZATRZYMANY):
                    return
                time.sleep(0.05)
                continue

            # Czekamy na nową klatkę - nie wysyłamy w kółko tej samej.
            if licznik == ostatni_licznik:
                time.sleep(0.01)
                continue
            ostatni_licznik = licznik

            naglowek = (
                b"--frame\r\n"
                b"Content-Type: image/jpeg\r\n"
                b"Content-Length: " + str(len(klatka)).encode("ascii") + b"\r\n\r\n"
            )
            yield naglowek + klatka + b"\r\n"
