"""Strumień klatek z kamery dla panelu GUI (obraz MJPEG + statystyki).

Moduł jest niezależny od interfejsu: panel webowy (gui/main.py) tylko
podłącza się do obiektu `StrumienKamery` i pobiera z niego najnowszą klatkę
JPEG. Dzięki temu ta sama logika może obsłużyć każde GUI.
"""

from __future__ import annotations

import threading
import time
from collections import deque

import cv2

from common.camera import otworz_kamere, wykryj_dostepne_kamery
from src.detection.detector import detect_face

SZEROKOSC_KAMERY = 1280
WYSOKOSC_KAMERY = 720
JAKOSC_JPEG = 80
MAKS_INDEKS_KAMERY = 5

STAN_ZATRZYMANY = "zatrzymany"
STAN_URUCHAMIANIE = "uruchamianie"
STAN_DZIALA = "dziala"
STAN_BLAD = "blad"

GRANICA_FPS = 1.0


class StrumienKamery:
    """Czyta klatki z kamery w wątku roboczym i udostępnia je jako JPEG.

    Wszystkie pola współdzielone między wątkami są chronione zamkiem, bo
    Flask obsługuje żądania w kilku wątkach naraz.
    """

    def __init__(self) -> None:
        self._zamek = threading.Lock()
        self._stop = threading.Event()
        self._watek: threading.Thread | None = None

        self._klatka: bytes | None = None
        self._licznik_klatek = 0
        self._czasy_klatek: deque[float] = deque(maxlen=120)
        self._stan = STAN_ZATRZYMANY
        self._blad: str | None = None
        self._twarze = 0
        self._prog = 0.6
        self._proporcja = 0.5
        self._indeks_kamery = 0

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
    def start(self, indeks: int, prog: float, proporcja: float) -> None:
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
            self._stan = STAN_URUCHAMIANIE
            self._blad = None
            self._klatka = None
            self._twarze = 0
            self._licznik_klatek = 0
            self._czasy_klatek.clear()

        self._stop.clear()
        self._watek = threading.Thread(
            target=self._petla,
            args=(kamera,),
            name="kamera-gui",
            daemon=True,
        )
        self._watek.start()

    def _petla(self, kamera) -> None:
        """Czyta klatki, rysuje ramki twarzy i zapisuje gotowy JPEG."""
        try:
            while not self._stop.is_set():
                czy_odczytano, klatka = kamera.read()
                if not czy_odczytano or klatka is None:
                    raise RuntimeError("Nie udało się odczytać klatki z kamery.")

                # Odbicie lustrzane - użytkownik widzi siebie naturalnie.
                klatka = cv2.flip(klatka, 1)

                twarze = detect_face(
                    klatka,
                    threshold=self._prog,
                    aspect_ratio=self._proporcja,
                )
                for twarz in twarze:
                    x, y, szerokosc, wysokosc = twarz["bbox"]
                    cv2.rectangle(
                        klatka,
                        (x, y),
                        (x + szerokosc, y + wysokosc),
                        (0, 200, 0),
                        2,
                    )
                    etykieta = f"twarz {twarz['confidence'] * 100:.0f}%"
                    cv2.putText(
                        klatka,
                        etykieta,
                        (x, max(18, y - 8)),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.6,
                        (0, 200, 0),
                        2,
                        cv2.LINE_AA,
                    )

                czy_ok, bufor = cv2.imencode(
                    ".jpg", klatka, [int(cv2.IMWRITE_JPEG_QUALITY), JAKOSC_JPEG]
                )
                if not czy_ok:
                    raise RuntimeError("Nie udało się zakodować klatki do JPEG.")

                teraz = time.monotonic()
                with self._zamek:
                    self._klatka = bufor.tobytes()
                    self._twarze = len(twarze)
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

    def stop(self) -> None:
        """Zatrzymuje wątek kamery i zwalnia urządzenie."""
        self._stop.set()
        watek = self._watek
        if watek is not None and watek.is_alive():
            watek.join(timeout=3.0)

        with self._zamek:
            self._watek = None
            if self._stan != STAN_BLAD:
                self._stan = STAN_ZATRZYMANY
            self._klatka = None
            self._twarze = 0
            self._czasy_klatek.clear()

    # ---------------------------------------------------------- odczyt
    def klatka(self) -> bytes | None:
        with self._zamek:
            return self._klatka

    def status(self) -> dict:
        with self._zamek:
            if self._stan in (STAN_ZATRZYMANY, STAN_BLAD):
                fps = 0.0
            else:
                teraz = time.monotonic()
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
