"""Panel GUI - okno desktopowe z interfejsem webowym (styl AdminLTE/Bootstrap).

Jak to działa:
1. W tle startuje mały serwer Flask na 127.0.0.1 (tylko lokalnie).
2. Serwer udostępnia stronę HTML (Bootstrap 5), strumień MJPEG z kamery
   oraz proste API sterujące detekcją.
3. pywebview otwiera natywne okno systemowe z wbudowanym WebView2, więc
   aplikacja wygląda jak desktopowa, ale interfejs jest zwykłym HTML/CSS.

Użycie:
    python gui/main.py

Awaryjnie (gdy WebView2 nie działa):
    python gui/tkinter_app.py
"""

from __future__ import annotations

import socket
import sys
import threading
import time
import urllib.error
import urllib.request
from pathlib import Path

# Uruchamiamy jako "python gui/main.py" - Python dodaje wtedy do sys.path
# folder gui/, a nie katalog projektu. Dodajemy katalog główny ręcznie.
KORZEN_PROJEKTU = Path(__file__).resolve().parent.parent
if str(KORZEN_PROJEKTU) not in sys.path:
    sys.path.insert(0, str(KORZEN_PROJEKTU))

from common.uruchom_w_venv import przelacz_na_venv_jesli_trzeba

# Musi być przed importem cv2/Flask - patrz common/uruchom_w_venv.py.
przelacz_na_venv_jesli_trzeba()

from werkzeug.serving import make_server

try:
    import webview
except ImportError as blad:  # pragma: no cover - zależne od środowiska
    raise SystemExit(
        "Brak biblioteki pywebview. Zainstaluj zależności:\n"
        "    pip install -r requirements.txt\n"
        "Albo uruchom awaryjne GUI w Tkinterze:\n"
        "    python gui/tkinter_app.py"
    ) from blad

from common.rejestr import RejestrObecnosci
from gui.camera_stream import StrumienKamery
from gui.rozpoznawanie import SilnikRozpoznawania
from gui.server import utworz_aplikacje

SZEROKOSC_OKNA = 1280
WYSOKOSC_OKNA = 820
MINIMALNY_ROZMIAR = (1000, 640)


def _wolny_port() -> int:
    """Zwraca wolny port TCP na loopbacku (system przydziela go sam)."""
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as gniazdo:
        gniazdo.bind(("127.0.0.1", 0))
        return int(gniazdo.getsockname()[1])


def _czekaj_na_serwer(adres: str, limit: float = 10.0) -> None:
    """Czeka, aż serwer zacznie odpowiadać, żeby okno nie pokazało błędu."""
    koniec = time.monotonic() + limit
    while time.monotonic() < koniec:
        try:
            with urllib.request.urlopen(adres, timeout=0.5) as odpowiedz:
                if odpowiedz.status == 200:
                    return
        except (urllib.error.URLError, OSError):
            time.sleep(0.1)
    raise RuntimeError("Serwer panelu nie wystartował w wyznaczonym czasie.")


def main() -> None:
    silnik = SilnikRozpoznawania()
    strumien = StrumienKamery(silnik)
    # Rejestr obecności to lokalna baza SQLite (data/rejestr/rejestr.db).
    # Tworzymy go PRZED serwerem, żeby endpointy i kamera używały tej samej
    # instancji (jeden zamek i jeden cooldown na osobę).
    rejestr = RejestrObecnosci()
    # Skanowanie kamer trwa kilka sekund - robimy je od razu w tle,
    # żeby panel po otwarciu miał już gotową listę.
    strumien.odswiez_kamery_w_tle()

    aplikacja = utworz_aplikacje(strumien, silnik, rejestr)
    menedzer = getattr(aplikacja, "menedzer_treningu", None)
    menedzer_ewaluacji = getattr(aplikacja, "menedzer_ewaluacji", None)
    menedzer_eksperymentow = getattr(aplikacja, "menedzer_eksperymentow", None)
    menedzer_testu_trudnego = getattr(aplikacja, "menedzer_testu_trudnego", None)
    port = _wolny_port()
    serwer = make_server("127.0.0.1", port, aplikacja, threaded=True)
    watek_serwera = threading.Thread(
        target=serwer.serve_forever, name="serwer-gui", daemon=True
    )
    watek_serwera.start()

    adres = f"http://127.0.0.1:{port}/"
    try:
        _czekaj_na_serwer(adres)
    except RuntimeError as blad:
        serwer.shutdown()
        raise SystemExit(str(blad)) from blad

    okno = webview.create_window(
        "Rozpoznawanie twarzy - panel",
        adres,
        width=SZEROKOSC_OKNA,
        height=WYSOKOSC_OKNA,
        min_size=MINIMALNY_ROZMIAR,
        background_color="#f4f6f9",
    )

    posprzatane = False

    def _posprzataj() -> None:
        nonlocal posprzatane
        if posprzatane:
            return
        posprzatane = True
        strumien.stop()
        if menedzer is not None:
            menedzer.zatrzymaj()
        if menedzer_ewaluacji is not None:
            menedzer_ewaluacji.zatrzymaj()
        if menedzer_eksperymentow is not None:
            menedzer_eksperymentow.zatrzymaj()
        if menedzer_testu_trudnego is not None:
            menedzer_testu_trudnego.zatrzymaj()
        serwer.shutdown()

    okno.events.closed += _posprzataj

    try:
        webview.start()
    finally:
        _posprzataj()


if __name__ == "__main__":
    main()
