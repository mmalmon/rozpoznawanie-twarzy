# GUI - panel desktopowy w stylu AdminLTE

Interfejs projektu to **natywne okno systemowe** z wnętrzem zrobionym w
HTML/CSS/JS (Bootstrap 5, wygląd inspirowany AdminLTE). Nie jest to aplikacja
przeglądarkowa: okno otwiera `pywebview`, a cały serwer działa lokalnie i
wyłącznie na `127.0.0.1`.

## Wymagania

Zależności instalujesz z głównego pliku projektu:

```bash
pip install -r requirements.txt
```

Panel używa `pywebview`, który na Windows korzysta z **WebView2** (jest
dołączony do Windows 10/11). Jeśli okno się nie otwiera, doinstaluj
[WebView2 Runtime](https://developer.microsoft.com/microsoft-edge/webview2/)
albo użyj awaryjnego GUI w Tkinterze (patrz niżej). `tkinter` jest częścią
standardowej biblioteki Pythona — nie instaluj go przez pip.

## Użycie

```bash
python gui/main.py
```

Awaryjnie (starszy panel Tkinter, bez WebView2):

```bash
python gui/tkinter_app.py
```

## Funkcje

- podgląd obrazu z wybranej kamery (strumień MJPEG),
- ramki wokół wykrytych twarzy z procentową pewnością,
- wybór kamery, progu pewności i minimalnej proporcji twarzy,
- statystyki na żywo: status, FPS, liczba twarzy, numer kamery,
- lista wykrytych kamer i przycisk odświeżenia,
- start/stop podglądu z poprawnym zwalnianiem kamery.

Panel pokazuje **detekcję** (gdzie jest twarz), a nie rozpoznawanie osób.
Pełne rozpoznawanie z wytrenowanymi wzorcami uruchamia
`04_rozpoznawanie_na_zywo.py`.

## Jak to działa

```
gui/main.py          uruchamia serwer Flask (wątek) i okno pywebview
gui/server.py        endpointy: /, /video, /api/kamery, /api/status,
                     /api/start, /api/stop
gui/camera_stream.py wątek kamery: odczyt klatki -> detekcja -> JPEG
gui/templates/       index.html (widok panelu)
gui/static/          Bootstrap 5 (lokalnie) + app.css + app.js
```

- Klatki nie przechodzą przez mostek JS↔Python — serwer wystawia je jako
  standardowy strumień `multipart/x-mixed-replace`, a `<img>` w HTML
  po prostu go wyświetla.
- Parametry detekcji są przekazywane przy starcie; zmiana progu w trakcie
  działania wymaga zatrzymania i ponownego uruchomienia podglądu.
- Serwer nasłuchuje tylko na `127.0.0.1` i nie wymaga logowania, bo steruje
  lokalną kamerą.

## Pliki Bootstrapa

`gui/static/vendor/` zawiera `bootstrap.min.css` i `bootstrap.bundle.min.js`
(5.3.3, licencja MIT). Trzymamy je lokalnie, żeby panel działał bez
internetu (np. w szkolnej pracowni). Szczegóły: `static/vendor/README.md`.
