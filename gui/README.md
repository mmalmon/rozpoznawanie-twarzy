# GUI — panel desktopowy w stylu AdminLTE

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

Najprościej: **dwuklik na `Uruchom panel.bat`** w głównym folderze projektu.
Plik sam wybiera Pythona z `.venv` (albo `py`/`python` z PATH), uruchamia
`gui/main.py` i przy błędzie zatrzymuje okno z komunikatem.

To samo z terminala:

```bash
python gui/main.py
```

Awaryjnie (starszy panel Tkinter, tylko detekcja):

```bash
python gui/tkinter_app.py
```

## Funkcje

Panel prowadzi cały proces — nie trzeba już przełączać się do konsoli:

| Zakładka | Co robi |
| --- | --- |
| **Detekcja na żywo** | Ramki wokół twarzy, procent pewności, FPS, liczba twarzy. Wybór kamery, progu i minimalnej proporcji twarzy. |
| **Rozpoznawanie** | Wczytuje `models/wzorce_osob.json` w tle i pokazuje imię + podobieństwo. Karta **Wynik rozpoznawania** pokazuje TOP-3 kandydatów: paski podobieństwa i znacznik progu. Suwak **Korekta progu** (jak `--korekta-progu` w `04`). Przełącznik **Wykrywanie emocji** (domyślnie OFF). |
| **Dodaj twarz** | Zapisuje wycięte twarze 224×224 do `data/raw/<osoba>/` co ok. 0,3 s, z licznikiem i celem. Działa jak automatyczny tryb `01_zbieranie_danych.py`. |
| **Trening** | Uruchamia `02_podzial_danych.py`, a potem `03_trenowanie_modelu.py --postep` w osobnym procesie. Pasek postępu, etap, log i przycisk **Przerwij**. |
| **Ewaluacja** | Uruchamia `07_ewaluacja.py --postep` w osobnym procesie. Pokazuje accuracy, FAR/FRR, EER, AUC, metryki na osobę, porównanie metod, wykresy PNG i galerię pomyłek. |
| **Baza twarzy** | Miniatury, liczba zdjęć, usuwanie osoby (opcjonalnie z ponownym treningiem). |
| **O projekcie** | Krótka instrukcja i uwagi o prywatności / AI Act. |

Dodatkowo:

- lista wykrytych kamer i przycisk odświeżenia (skanowanie w tle),
- start/stop podglądu z poprawnym zwalnianiem kamery,
- statusy i komunikaty błędów w postaci toastów,
- po udanym treningu silnik **przeładowuje wzorce bez restartu** panelu,
- TOP-3 kandydatów w rozpoznawaniu (bez zmiany logiki decyzyjnej `04`).

## Jak to działa

```
gui/main.py            uruchamia serwer Flask (wątek) i okno pywebview
gui/server.py          endpointy API + strumień MJPEG (tylko 127.0.0.1)
gui/camera_stream.py   wątek kamery: detekcja / rozpoznawanie / zbieranie zdjęć
gui/rozpoznawanie.py   silnik: modele, wzorce, emocje, korekta progu
gui/trening.py         subprocess 02 + 03, parser [postep], log i przerwanie
gui/ewaluacja.py       subprocess 07, parser [postep], log i przerwanie
gui/templates/         index.html (jeden widok, przełączane zakładki)
gui/static/            Bootstrap 5 (lokalnie) + app.css + app.js
common/rozpoznawanie.py wspólna logika dopasowania używana też przez 04
```

- Klatki nie przechodzą przez mostek JS↔Python — serwer wystawia je jako
  standardowy strumień `multipart/x-mixed-replace`, a `<img>` w HTML
  po prostu go wyświetla.
- W trybie rozpoznawania analiza działa w **osobnym wątku** (`ThreadPoolExecutor`
  z jednym robotnikiem): podgląd pozostaje płynny, a etykiety mogą być
  opóźnione o ułamek sekundy.
- Zbieranie zdjęć wycina twarz z **czystej klatki** — ramki rysowane dla oka
  nie trafiają do zbioru treningowego.
- Parametry detekcji są przekazywane przy starcie; zmiana progu w trakcie
  działania wymaga zatrzymania i ponownego uruchomienia podglądu.
- Serwer nasłuchuje tylko na `127.0.0.1` i nie wymaga logowania, bo steruje
  lokalną kamerą. Endpointy API zwracają `Cache-Control: no-store`.

## Endpointy API

```
GET  /                          widok panelu
GET  /video                     strumień MJPEG
GET  /api/kamery                lista kamer (skanowanie w tle)
POST /api/kamery/odswiez        ponowne skanowanie
GET  /api/status                stan kamery i trybu
POST /api/start                 start podglądu (tryb, emocje, osoba, cel)
POST /api/stop                  stop podglądu

GET  /api/model                 stan silnika rozpoznawania
POST /api/model/przygotuj       wczytanie modeli w tle
POST /api/model/emocje          włącz/wyłącz emocje
POST /api/model/korekta         korekta progu podobieństwa

POST /api/zbieranie/start       start zbierania do data/raw
POST /api/zbieranie/stop        stop zbierania

GET  /api/trening/status        stan treningu (etap, %, log)
POST /api/trening/start         uruchom 02 + 03
POST /api/trening/stop          przerwij trening

GET  /api/ewaluacja/status      stan ewaluacji (etap, %, log)
POST /api/ewaluacja/start       uruchom 07_ewaluacja.py
POST /api/ewaluacja/stop        przerwij ewaluację
GET  /api/ewaluacja/raport      ostatni raport JSON
GET  /api/ewaluacja/wykres/<n>  wykres PNG z models/ewaluacja/wykresy
GET  /api/ewaluacja/pomylka/<n> zdjęcie z galerii pomyłek

GET  /api/baza                  osoby, liczba zdjęć, informacje o modelu
GET  /api/baza/<osoba>/zdjecie  przykładowe zdjęcie osoby
POST /api/baza/usun             usuń osobę (wymaga potwierdzenia nazwą)
```

## Trening i pasek postępu

`gui/trening.py` uruchamia dokładnie te same skrypty co konsola, ale czyta
ich wyjście na bieżąco. `03_trenowanie_modelu.py --postep` wypisuje linie:

```
[postep-start] embeddingi 1234
[postep] embeddingi 32
[postep] embeddingi 64
...
```

Panel mapuje etap `02` na 0–10%, `03` na 10–95%, a zapis wyniku na 95–100%.
Linie maszynowe nie trafiają do logu widocznego dla ucznia. Przerwanie
przyciskiem **Przerwij** zabija proces potomny i nie nadpisuje modelu.

## Ewaluacja

`gui/ewaluacja.py` uruchamia `07_ewaluacja.py --postep` w osobnym procesie
i czyta te same maszynowe linie postępu:

```
[postep-start] embeddingi 2785
[postep] embeddingi 32
...
[postep-start] metryki 4
```

Faza `embeddingi` odpowiada za 0–82% paska, a `metryki` (prototypy, k-NN,
regresja logistyczna, wykresy i zapis raportu) za 82–97%. Po sukcesie pasek
pokazuje 100%, a panel odświeża raport.

Zakładka **Ewaluacja** pokazuje:

- badge stanu i etap, procent, czas oraz log (bez linii maszynowych),
- podsumowanie: accuracy, FAR, FRR, EER, AUC,
- tabelę precision / recall / F1 na osobę,
- tabelę porównania metod (prototypy, k-NN, regresja logistyczna),
- wykresy PNG i galerię pomyłek z podpisami „prawdziwa → przewidziana”,
- ostrzeżenia metodologiczne z raportu (wyniki na `val` są optymistyczne,
  bo ten zbiór posłużył do kalibracji progów).

Raport i obrazy są serwowane wyłącznie z `models/ewaluacja/`; endpointy
sprawdzają rozszerzenie pliku i odrzucają próby wyjścia poza folder
(`../`). Przyciski **Uruchom ewaluację** / **Przerwij** działają tak samo jak
w treningu — przerwanie zabija proces potomny i nie zostawia połowy raportu
(zapis odbywa się atomowo przez plik tymczasowy).

## Pliki Bootstrapa

`gui/static/vendor/` zawiera `bootstrap.min.css` i `bootstrap.bundle.min.js`
(5.3.3, licencja MIT). Trzymamy je lokalnie, żeby panel działał bez
internetu (np. w szkolnej pracowni). Szczegóły: `static/vendor/README.md`.
