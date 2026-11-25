# Rozpoznawanie twarzy — projekt edukacyjny (uczenie maszynowe)

Pokazowy, w pełni działający system rozpoznawania twarzy przygotowany na
zajęcia w technikum. W odróżnieniu od gotowych bibliotek typu
`face_recognition` (które tylko *wywołują* gotowy model), tutaj uczniowie
**sami trenują własną sieć neuronową** na zdjęciach zebranych z kamery —
dzięki temu widać cały proces uczenia maszynowego: zbieranie danych,
przygotowanie zbioru, trening, ewaluację i wdrożenie modelu do działania
w czasie rzeczywistym.

## Spis treści

1. [Jak to działa (teoria dla uczniów)](#jak-to-działa-teoria-dla-uczniów)
2. [Wymagania sprzętowe i sprzęt użyty w projekcie](#wymagania-sprzętowe-i-sprzęt-użyty-w-projekcie)
3. [Struktura projektu](#struktura-projektu)
4. [Instalacja środowiska](#instalacja-środowiska)
5. [Tutorial krok po kroku](#tutorial-krok-po-kroku)
6. [Jak przygotować dobre materiały treningowe](#jak-przygotować-dobre-materiały-treningowe)
7. [Rozwiązywanie problemów (FAQ)](#rozwiązywanie-problemów-faq)
8. [Pomysły na rozszerzenie projektu (zadania dla uczniów)](#pomysły-na-rozszerzenie-projektu-zadania-dla-uczniów)
9. [Ekran z kawałami sterowany gestem (05_ekran_z_kawalami.py)](#ekran-z-kawałami-sterowany-gestem-05_ekran_z_kawalami-py)
10. [Usuwanie danych osób po zakończeniu](#usuwanie-danych-osób-po-zakończeniu)
11. [Krok 5 — ewaluacja modelu](#krok-5--ewaluacja-modelu)
12. [Krok 6 — eksperymenty](#krok-6--eksperymenty)
13. [Krok 7 — test trudny](#krok-7--test-trudny)
14. [Przedziały ufności — jak czytać niepewność wyniku](#przedziały-ufności--jak-czytać-niepewność-wyniku)
15. [Krok 8 — rejestr rozpoznań i frekwencja](#krok-8--rejestr-rozpoznań-i-frekwencja)

## Jak to działa (teoria dla uczniów)

System składa się z dwóch niezależnych etapów, które warto odróżniać:

1. **Detekcja twarzy** — znalezienie *gdzie* na obrazie jest twarz (bez
   rozpoznawania *kogo*). Główny detektor to mała sieć YuNet
   (`common/modele/face_detection_yunet_2023mar.onnx`, ~0,1 s na klatce 4K
   zamiast kilku sekund); gdy pliku brak, używana jest wolniejsza klasyczna
   kaskada Haara z OpenCV (algorytm Violi-Jonesa — cechy Haara + boosting +
   okno przesuwne po obrazie w wielu skalach). Kaskada jest szybka, nie wymaga
   GPU i świetnie nadaje się do tłumaczenia na lekcjach.
2. **Rozpoznawanie (klasyfikacja) twarzy** — mając już wycięty fragment
   obrazu z twarzą, sieć neuronowa decyduje, **do kogo** ona należy.
   To właśnie ten etap trenujemy sami.

Emocje (neutralna, radość, zdziwienie, smutek, złość, wstręt, strach, pogarda)
rozpoznaje gotowy model FER+ (`common/modele/emotion-ferplus-8.onnx`,
`common/emotions.py`); wynik jest wygładzany w czasie i pokazywany obok nazwy
osoby. **Domyślnie wyłączone** - włącza je dopiero flaga `--emocje`
w `04_rozpoznawanie_na_zywo.py`. Używaj wyłącznie do nauki poza szkołą lub na
własnej twarzy: w szkole AI Act (art. 5 ust. 1 lit. f) zakazuje rozpoznawania
emocji.

Do dopasowania używamy metody **transfer learning**: zamiast trenować sieć
od zera (co wymagałoby milionów zdjęć), bierzemy sieć **MobileNetV3-Small**
wytrenowaną wcześniej na dużym zbiorze ImageNet (1000 klas ogólnych
obiektów: koty, samochody, itd.). Taka sieć "umie już" wyciągać uniwersalne
cechy wizualne (krawędzie, kształty, tekstury).

Zamiast (jak w pierwszej wersji tego projektu) douczać nową warstwę
klasyfikującą "kto z tych N znanych osób jest na zdjęciu", używamy
podejścia **rozpoznawania przez podobieństwo** (verification), które pozwala
dodawać osoby bez dotrenowywania sieci:

- Sieć (bez żadnego dotrenowywania — wagi zostają dokładnie takie, jak po
  treningu na ImageNet) zamienia każde zdjęcie twarzy na **embedding** —
  wektor kilkuset liczb opisujący jej wygląd.
- Dla każdej znanej osoby zachowujemy embeddingi jej zdjęć treningowych;
  system dopasowuje twarz do najbardziej podobnego zdjęcia, zamiast jednego
  uśrednionego wzorca.
- Podczas rozpoznawania na żywo liczymy embedding nowej twarzy i sprawdzamy
  jego **podobieństwo kosinusowe** do wzorca każdej znanej osoby.
- Przed ekstrakcją cech obrazy są kadrowane do środkowych 82% wycinka
  detektora, aby ograniczyć wpływ tła; ten sam krok działa przy budowaniu
  wzorców i rozpoznawaniu na żywo.

Dzięki temu wystarczy 150–300 zdjęć na osobę, a "trening" (w praktyce:
liczenie embeddingów i kalibracja progu) zajmuje sekundy zamiast minut,
nawet na samym CPU.

> **Dlaczego nie zwykła klasyfikacja?** Pierwsza wersja tego projektu
> uczyła sieć rozróżniać "znaną osobę" od zdjęć klasy "nieznajomy" pobranych
> z internetu. W praktyce sieć nauczyła się rozróżniać *styl zdjęcia*
> (ostre zdjęcie z kamery vs. skompresowane zdjęcie stockowe), a nie same
> rysy twarzy — to tzw. **uczenie się skrótów** (*shortcut learning*), bardzo
> częsty błąd w uczeniu maszynowym. Zamrożony ekstraktor i kadrowanie
> ograniczają ten problem, ale go nie eliminują: MobileNet jest modelem
> ogólnego przeznaczenia, a nie specjalistycznym modelem do twarzy. Dlatego
> ważne są przykłady z różnych teł, oświetlenia i kątów.

Bardzo ważny jest też **próg podobieństwa** przy rozpoznawaniu: dla każdej
osoby jest on automatycznie **skalibrowany** w kroku 3 na podstawie danych
walidacyjnych (patrz `models/kalibracja_progu.png`), a nie ustawiony "na
oko". Jeśli podobieństwo nowej twarzy do żadnego wzorca nie przekracza jego
progu, program pokazuje etykietę "Nieznana osoba" zamiast zgadywać.

## Wymagania sprzętowe i sprzęt użyty w projekcie

- **GPU:** NVIDIA RTX PRO 500 (architektura Blackwell), 6 GB VRAM.
  MobileNetV3-Small to bardzo mały model — 6 GB VRAM to wielokrotność tego,
  czego potrzeba do treningu z tego tutorialu (batch size 32, obrazy
  224×224). To dobry sprzęt dydaktyczny: trening całego modelu zajmuje
  zwykle 2–5 minut na kilka osób.
- **Kamera:** RoWave RC16, USB UVC 4K. Rozpoznawanie na żywo domyślnie
  prosi o 3840×2160 i przekazuje klatki w pełnej, odebranej rozdzielczości
  do detekcji oraz rozpoznawania; zmniejszany jest wyłącznie obraz
  wyświetlany na ekranie. Kod wymusza kodek `MJPG`, bo bez tego niektóre
  kamery UVC 4K potrafią działać w trybie zaledwie kilku klatek na sekundę.
  Rzeczywistą rozdzielczość klatek program potwierdza po uruchomieniu.
  Jeśli sterownik nie udostępni 4K, program wypisze rzeczywistą rozdzielczość
  i ostrzeżenie. Przetwarzanie 4K może zmniejszyć płynność analizy, a podgląd
  jest zmniejszany do rozmiaru ekranu. Podgląd działa niezależnie od
  wolniejszej analizy; program pokazuje osobno FPS
  podglądu i rozpoznawania. Podczas dopasowania twarz jest sprawdzana w
  dwóch centralnych kadrach o różnym zbliżeniu, co ogranicza wpływ dłoni i
  innych przeszkód blisko krawędzi twarzy.
- **System:** Windows 10/11.
- **Python:** 3.10–3.13.

> RTX Blackwell wymaga paczki PyTorch z CUDA 13.x, zgodnie z instrukcją
> instalacji poniżej. Samo wykrycie karty przez `torch.cuda.is_available()`
> nie potwierdza, że paczka zawiera kernele dla tej architektury.

## Struktura projektu

```
rozpoznawanie twarzy/
├── 00_test_detekcji.py          # krok 0: szybki test detekcji (bez rozpoznawania)
├── 01_zbieranie_danych.py       # krok 1: zbieranie zdjęć z kamery
├── 02_podzial_danych.py         # krok 2: podział na train/val/test
├── 03_trenowanie_modelu.py      # krok 3: wzorce tozsamosci (embeddingi) + kalibracja progu
├── 04_rozpoznawanie_na_zywo.py  # krok 4: rozpoznawanie na żywo z kamery
├── 05_ekran_z_kawalami.py       # pomocniczy: ekran z kawałami sterowany gestem (MediaPipe)
├── 06_usun_dane_osoby.py        # pomocniczy: usuwa zdjęcia i wzorce wybranych osób
├── 07_ewaluacja.py              # krok 5: metryki, macierz pomyłek, porównanie metod, CI
├── 08_eksperymenty.py           # krok 6: wpływ liczby zdjęć, jasności, k w k-NN i progu
├── 09_test_trudny.py            # krok 7: osobny, trudny zbiór testowy + CI
├── 10_rejestr_obecnosci.py      # krok 8: podgląd, eksport i porządki rejestru obecności
├── listuj_kamery.py             # pomocniczy: wypisuje dostępne kamery i ich rozdzielczości
├── pobierz_zdjecia_nieznajomych.py  # pomocniczy: pobiera zdjecia "obcych" osob do kalibracji progu
├── Uruchom panel.bat            # dwuklik: uruchamia panel GUI (szuka Pythona z .venv)
├── common/
│   ├── camera.py                # obsługa kamery (wybór, otwieranie, wymuszenie MJPG/4K)
│   ├── dane_osob.py             # lista osób, numeracja zdjęć, bezpieczny odczyt JSON
│   ├── embeddingi.py            # wspólne liczenie embeddingów (07, 08, 09 i GUI)
│   ├── emotions.py              # rozpoznawanie emocji (FER+)
│   ├── face_detector.py         # detekcja twarzy (YuNet z modele/, zapas: kaskada Haara)
│   ├── gesture.py               # wykrywanie gestu "ręka nad głową" (MediaPipe Pose)
│   ├── imgio.py                 # zapis/odczyt zdjęć odporny na polskie znaki w ścieżce
│   ├── metryki.py               # metryki FAR/FRR/EER/AUC, k-NN, prototypy i przedziały ufności
│   ├── modele/                  # modele ONNX/task (YuNet, FER+, MediaPipe Pose; poza Git)
│   ├── model.py                 # definicja modelu (ekstraktor cech / embeddingi)
│   ├── rejestr.py               # rejestr obecności w SQLite (bez zdjęć i emocji)
│   ├── rozpoznawanie.py         # wspólna logika: wzorce, dopasowanie, emocje (04 + GUI)
│   └── uruchom_w_venv.py        # restart skryptu w środowisku .venv
├── gui/
│   ├── main.py                  # panel desktopowy: pywebview + Flask + Bootstrap (styl AdminLTE)
│   ├── server.py                # lokalne API i strumień MJPEG (tylko 127.0.0.1)
│   ├── camera_stream.py         # wątek kamery: detekcja, rozpoznawanie i zbieranie zdjęć
│   ├── rozpoznawanie.py         # silnik modelu dla panelu (wczytywanie, przeładowanie wzorców)
│   ├── trening.py               # uruchamianie 02 + 03 z paskiem postępu i logiem
│   ├── ewaluacja.py             # uruchamianie 07 z paskiem postępu i logiem
│   ├── eksperymenty.py          # uruchamianie 08 z paskiem postępu i logiem
│   ├── test_trudny.py           # uruchamianie 09 z paskiem postępu i logiem
│   ├── tkinter_app.py           # awaryjne GUI w Tkinterze
│   ├── templates/index.html     # widok panelu
│   ├── static/                  # Bootstrap 5 (lokalnie), własny CSS i JS
│   ├── config.yaml              # konfiguracja starego GUI Tkinter
│   └── README.md                # opis GUI
├── src/
│   └── detection/detector.py    # nakładka na common/face_detector.py używana przez GUI
├── detection/                   # zgodność wsteczna: re-eksport src/detection/detector.py
├── data/
│   ├── raw/<osoba>/             # surowe zdjęcia z kamery (per osoba)
│   ├── raw_test/<osoba>/        # zdjęcia do testu trudnego (nie wchodzą do treningu)
│   ├── processed/{train,val,test}/  # dane po podziale, gotowe do treningu i oceny
│   └── rejestr/rejestr.db       # lokalny rejestr obecności (SQLite, poza Git)
├── models/
│   ├── wzorce_osob.json         # wzorce zdjec tozsamosci + skalibrowane progi
│   ├── kalibracja_progu.png     # wykres pomagajacy zrozumiec dobor progu
│   ├── ewaluacja/               # raport.json, wykresy/*.png, pomylki/*.jpg (po 07)
│   ├── eksperymenty/            # raport_eksperymentow.json, wyniki.csv, wykresy/*.png (po 08)
│   └── test_trudny/             # raport_test_trudny.json, wykresy/*.png (po 09)
├── requirements.txt
└── README.md
```

> **Panel GUI** (`gui/main.py`) prowadzi cały proces bez konsoli: zbieranie
> zdjęć do `data/raw`, trening z paskiem postępu (te same skrypty `02` i
> `03`), rozpoznawanie wyuczonych osób na żywo, opcjonalne emocje,
> przeglądanie i usuwanie bazy twarzy oraz rejestr frekwencji z eksportem
> CSV. Skrypty `01`–`04` nadal działają i
> uczą dokładnie tego samego pipeline'u — panel jest wygodnym skrótem, a nie
> osobną implementacją. Gdy WebView2 nie działa, uruchom awaryjne GUI:
> `python gui/tkinter_app.py`.

Foldery `data/raw`, `data/raw_test`, `data/processed`, `data/rejestr`,
`models` oraz `common/modele` są celowo puste w repozytorium (`.gitignore`) —
zawierają dane osobowe (zdjęcia twarzy, wpisy obecności) i duże pliki binarne,
których nie powinno się trzymać w Git.

Pliki modeli pobierz osobno i wgraj do `common/modele/`:

- `face_detection_yunet_2023mar.onnx` — [OpenCV Zoo, model
  `face_detection_yunet`](https://github.com/opencv/opencv_zoo/tree/main/models/face_detection_yunet),
- `emotion-ferplus-8.onnx` — [ONNX Model Zoo, `emotion_ferplus`](https://github.com/onnx/models/tree/main/validated/vision/body_analysis/emotion_ferplus),
- `pose_landmarker_lite.task` — [MediaPipe Pose Landmarker
  (lite)](https://storage.googleapis.com/mediapipe-models/pose_landmarker/pose_landmarker_lite/float16/1/pose_landmarker_lite.task).

Gdy któregoś pliku brak, program nadal działa: detekcja spada na kaskadę
Haara, a rozpoznawanie emocji (`--emocje`) i ekran `05` są pomijane.

> **Uwaga o ścieżce projektu:** biblioteka OpenCV na Windows ma problem z
> odczytem/zapisem plików, gdy pełna ścieżka do projektu zawiera znaki
> spoza ASCII (np. polskie „ł”, „ó” — jak w folderze
> `OneDrive - Zespół Szkół...`). Kod w tym repo radzi sobie z tym
> automatycznie (patrz [common/imgio.py](common/imgio.py) oraz mechanizm
> awaryjnego kopiowania kaskady Haara w
> [common/face_detector.py](common/face_detector.py)), więc nie trzeba
> przenosić projektu do innego folderu — ale warto o tym pamiętać, pisząc
> własny kod z `cv2.imread`/`cv2.imwrite`.

## Instalacja środowiska

### 1. Utwórz wirtualne środowisko

```powershell
cd "rozpoznawanie twarzy"
python -m venv .venv
.venv\Scripts\Activate.ps1
```

> **Uwaga dla uczniów — nie musisz pamiętać o aktywowaniu środowiska za
> każdym razem.** Na komputerze może być kilka różnych instalacji Pythona
> (np. Miniconda/Anaconda obok tego wirtualnego środowiska) — jeśli
> uruchomisz skrypt zwykłym poleceniem `python 04_rozpoznawanie_na_zywo.py`
> w terminalu, w którym akurat aktywny jest INNY Python (bez
> zainstalowanego `cv2`/`torch`), zobaczysz błąd
> `ModuleNotFoundError: No module named 'cv2'`. Każdy skrypt startowy tego
> projektu (`00_*`...`04_*`, `listuj_kamery.py`,
> `pobierz_zdjecia_nieznajomych.py`) sam to wykrywa i automatycznie
> uruchamia się ponownie właściwym interpreterem z `.venv` — zobaczysz
> wtedy tylko dodatkowy komunikat `[info] Uruchomiono przez... przelaczam
> na wirtualne srodowisko projektu`. Mechanizm ten opisany jest w
> [common/uruchom_w_venv.py](common/uruchom_w_venv.py).

### 2. Instalacja PyTorch z obsługą CUDA (GPU)

Sprzęt z GPU Blackwell (RTX PRO 500, `sm_120`) wymaga wersji PyTorch
zawierającej kernele dla CUDA 13.x; starsza paczka CUDA 12.6 może wykrywać
kartę, ale nie potrafi na niej wykonywać obliczeń. Dla Windows, pip i CUDA
13.0 zainstaluj:

```powershell
.\.venv\Scripts\python.exe -m pip install --upgrade torch torchvision --index-url https://download.pytorch.org/whl/cu130
```

Nie instaluj CUDA 12.6 ani wersji nightly, jeśli powyższa stabilna paczka
CUDA 13.0 jest dostępna. Sprawdź faktyczne obliczenie na karcie:

```powershell
.\.venv\Scripts\python.exe -c "import torch; x=torch.randn(1024,1024,device='cuda'); print(torch.__version__, torch.cuda.get_device_name(0), (x@x).mean().item())"
```

Samo `torch.cuda.is_available()` nie wystarcza: może zwrócić `True`, mimo że
zainstalowanej paczce brakuje kerneli dla `sm_120`.

### 3. Instalacja pozostałych zależności

```powershell
pip install -r requirements.txt
```

(Pip zauważy, że `torch`/`torchvision` są już zainstalowane we właściwej
wersji z GPU i ich nie nadpisze, o ile spełniają minimalne wersje z pliku.)

### 4. Podłącz kamerę RoWave RC16

Podłącz kamerę do portu USB (najlepiej USB 3.0, żeby zapewnić przepustowość
dla 4K/MJPG). Sprawdź w Menedżerze urządzeń Windows, że kamera jest
widoczna, i zamknij inne aplikacje, które mogą jej używać (Kamera Windows,
Teams, Zoom, OBS itd.) — tylko jedna aplikacja naraz może korzystać z
kamery.

Przy rozpoznawaniu na żywo program domyślnie zamawia 3840×2160 i wykonuje
detekcję na pełnej rozdzielczości odebranej klatki. Jeśli wybrana kamera
udostępnia inny tryb maksymalny, ustaw go przez `--szerokosc` i
`--wysokosc`; rzeczywisty rozmiar strumienia jest wypisywany po starcie.

Na laptopach z wbudowaną kamerą w systemie widoczne będą **co najmniej
dwie kamery** — wbudowana i RoWave RC16. Wszystkie skrypty w projekcie
(`00`, `01`, `04`) domyślnie same wykrywają dostępne kamery i **pytają w
konsoli, której użyć** (kamera z wykrytą rozdzielczością 3840×2160 jest
oznaczana jako najprawdopodobniej RoWave RC16). Możesz też pominąć to
pytanie, podając indeks wprost, np. `--kamera 1`. Aby sprawdzić indeksy
kamer z wyprzedzeniem, uruchom:

```powershell
python listuj_kamery.py
```

### 5. Panel GUI (opcjonalnie, ale wygodnie)

Panel to natywne okno systemowe z interfejsem webowym w stylu AdminLTE
(Bootstrap 5). Najprostsze uruchomienie: **dwuklik na pliku `Uruchom panel.bat`**
w folderze projektu. Plik sam znajduje Pythona z `.venv` (a gdy go nie ma —
launcher `py`/`python` z PATH) i otwiera okno panelu; przy błędzie pokazuje
czytelny komunikat i nie zamyka konsoli.

To samo z terminala:

```powershell
python gui/main.py
```

Panel prowadzi cały proces:

- **Detekcja na żywo** — ramki wokół twarzy, próg pewności, minimalna
  proporcja, FPS i liczba twarzy.
- **Rozpoznawanie** — po wczytaniu `models/wzorce_osob.json` pokazuje imię
  i podobieństwo; model wczytuje się w tle, a po treningu przeładowuje się
  bez restartu panelu. Suwak „korekta progu” działa jak `--korekta-progu`
  w `04_rozpoznawanie_na_zywo.py`. Karta **Wynik rozpoznawania** pokazuje
  TOP-3 kandydatów: pasek podobieństwa każdej osoby i pionowy znacznik progu
  decyzyjnego, dzięki czemu widać, jak blisko progu byli pozostali.
- **Dodaj twarz** — wpisujesz nazwę osoby i cel (np. 300 zdjęć), a panel
  zapisuje wycięte twarze 224×224 do `data/raw/<osoba>/` co ok. 0,3 s —
  dokładnie tak jak `01_zbieranie_danych.py` w trybie automatycznym.
  Przełącznik **Zapisz jako test trudny** kieruje zdjęcia do
  `data/raw_test/<osoba>/` — osobnego zbioru, którego nie używamy do treningu
  ani kalibracji progu.
- **Trening** — uruchamia `02_podzial_danych.py`, a potem
  `03_trenowanie_modelu.py --postep` w osobnym procesie; czyta maszynowe
  linie postępu (`[postep-start]`, `[postep]`), pokazuje pasek, etap i log.
  Trening można przerwać przyciskiem „Przerwij”.
- **Ewaluacja** — uruchamia `07_ewaluacja.py --postep` w osobnym procesie
  i pokazuje raport: accuracy, FAR/FRR, EER, AUC, metryki na osobę,
  porównanie metod, wykresy PNG i galerię pomyłek. Karty i tabele pokazują
  też **95% przedziały ufności**. Szczegóły w sekcji
  [Krok 5 — ewaluacja modelu](#krok-5--ewaluacja-modelu).
- **Eksperymenty** — uruchamia `08_eksperymenty.py --postep` i pokazuje
  wpływ liczby zdjęć, jasności, `k` w k-NN i progu decyzyjnego: tabelę
  wyników, wykresy i automatyczne wnioski. Tabela wariantów ma kolumnę
  przedziału ufności accuracy. Szczegóły w sekcji
  [Krok 6 — eksperymenty](#krok-6--eksperymenty).
- **Test trudny** — uruchamia `09_test_trudny.py --postep --folder
  data/raw_test` i porównuje wynik z łatwym testem z kroku 5: tabela
  „łatwy vs trudny”, metryki na osobę, wykresy i przedziały ufności.
  Szczegóły w sekcji [Krok 7 — test trudny](#krok-7--test-trudny).
- **Baza twarzy** — miniatury, liczba zdjęć i usuwanie osoby (opcjonalnie z
  ponownym treningiem). Usunięcie ostatniej osoby czyści wzorce.

W oknie wybierasz kamerę, ustawiasz próg pewności i proporcję twarzy, a
podgląd pokazuje ramki i liczbę wykrytych twarzy. Panel korzysta z
`pywebview` i WebView2 (dołączony do Windows 10/11); pliki Bootstrapa leżą
w repo, więc internet nie jest potrzebny. Serwer działa wyłącznie na
`127.0.0.1`. Gdy WebView2 jest niedostępny, uruchom awaryjne GUI:

```powershell
python gui/tkinter_app.py
```

Emocje są w panelu **domyślnie wyłączone** — włączasz je przełącznikiem w
zakładce Rozpoznawanie. W szkole ich nie używaj (AI Act, art. 5 ust. 1
lit. f); patrz uwagi o prywatności wyżej.

## Tutorial krok po kroku

> Ten sam proces możesz przejść w panelu **Dodaj twarz → Trening →
> Ewaluacja → Eksperymenty → Test trudny → Rozpoznawanie** (patrz sekcja
> [Panel GUI](#5-panel-gui-opcjonalnie-ale-wygodnie)).
> Poniższe kroki pokazują, co dokładnie dzieje się pod spodem — warto je
> przejść przynajmniej raz z konsoli, żeby zobaczyć surowe komunikaty.

### Krok 0 — szybki test detekcji (bez rozpoznawania)

Zanim zaczniecie zbierać dane i trenować model, warto sprawdzić, że kamera
i sama detekcja twarzy działają:

```powershell
python 00_test_detekcji.py
```

Powinno pojawić się okno podglądu z kamery z **żółtą ramką** wokół każdej
wykrytej twarzy (bez żadnego imienia — na tym etapie system jeszcze nikogo
nie rozpoznaje, tylko wykrywa, że w kadrze *jest jakaś* twarz). Naciśnij
`q`, aby zakończyć.

### Krok 1 — zbieranie danych treningowych

Dla **każdej osoby**, którą system ma rozpoznawać, uruchom:

```powershell
python 01_zbieranie_danych.py --osoba jan_kowalski --cel 250
```

Jeśli chcesz zapisywać materiał z pełnego 4K, dopisz
`--szerokosc 3840 --wysokosc 2160`.

- Pojawi się podgląd z kamery z zieloną ramką wokół wykrytej twarzy.
- Naciśnij `SPACJĘ`, aby zapisać pojedyncze zdjęcie, albo `a`, aby włączyć
  automatyczne zapisywanie co ok. 0.3 sekundy.
- Podczas zbierania **ruszaj lekko głową** (patrz w bok, w górę, w dół),
  zmieniaj wyraz twarzy, jeśli to możliwe — poproś o zmianę oświetlenia.
- Zbieraj zdjęcia także na **różnych tłach**. Kadrowanie ogranicza ich wpływ,
  ale nie zastąpi przykładów twarzy w warunkach, w których system będzie używany.
- Zdjęcia trafiają do `data/raw/jan_kowalski/`.
- Powtórz dla każdej osoby (np. wszystkich uczniów w grupie).

> Wskazówka dydaktyczna: warto też zebrać folder osoby "obcy"/"nieznany" ze
> zdjęciami osób spoza grupy (np. z internetu, za zgodą, albo zdjęcia
> nauczyciela) — pomocniczy skrypt `pobierz_zdjecia_nieznajomych.py` robi to
> automatycznie. Te zdjęcia NIE są używane do trenowania żadnej wagi — służą
> wyłącznie do automatycznej kalibracji progu podobieństwa w kroku 3 (patrz
> niżej), więc ich stylistyczna odmienność od zdjęć z kamery nie stanowi
> już problemu (w odróżnieniu od podejścia klasyfikacyjnego).
>
> **Test trudny zbieraj osobno.** Zdjęcia do kroku 7 (inny dzień, inne
> światło, inne urządzenie) zapisuj do `data/raw_test`, a nie do
> `data/raw` — dopisz `--folder data/raw_test`:
>
> ```powershell
> python 01_zbieranie_danych.py --osoba jan_kowalski --cel 80 --folder data/raw_test
> ```
>
> Dzięki temu zdjęcia testowe nigdy nie trafią do treningu ani do kalibracji
> progu. W panelu robi to przełącznik **Zapisz jako test trudny**.

### Krok 2 — podział na zbiór treningowy, walidacyjny i testowy

```powershell
python 02_podzial_danych.py
```

Domyślnie **70%** zdjęć każdej osoby trafia do treningu, **15%** do
walidacji, a **15%** do testu. Można to zmienić parametrami `--val-split`
i `--test-split` (np. `--val-split 0.1 --test-split 0.2`). Wynik trafia do
`data/processed/train/`, `data/processed/val/` i `data/processed/test/`.

Po co aż trzy zbiory?

- **train** — z niego krok 3 buduje wzorce tożsamości,
- **val** — na nim kalibrujemy próg decyzyjny; model nie uczy się na tych
  zdjęciach, ale „widzi” je pośrednio przy wyborze progu,
- **test** — leży nietknięty do samego końca. Sięgają po niego dopiero
  skrypty `07` i `08`, żeby zmierzyć ostateczną jakość.

Gdyby próg ustawić i raportować na tym samym zbiorze, wynik byłby
**optymistyczny** — próg „dopasowałby się” do tych zdjęć. Test ma to
wyeliminować. To jedna z najważniejszych zasad oceny modeli.

> Przy małych zbiorach podział 15/15% daje kilkadziesiąt zdjęć testowych.
> Jedna pomyłka zmienia wtedy wynik o kilka punktów procentowych — dlatego
> w raportach podajemy też surowe liczby, a nie tylko procenty.

### Krok 3 — budowanie wzorców tożsamości i kalibracja progu

```powershell
python 03_trenowanie_modelu.py
```

Skrypt:

1. Liczy embeddingi (wektory opisujące wygląd twarzy) wszystkich zdjęć
   treningowych, używając zamrożonej sieci MobileNetV3-Small (wagi
   ImageNet, bez żadnego dotrenowywania). Przed ekstrakcją zmniejsza wycinek
   do środkowych 82% szerokości i wysokości, żeby skupić cechy na twarzy.
   Każde zdjęcie liczy także przy trzech poziomach jasności, żeby wzorce
   obejmowały różnice ekspozycji.
2. Zachowuje wiele wzorców zdjęć każdej znanej osoby, aby objąć różne pozy
   i oświetlenie zamiast sprowadzać je do jednego średniego wektora.
3. Na zbiorze walidacyjnym (w tym zdjęciach "nieznajomy", jeśli je zebrano)
   automatycznie **kalibruje próg podobieństwa** oddzielający "to ta osoba"
   od "to ktoś inny"; próg jest ograniczony maksymalnie do **87%**, żeby
   łatwiej rozpoznać twarz przy zmianie pozy lub oświetlenia — patrz wartości w konsoli i wykres
   `models/kalibracja_progu.png`.
4. Zapisuje wzorce i progi do `models/wzorce_osob.json`.

Po zmianie sposobu kadrowania trzeba ponownie uruchomić ten krok — starsze
wzorce są celowo odrzucane, ponieważ nie pasują do nowych embeddingów.

Opcja `--postep` wypisuje maszynowe linie `[postep-start] <faza> <total>` i
`[postep] <faza> <zrobione>` zamiast zwykłego postępu w konsoli. Używa jej
panel GUI, żeby przesuwać pasek treningu; ręcznie zwykle nie jest potrzebna.

Ten krok trwa sekundy do (przy bardzo dużych zbiorach) kilkudziesięciu
sekund — w odróżnieniu od poprzedniego podejścia (klasyfikacja) nie ma tu
żadnej pętli uczenia, ale GPU przyspiesza liczenie embeddingów. Zwróćcie uwagę
na komunikat `[uwaga] Slaba separowalnosc...`, jeśli się pojawi — oznacza
on, że zdjęcia jakiejś osoby są zbyt podobne do zdjęć "obcych" (np. za mało
zróżnicowane oświetlenie/kąty) i warto zebrać więcej danych.

### Krok 4 — rozpoznawanie na żywo

```powershell
python 04_rozpoznawanie_na_zywo.py --korekta-progu 0.05
```

Domyślnie skrypt zamawia 4K (3840×2160), a detektor i rozpoznawanie pracują
na pełnym rozmiarze odebranej klatki. Po starcie konsola i podgląd pokażą
faktyczne wymiary klatek; do innego trybu użyj `--szerokosc` i `--wysokosc`.
Obraz jest skalowany wyłącznie na potrzeby wyświetlania w oknie.

Otworzy się podgląd z kamery z ramkami i etykietami rozpoznanych osób wraz
z procentowym podobieństwem. `--korekta-progu` pozwala szybko zaostrzyć
(wartość dodatnia) lub złagodzić (wartość ujemna) rozpoznawanie bez
ponownego uruchamiania kroku 3. Naciśnij `q`, aby zakończyć.

### Krok 5 — ewaluacja modelu

Skrypt `07_ewaluacja.py` odpowiada na pytanie **„jak dobrze działa model
i gdzie się myli?”** — zamiast tylko „czy działa”. Uruchom po treningu:

```powershell
python 07_ewaluacja.py
```

Skrypt liczy embeddingi zbioru **testowego** (`data/processed/test`),
a progi decyzyjne dobiera na zbiorze **walidacyjnym**. Jako „obcych” używa
zdjęć klasy `nieznajomy` ze zbiorów `train` i `test` — celowo **bez**
`val`, bo walidacja posłużyła do kalibracji progów. Wyznacza:

- **accuracy** — odsetek znanych twarzy rozpoznanych poprawnie,
- **precision / recall / F1** dla każdej osoby (czy model nie myli jej
  z innymi i czy jej nie odrzuca),
- **macierz pomyłek** — tabelę „kogo z kim myli” (wiersze = prawda,
  kolumny = decyzja modelu),
- **FAR** (*False Acceptance Rate*) — odsetek obcych zaakceptowanych jako
  znana osoba. W szkole to najważniejszy błąd: system wpuszcza kogoś, kogo
  nie powinien,
- **FRR** (*False Rejection Rate*) — odsetek znanych osób odrzuconych jako
  „Nieznana osoba”,
- **EER** — punkt, w którym FAR = FRR; jedna liczba do porównywania metod,
- **AUC** — jak dobrze sam ranking podobieństw rozdziela swoich od obcych
  (1,0 = idealnie, 0,5 = losowo),
- **95% przedziały ufności** dla accuracy, FAR i FRR (metoda Wilsona) oraz
  dla EER i AUC (bootstrap) — czyli informację, jak niepewny jest każdy
  wynik przy tej liczbie zdjęć. Wyjaśnienie w sekcji
  [Przedziały ufności](#przedziały-ufności--jak-czytać-niepewność-wyniku).

Na koniec porównuje trzy metody na **tych samych embeddingach**:
prototypy z pliku wzorców (tak jak `04`), **k-NN** oraz **regresję
logistyczną** z `scikit-learn`. To dobra lekcja: te same cechy i różne
sposoby podejmowania decyzji dają różne wyniki — deep learning nie zawsze
jest ostatnim słowem.

Przykładowe wyniki na zbiorze testowym (130 znanych zdjęć + 128 obcych):

| Metoda | Accuracy | 95% CI accuracy | FAR | FRR | EER | AUC |
| --- | --- | --- | --- | --- | --- | --- |
| Prototypy (progi z `03`) | 100,0% | 97,1–100,0% | 0,0% | 0,0% | 0,0% | 1,000 |
| k-NN (k = 5) | 100,0% | 97,1–100,0% | 4,7% | 0,0% | 0,0% | 1,000 |
| Regresja logistyczna | 96,9% | 92,4–98,8% | 0,8% | 3,1% | 2,7% | 0,999 |

Widać ciekawą rzecz: k-NN ma idealne accuracy, ale wpuszcza 4,7% obcych,
a regresja jest ostrożniejsza — rzadziej wpuszcza obcego, za to częściej
odrzuca swoich. **Nie ma jednej „najlepszej” metody** — wybór zależy od
tego, który błąd jest w danym zastosowaniu gorszy.

Przedział ufności mówi, jak precyzyjny jest pomiar. Dla prototypów accuracy
100% ma przedział 97,1–100,0%, a FAR 0% — przedział 0,0–2,9%. To znaczy:
przy innych 130 zdjęciach testowych wynik mógłby wyjść trochę gorszy, nawet
gdyby model się nie zmienił. Przy 128 obcych zdjęciach jeden fałszywy alarm
to skok FAR o 0,8 pkt proc., więc przedział jest szeroki. Przedziały metod
mogą na siebie nachodzić — wtedy **nie wolno** ogłaszać, że jedna metoda
jest lepsza. Potrzeba więcej danych albo trudniejszego testu.

Wyniki trafiają do:

```
models/ewaluacja/raport_ewaluacji.json   # wszystkie liczby, przedziały ufności i ostrzeżenia
models/ewaluacja/wykresy/*.png           # macierz, rozkład, FAR/FRR, PCA, porównanie, CI
models/ewaluacja/pomylki/*.jpg           # galeria najciekawszych błędów
```

Opcje przydatne na lekcji: `--k-sasiadow 7` (inne `k` w k-NN),
`--maks-pomylek 12` (ile zdjęć zapisać do galerii; `0` = nie zapisuj),
`--n-bootstrap 400` (ile losowań bootstrapu dla EER/AUC; mniej = szybciej,
ale szerszy i mniej stabilny przedział) oraz `--postep` (maszynowe linie
postępu dla panelu GUI).

> **Metodologia — dlaczego ten wynik jest uczciwy.** Próg każdej osoby
> pochodzi z kroku 3 i został skalibrowany na zbiorze walidacyjnym. Próg
> EER dla k-NN i regresji logistycznej także wybieramy na walidacji, a
> dopiero potem **bez zmian** stosujemy do testu. Same metryki liczymy
> wyłącznie na zbiorze testowym, którego nie użyto ani do budowy wzorców,
> ani do wyboru progu — dlatego nie jest on „dopasowany” do danych
> kalibracyjnych.
>
> To różnica między eksperymentem a chwaleniem się wynikiem. Zdjęcia obcych
> pochodzą z klas `nieznajomy` w `train` i `test` (bez `val`), więc FAR
> też nie jest policzony na danych kalibracyjnych.
>
> **Test jest mały.** Przy 15% zbioru to zwykle kilkadziesiąt zdjęć na
> osobę. Jedna pomyłka zmienia accuracy o kilka punktów procentowych.
> Dlatego wynik 100% na teście nie oznacza „model jest idealny” — oznacza
> „ten konkretny test nie pokazał błędu”. Ostrzeżenia są zapisane w
> raporcie JSON i pokazane w panelu.
>
> **Przedział ufności to nie ozdoba.** Każda liczba w raporcie ma obok
> siebie przedział 95%. Jeśli dwa warianty mają accuracy 100% i 98%, ale
> ich przedziały mocno na siebie nachodzą, to różnica może być zwykłym
> przypadkiem — zależy, które konkretnie zdjęcia wylosowano do testu.
> Porównuj przedziały, nie tylko wartości środkowe.

### Jak czytać wykresy z ewaluacji

- **Rozkład podobieństw** — niebieski słupek po prawej to znane twarze,
  pomarańczowy po lewej to obcy. Im mniejsze nachodzenie na siebie, tym
  łatwiej dobrać próg.
- **Kompromis FAR / FRR** — przesuwając próg w prawo, rzadziej wpuszczamy
  obcych (FAR spada), ale częściej odrzucamy swoich (FRR rośnie).
  Czerwona i niebieska linia przecinają się w punkcie EER.
- **Macierz pomyłek** — duże liczby na przekątnej to trafienia; wszystko
  poza przekątną to błędy. Kolumna „Nieznana osoba” pokazuje, ile twarzy
  danej osoby model odrzucił.
- **PCA** — rzutowanie embeddingów na 2 wymiary. Jeśli klastry osób są
  rozdzielone, model dobrze opisuje różnice między ludźmi; punkty obcych
  powinny leżeć poza klastrami.
- **Porównanie metod** — accuracy (zamknięta, bez opcji „nieznana”)
  i EER dla prototypów, k-NN i regresji logistycznej.
- **Przedziały ufności (95%)** — słupek to zmierzona wartość, a czarny
  odcinek to przedział. Krótki odcinek = precyzyjny pomiar; długi = mało
  danych albo wynik blisko 0%/100%, gdzie przedział z natury jest szerszy.
  Jeśli przedziały dwóch metod zachodzą na siebie, nie twierdź, że jedna
  jest lepsza.

### Krok 6 — eksperymenty

Skrypt `08_eksperymenty.py` zadaje pytanie, które jest sednem nauki:
**co się stanie, jeśli zmienimy jedną rzecz?** Zmieniamy tylko jeden
parametr, resztę zostawiamy bez zmian, mierzymy wynik i porównujemy.
Jeśli zmienimy wszystko naraz, nie będziemy wiedzieć, co naprawdę
wpłynęło na wynik.

```powershell
python 08_eksperymenty.py
```

Skrypt przeprowadza cztery eksperymenty:

1. **Liczba zdjęć** — 50, 100, 200 lub wszystkie zdjęcia na osobę.
   Pytanie: czy więcej danych zawsze oznacza lepszy model?
2. **Jasność** — tylko oryginalne zdjęcia vs trzy warianty jasności
   (0,65 / 1,0 / 1,4), tak jak w treningu `03`.
   Pytanie: czy augmentacja jasności naprawdę pomaga?
3. **`k` w k-NN** — 1, 3, 5, 9 i 15 sąsiadów.
   Pytanie: czy patrzeć tylko na najbliższe zdjęcie, czy na kilka
   i uśredniać decyzję?
4. **Próg decyzyjny** — od 0,50 do 0,95 co 0,05.
   Pytanie: gdzie jest granica między „wpuszczam swoich” a „nie wpuszczam
   obcych”?

Dla każdego wariantu liczone są te same metryki co w kroku 5: accuracy
zamknięta (kogo-kim, bez odrzucania), accuracy otwarta (z opcją
„nieznana osoba”), FAR, FRR, EER i AUC — każda z 95% przedziałem ufności.
Próg decyzyjny wybieramy w punkcie EER **na zbiorze walidacyjnym** i
stosujemy bez zmian do **testu** — tam też liczymy wszystkie metryki. Dzięki
temu eksperymenty oceniają uczciwie, a nie na danych, z których wybierano
próg.

Przykładowe wyniki na danych projektu (3 osoby, 615 znanych zdjęć
treningowych + 106 obcych, 130 znanych zdjęć testowych + 22 obcych,
ok. 44 s na GPU):

| Eksperyment | Wariant | Acc. zamk. | Acc. otw. | EER |
| --- | --- | --- | --- | --- |
| Liczba zdjęć | 50 | 100,0% | 100,0% | 0,4% |
| Liczba zdjęć | 100 | 100,0% | 100,0% | 0,0% |
| Liczba zdjęć | 200 / wszystkie | 100,0% | 100,0% | 0,0% |
| Jasność | tylko oryginał | 100,0% | 100,0% | 0,0% |
| Jasność | 3 warianty | 100,0% | 100,0% | 0,0% |
| k-NN | k = 1 / 3 / 5 / 9 / 15 | 100,0% | 100,0% | 0,0% |
| Próg 0,50 | — | 100,0% | 100,0% | FAR 100%, FRR 0% |
| Próg 0,85–0,90 | — | 100,0% | 100,0% | FAR 0%, FRR 0% |
| Próg 0,95 | — | 93,8% | 93,8% | FAR 0%, FRR 6,2% |

Na tych danych accuracy jest **sufitowa** (100% w każdym wariancie), więc
różnice widać dopiero w EER, FAR i w tabeli progów. To też jest lekcja:
przy łatwym zbiorze trzeba patrzeć na trudniejsze miary, a nie tylko na
jedną „ładną” liczbę.

Przedział 95% accuracy dla wariantu „wszystkie” to 97,1–100,0%, czyli ma
szerokość ok. 2,9 pkt proc. **Każda różnica między wariantami mniejsza niż
ta szerokość może być tylko szumem losowania zdjęć testowych**, a nie
prawdziwą różnicą. To dlatego tabela wariantów ma kolumnę CI — bez niej
łatwo przeceniłoby się znaczenie drobnych skoków.

Dwa efekty warte omówienia z klasą:

- **Sufit danych.** Skrypt `03` ogranicza liczbę prototypów do 128 na
  wariant jasności. Dlatego warianty „200” i „wszystkie” mają tyle samo
  prototypów (1152) i dają identyczny wynik — dodatkowe zdjęcia nie
  zmieniają decyzji, dopóki nie podniesiemy tego limitu. „Więcej danych”
  działa tylko do granicy ustawionej przez projekt modelu. Przy 50
  zdjęciach na osobę EER rośnie tylko do 0,4%, a FAR wynosi 3,1% zamiast
  4,7–6,3% — pozornie „lepiej”. Przy 128 zdjęciach testowych jedna
  zaakceptowana twarz zmienia FAR o 0,8 pkt proc., więc takie różnice to
  szum, nie dowód. Na trudniejszym zbiorze różnica byłaby większa.
- **Próg to wybór, nie prawda.** Przy progu 0,50 model wpuszcza
  wszystkich obcych (FAR 100%). Przy 0,85–0,90 nie wpuszcza żadnego
  obcego i nie odrzuca swoich (FAR 0%, FRR 0%). Przy 0,95 zaczyna
  odrzucać 6,2% swoich, bo próg jest zbyt ostry. Nie ma progu idealnego —
  wybór zależy od tego, co w danym zastosowaniu jest gorsze.

Wyniki trafiają do:

```
models/eksperymenty/raport_eksperymentow.json  # wszystkie liczby, wnioski, ostrzeżenia
models/eksperymenty/wyniki.csv                 # tabela do Excela (separator ";")
models/eksperymenty/wykresy/*.png              # cztery wykresy opisane niżej
```

Opcje: `--batch-size 16` (ile zdjęć naraz trafia do sieci), `--seed 42`
(powtarzalność), `--n-bootstrap 300` (losowania bootstrapu dla EER/AUC;
zmniejszenie przyspiesza kosztem mniej stabilnego przedziału) oraz
`--postep` (maszynowe linie postępu dla panelu GUI).

### Jak czytać wykresy z eksperymentów

- **Wpływ liczby zdjęć** — trzy linie: accuracy zamknięta, accuracy
  otwarta i EER (im niżej, tym lepiej). Płaska linia przy 200 i
  „wszystkie” to właśnie efekt limitu prototypów.
- **Wpływ jasności** — słupki obok siebie dla samego oryginału i dla
  trzech wariantów. Jeśli słupki są prawie równe, augmentacja nie wniosła
  wiele *na tym zbiorze* — nie znaczy to, że nigdy nie pomaga.
- **Wpływ `k` w k-NN** — małe `k` jest czułe na pojedyncze zdjęcie
  (szum), duże `k` wygładza decyzję, ale może mieszać osoby podobne.
- **Próg decyzyjny** — trzy linie: accuracy otwarta, FAR i FRR.
  Widać, gdzie FAR spada do zera i ile kosztuje to w FRR.

> **Uwaga metodologiczna.** Podobnie jak w kroku 5, próg EER jest
> wybierany na zbiorze **walidacyjnym**, a metryki liczymy na **testowym**.
> Test nie bierze udziału ani w budowie prototypów, ani w wyborze progu —
> dlatego wynik jest uczciwy. Jest jednak mały, więc pojedyncza pomyłka
> zmienia procent o kilka punktów. Ostrzeżenia są zapisane w raporcie JSON
> i pokazane w panelu.
>
> **Przedziały ufności a wnioski.** W eksperymentach porównujemy warianty,
> które różnią się jednym parametrem. Jeśli ich przedziały ufności zachodzą
> na siebie, wniosek brzmi „na tym zbiorze nie widać różnicy”, a nie
> „parametr X jest lepszy”. To najczęstszy błąd w szkolnych prezentacjach
> wyników — i dlatego raport wypisuje ostrzeżenie o szerokości CI.

### Krok 7 — test trudny

Krok 5 odpowiada na pytanie „jak model działa na zdjęciach z tej samej
sesji?”. Krok 7 zadaje trudniejsze pytanie: **czy model działa też poza
warunkami treningu?** Do tego potrzebny jest osobny zbiór zdjęć, którego
model nigdy nie widział — z innego dnia, innego oświetlenia, innego kąta,
a najlepiej z innego urządzenia.

```powershell
python 09_test_trudny.py
```

Skrypt domyślnie czyta `data/raw_test`, a nie `data/processed/test`.
To celowe: zdjęcia z `data/raw_test` **nie przechodzą przez krok 2**, więc
nie trafiają ani do treningu, ani do walidacji, ani do kalibracji progu.

**Skąd wziąć zdjęcia do testu trudnego?**

- Z konsoli: `python 01_zbieranie_danych.py --osoba jan_kowalski --cel 80
  --folder data/raw_test` (parametr `--folder` zmienia folder docelowy).
- W panelu: zakładka **Dodaj twarz** → przełącznik **Zapisz jako test
  trudny**. Zdjęcia trafią do `data/raw_test/<osoba>/`.
- Zbierz je **później niż dane treningowe** — po kilku dniach, w innym
  pomieszczeniu, przy innym świetle, ewentualnie inną kamerą. Jeśli zbierzesz
  je w tej samej sesji, test nie będzie trudny, choć będzie miał inną nazwę.
- Warto też dodać folder `data/raw_test/nieznajomy/` ze zdjęciami osób
  spoza bazy — bez nich skrypt nie policzy FAR, EER ani AUC.
- Nie musi być dużo: 30–80 zdjęć na osobę wystarczy, żeby zobaczyć kierunek
  zmiany. Pamiętaj jednak, że mała próba = szeroki przedział ufności.

Co robi `09_test_trudny.py`:

1. Wczytuje wzorce i progi z `models/wzorce_osob.json` (model z kroku 3 —
   **nic nie jest dotrenowywane**).
2. Liczy embeddingi zdjęć z `data/raw_test`. Podfoldery o nazwach osób z
   modelu to twarze znane. Foldery spoza modelu (np. `nieznajomy`, ale też
   każda nowa osoba) są traktowane jako **obcy**.
3. Liczy accuracy, FAR, FRR, EER i AUC oraz 95% przedziały ufności, a także
   recall dla każdej osoby i macierz pomyłek.
4. Jeśli istnieje raport z kroku 5 (`models/ewaluacja/raport_ewaluacji.json`),
   porównuje wynik trudny z łatwym i pokazuje różnicę w punktach
   procentowych.
5. Zapisuje raport i wykresy:

```
models/test_trudny/raport_test_trudny.json  # liczby, CI, porównanie, ostrzeżenia
models/test_trudny/wykresy/*.png            # podsumowanie, porównanie, rozkład, macierz
```

Opcje: `--folder data/raw_test` (inny folder testu), `--n-bootstrap 400`
(liczba losowań dla EER/AUC), `--batch-size 32`, `--seed 42` oraz `--postep`
(dla panelu GUI).

**Jak interpretować wynik?**

- Wynik **taki sam jak w kroku 5** (np. nadal 100%) oznacza, że albo nowe
  zdjęcia nie są wystarczająco trudne, albo model naprawdę się generalizuje.
  Sprawdź, czy na pewno różnią się warunki — inaczej test nic nie wnosi.
- Wynik **gorszy** to nie awaria, tylko prawdziwa informacja: model częściowo
  nauczył się warunków sesji, a nie samych twarzy. To najważniejsza lekcja
  tego kroku.
- Brak zdjęć obcych → FAR, EER i AUC są pomijane (`null`), a skrypt wypisuje
  ostrzeżenie. To nie błąd, tylko brak danych do pomiaru.
- Brak zdjęć niektórych osób → skrypt ostrzega, ale liczy dalej na tym, co
  jest. Accuracy dotyczy wtedy tylko obecnych osób.
- **Nie poprawiaj modelu „pod” test trudny.** Jeśli po zobaczeniu wyniku
  zmienisz próg albo dobierzesz zdjęcia, zbiór przestaje być testem, a staje
  się kolejnym zbiorem walidacyjnym. Test trudny to egzamin, nie trening.

### Jak czytać wykresy z testu trudnego

- **Podsumowanie testu trudnego** — słupki accuracy, FAR, FRR i EER z
  pionowymi odcinkami 95% CI. Długi odcinek oznacza, że zdjęć jest mało
  i wynik jest niepewny.
- **Łatwy test vs trudny test** — te same metryki obok siebie. Dopiero ta
  para pokazuje, ile „wart” był łatwy wynik z kroku 5.
- **Rozkład podobieństw** — czy znane twarze z nowych warunków nadal leżą
  po prawej, a obcy po lewej. Jeśli rozkłady się zlewają, próg z kroku 3
  jest za niski dla tych warunków.
- **Macierz pomyłek** — wiersze to prawda, kolumny to decyzja modelu.
  Kolumna „Nieznana osoba” pokazuje, ile razy model odrzucił znaną twarz,
  a wiersz „obcy” — ile razy wpuścił kogoś spoza bazy.

## Przedziały ufności — jak czytać niepewność wyniku

Każda liczba w raportach (`100%`, `0%`, `EER 2,7%`) jest **oszacowaniem
z próby**, a nie prawdą absolutną. Gdybyśmy zebrali inne 130 zdjęć
testowych, wynik mógłby wyjść trochę inny — nawet gdyby model się nie
zmienił. Przedział ufności mówi właśnie, **jak dużej zmiany można się
spodziewać od samego losowania zdjęć**.

W projekcie używamy dwóch metod:

- **Wilsona** — dla odsetków: accuracy, FAR, FRR. To ten sam rachunek, co
  „poparcie 45% ± 3 pkt proc.” w sondażach. Działa dobrze także przy
  wynikach 0% i 100%, gdzie naiwny wzór daje przedział o szerokości zera.
- **Bootstrap** — dla EER i AUC. Losujemy ze zwracaniem tyle samo zdjęć,
  ile mamy, liczymy metrykę, powtarzamy `--n-bootstrap` razy (domyślnie
  400) i odcinamy skrajne 2,5% wyników z każdej strony. To odpowiedź na
  pytanie: „gdyby test wyglądał trochę inaczej, jaki wynik moglibyśmy
  dostać?”. Ziarno `--seed 42` sprawia, że wynik jest powtarzalny.

Przykład z danych projektu (130 znanych zdjęć, 128 obcych):

| Pomiar | Wynik | 95% CI | Jak to czytać |
| --- | --- | --- | --- |
| Accuracy | 100,0% | 97,1–100,0% | prawdziwa skuteczność może być bliżej 97% |
| FAR | 0,0% | 0,0–2,9% | w innych 128 obcych zdjęciach mogłoby wpaść ok. 4 osoby |
| FRR | 0,0% | 0,0–2,9% | model mógłby odrzucić do ok. 4 znanych zdjęć |
| EER | 0,0% | 0,0–0,0% | przy idealnym rozdzieleniu bootstrap też daje same zera |

Ostatni wiersz to ważna pułapka: **przedział 0–0 nie znaczy „pewność
absolutna”**. Przy idealnym rozdzieleniu każda próbka bootstrapu daje EER
0%, więc metoda nie widzi niepewności. Dlatego accuracy, FAR i FRR liczymy
metodą Wilsona, która przy 0 błędach nadal daje sensowny przedział.

Trzy zasady, które warto zapamiętać:

1. **Mała próba = szeroki przedział.** Na mini-teście 8 znanych zdjęć
   accuracy 100% ma przedział 67,6–100,0% — prawie nic nie wiadomo.
2. **Nachodzące przedziały = brak dowodu różnicy.** Jeśli metoda A ma
   97–100%, a metoda B 92–99%, nie można ogłosić, że A jest lepsza.
3. **Przedział nie opisuje pojedynczej osoby.** To niepewność średniej
   dla całego zbioru, a nie „ten uczeń zostanie rozpoznany w 97%”.

## Krok 8 — rejestr rozpoznań i frekwencja

Model odpowiada na pytanie „kto jest przed kamerą?”. Krok 8 zamienia tę
odpowiedź w coś użytecznego: **dziennik obecności**. Panel (zakładka
**Frekwencja**) i skrypt `10_rejestr_obecnosci.py` zapisują rozpoznania do
lokalnej bazy SQLite `data/rejestr/rejestr.db`.

```powershell
python 10_rejestr_obecnosci.py
python 10_rejestr_obecnosci.py --data 2026-11-24
python 10_rejestr_obecnosci.py --ostatnie 20
python 10_rejestr_obecnosci.py --eksport frekwencja.csv --data 2026-11-24
python 10_rejestr_obecnosci.py --wyczysc-dzien 2026-11-24 --tak
python 10_rejestr_obecnosci.py --wyczysc-wszystko --tak
```

**Co trafia do rejestru?** Tylko tekst i liczby: nazwa osoby z modelu, czas
rozpoznania, podobieństwo do wzorca, numer kamery i źródło wpisu (kamera albo
ręcznie). **Nie zapisujemy zdjęć, embeddingów ani emocji** — im mniej danych
osobowych, tym mniej ryzyka. Plik bazy jest w `.gitignore`.

**Dlaczego nie każda klatka?** Rozpoznawanie liczy się kilka razy na sekundę.
Bez ograniczenia baza zapełniłaby się tysiącami identycznych wpisów. Dlatego
ta sama osoba jest zapisywana najwyżej **raz na 5 minut** (`ODSTEP_ZAPISU_S`
w `common/rejestr.py`). To wystarcza, żeby zobaczyć pierwsze i ostatnie
pojawienie się osoby w ciągu dnia.

**Frekwencja dnia** to osoby, które mają co najmniej jedno rozpoznanie w
wybranym dniu. „Nieobecni” to osoby z `models/wzorce_osob.json`, których w
tym dniu nie było. Lista pochodzi z modelu, a nie z folderów w `data/` —
uczniowie z testu trudnego nie są przecież częścią klasy.

**Zasady prywatności i retencji:**

- Zapis można w każdej chwili wyłączyć przełącznikiem w panelu; ustawienie
  jest trwałe i dotyczy też kolejnych uruchomień.
- Wpisy starsze niż **90 dni** są usuwane automatycznie przy starcie i przy
  zapisie (retencję można zmienić w kodzie lub przez `--retencja`).
- Dzień albo cały rejestr można usunąć z panelu lub z konsoli. Usunięcie jest
  trwałe — dlatego wymaga potwierdzenia (wpisania daty lub słowa `WYCZYSC`).
- Rejestr jest lokalny: serwer nasłuchuje tylko na `127.0.0.1`, a baza nie
  opuszcza komputera.
- **Eksport CSV** używa średnika i BOM, więc polskie znaki poprawnie otwierają
  się w Excelu. To wygodny sposób przekazania frekwencji, ale pamiętaj, że
  eksportowany plik też zawiera dane osobowe.

> **Ćwiczenie z etyki.** Zastanówcie się, czy system obecności oparty na
> kamerze jest w Waszej szkole w ogóle dopuszczalny. Kto wyraża zgodę? Jak
> długo wolno trzymać wpisy? Co się stanie, gdy model pomyli dwie osoby?
> Rejestr pokazuje, że techniczna poprawność to dopiero połowa decyzji —
> druga połowa to prawo, zgoda i zaufanie.

## Jak przygotować dobre materiały treningowe

Jakość danych ma większy wpływ na wynik niż wybór architektury sieci.
Zalecenia:

- **Ilość:** minimum 100–150 zdjęć na osobę, zalecane 200–300.
- **Różnorodność ustawienia głowy:** patrzenie prosto, lekko w lewo/prawo,
  lekko w górę/dół — sieć musi nauczyć się rozpoznawać osobę z różnych
  kątów, nie tylko "en face".
- **Różnorodność tła i oświetlenia:** zbierz dane na różnych tłach i w
  różnych porach dnia / przy
  różnym świetle sztucznym, jeśli to możliwe. Kaskada Haara i sieć
  klasyfikująca są wrażliwe na skrajne warunki oświetleniowe.
- **Mimika i dodatki:** uśmiech, neutralny wyraz twarzy, z okularami/bez
  (jeśli dana osoba je nosi), z różną fryzurą, jeśli się zmienia.
- **Unikaj duplikatów:** seria 50 identycznych, nieruchomych klatek pod
  rząd nie wnosi tyle, co 50 zdjęć z różnicami — lepiej korzystać z trybu
  automatycznego zapisu (`a`) i cały czas lekko się poruszać.
- **Balans klas:** staraj się zbierać podobną liczbę zdjęć dla każdej
  osoby — duża dysproporcja (np. 500 zdjęć jednej osoby, 50 innej) sprawia,
  że model będzie "faworyzował" liczniejszą klasę.
- **RODO / zgody:** to zdjęcia twarzy uczniów — omówcie z klasą kwestie
  zgody na przetwarzanie wizerunku, poinformujcie, że dane są lokalne (nie
  trafiają do repozytorium Git dzięki `.gitignore`), i usuńcie/dodatkowo
  zabezpieczcie folder `data/` po zajęciach.

## Rozwiązywanie problemów (FAQ)

**Błąd `no kernel image is available` przy `torch.cuda.is_available() == True`.**
Zainstaluj stabilne `torch` i `torchvision` z CUDA 13.0 (patrz sekcja
instalacji), a następnie sprawdź mnożenie macierzy na GPU. Samo wykrycie
karty nie oznacza, że zainstalowany wheel obsługuje Blackwell.

**Kamera nie otwiera się / błąd `Nie udalo sie otworzyc kamery`.**
Sprawdź, czy żadna inna aplikacja (Kamera, Teams, Zoom, przeglądarka z
otwartą stroną wideokonferencji) nie korzysta z kamery. Spróbuj innego
portu USB (najlepiej USB 3.0, kolor niebieski). Jeśli w systemie jest
więcej niż jedna kamera, zmień `--kamera 1`, `--kamera 2` itd.

**Program pyta o wybór kamery, ale i tak włącza się zła kamera (np. wbudowana
zamiast RoWave RC16).**
Uruchom `python listuj_kamery.py`, żeby zobaczyć indeksy i maksymalne
rozdzielczości wszystkich wykrytych kamer — RoWave RC16 powinna pokazać
prawdziwe `3840x2160`, a kamera wbudowana w laptop zwykle mniej (np.
`2560x1440`, `1920x1080` czy `1280x720`). Zanotuj właściwy numer i podaj go
jawnie, np. `python 00_test_detekcji.py --kamera 1` — wtedy program nie
będzie pytał interaktywnie.

**Bardzo niskie FPS podczas podglądu.**
Upewnij się, że kamera pracuje w trybie MJPG (skrypt ustawia to
automatycznie) i że USB jest podłączone do portu 3.0. Możesz też zmniejszyć
rozdzielczość podglądu, np. `--szerokosc 1280 --wysokosc 720`.

**Model myli osoby / dużo "Nieznana osoba" dla znanych osób.**
Zbierz więcej i bardziej zróżnicowanych danych treningowych (patrz sekcja
wyżej — różne kąty głowy, oświetlenie), uruchom ponownie krok 3 (kalibracja
progu jest automatyczna) albo obniż próg ręcznie przy starcie kroku 4:
`python 04_rozpoznawanie_na_zywo.py --korekta-progu -0.05` (w panelu GUI
służy do tego suwak **Korekta progu** w zakładce Rozpoznawanie).

**Model rozpoznaje jako znaną osobę kogoś zupełnie innego (albo nawet
fragment czoła/dłoni).** To był częsty problem poprzedniej wersji projektu
(klasyfikacja) — sieć uczyła się rozróżniać *styl* zdjęć zamiast rysów
twarzy (patrz sekcja "Jak to działa"). Aktualne podejście (embeddingi +
skalibrowany próg podobieństwa) jest na to znacznie odporniejsze, ale jeśli
mimo to się zdarza:
- Sprawdźcie wykres `models/kalibracja_progu.png` — czy grupy "własne" i
  "obce" faktycznie się rozdzielają? Jeśli mocno na siebie nachodzą,
  zbierzcie więcej/różnorodniejszych zdjęć danej osoby.
- Podnieście próg ręcznie: `--korekta-progu 0.05` (lub więcej).
- Upewnijcie się, że folder "nieznajomy" (patrz
  `pobierz_zdjecia_nieznajomych.py`) zawiera wystarczająco dużo zdjęć — im
  więcej i różnorodniejszych "obcych" przykładów w kroku 3, tym dokładniej
  skalibrowany próg.
- Detektor potwierdza wykrycie drugą, niezależną kaskadą, aby ograniczyć
  fałszywe ramki na zegarach i kafelkach. Jeśli prawdziwe twarze w oddali są
  pomijane, sprawdź ostrość obiektywu i światło — przy bardzo małej liczbie
  pikseli detekcja pozostaje ograniczona.
- Sprawdźcie jakość zdjęć w `data/raw/<osoba>` — zamazane, ciemne albo
  puste (samo tło, bez wyraźnej twarzy) zdjęcia zaniżają "podobienstwo
  wlasne min" podczas kalibracji, co obniża wyliczony próg i ułatwia
  fałszywe rozpoznania. Usuńcie słabe zdjęcia z `data/raw`, a potem
  koniecznie uruchomcie PONOWNIE `python 02_podzial_danych.py` (żeby
  odświeżyć `data/processed` z aktualnej zawartości `data/raw` — sam
  `03_trenowanie_modelu.py` tego nie robi) i dopiero potem
  `python 03_trenowanie_modelu.py`.

**Model myli mnie z bliską osobą (np. rodzeństwem/dzieckiem), które
fizycznie jest do mnie podobne — pokazuje ją jako mnie z wysokim
procentem, albo tuż pod progiem jako "nieznana osoba".** To nie błąd, tylko
naturalne ograniczenie automatycznej kalibracji: próg jest dobierany na
podstawie zdjęć folderu "nieznajomy" (przypadkowe twarze z internetu) —
a bliscy krewni są znacznie trudniejszym "negatywem" niż losowa obca
twarz, bo mają realnie podobne rysy. Dwa sposoby na poprawę:
1. **Szybki (bez nowych zdjęć):** podnieście margines bezpieczeństwa przy
   kalibracji, np. `python 03_trenowanie_modelu.py --margines-bezpieczenstwa 0.05`
   — ale uważajcie: zbyt duży margines może sprawić, że próg przekroczy
   nawet Wasze WŁASNE najsłabsze podobieństwo (widoczne w logu jako
   "podobienstwo wlasne min") i program zacznie odrzucać też Was. Program
   wypisuje to minimum w logu — dobierzcie margines tak, żeby
   `środek(min_wlasne, maks_obce) + margines` został WYRAŹNIE poniżej
   min_wlasne.
2. **Najskuteczniejszy (edukacyjnie bardzo pouczający — "hard negative
   mining"):** dorzućcie kilka-kilkanaście zdjęć tej podobnej osoby (np.
   rodzeństwa) do folderu `data/raw/nieznajomy/` PRZED krokiem 2/3.
   Kalibracja "zobaczy" wtedy prawdziwie trudny przypadek i dobierze
   wyższy, bezpieczniejszy próg — dokładnie tak, jak w prawdziwych
   systemach rozpoznawania twarzy uczy się je odróżniać bliźnięta czy
   rodzeństwo.

**Podgląd z kamery wygląda na przycięty/przesunięty w prawo-dół, mimo że
kamera "widzi" całą scenę.** To najczęściej NIE jest błąd w kodzie ani
utrata danych z kamery, tylko skalowanie DPI Windows. Jeśli ekran ma
włączone skalowanie (np. 125%/150% w Ustawienia > System > Ekran), a
aplikacja nie zgłosi się jako "DPI aware", Windows sam przeskalowuje/przycina
okna takiej aplikacji — dokładnie to wygląda jak "zły crop" obrazu z kamery.
Ten projekt naprawia to automatycznie (`common/camera.py` wywołuje
`ustaw_swiadomosc_dpi()` już przy starcie każdego skryptu), a dodatkowo:
- Okno podglądu (`cv2.WINDOW_NORMAL`) można teraz swobodnie zmieniać
  rozmiarem myszką (przeciągając krawędź/róg) — obraz automatycznie
  skaluje się do nowego rozmiaru okna, bez przycinania.
- Funkcja `dopasuj_do_ekranu()` w `common/camera.py` dodatkowo zmniejsza
  (nigdy nie powiększa) obraz przy starcie, żeby okno od razu mieściło się
  na ekranie, nawet gdy kamera pracuje w rozdzielczości większej niż ekran
  (np. kamera 4K na ekranie Full HD).
- Ważne: samo skalowanie/zmiana rozmiaru okna dotyczy WYŁĄCZNIE tego, co
  widać na ekranie — detekcja i rozpoznawanie twarzy zawsze działają na
  oryginalnej, pełnej klatce z kamery, więc jakość rozpoznawania się nie
  zmienia.
- Jeśli mimo to nadal brakuje kawałka sceny, to prawdopodobnie
  ograniczenie fizyczne/sterownika samej kamery (np. domyślny cyfrowy
  zoom/crop w ustawieniach kamery w Windows) — sprawdźcie Ustawienia >
  Bluetooth i urządzenia > Kamery > [Wasza kamera] > Ustawienia
  zaawansowane.

**Test trudny nie chce się uruchomić: „Folder data/raw_test nie ma zdjęć
w podfolderach osób”.**
Skrypt `09` czyta wyłącznie `data/raw_test`, a nie `data/raw` ani
`data/processed/test`. Zbierz zdjęcia z `--folder data/raw_test` (albo
włącz **Zapisz jako test trudny** w panelu). Każda osoba musi mieć własny
podfolder, np. `data/raw_test/jan_kowalski/`. Folder `nieznajomy` jest
opcjonalny, ale bez niego nie da się policzyć FAR, EER ani AUC.

**W teście trudnym mam tylko kilka zdjęć i przedziały ufności są ogromne.**
To normalne: przy 8 zdjęciach accuracy 100% ma przedział 67,6–100,0%.
Do wiarygodnego wniosku potrzeba kilkudziesięciu zdjęć na osobę. Możesz też
uruchomić `09` kilka razy z różnymi `--seed`, żeby zobaczyć, jak bardzo
wynik zależy od losowania.

## Pomysły na rozszerzenie projektu (zadania dla uczniów)

- Zamienić kaskadę Haara na nowocześniejszy detektor (np. MediaPipe Face
  Detector) i porównać szybkość/skuteczność.
- Zamienić ogólny ekstraktor cech ImageNet na sieć wytrenowaną
  SPECJALNIE do rozpoznawania twarzy (np. FaceNet/ArcFace) — powinno dać
  jeszcze lepszą separację embeddingów niż obecne podejście.
  To bardziej zaawansowany, ale bardzo pouczający temat "one-shot/few-shot
  learning".
- **Dodać zapis logów rozpoznań** (kto i kiedy został rozpoznany) do pliku
  CSV/bazy danych — prosty system obecności jest już w kroku 8
  (`common/rejestr.py`). Rozbudujcie go: dodajcie drugi tryb zapisu (tylko
  wejście/wyjście), raport tygodniowy albo wykres frekwencji w czasie.
- Poeksperymentować z inną architekturą backbone'u (np.
  `resnet18`, `efficientnet_b0`) i porównać dokładność/szybkość.
- **Zebrać trudniejszy zbiór testowy** (zdjęcia z innego dnia, innego
  urządzenia, gorszego światła) do `data/raw_test` i uruchomić
  `python 09_test_trudny.py`. Sprawdźcie, o ile wynik spadnie względem
  łatwego testu z `07` i czy przedziały ufności nadal się nie nakładają.
  To najważniejsze ćwiczenie z metodologii eksperymentu.
- **Zbadać wpływ progu decyzyjnego na FAR/FRR w praktyce:** przy jakim progu
  system wpuszcza obcych, a przy jakim odmawia dostępu własnym osobom?
- **Zwiększyć `--n-bootstrap`** (np. do 1000) i sprawdzić, czy przedział
  ufności EER/AUC się uspokaja. Zmienić `--seed` i zobaczyć, o ile wynik
  się przesuwa — to pokazuje, że bootstrap też jest losowaniem.
- **Dodać do raportu nową metrykę** (np. dokładność dla każdej minuty
  nagrania albo dla każdego rodzaju oświetlenia) i policzyć dla niej
  przedział ufności. Zastanówcie się, czy metoda Wilsona ma tu sens.
- **Porównać przedziały zamiast średnich:** przygotować tabelę, w której
  dla każdej pary metod sprawdzacie, czy ich przedziały accuracy zachodzą
  na siebie. Wnioski zapisać słowami „nie widać różnicy” albo „różnica
  większa niż niepewność pomiaru”.

## Ekran z kawałami sterowany gestem (`05_ekran_z_kawalami.py`)

Ekran na korytarz bez dotykania: **podnieś rękę nad głowę**, żeby przejść dalej
(„Podnieś rękę” – pytanie – podnieś rękę – puenta). Kamera służy wyłącznie do
wykrycia gestu (model pozy MediaPipe, `common/gesture.py`,
`common/modele/pose_landmarker_lite.task`); program nie rozpoznaje osób ani
emocji i nie zapisuje obrazu. Wymaga `pip install mediapipe` - jeśli pip
doinstaluje `opencv-contrib-python` 5.x, usuń je i zainstaluj ponownie
`opencv-python>=4.9,<5.0`. Awaryjnie: SPACJA = dalej, q/ESC = koniec.

## Usuwanie danych osób po zakończeniu

Po zamknięciu podglądu (q/ESC) `04_rozpoznawanie_na_zywo.py` pyta w konsoli, czy
usunąć dane osób. Po wskazaniu osób i wpisaniu `tak` kasuje ich zdjęcia
(`data/raw`, `data/processed`) oraz nieużywane stare pliki modelu, a następnie
uruchamia ponownie `02_podzial_danych.py` i `03_trenowanie_modelu.py`, więc
wzorce powstają już bez tych osób. To samo bez kamery:
`python 06_usun_dane_osoby.py NAZWA [NAZWA ...] --tak` (bez `--tak` tylko podgląd).
