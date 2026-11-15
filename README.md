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
├── 02_podzial_danych.py         # krok 2: podział na train/val
├── 03_trenowanie_modelu.py      # krok 3: wzorce tozsamosci (embeddingi) + kalibracja progu
├── 04_rozpoznawanie_na_zywo.py  # krok 4: rozpoznawanie na żywo z kamery
├── 05_ekran_z_kawalami.py       # pomocniczy: ekran z kawałami sterowany gestem (MediaPipe)
├── 06_usun_dane_osoby.py        # pomocniczy: usuwa zdjęcia i wzorce wybranych osób
├── listuj_kamery.py             # pomocniczy: wypisuje dostępne kamery i ich rozdzielczości
├── pobierz_zdjecia_nieznajomych.py  # pomocniczy: pobiera zdjecia "obcych" osob do kalibracji progu
├── common/
│   ├── camera.py                # obsługa kamery (wybór, otwieranie, wymuszenie MJPG/4K)
│   ├── dane_osob.py             # lista osób, numeracja zdjęć, bezpieczny odczyt JSON
│   ├── emotions.py              # rozpoznawanie emocji (FER+)
│   ├── face_detector.py         # detekcja twarzy (YuNet z modele/, zapas: kaskada Haara)
│   ├── gesture.py               # wykrywanie gestu "ręka nad głową" (MediaPipe Pose)
│   ├── imgio.py                 # zapis/odczyt zdjęć odporny na polskie znaki w ścieżce
│   ├── modele/                  # modele ONNX/task (YuNet, FER+, MediaPipe Pose; poza Git)
│   ├── model.py                 # definicja modelu (ekstraktor cech / embeddingi)
│   └── uruchom_w_venv.py        # restart skryptu w środowisku .venv
├── gui/
│   ├── main.py                  # panel desktopowy: pywebview + Flask + Bootstrap (styl AdminLTE)
│   ├── server.py                # lokalne API i strumień MJPEG (tylko 127.0.0.1)
│   ├── camera_stream.py         # wątek kamery, detekcja i klatki JPEG
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
│   └── processed/{train,val}/   # dane po podziale, gotowe do treningu
├── models/
│   ├── wzorce_osob.json         # wzorce zdjec tozsamosci + skalibrowane progi
│   └── kalibracja_progu.png     # wykres pomagajacy zrozumiec dobor progu
├── requirements.txt
└── README.md
```

> **Panel GUI** (`gui/main.py`) to okno desktopowe z interfejsem webowym w
> stylu AdminLTE (Bootstrap 5). Pokazuje podgląd z kamery i rysuje ramki
> wokół wykrytych twarzy, ale nie prowadzi jeszcze rozpoznawania osób ani
> nie steruje zbieraniem danych — do pełnego pipeline'u używaj skryptów
> `01`–`04` z konsoli. Gdy WebView2 nie działa, uruchom awaryjne GUI:
> `python gui/tkinter_app.py`.

Foldery `data/raw`, `data/processed`, `models` oraz `common/modele` są celowo
puste w repozytorium (`.gitignore`) — zawierają dane osobowe (zdjęcia twarzy)
i duże pliki binarne, których nie powinno się trzymać w Git.

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

### 5. Panel GUI (opcjonalnie)

Panel to natywne okno systemowe z interfejsem webowym w stylu AdminLTE
(Bootstrap 5). Uruchomienie:

```powershell
python gui/main.py
```

W oknie wybierasz kamerę, ustawiasz próg pewności i proporcję twarzy, a
podgląd pokazuje ramki i liczbę wykrytych twarzy. Panel korzysta z
`pywebview` i WebView2 (dołączony do Windows 10/11); pliki Bootstrapa leżą
w repo, więc internet nie jest potrzebny. Serwer działa wyłącznie na
`127.0.0.1`. Gdy WebView2 jest niedostępny, uruchom awaryjne GUI:

```powershell
python gui/tkinter_app.py
```

## Tutorial krok po kroku

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

### Krok 2 — podział na zbiór treningowy i walidacyjny

```powershell
python 02_podzial_danych.py
```

Domyślnie 80% zdjęć każdej osoby trafia do treningu, a 20% do walidacji
(używanej do sprawdzania, czy model się nie przeucza). Wynik trafia do
`data/processed/train/` i `data/processed/val/`.

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
`python 04_rozpoznawanie_na_zywo.py --korekta-progu -0.05`.

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

## Pomysły na rozszerzenie projektu (zadania dla uczniów)

- Zamienić kaskadę Haara na nowocześniejszy detektor (np. MediaPipe Face
  Detector) i porównać szybkość/skuteczność.
- Zamienić ogólny ekstraktor cech ImageNet na sieć wytrenowaną
  SPECJALNIE do rozpoznawania twarzy (np. FaceNet/ArcFace) — powinno dać
  jeszcze lepszą separację embeddingów niż obecne podejście.
  To bardziej zaawansowany, ale bardzo pouczający temat "one-shot/few-shot
  learning".
- Dodać zapis logów rozpoznań (kto i kiedy został rozpoznany) do pliku
  CSV/bazy danych — np. jako prosty system obecności na zajęciach.
- Poeksperymentować z inną architekturą backbone'u (np.
  `resnet18`, `efficientnet_b0`) i porównać dokładność/szybkość.
- Dodać wizualizację macierzy pomyłek (confusion matrix) na zbiorze
  walidacyjnym za pomocą `scikit-learn`.
- Zmierzyć wpływ liczby zdjęć treningowych na dokładność (np. wytrenować
  model na 50, 100, 200 zdjęciach na osobę i porównać wyniki).

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
