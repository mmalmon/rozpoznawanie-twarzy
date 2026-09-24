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

## Jak to działa (teoria dla uczniów)

System składa się z dwóch niezależnych etapów, które warto odróżniać:

1. **Detekcja twarzy** — znalezienie *gdzie* na obrazie jest twarz (bez
   rozpoznawania *kogo*). Używamy klasycznej kaskady Haara z OpenCV
   (algorytm Violi-Jonesa) — jest szybka, nie wymaga GPU i świetnie nadaje
   się do tłumaczenia na lekcjach (cechy Haara + boosting + okno
   przesuwne po obrazie w wielu skalach).
2. **Rozpoznawanie (klasyfikacja) twarzy** — mając już wycięty fragment
   obrazu z twarzą, sieć neuronowa decyduje, **do kogo** ona należy.
   To właśnie ten etap trenujemy sami.

Do klasyfikacji używamy metody **transfer learning**: zamiast trenować sieć
od zera (co wymagałoby milionów zdjęć), bierzemy sieć **MobileNetV3-Small**
wytrenowaną wcześniej na dużym zbiorze ImageNet (1000 klas ogólnych
obiektów: koty, samochody, itd.). Taka sieć "umie już" wyciągać uniwersalne
cechy wizualne (krawędzie, kształty, tekstury). My tylko:

- zamrażamy (nie trenujemy) większość jej warstw,
- podmieniamy jej ostatnią warstwę na nową, z liczbą wyjść równą liczbie
  osób w naszej bazie,
- trenujemy tę nową warstwę na naszych zdjęciach (etap 1),
- na koniec lekko doszkalamy też kilka ostatnich warstw oryginalnej sieci
  (etap 2, tzw. *fine-tuning*), żeby dopasować ją dokładniej do specyfiki
  twarzy.

Dzięki temu wystarczy 150–300 zdjęć na osobę, a trening zajmuje minuty
zamiast dni.

Bardzo ważny jest też **próg pewności** przy rozpoznawaniu: sieć zawsze
zwróci "najbardziej prawdopodobną" znaną jej osobę, nawet jeśli w kadrze
jest ktoś zupełnie inny. Dlatego jeśli pewność klasyfikacji jest niska
(poniżej ustawionego progu, domyślnie 60%), program pokazuje etykietę
"Nieznana osoba" zamiast zgadywać.

## Wymagania sprzętowe i sprzęt użyty w projekcie

- **GPU:** NVIDIA RTX PRO 500 (architektura Blackwell), 6 GB VRAM.
  MobileNetV3-Small to bardzo mały model — 6 GB VRAM to wielokrotność tego,
  czego potrzeba do treningu z tego tutorialu (batch size 32, obrazy
  224×224). To dobry sprzęt dydaktyczny: trening całego modelu zajmuje
  zwykle 2–5 minut na kilka osób.
- **Kamera:** RoWave RC16, USB UVC 4K. W skryptach domyślnie ustawiamy
  podgląd na 1920×1080 (Full HD) — do detekcji i rozpoznawania twarzy pełne
  4K nie jest potrzebne, a mniejsza rozdzielczość oznacza więcej klatek na
  sekundę. Kod wymusza kodek `MJPG`, bo bez tego niektóre kamery UVC 4K
  potrafią działać w trybie zaledwie kilku klatek na sekundę.
- **System:** Windows 10/11.
- **Python:** 3.10–3.13.

> Uwaga o architekturze Blackwell: to bardzo nowa generacja GPU. Jeśli
> stabilna wersja PyTorch z `pip install torch` nie wykrywa GPU
> (`torch.cuda.is_available()` zwraca `False`) lub zgłasza błąd o
> nieobsługiwanej "compute capability", zainstaluj najnowszą wersję nightly
> zgodnie z instrukcją w sekcji [Instalacja środowiska](#instalacja-środowiska).

## Struktura projektu

```
rozpoznawanie twarzy/
├── 01_zbieranie_danych.py       # krok 1: zbieranie zdjęć z kamery
├── 02_podzial_danych.py         # krok 2: podział na train/val
├── 03_trenowanie_modelu.py      # krok 3: trening sieci (transfer learning)
├── 04_rozpoznawanie_na_zywo.py  # krok 4: rozpoznawanie na żywo z kamery
├── common/
│   ├── camera.py                # obsługa kamery RoWave RC16 (4K, UVC)
│   ├── face_detector.py         # detekcja twarzy (kaskada Haara)
│   └── model.py                 # definicja modelu (transfer learning)
├── data/
│   ├── raw/<osoba>/             # surowe zdjęcia z kamery (per osoba)
│   └── processed/{train,val}/   # dane po podziale, gotowe do treningu
├── models/
│   ├── model_twarzy.pt          # wagi wytrenowanego modelu
│   ├── klasy.json               # lista rozpoznawanych osób
│   └── krzywe_uczenia.png       # wykres accuracy/loss
├── requirements.txt
└── README.md
```

Foldery `data/raw`, `data/processed` i `models` są celowo puste w
repozytorium (`.gitignore`) — zawierają dane osobowe (zdjęcia twarzy) i
duże pliki binarne, których nie powinno się trzymać w Git.

## Instalacja środowiska

### 1. Utwórz wirtualne środowisko

```powershell
cd "rozpoznawanie twarzy"
python -m venv .venv
.venv\Scripts\Activate.ps1
```

### 2. Instalacja PyTorch z obsługą CUDA (GPU)

Sprzęt z GPU Blackwell wymaga **aktualnej** wersji PyTorch z obsługą CUDA
12.x. Zalecany sposób — użyć konfiguratora na stronie PyTorch
(https://pytorch.org/get-started/locally/), wybierając: *Stable*, *Windows*,
*Pip*, *Python*, *CUDA 12.x*. Przykładowa komenda (sprawdź na stronie
aktualny numer wersji CUDA):

```powershell
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu126
```

Jeśli po instalacji `torch.cuda.is_available()` zwraca `False` albo pojawia
się błąd o nieobsługiwanym "sm_120"/"compute capability", oznacza to, że
stabilna wersja jeszcze nie wspiera Twojego GPU — zainstaluj wersję nightly:

```powershell
pip uninstall torch torchvision -y
pip install --pre torch torchvision --index-url https://download.pytorch.org/whl/nightly/cu126
```

Sprawdź, czy GPU zostało wykryte:

```powershell
python -c "import torch; print(torch.__version__, torch.cuda.is_available(), torch.cuda.get_device_name(0) if torch.cuda.is_available() else '')"
```

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

## Tutorial krok po kroku

### Krok 1 — zbieranie danych treningowych

Dla **każdej osoby**, którą system ma rozpoznawać, uruchom:

```powershell
python 01_zbieranie_danych.py --osoba jan_kowalski --cel 250
```

- Pojawi się podgląd z kamery z zieloną ramką wokół wykrytej twarzy.
- Naciśnij `SPACJĘ`, aby zapisać pojedyncze zdjęcie, albo `a`, aby włączyć
  automatyczne zapisywanie co ok. 0.3 sekundy.
- Podczas zbierania **ruszaj lekko głową** (patrz w bok, w górę, w dół),
  zmieniaj wyraz twarzy, jeśli to możliwe — poproś o zmianę oświetlenia.
- Zdjęcia trafiają do `data/raw/jan_kowalski/`.
- Powtórz dla każdej osoby (np. wszystkich uczniów w grupie).

> Wskazówka dydaktyczna: warto też zebrać folder osoby "obcy"/"nieznany" ze
> zdjęciami osób spoza grupy (np. z internetu, za zgodą, albo zdjęcia
> nauczyciela) — pomaga to ocenić, jak model radzi sobie z progiem pewności.

### Krok 2 — podział na zbiór treningowy i walidacyjny

```powershell
python 02_podzial_danych.py
```

Domyślnie 80% zdjęć każdej osoby trafia do treningu, a 20% do walidacji
(używanej do sprawdzania, czy model się nie przeucza). Wynik trafia do
`data/processed/train/` i `data/processed/val/`.

### Krok 3 — trening modelu

```powershell
python 03_trenowanie_modelu.py --epoki 15 --epoki-finetuning 8
```

Skrypt:

1. Trenuje nowy klasyfikator na zamrożonym backbone'ie (szybkie epoki).
2. Odmraża kilka ostatnich warstw sieci i dotrenowuje je (fine-tuning) z
   mniejszym learning rate.
3. Zapisuje najlepszy (wg dokładności walidacyjnej) model do
   `models/model_twarzy.pt`, listę klas do `models/klasy.json` oraz wykres
   krzywych uczenia do `models/krzywe_uczenia.png`.

Na RTX PRO 500 (Blackwell) trening dla kilku osób (kilkaset zdjęć łącznie)
powinien zająć od ok. 1 do kilku minut. Obserwujcie na wykresie, czy
`val_acc` rośnie razem z `train_acc` (dobrze) czy zaczyna spadać, gdy
`train_acc` dalej rośnie (przeuczenie — warto wtedy zebrać więcej danych
albo skrócić trening).

### Krok 4 — rozpoznawanie na żywo

```powershell
python 04_rozpoznawanie_na_zywo.py --prog-pewnosci 0.6
```

Otworzy się podgląd z kamery z ramkami i etykietami rozpoznanych osób wraz
z procentową pewnością. Naciśnij `q`, aby zakończyć.

## Jak przygotować dobre materiały treningowe

Jakość danych ma większy wpływ na wynik niż wybór architektury sieci.
Zalecenia:

- **Ilość:** minimum 100–150 zdjęć na osobę, zalecane 200–300.
- **Różnorodność ustawienia głowy:** patrzenie prosto, lekko w lewo/prawo,
  lekko w górę/dół — sieć musi nauczyć się rozpoznawać osobę z różnych
  kątów, nie tylko "en face".
- **Różnorodność oświetlenia:** zbierz dane w różnych porach dnia / przy
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

**`torch.cuda.is_available()` zwraca `False`.**
Zainstaluj wersję nightly PyTorch (patrz sekcja instalacji) — architektura
Blackwell może nie być jeszcze wspierana przez aktualną wersję stabilną w
momencie, gdy z tego korzystacie. Program i tak zadziała na CPU, tylko
wolniej.

**Kamera nie otwiera się / błąd `Nie udalo sie otworzyc kamery`.**
Sprawdź, czy żadna inna aplikacja (Kamera, Teams, Zoom, przeglądarka z
otwartą stroną wideokonferencji) nie korzysta z kamery. Spróbuj innego
portu USB (najlepiej USB 3.0, kolor niebieski). Jeśli w systemie jest
więcej niż jedna kamera, zmień `--kamera 1`, `--kamera 2` itd.

**Bardzo niskie FPS podczas podglądu.**
Upewnij się, że kamera pracuje w trybie MJPG (skrypt ustawia to
automatycznie) i że USB jest podłączone do portu 3.0. Możesz też zmniejszyć
rozdzielczość podglądu, np. `--szerokosc 1280 --wysokosc 720`.

**Model myli osoby / dużo "Nieznana osoba" dla znanych osób.**
Zbierz więcej i bardziej zróżnicowanych danych treningowych (patrz sekcja
wyżej), wydłuż trening (`--epoki`, `--epoki-finetuning`) albo obniż
`--prog-pewnosci` w kroku 4 (kosztem większego ryzyka pomyłek).

**Model bardzo pewnie (>90%) rozpoznaje osobę, której nie ma w kadrze.**
To spodziewane zachowanie sieci klasyfikującej wśród znanych jej klas —
dlatego właśnie stosujemy próg pewności. Rozważcie też dodanie klasy
"obcy"/"nieznany" z różnorodnymi zdjęciami innych osób podczas zbierania
danych, żeby model miał szansę nauczyć się jej rozpoznawać.

## Pomysły na rozszerzenie projektu (zadania dla uczniów)

- Zamienić kaskadę Haara na nowocześniejszy detektor (np. MediaPipe Face
  Detector) i porównać szybkość/skuteczność.
- Zamiast klasyfikacji (stała lista osób) zaimplementować podejście oparte
  o **embeddingi** (np. wytrenować/wykorzystać sieć FaceNet/ArcFace) —
  pozwala dodawać nowe osoby bez ponownego treningu całej sieci.
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
