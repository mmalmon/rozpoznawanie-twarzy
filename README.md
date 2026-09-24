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
cechy wizualne (krawędzie, kształty, tekstury).

Zamiast (jak w pierwszej wersji tego projektu) douczać nową warstwę
klasyfikującą "kto z tych N znanych osób jest na zdjęciu", używamy
podejścia **rozpoznawania przez podobieństwo** (verification), znacznie
odporniejszego na małe, niezbalansowane zbiory danych:

- Sieć (bez żadnego dotrenowywania — wagi zostają dokładnie takie, jak po
  treningu na ImageNet) zamienia każde zdjęcie twarzy na **embedding** —
  wektor kilkuset liczb opisujący jej wygląd.
- Dla każdej znanej osoby uśredniamy embeddingi wszystkich jej zdjęć
  treningowych, tworząc jej **wzorzec** (centroid).
- Podczas rozpoznawania na żywo liczymy embedding nowej twarzy i sprawdzamy
  jego **podobieństwo kosinusowe** do wzorca każdej znanej osoby.

Dzięki temu wystarczy 150–300 zdjęć na osobę, a "trening" (w praktyce:
liczenie embeddingów i kalibracja progu) zajmuje sekundy zamiast minut,
nawet na samym CPU.

> **Dlaczego nie zwykła klasyfikacja?** Pierwsza wersja tego projektu
> uczyła sieć rozróżniać "znaną osobę" od zdjęć klasy "nieznajomy" pobranych
> z internetu. W praktyce sieć nauczyła się rozróżniać *styl zdjęcia*
> (ostre zdjęcie z kamery vs. skompresowane zdjęcie stockowe), a nie same
> rysy twarzy — to tzw. **uczenie się skrótów** (*shortcut learning*), bardzo
> częsty i pouczający błąd w uczeniu maszynowym. Podejście oparte o
> embeddingi i próg podobieństwa nie ma tej wady, bo nigdy nie trenujemy
> żadnej wagi na parze (moja twarz vs. zdjęcia z internetu) — o dopasowaniu
> decyduje wyłącznie ogólna wiedza sieci o obrazach.

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
├── 00_test_detekcji.py          # krok 0: szybki test detekcji (bez rozpoznawania)
├── 01_zbieranie_danych.py       # krok 1: zbieranie zdjęć z kamery
├── 02_podzial_danych.py         # krok 2: podział na train/val
├── 03_trenowanie_modelu.py      # krok 3: wzorce tozsamosci (embeddingi) + kalibracja progu
├── 04_rozpoznawanie_na_zywo.py  # krok 4: rozpoznawanie na żywo z kamery
├── listuj_kamery.py             # pomocniczy: wypisuje dostępne kamery i ich rozdzielczości
├── pobierz_zdjecia_nieznajomych.py  # pomocniczy: pobiera zdjecia "obcych" osob do kalibracji progu
├── common/
│   ├── camera.py                # obsługa kamery (wybór, otwieranie, wymuszenie MJPG/4K)
│   ├── face_detector.py         # detekcja twarzy (kaskada Haara)
│   ├── imgio.py                 # zapis/odczyt zdjęć odporny na polskie znaki w ścieżce
│   └── model.py                 # definicja modelu (ekstraktor cech / embeddingi)
├── data/
│   ├── raw/<osoba>/             # surowe zdjęcia z kamery (per osoba)
│   └── processed/{train,val}/   # dane po podziale, gotowe do treningu
├── models/
│   ├── wzorce_osob.json         # wzorce (centroidy) tozsamosci + skalibrowane progi
│   └── kalibracja_progu.png     # wykres pomagajacy zrozumiec dobor progu
├── requirements.txt
└── README.md
```

Foldery `data/raw`, `data/processed` i `models` są celowo puste w
repozytorium (`.gitignore`) — zawierają dane osobowe (zdjęcia twarzy) i
duże pliki binarne, których nie powinno się trzymać w Git.

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

- Pojawi się podgląd z kamery z zieloną ramką wokół wykrytej twarzy.
- Naciśnij `SPACJĘ`, aby zapisać pojedyncze zdjęcie, albo `a`, aby włączyć
  automatyczne zapisywanie co ok. 0.3 sekundy.
- Podczas zbierania **ruszaj lekko głową** (patrz w bok, w górę, w dół),
  zmieniaj wyraz twarzy, jeśli to możliwe — poproś o zmianę oświetlenia.
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
   ImageNet, bez żadnego dotrenowywania).
2. Uśrednia embeddingi każdej znanej osoby w jeden **wzorzec** (centroid).
3. Na zbiorze walidacyjnym (w tym zdjęciach "nieznajomy", jeśli je zebrano)
   automatycznie **kalibruje próg podobieństwa** oddzielający "to ta osoba"
   od "to ktoś inny" — patrz wypisane w konsoli wartości i wykres
   `models/kalibracja_progu.png`.
4. Zapisuje wzorce i progi do `models/wzorce_osob.json`.

Ten krok trwa sekundy do (przy bardzo dużych zbiorach) kilkudziesięciu
sekund — w odróżnieniu od poprzedniego podejścia (klasyfikacja) nie ma tu
żadnej pętli uczenia, więc nie potrzeba GPU ani wielu epok. Zwróćcie uwagę
na komunikat `[uwaga] Slaba separowalnosc...`, jeśli się pojawi — oznacza
on, że zdjęcia jakiejś osoby są zbyt podobne do zdjęć "obcych" (np. za mało
zróżnicowane oświetlenie/kąty) i warto zebrać więcej danych.

### Krok 4 — rozpoznawanie na żywo

```powershell
python 04_rozpoznawanie_na_zywo.py --korekta-progu 0.05
```

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
- Jeśli detektor Haar łapie fragment twarzy (np. samo czoło) jako całą
  twarz, sprawdźcie `min_rozmiar_wzgledny` w `common/face_detector.py` —
  zwiększenie go wymusza większe (a więc bardziej kompletne) wykrycia.

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
