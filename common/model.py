"""
Definicja modelu do rozpoznawania twarzy (klasyfikacja obrazu twarzy -> osoba).

DLA UCZNIOW - czym jest "transfer learning" (uczenie transferowe)?

Zamiast trenowac siec neuronowa od zera (co wymagaloby milionow zdjec i
tygodni obliczen), bierzemy siec MobileNetV3-Small, ktora zostala juz
wczesniej wytrenowana przez kogos innego na ogromnym zbiorze ImageNet
(1000 klas ogolnych obiektow: koty, psy, samochody, naczynia, itd.). Taka
siec "nauczyla sie juz" wykrywac uniwersalne cechy wizualne - krawedzie,
ksztalty, tekstury, kolory - ktore przydaja sie przy rozpoznawaniu
praktycznie dowolnych obiektow, w tym twarzy.

Robimy z nia dwie rzeczy:
  1. Zamrazamy (nie trenujemy) wiekszosc jej warstw - tzw. "ekstraktor cech"
     (backbone) - zostawiajac je dokladnie takie, jakie byly po treningu na
     ImageNet.
  2. Podmieniamy jej ostatnia warstwe (klasyfikator) na nowa, z liczba wyjsc
     rowna liczbie osob w naszej bazie, i trenujemy TYLKO ta nowa warstwe na
     naszych zdjeciach.

Dzieki temu wystarczy 150-300 zdjec na osobe (zamiast milionow), a trening
zajmuje minuty zamiast dni, nawet na sredniej klasy laptopowym GPU.

UWAGA - DWA PODEJSCIA W TYM MODULE:
  - zbuduj_model() + odmroz_ekstraktor_cech() to podejscie "klasyczne"
    (klasyfikacja zamknieta, softmax) - zostawione tutaj jako material do
    eksperymentow dla zainteresowanych uczniow (patrz pomysly na
    rozszerzenie w README.md), ale skrypty 03/04 juz go NIE uzywaja.
  - zbuduj_ekstraktor_cech() + oblicz_znormalizowane_cechy() to aktualne
    podejscie uzywane w skryptach 03/04 - rozpoznawanie przez PODOBIENSTWO
    (embeddingi + prog kosinusowy), odporne na "uczenie sie skrotow"
    miedzy domenami zdjec. Szczegoly - patrz komentarz przy
    zbuduj_ekstraktor_cech() ponizej oraz naglowek 03_trenowanie_modelu.py.
"""

from __future__ import annotations

# torch to glowna biblioteka PyTorch - silnik do budowy i trenowania sieci
# neuronowych, dzialajacy zarowno na CPU, jak i na GPU (kartach graficznych
# obslugujacych CUDA, np. RTX PRO Blackwell).
import torch
# nn (od "neural network") to podmodul torch zawierajacy gotowe "klocki" do
# budowy sieci neuronowych - warstwy (np. nn.Linear), funkcje straty
# (np. nn.CrossEntropyLoss) itd.
from torch import nn
# functional (importowany jako "F" - powszechna konwencja) zawiera funkcje
# matematyczne uzywane w sieciach neuronowych - tutaj potrzebujemy
# F.normalize() do zamiany wektora "cech twarzy" na wektor jednostkowy
# (dlugosci 1), co jest niezbedne, zeby podobienstwo kosinusowe dzialalo
# poprawnie (patrz oblicz_znormalizowane_cechy ponizej).
from torch.nn import functional as F
# models to podmodul biblioteki torchvision z gotowymi, popularnymi
# architekturami sieci neuronowych do przetwarzania obrazu (w tym
# MobileNetV3-Small uzywana w tym projekcie), czesto razem z wagami
# wytrenowanymi wczesniej na zbiorze ImageNet.
from torchvision import models

# Standardowy rozmiar wejscia (szerokosc x wysokosc w pikselach) dla sieci
# trenowanych na zbiorze ImageNet - kazdy obraz musi zostac przeskalowany
# do tego rozmiaru, zanim trafi do sieci.
ROZMIAR_OBRAZU = 224

# Srednia i odchylenie standardowe jasnosci kanalow R, G, B, wyliczone na
# calym zbiorze ImageNet podczas oryginalnego treningu. Musimy znormalizowac
# nasze obrazy DOKLADNIE w ten sam sposob - siec "widziala" podczas treningu
# tylko dane w tym zakresie wartosci i inaczej znormalizowane obrazy
# zaburzylyby jej dzialanie.
SREDNIA_IMAGENET = [0.485, 0.456, 0.406]
ODCHYLENIE_IMAGENET = [0.229, 0.224, 0.225]


def zbuduj_model(
    liczba_klas: int, pretrenowany: bool = True, zamroz_ekstraktor_cech: bool = True
) -> nn.Module:
    """Tworzy model MobileNetV3-Small z podmienionym klasyfikatorem koncowym.

    liczba_klas: ile osob ma rozpoznawac model (tyle bedzie mial neuronow
        wyjsciowych w ostatniej warstwie).
    pretrenowany: czy zaladowac wagi wytrenowane wczesniej na ImageNet
        (prawie zawsze chcemy True - to sedno transfer learningu).
    zamroz_ekstraktor_cech: czy zablokowac trenowanie warstw ekstraktora cech
        (zostawiajac je takimi, jakie byly po treningu na ImageNet) i
        trenowac wylacznie nowy klasyfikator.
    """
    wagi = models.MobileNet_V3_Small_Weights.DEFAULT if pretrenowany else None
    # MobileNet_V3_Small_Weights.DEFAULT to obiekt opisujacy, ktore
    # konkretnie wagi pobrac (torchvision zrobi to automatycznie przy
    # pierwszym uruchomieniu i zapamieta je w pamieci podrecznej systemu -
    # kolejne uruchomienia beda juz szybsze, bo nie trzeba niczego pobierac
    # ponownie).
    model = models.mobilenet_v3_small(weights=wagi)

    if zamroz_ekstraktor_cech:
        # "Zamrazanie" oznacza ustawienie requires_grad = False - PyTorch nie
        # bedzie wtedy liczyl gradientow dla tych parametrow ani ich
        # aktualizowal podczas treningu. Przyspiesza to trening i zmniejsza
        # ryzyko przeuczenia przy malym zbiorze danych (kilkaset zdjec na osobe).
        for parametr in model.features.parameters():
            parametr.requires_grad = False

    # Ostatnia warstwa oryginalnej sieci ma 1000 wyjsc (tyle, ile klas w
    # ImageNet). Podmieniamy ja na nowa warstwe liniowa z liczba wyjsc rowna
    # liczbie osob, ktore chcemy rozpoznawac.
    liczba_cech_wejsciowych = model.classifier[-1].in_features
    model.classifier[-1] = nn.Linear(liczba_cech_wejsciowych, liczba_klas)

    return model


def odmroz_ekstraktor_cech(model: nn.Module, liczba_ostatnich_blokow: int = 3) -> None:
    """Odmraza (wlacza trenowanie) ostatnich N blokow ekstraktora cech -
    tzw. fine-tuning (douczanie) drugiego etapu.

    DLA UCZNIOW: wywolujemy to po kilku epokach treningu samego
    klasyfikatora. Pozwala to lekko doszkolic gorne (najbardziej
    wyspecjalizowane) warstwy sieci pod nasze konkretne twarze, zamiast
    polegac wylacznie na ogolnych cechach z ImageNet. Zwykle podnosi to
    dokladnosc o kilka punktow procentowych, kosztem nieco dluzszego treningu.
    """
    bloki = list(model.features.children())
    for blok in bloki[-liczba_ostatnich_blokow:]:
        for parametr in blok.parameters():
            parametr.requires_grad = True


def zbuduj_ekstraktor_cech() -> nn.Module:
    """Tworzy "goly" ekstraktor cech (embeddingow) twarzy - MobileNetV3-Small
    z wagami ImageNet, ale BEZ warstwy klasyfikujacej na koncu.

    DLA UCZNIOW - dlaczego w ogole potrzebujemy tej funkcji, skoro mamy juz
    zbuduj_model() powyzej?

    Pierwsza wersja tego projektu uczyla siec ROZPOZNAWAC (klasyfikowac)
    twarze metoda "kto z tych N znanych osob jest na zdjeciu" (softmax na
    koncu sieci). To podejscie ma powazna wade przy MALEJ liczbie znanych
    osob: siec musi jakos "wyobrazic sobie" wszystkich pozostalych ludzi na
    swiecie jako jedna, druga klase "nieznajomy". Jesli zdjecia tej klasy
    roznia sie od zdjec znanej osoby jakas przypadkowa cecha (np. ostroscia,
    kompresja, tlem) zamiast samymi rysami twarzy, siec nauczy sie tej
    LATWIEJSZEJ, przypadkowej cechy zamiast prawdziwej tozsamosci - to
    zjawisko nazywa sie "uczeniem sie skrotow" (shortcut learning) i wlasnie
    ono powodowalo, ze KAZDA nowa, prawdziwa twarz przed kamera byla
    rozpoznawana jako znana osoba (obie byly "ostrymi zdjeciami z tego
    samego pokoju", w odroznieniu od rozmazanych zdjec stockowych z
    internetu).

    ROZWIAZANIE - rozpoznawanie przez PODOBIENSTWO (verification), a nie
    KLASYFIKACJE: zamiast uczyc siec odrozniac "moja twarz" od "zdjec z
    internetu", bierzemy siec NIGDY nie trenowana na naszych konkretnych
    danych (czysto ImageNet) i uzywamy jej jako "generatora opisu wizualnego"
    (embeddingu) kazdego zdjecia twarzy - wektora kilkuset liczb opisujacego
    jego wyglad. Nastepnie:
      1. Dla znanej osoby liczymy embeddingi WSZYSTKICH jej zdjec
         treningowych i usredniamy je -> "wzorzec" (centroid) tej osoby.
      2. Podczas rozpoznawania na zywo liczymy embedding nowej twarzy i
         sprawdzamy, jak bardzo jest PODOBNY (podobienstwo kosinusowe) do
         wzorca kazdej znanej osoby.
      3. Jesli podobienstwo przekracza ustalony (skalibrowany) prog - to ta
         osoba. Jesli nie - "Nieznana osoba".

    Kluczowa zaleta: NIGDY nie trenujemy zadnej warstwy na parze
    (moja twarz vs internetowe zdjecia), wiec nie ma jak "nauczyc sie na
    skrotow" tej konkretnej, mylacej roznicy w stylu zdjec - embedding
    opiera sie wylacznie na ogolnej wiedzy sieci o tym, jak wygladaja
    obrazy (w tym twarze), wyniesionej z treningu na milionach zdjec
    ImageNet.
    """
    wagi = models.MobileNet_V3_Small_Weights.DEFAULT
    model = models.mobilenet_v3_small(weights=wagi)

    # model.features to wszystkie warstwy konwolucyjne (wyciaganie cech
    # wizualnych), model.avgpool usrednia mape cech do jednego wektora na
    # zdjecie (tzw. global average pooling), a nn.Flatten() "spopla" wynik
    # z ksztaltu [batch, kanaly, 1, 1] do prostego wektora [batch, kanaly].
    # Pomijamy CALKOWICIE oryginalny model.classifier (1000 klas ImageNet) -
    # nie jest nam do niczego potrzebny.
    ekstraktor = nn.Sequential(model.features, model.avgpool, nn.Flatten())

    # Zamrazamy WSZYSTKIE parametry - ten ekstraktor nigdy nie jest
    # trenowany (uczony), uzywamy go wylacznie do "odczytu" (inferencji).
    for parametr in ekstraktor.parameters():
        parametr.requires_grad = False
    ekstraktor.eval()

    return ekstraktor


def oblicz_znormalizowane_cechy(ekstraktor: nn.Module, tensor_wsadowy: torch.Tensor) -> torch.Tensor:
    """Przepuszcza paczke obrazow przez ekstraktor cech i zwraca ich
    embeddingi ZNORMALIZOWANE do dlugosci 1 (tzw. wektory jednostkowe).

    DLA UCZNIOW: normalizacja do dlugosci 1 jest niezbedna, zeby zwykly
    iloczyn skalarny dwoch wektorow (torch.dot / mnozenie macierzy) byl
    rowny ich PODOBIENSTWU KOSINUSOWEMU (cosine similarity) - liczbie z
    zakresu od -1 (przeciwne kierunki) do 1 (identyczny kierunek), niezaleznej
    od "dlugosci"/skali samych wektorow, a jedynie od kata miedzy nimi.
    Dzieki temu porownywanie "jak bardzo podobne sa te dwie twarze" sprowadza
    sie do prostego mnozenia i sprawdzenia, czy wynik przekracza prog.
    """
    with torch.no_grad():
        # torch.no_grad() - patrz wyjasnienie w 04_rozpoznawanie_na_zywo.py -
        # nie trenujemy tego modelu, wiec gradienty sa zbedne.
        cechy = ekstraktor(tensor_wsadowy)
    return F.normalize(cechy, p=2, dim=1)


def pobierz_urzadzenie() -> torch.device:
    """Zwraca urzadzenie obliczeniowe, na ktorym powinien dzialac model:
    karte graficzna (GPU, jesli jest dostepna i realnie obslugiwana przez
    zainstalowana wersje PyTorch) albo procesor (CPU) w przeciwnym razie.

    DLA UCZNIOW: trening sieci neuronowej to w gruncie rzeczy ogromna liczba
    mnozen macierzy - GPU jest zaprojektowane do wykonywania wielu takich
    mnozen rownolegle, dzieki czemu trening potrafi byc 10-50x szybszy niz
    na CPU.

    UWAGA TECHNICZNA: `torch.cuda.is_available()` sprawdza tylko, czy sterownik
    NVIDIA i biblioteka CUDA w ogole dzialaja - NIE sprawdza, czy zainstalowana
    "paczka" PyTorch zawiera skompilowane "kernele" (gotowe procedury
    obliczeniowe) dla konkretnego, czasem bardzo nowego, modelu karty
    graficznej (tak jak np. przy najnowszej architekturze Blackwell w chwili
    pisania tego kodu). Jesli kernele brakuja, `is_available()` nadal zwroci
    True, ale pierwsza proba wykonania obliczen na GPU zakonczy sie bledem
    w trakcie treningu. Dlatego ponizej dodatkowo wykonujemy proba malego,
    "testowego" mnozenia macierzy na GPU - dopiero jej powodzenie oznacza, ze
    GPU faktycznie mozna bezpiecznie uzyc.
    """
    if not torch.cuda.is_available():
        print(
            "[model] UWAGA: nie wykryto GPU z obsluga CUDA - trening/inferencja "
            "beda dzialac na CPU i moga byc znacznie wolniejsze."
        )
        return torch.device("cpu")

    try:
        # Male, testowe mnozenie macierzy 8x8 na GPU - jesli zainstalowana
        # wersja PyTorch nie ma kernelow dla tej karty graficznej, ten
        # wiersz zglosi wyjatek (RuntimeError), zamiast wywalic sie dopiero
        # w srodku dlugiego treningu.
        testowy_tensor = torch.rand(8, 8, device="cuda")
        _ = testowy_tensor @ testowy_tensor
        torch.cuda.synchronize()
    except RuntimeError as blad:
        print(
            "[model] UWAGA: wykryto GPU, ale zainstalowana wersja PyTorch nie "
            f"potrafi na nim liczyc ({blad}). "
            "Prawdopodobnie to bardzo nowy model karty graficznej, dla ktorego "
            "nie ma jeszcze gotowych 'kernelow' CUDA w tej wersji PyTorch. "
            "Sprawdz sekcje o instalacji PyTorch w README.md - przelaczam sie "
            "tymczasowo na CPU."
        )
        return torch.device("cpu")

    return torch.device("cuda")
