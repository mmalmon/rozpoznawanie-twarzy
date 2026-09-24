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


def pobierz_urzadzenie() -> torch.device:
    """Zwraca urzadzenie obliczeniowe, na ktorym powinien dzialac model:
    karte graficzna (GPU, jesli jest dostepna i obslugiwana przez CUDA) albo
    procesor (CPU) w przeciwnym razie.

    DLA UCZNIOW: trening sieci neuronowej to w gruncie rzeczy ogromna liczba
    mnozen macierzy - GPU jest zaprojektowane do wykonywania wielu takich
    mnozen rownolegle, dzieki czemu trening potrafi byc 10-50x szybszy niz
    na CPU.
    """
    if torch.cuda.is_available():
        return torch.device("cuda")
    print(
        "[model] UWAGA: nie wykryto GPU z obsluga CUDA - trening/inferencja "
        "beda dzialac na CPU i moga byc znacznie wolniejsze."
    )
    return torch.device("cpu")
