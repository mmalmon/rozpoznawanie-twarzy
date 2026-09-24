"""
Krok 3: Trening modelu rozpoznawania twarzy (transfer learning, PyTorch).

DLA UCZNIOW - jak przebiega ten trening w dwoch etapach?

  ETAP 1 (trening klasyfikatora): ekstraktor cech (backbone) jest zamrozony,
    trenujemy tylko nowa, ostatnia warstwe siec. To szybkie i bezpieczne -
    male ryzyko przeuczenia, nawet przy niewielkiej liczbie zdjec.

  ETAP 2 (fine-tuning / douczanie): odmrazamy kilka ostatnich blokow
    ekstraktora cech i douczamy je z bardzo malym wspolczynnikiem uczenia
    (learning rate). To pozwala sieci lekko dostosowac sie do specyfiki
    naszych twarzy, ale robimy to ostroznie, zeby nie "zepsuc" wartosciowej
    wiedzy wyniesionej z treningu na ImageNet.

Wymaga wczesniejszego uruchomienia:
    01_zbieranie_danych.py (dla kazdej osoby)
    02_podzial_danych.py

Uzycie:
    python 03_trenowanie_modelu.py
    python 03_trenowanie_modelu.py --epoki 20 --batch-size 32

Wynik:
    models/model_twarzy.pt   - wagi wytrenowanego modelu
    models/klasy.json        - mapowanie indeks -> nazwa osoby
    models/krzywe_uczenia.png - wykres accuracy/loss w czasie
"""

from __future__ import annotations

# argparse - obsluga argumentow uruchomieniowych (--epoki, --batch-size itd.).
import argparse
# json - zapis/odczyt danych w formacie JSON (tu: listy nazw rozpoznawanych
# osob, zeby skrypt 04 wiedzial, ktory numer wyjscia sieci odpowiada ktorej
# osobie).
import json
# time - do mierzenia, ile trwala kazda epoka treningu.
import time
# Path - obiektowa reprezentacja sciezek plikow/folderow.
from pathlib import Path

# torch - glowny silnik PyTorch do obliczen na tensorach (tablicach
# wielowymiarowych) i budowy/trenowania sieci neuronowych.
import torch
# nn - "klocki" do budowy sieci (funkcje straty itp.).
# optim - podmodul z algorytmami optymalizacji (tu: Adam), czyli metodami
# aktualizacji wag sieci na podstawie wyliczonych gradientow.
from torch import nn, optim
# DataLoader automatycznie dzieli caly zbior danych na mniejsze paczki
# (batch) i w tle rownolegle wczytuje/przygotowuje kolejne paczki, zeby
# karta graficzna nigdy nie czekala bezczynnie na dane.
from torch.utils.data import DataLoader
# datasets.ImageFolder to gotowa klasa z torchvision, ktora automatycznie
# wczytuje zdjecia z folderow (klasa = nazwa podfolderu). transforms to
# zestaw gotowych przeksztalcen obrazu (zmiana rozmiaru, augmentacja,
# normalizacja) stosowanych przed podaniem zdjecia do sieci.
from torchvision import datasets, transforms

from common.model import (
    ODCHYLENIE_IMAGENET,
    ROZMIAR_OBRAZU,
    SREDNIA_IMAGENET,
    odmroz_ekstraktor_cech,
    pobierz_urzadzenie,
    zbuduj_model,
)

FOLDER_PRZETWORZONYCH_DANYCH = Path(__file__).parent / "data" / "processed"
FOLDER_MODELI = Path(__file__).parent / "models"


def parsuj_argumenty() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Trening modelu rozpoznawania twarzy.")
    parser.add_argument("--epoki", type=int, default=15, help="Liczba epok treningu klasyfikatora")
    parser.add_argument(
        "--epoki-finetuning", type=int, default=8, help="Liczba epok douczania (fine-tuningu) ekstraktora cech"
    )
    parser.add_argument("--batch-size", type=int, default=32, help="Ile zdjec naraz trafia do sieci podczas jednego kroku uczenia")
    parser.add_argument("--lr", type=float, default=1e-3, help="Wspolczynnik uczenia (learning rate) klasyfikatora")
    parser.add_argument(
        "--lr-finetuning", type=float, default=1e-4, help="Wspolczynnik uczenia dla etapu douczania (fine-tuningu)"
    )
    return parser.parse_args()


def przygotuj_zbiory_danych(rozmiar_batcha: int) -> tuple[DataLoader, DataLoader, list[str]]:
    """Przygotowuje ladowarki danych (DataLoader) dla zbioru treningowego
    i walidacyjnego, wraz z lista rozpoznawanych osob (klas)."""

    # DLA UCZNIOW - "augmentacja danych": sztucznie tworzymy nowe warianty
    # tego samego zdjecia (odbicie lustrzane, lekki obrot, zmiana jasnosci
    # i kontrastu). Dzieki temu ta sama twarz "wyglada" dla sieci troche
    # inaczej za kazdym razem, co uczy ja ignorowac drobne, nieistotne
    # roznice zamiast "zapamietywac" pojedyncze zdjecia. Augmentacje
    # stosujemy TYLKO na zbiorze treningowym - zbior walidacyjny musi
    # pozostac niezmieniony, bo sluzy do uczciwej oceny modelu.
    przeksztalcenia_treningowe = transforms.Compose(
        [
            # transforms.Compose laczy liste pojedynczych przeksztalcen w
            # jeden "potok" - kazde zdjecie przejdzie przez nie po kolei,
            # od gory do dolu.
            transforms.Resize((ROZMIAR_OBRAZU, ROZMIAR_OBRAZU)),
            # RandomHorizontalFlip - z prawdopodobienstwem 50% odbija zdjecie
            # w poziomie (lustrzane odbicie) - twarz wyglada naturalnie
            # zarowno w oryginale, jak i po takim odbiciu.
            transforms.RandomHorizontalFlip(p=0.5),
            # RandomRotation - losowo obraca zdjecie o maksymalnie 10 stopni
            # w dowolna strone, symulujac lekkie przechylenie glowy.
            transforms.RandomRotation(10),
            # ColorJitter - losowo zmienia jasnosc/kontrast/nasycenie
            # kolorow, symulujac rozne warunki oswietlenia w sali lekcyjnej.
            transforms.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.2),
            # ToTensor zamienia obraz (dotychczas w formacie PIL/numpy) na
            # tensor PyTorch - podstawowy typ danych, na ktorym dziala siec
            # neuronowa - i skaluje wartosci pikseli z zakresu 0-255 do 0-1.
            transforms.ToTensor(),
            # Normalize odejmuje srednia i dzieli przez odchylenie standardowe
            # dla kazdego kanalu koloru - patrz wyjasnienie w common/model.py
            # (musi byc identyczne jak podczas oryginalnego treningu na ImageNet).
            transforms.Normalize(SREDNIA_IMAGENET, ODCHYLENIE_IMAGENET),
        ]
    )
    przeksztalcenia_walidacyjne = transforms.Compose(
        [
            # Zbior walidacyjny NIE dostaje augmentacji (RandomHorizontalFlip,
            # RandomRotation, ColorJitter) - tylko przeskalowanie i normalizacje,
            # bo chcemy oceniac model na "czystych", niezmienionych zdjeciach.
            transforms.Resize((ROZMIAR_OBRAZU, ROZMIAR_OBRAZU)),
            transforms.ToTensor(),
            transforms.Normalize(SREDNIA_IMAGENET, ODCHYLENIE_IMAGENET),
        ]
    )

    # ImageFolder automatycznie tworzy klasy na podstawie nazw podfolderow -
    # np. data/processed/train/jan_kowalski/*.jpg -> klasa "jan_kowalski".
    # Parametr "transform" mowi, jaki potok przeksztalcen zastosowac do
    # kazdego wczytywanego zdjecia.
    zbior_treningowy = datasets.ImageFolder(
        FOLDER_PRZETWORZONYCH_DANYCH / "train", transform=przeksztalcenia_treningowe
    )
    zbior_walidacyjny = datasets.ImageFolder(
        FOLDER_PRZETWORZONYCH_DANYCH / "val", transform=przeksztalcenia_walidacyjne
    )

    # DataLoader dzieli caly zbior na paczki (batch) po "rozmiar_batcha"
    # zdjec i podaje je sieci po kolei. shuffle=True losowo miesza
    # kolejnosc zdjec w kazdej epoce treningowej (zeby siec nie uczyla sie
    # przypadkiem "kolejnosci" danych) - dla walidacji shuffle nie jest
    # potrzebne, bo tylko oceniamy model, nie trenujemy go. num_workers=2
    # oznacza, ze wczytywaniem/przygotowywaniem kolejnych paczek zajmuja sie
    # 2 dodatkowe procesy rownolegle, w tle, podczas gdy GPU liczy poprzednia paczke.
    ladowarka_treningowa = DataLoader(
        zbior_treningowy, batch_size=rozmiar_batcha, shuffle=True, num_workers=2
    )
    ladowarka_walidacyjna = DataLoader(
        zbior_walidacyjny, batch_size=rozmiar_batcha, shuffle=False, num_workers=2
    )

    return ladowarka_treningowa, ladowarka_walidacyjna, zbior_treningowy.classes


def wykonaj_epoke(
    model: nn.Module,
    ladowarka: DataLoader,
    urzadzenie: torch.device,
    funkcja_straty: nn.Module,
    optymalizator: optim.Optimizer | None,
) -> tuple[float, float]:
    """Wykonuje jeden pelny przebieg (epoke) po podanym zbiorze danych.

    Jesli podano optymalizator, jest to epoka TRENINGOWA (model uczy sie -
    aktualizuje swoje wagi). Jesli optymalizator to None, jest to epoka
    WALIDACYJNA (model tylko ocenia dane, bez uczenia sie na nich).

    Zwraca srednia wartosc funkcji straty (loss) oraz dokladnosc (accuracy)
    na calym zbiorze.
    """
    czy_trening = optymalizator is not None
    # model.train(True/False) przelacza siec miedzy trybem treningowym a
    # ewaluacyjnym - wplywa to na warstwy takie jak Dropout czy BatchNorm,
    # ktore powinny zachowywac sie inaczej podczas uczenia niz podczas
    # zwyklego uzywania/oceny modelu.
    model.train(czy_trening)

    suma_straty, liczba_trafien, liczba_probek = 0.0, 0, 0

    # torch.set_grad_enabled(False) podczas walidacji wylacza liczenie
    # gradientow - nie sa one potrzebne, gdy tylko oceniamy model, a ich
    # pominiecie oszczedza pamiec i przyspiesza obliczenia.
    with torch.set_grad_enabled(czy_trening):
        # Petla po kolejnych paczkach (batch) danych z ladowarki - kazda
        # paczka to para: "obrazy" (tensor kilkudziesieciu zdjec naraz) i
        # "etykiety" (numery klas/osob, do ktorych te zdjecia naleza).
        for obrazy, etykiety in ladowarka:
            # .to(urzadzenie) przenosi dane na wybrane urzadzenie
            # obliczeniowe (GPU albo CPU) - obliczenia moga sie odbywac
            # tylko wtedy, gdy model i dane znajduja sie na tym samym
            # urzadzeniu.
            obrazy, etykiety = obrazy.to(urzadzenie), etykiety.to(urzadzenie)

            # "Przejscie w przod" (forward pass) - podajemy obrazy do sieci
            # i dostajemy jej surowe przewidywania (logity) dla kazdej klasy.
            wyniki = model(obrazy)
            strata = funkcja_straty(wyniki, etykiety)

            if czy_trening:
                # Klasyczny "krok" uczenia sieci neuronowej metoda propagacji
                # wstecznej (backpropagation):
                optymalizator.zero_grad()  # 1. wyzeruj gradienty z poprzedniego kroku
                strata.backward()  # 2. wylicz gradienty (jak zmienic wagi, by zmniejszyc strate)
                optymalizator.step()  # 3. zaktualizuj wagi modelu

            suma_straty += strata.item() * obrazy.size(0)
            liczba_trafien += (wyniki.argmax(1) == etykiety).sum().item()
            liczba_probek += obrazy.size(0)

    return suma_straty / liczba_probek, liczba_trafien / liczba_probek


def main() -> None:
    argumenty = parsuj_argumenty()
    FOLDER_MODELI.mkdir(parents=True, exist_ok=True)

    if not (FOLDER_PRZETWORZONYCH_DANYCH / "train").exists():
        raise SystemExit(
            f"Brak danych w {FOLDER_PRZETWORZONYCH_DANYCH}. Najpierw uruchom 02_podzial_danych.py."
        )

    urzadzenie = pobierz_urzadzenie()
    ladowarka_treningowa, ladowarka_walidacyjna, klasy = przygotuj_zbiory_danych(argumenty.batch_size)
    print(f"[info] Klasy ({len(klasy)}): {klasy}")

    model = zbuduj_model(liczba_klas=len(klasy), pretrenowany=True, zamroz_ekstraktor_cech=True)
    model.to(urzadzenie)

    # CrossEntropyLoss to standardowa funkcja straty dla klasyfikacji
    # wieloklasowej - kara model tym mocniej, im bardziej byl "pewny"
    # blednej odpowiedzi.
    funkcja_straty = nn.CrossEntropyLoss()

    historia = {"strata_trening": [], "trafnosc_trening": [], "strata_walidacja": [], "trafnosc_walidacja": []}
    najlepsza_trafnosc_walidacyjna = 0.0

    # --- Etap 1: trening samego klasyfikatora (ekstraktor cech zamrozony) ---
    # optim.Adam to popularny algorytm optymalizacji (aktualizacji wag sieci)
    # - automatycznie dostosowuje "krok" uczenia dla kazdego parametru osobno,
    # co zwykle daje szybsza i stabilniejsza zbieznosc niz prostszy SGD.
    # Wyrazenie generatorowe "(parametr for parametr in model.parameters()
    # if parametr.requires_grad)" przekazuje optymalizatorowi TYLKO te
    # parametry, ktore nie sa zamrozone (patrz common/model.py) - dzieki
    # temu optymalizator w ogole nie probuje aktualizowac zamrozonego
    # ekstraktora cech.
    optymalizator = optim.Adam(
        (parametr for parametr in model.parameters() if parametr.requires_grad), lr=argumenty.lr
    )
    print("\n[etap 1/2] Trening klasyfikatora (ekstraktor cech zamrozony)")
    for numer_epoki in range(1, argumenty.epoki + 1):
        czas_startu = time.time()
        strata_trening, trafnosc_trening = wykonaj_epoke(
            model, ladowarka_treningowa, urzadzenie, funkcja_straty, optymalizator
        )
        strata_walidacja, trafnosc_walidacja = wykonaj_epoke(
            model, ladowarka_walidacyjna, urzadzenie, funkcja_straty, None
        )
        czas_trwania = time.time() - czas_startu

        historia["strata_trening"].append(strata_trening)
        historia["trafnosc_trening"].append(trafnosc_trening)
        historia["strata_walidacja"].append(strata_walidacja)
        historia["trafnosc_walidacja"].append(trafnosc_walidacja)

        print(
            f"epoka {numer_epoki:02d}/{argumenty.epoki} | "
            f"strata_trening={strata_trening:.3f} trafnosc_trening={trafnosc_trening:.3f} | "
            f"strata_walidacja={strata_walidacja:.3f} trafnosc_walidacja={trafnosc_walidacja:.3f} | "
            f"{czas_trwania:.1f}s"
        )

        # Zapisujemy model TYLKO wtedy, gdy poprawil sie wynik na zbiorze
        # walidacyjnym - to gwarantuje, ze na koncu zostanie nam najlepsza
        # (a nie ostatnia) wersja modelu, nawet jesli pozniej zaczalby sie
        # przeuczac.
        if trafnosc_walidacja > najlepsza_trafnosc_walidacyjna:
            najlepsza_trafnosc_walidacyjna = trafnosc_walidacja
            torch.save(model.state_dict(), FOLDER_MODELI / "model_twarzy.pt")

    # --- Etap 2: fine-tuning (douczanie) ostatnich blokow ekstraktora cech ---
    if argumenty.epoki_finetuning > 0:
        print("\n[etap 2/2] Fine-tuning (douczanie) ostatnich warstw ekstraktora cech")
        odmroz_ekstraktor_cech(model, liczba_ostatnich_blokow=3)
        optymalizator = optim.Adam(
            (parametr for parametr in model.parameters() if parametr.requires_grad),
            lr=argumenty.lr_finetuning,
        )

        for numer_epoki in range(1, argumenty.epoki_finetuning + 1):
            czas_startu = time.time()
            strata_trening, trafnosc_trening = wykonaj_epoke(
                model, ladowarka_treningowa, urzadzenie, funkcja_straty, optymalizator
            )
            strata_walidacja, trafnosc_walidacja = wykonaj_epoke(
                model, ladowarka_walidacyjna, urzadzenie, funkcja_straty, None
            )
            czas_trwania = time.time() - czas_startu

            historia["strata_trening"].append(strata_trening)
            historia["trafnosc_trening"].append(trafnosc_trening)
            historia["strata_walidacja"].append(strata_walidacja)
            historia["trafnosc_walidacja"].append(trafnosc_walidacja)

            print(
                f"epoka ft {numer_epoki:02d}/{argumenty.epoki_finetuning} | "
                f"strata_trening={strata_trening:.3f} trafnosc_trening={trafnosc_trening:.3f} | "
                f"strata_walidacja={strata_walidacja:.3f} trafnosc_walidacja={trafnosc_walidacja:.3f} | "
                f"{czas_trwania:.1f}s"
            )

            if trafnosc_walidacja > najlepsza_trafnosc_walidacyjna:
                najlepsza_trafnosc_walidacyjna = trafnosc_walidacja
                torch.save(model.state_dict(), FOLDER_MODELI / "model_twarzy.pt")

    # Zapisujemy liste klas (nazw osob) w tej samej kolejnosci, w jakiej
    # widzial je model - bez tego nie moglibysmy pozniej odczytac, ktory
    # numer wyjscia sieci odpowiada ktorej osobie.
    with open(FOLDER_MODELI / "klasy.json", "w", encoding="utf-8") as plik:
        json.dump(klasy, plik, ensure_ascii=False, indent=2)

    print(f"\n[gotowe] Najlepsza dokladnosc walidacyjna: {najlepsza_trafnosc_walidacyjna:.3f}")
    print(f"[gotowe] Model zapisany w: {FOLDER_MODELI / 'model_twarzy.pt'}")

    # Rysujemy wykres krzywych uczenia - bardzo przydatny do dydaktyki:
    # uczniowie moga na wlasne oczy zobaczyc, jak wyglada przeuczenie
    # (train_acc rosnie, val_acc przestaje rosnac albo spada).
    try:
        import matplotlib.pyplot as plt

        zakres_epok = range(1, len(historia["trafnosc_trening"]) + 1)
        rysunek, osie = plt.subplots(1, 2, figsize=(10, 4))

        osie[0].plot(zakres_epok, historia["strata_trening"], label="trening")
        osie[0].plot(zakres_epok, historia["strata_walidacja"], label="walidacja")
        osie[0].set_title("Funkcja straty (loss)")
        osie[0].set_xlabel("epoka")
        osie[0].legend()

        osie[1].plot(zakres_epok, historia["trafnosc_trening"], label="trening")
        osie[1].plot(zakres_epok, historia["trafnosc_walidacja"], label="walidacja")
        osie[1].set_title("Dokladnosc (accuracy)")
        osie[1].set_xlabel("epoka")
        osie[1].legend()

        rysunek.tight_layout()
        rysunek.savefig(FOLDER_MODELI / "krzywe_uczenia.png")
        print(f"[gotowe] Wykres krzywych uczenia: {FOLDER_MODELI / 'krzywe_uczenia.png'}")
    except ImportError:
        print("[uwaga] matplotlib niedostepny - pomijam wykres krzywych uczenia.")


if __name__ == "__main__":
    main()
