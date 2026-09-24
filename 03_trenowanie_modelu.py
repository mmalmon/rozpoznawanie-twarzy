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

import argparse
import json
import time
from pathlib import Path

import torch
from torch import nn, optim
from torch.utils.data import DataLoader
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
            transforms.Resize((ROZMIAR_OBRAZU, ROZMIAR_OBRAZU)),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomRotation(10),
            transforms.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.2),
            transforms.ToTensor(),
            transforms.Normalize(SREDNIA_IMAGENET, ODCHYLENIE_IMAGENET),
        ]
    )
    przeksztalcenia_walidacyjne = transforms.Compose(
        [
            transforms.Resize((ROZMIAR_OBRAZU, ROZMIAR_OBRAZU)),
            transforms.ToTensor(),
            transforms.Normalize(SREDNIA_IMAGENET, ODCHYLENIE_IMAGENET),
        ]
    )

    # ImageFolder automatycznie tworzy klasy na podstawie nazw podfolderow -
    # np. data/processed/train/jan_kowalski/*.jpg -> klasa "jan_kowalski".
    zbior_treningowy = datasets.ImageFolder(
        FOLDER_PRZETWORZONYCH_DANYCH / "train", transform=przeksztalcenia_treningowe
    )
    zbior_walidacyjny = datasets.ImageFolder(
        FOLDER_PRZETWORZONYCH_DANYCH / "val", transform=przeksztalcenia_walidacyjne
    )

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
    model.train(czy_trening)

    suma_straty, liczba_trafien, liczba_probek = 0.0, 0, 0

    # torch.set_grad_enabled(False) podczas walidacji wylacza liczenie
    # gradientow - nie sa one potrzebne, gdy tylko oceniamy model, a ich
    # pominiecie oszczedza pamiec i przyspiesza obliczenia.
    with torch.set_grad_enabled(czy_trening):
        for obrazy, etykiety in ladowarka:
            obrazy, etykiety = obrazy.to(urzadzenie), etykiety.to(urzadzenie)

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
