"""
Krok 4: Rozpoznawanie twarzy na zywo z kamery.

Wymaga wczesniejszego wytrenowania modelu (03_trenowanie_modelu.py).

Uzycie:
    python 04_rozpoznawanie_na_zywo.py
    python 04_rozpoznawanie_na_zywo.py --prog-pewnosci 0.75

DLA UCZNIOW - dlaczego potrzebny jest "prog pewnosci"?

Siec klasyfikujaca ZAWSZE zwroci "najbardziej prawdopodobna" osobe sposrod
tych, ktore widziala podczas treningu - nawet jesli w kadrze jest ktos
zupelnie inny, kogo model nigdy nie widzial! Dlatego oprocz samej
przewidywanej klasy patrzymy tez na PEWNOSC modelu (prawdopodobienstwo
z funkcji softmax). Jesli pewnosc jest ponizej ustalonego progu (domyslnie
60%), pokazujemy etykiete "Nieznana osoba" zamiast zgadywac. To bardzo
wazny mechanizm bezpieczenstwa w kazdym realnym systemie rozpoznawania
twarzy.

Kolory ramek:
  - ZIELONA + imie   -> pewnosc modelu >= progu pewnosci (osoba rozpoznana),
  - CZERWONA + "Nieznana osoba" -> pewnosc ponizej progu.
"""

from __future__ import annotations

# argparse - obsluga argumentow uruchomieniowych (--kamera, --prog-pewnosci itd.).
import argparse
# json - do wczytania listy nazw osob zapisanej przez skrypt 03 (klasy.json).
import json
# time - do mierzenia FPS (klatek na sekunde).
import time
# Path - obiektowa reprezentacja sciezek plikow/folderow.
from pathlib import Path

# cv2 - OpenCV: obsluga kamery, rysowanie ramek/tekstu, konwersja kolorow.
import cv2
# torch - glowny silnik PyTorch, tu uzywany do wczytania wag modelu i
# uruchomienia go na obrazie z kamery (tzw. inferencja).
import torch
# nn.functional (importowany jako "F" - to powszechna konwencja) zawiera
# funkcje matematyczne uzywane w sieciach neuronowych, tu: softmax do
# zamiany surowych wynikow sieci na prawdopodobienstwa.
import torch.nn.functional as F
# transforms - przeksztalcenia obrazu (zmiana rozmiaru, normalizacja),
# musza byc identyczne jak podczas treningu w skrypcie 03.
from torchvision import transforms

from common.camera import otworz_kamere, wybierz_kamere_interaktywnie
from common.face_detector import DetektorTwarzy
from common.model import ODCHYLENIE_IMAGENET, ROZMIAR_OBRAZU, SREDNIA_IMAGENET, pobierz_urzadzenie, zbuduj_model

FOLDER_MODELI = Path(__file__).parent / "models"

# Kolory ramek w formacie BGR (tak przechowuje kolory OpenCV - odwrotnie niz RGB).
KOLOR_ROZPOZNANY = (0, 200, 0)  # zielony
KOLOR_NIEZNANY = (0, 0, 220)  # czerwony


def parsuj_argumenty() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Rozpoznawanie twarzy na zywo.")
    parser.add_argument(
        "--kamera",
        type=int,
        default=None,
        help="Indeks kamery. Jesli pominiety, program zapyta interaktywnie.",
    )
    parser.add_argument("--szerokosc", type=int, default=1920)
    parser.add_argument("--wysokosc", type=int, default=1080)
    parser.add_argument(
        "--prog-pewnosci",
        type=float,
        default=0.6,
        help="Minimalna pewnosc (0-1), ponizej ktorej osoba jest oznaczana jako 'Nieznana'",
    )
    return parser.parse_args()


def wczytaj_model(urzadzenie: torch.device) -> tuple[torch.nn.Module, list[str]]:
    """Wczytuje wytrenowany model oraz liste rozpoznawanych osob z dysku."""
    sciezka_klas = FOLDER_MODELI / "klasy.json"
    sciezka_wag = FOLDER_MODELI / "model_twarzy.pt"

    if not sciezka_klas.exists() or not sciezka_wag.exists():
        raise SystemExit(
            "Brak wytrenowanego modelu. Najpierw uruchom 03_trenowanie_modelu.py."
        )

    with open(sciezka_klas, "r", encoding="utf-8") as plik:
        # json.load() odczytuje plik tekstowy w formacie JSON i zamienia go
        # z powrotem na obiekt Pythona (tu: liste napisow - nazw osob).
        klasy = json.load(plik)

    # pretrenowany=False, bo zaraz wczytamy WLASNE wagi (te wytrenowane w
    # skrypcie 03) - nie ma sensu najpierw pobierac oryginalnych wag
    # ImageNet, skoro i tak zostana one zaraz nadpisane.
    model = zbuduj_model(liczba_klas=len(klasy), pretrenowany=False)
    # torch.load wczytuje z dysku zapisany wczesniej "stan" modelu (wartosci
    # wszystkich jego wag) - map_location=urzadzenie gwarantuje, ze zadziala
    # to poprawnie niezaleznie od tego, czy model byl trenowany na GPU, a
    # teraz uruchamiamy go na CPU (lub odwrotnie).
    model.load_state_dict(torch.load(sciezka_wag, map_location=urzadzenie))
    model.to(urzadzenie)
    # Tryb .eval() wylacza mechanizmy uzywane tylko podczas treningu
    # (np. dropout) - podczas rozpoznawania na zywo chcemy zawsze
    # deterministycznego, "najlepszego" zachowania sieci.
    model.eval()

    return model, klasy


def main() -> None:
    argumenty = parsuj_argumenty()
    urzadzenie = pobierz_urzadzenie()

    model, klasy = wczytaj_model(urzadzenie)
    detektor = DetektorTwarzy()

    # Przeksztalcenia musza byc IDENTYCZNE jak te uzyte przy walidacji modelu
    # w kroku 3 (bez augmentacji losowych) - inaczej porownywalibysmy jablka
    # z gruszkami.
    przygotuj_obraz = transforms.Compose(
        [
            # ToPILImage zamienia tablice numpy (obraz z OpenCV) na obiekt
            # biblioteki Pillow (PIL) - reszta przeksztalcen torchvision
            # oczekuje wlasnie takiego formatu wejsciowego.
            transforms.ToPILImage(),
            transforms.Resize((ROZMIAR_OBRAZU, ROZMIAR_OBRAZU)),
            transforms.ToTensor(),
            transforms.Normalize(SREDNIA_IMAGENET, ODCHYLENIE_IMAGENET),
        ]
    )

    kamera = otworz_kamere(
        argumenty.kamera if argumenty.kamera is not None else wybierz_kamere_interaktywnie(),
        szerokosc=argumenty.szerokosc,
        wysokosc=argumenty.wysokosc,
    )

    print("[info] Nacisnij 'q' lub ESC, aby zakonczyc.")
    poprzedni_czas = time.time()

    try:
        while True:
            czy_odczytano, klatka = kamera.read()
            if not czy_odczytano:
                print("[blad] Nie udalo sie odczytac klatki z kamery.")
                break

            wykryte_twarze = detektor.wykryj(klatka)

            # Rozpoznawanie osob dzieje sie osobno dla kazdej wykrytej
            # twarzy w kadrze - jesli w kadrze jest kilka osob, kazda
            # zostanie sklasyfikowana niezaleznie.
            for twarz in wykryte_twarze:
                wycinek = twarz.wytnij(klatka, margines=0.2)
                if wycinek.size == 0:
                    # size == 0 oznacza pusty wycinek (np. wykrycie tuz przy
                    # samej krawedzi obrazu) - pomijamy taka twarz, zeby
                    # uniknac bledu przy dalszym przetwarzaniu.
                    continue

                # OpenCV przechowuje obrazy w formacie BGR, a sieci
                # neuronowe trenowane na ImageNet oczekuja formatu RGB -
                # trzeba zamienic kolejnosc kanalow przed podaniem do modelu.
                wycinek_rgb = cv2.cvtColor(wycinek, cv2.COLOR_BGR2RGB)
                # .unsqueeze(0) dodaje na poczatku dodatkowy, "sztuczny" wymiar
                # (tzw. wymiar paczki/batch) - model oczekuje wejscia w formacie
                # [liczba_zdjec, kanaly, wysokosc, szerokosc], a my mamy tylko
                # jedno pojedyncze zdjecie, wiec udajemy "paczke" o rozmiarze 1.
                tensor_wejsciowy = przygotuj_obraz(wycinek_rgb).unsqueeze(0).to(urzadzenie)

                # torch.no_grad() wylacza sledzenie gradientow - podczas
                # samego rozpoznawania (inferencji) nie trenujemy modelu,
                # wiec gradienty sa niepotrzebne i tylko zajmowalyby pamiec.
                with torch.no_grad():
                    surowe_wyniki = model(tensor_wejsciowy)
                    # Funkcja softmax zamienia surowe wyniki sieci (tzw.
                    # logity) na prawdopodobienstwa, ktore sumuja sie do 1.0
                    # - dzieki temu mozemy mowic o "pewnosci" w procentach.
                    # dim=1 mowi, ze normalizujemy wzdluz wymiaru klas (a nie
                    # wzdluz wymiaru paczki), a "[0]" wyciaga wynik dla
                    # jedynego zdjecia w naszej "paczce" o rozmiarze 1.
                    prawdopodobienstwa = F.softmax(surowe_wyniki, dim=1)[0]
                    # torch.max zwraca jednoczesnie najwieksza wartosc
                    # (pewnosc) oraz jej indeks (indeks_klasy) - czyli ktora
                    # osoba zostala uznana za "najbardziej prawdopodobna".
                    pewnosc, indeks_klasy = torch.max(prawdopodobienstwa, dim=0)

                # .item() wyciaga zwykla liczbe Pythona z jednoelementowego
                # tensora PyTorch - potrzebne do dalszych porownan/formatowania.
                pewnosc = pewnosc.item()
                if pewnosc >= argumenty.prog_pewnosci:
                    etykieta = f"{klasy[indeks_klasy.item()]} ({pewnosc * 100:.0f}%)"
                    kolor_ramki = KOLOR_ROZPOZNANY
                else:
                    etykieta = f"Nieznana osoba ({pewnosc * 100:.0f}%)"
                    kolor_ramki = KOLOR_NIEZNANY

                cv2.rectangle(
                    klatka,
                    (twarz.x, twarz.y),
                    (twarz.x + twarz.w, twarz.y + twarz.h),
                    kolor_ramki,
                    2,
                )
                cv2.putText(
                    klatka,
                    etykieta,
                    (twarz.x, max(0, twarz.y - 10)),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.7,
                    kolor_ramki,
                    2,
                )

            teraz = time.time()
            fps = 1.0 / max(1e-6, teraz - poprzedni_czas)
            poprzedni_czas = teraz
            cv2.putText(
                klatka,
                f"FPS: {fps:.1f}",
                (20, 40),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                (255, 255, 0),
                2,
            )

            cv2.imshow("Rozpoznawanie twarzy (q=koniec)", klatka)

            klawisz = cv2.waitKey(1) & 0xFF
            if klawisz in (ord("q"), 27):
                break
    finally:
        kamera.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
