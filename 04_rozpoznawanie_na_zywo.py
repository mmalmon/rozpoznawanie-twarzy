"""
Krok 4: Rozpoznawanie twarzy na zywo z kamery.

Wymaga wczesniejszego zbudowania wzorcow tozsamosci (03_trenowanie_modelu.py).

Uzycie:
    python 04_rozpoznawanie_na_zywo.py
    python 04_rozpoznawanie_na_zywo.py --korekta-progu 0.05

DLA UCZNIOW - jak dziala rozpoznawanie w tej wersji projektu?

W odroznieniu od "klasycznej" klasyfikacji (siec zawsze wybiera
"najbardziej prawdopodobna" osobe sposrod znanych jej klas, nawet gdy w
kadrze jest ktos zupelnie inny), tutaj uzywamy podejscia zwanego
"rozpoznawaniem przez podobienstwo" (verification):
  1. Kazda wykryta twarz jest zamieniana na embedding - wektor kilkuset
     liczb opisujacy jej wyglad (uzywajac tej samej, zamrozonej sieci
     ImageNet, co przy budowaniu wzorcow w skrypcie 03).
  2. Ten embedding jest porownywany (podobienstwo kosinusowe) ze wzorcem
     (centroidem) kazdej znanej osoby, wczytanym z models/wzorce_osob.json.
  3. Jesli NAJWYZSZE znalezione podobienstwo przekracza wczesniej
     skalibrowany prog danej osoby - twarz zostaje rozpoznana jako ta
     osoba. Jesli nie przekracza progu ZADNEJ znanej osoby - etykieta
     "Nieznana osoba".

Kolory ramek:
  - ZIELONA + imie   -> podobienstwo do wzorca >= skalibrowanego progu,
  - CZERWONA + "Nieznana osoba" -> podobienstwo ponizej progu kazdej osoby.
"""

from __future__ import annotations

# argparse - obsluga argumentow uruchomieniowych (--kamera, --korekta-progu itd.).
import argparse
# json - do wczytania wzorcow tozsamosci (centroidow) i progow zapisanych
# przez skrypt 03 (wzorce_osob.json).
import json
# time - do mierzenia FPS (klatek na sekunde).
import time
# Path - obiektowa reprezentacja sciezek plikow/folderow.
from pathlib import Path

# cv2 - OpenCV: obsluga kamery, rysowanie ramek/tekstu, konwersja kolorow.
import cv2
# torch - glowny silnik PyTorch, tu uzywany do uruchomienia zamrozonego
# ekstraktora cech na obrazie z kamery (tzw. inferencja).
import torch
# transforms - przeksztalcenia obrazu (zmiana rozmiaru, normalizacja),
# musza byc identyczne jak przy liczeniu wzorcow w skrypcie 03.
from torchvision import transforms

from common.camera import otworz_kamere, wybierz_kamere_interaktywnie
from common.face_detector import DetektorTwarzy
from common.model import (
    ODCHYLENIE_IMAGENET,
    ROZMIAR_OBRAZU,
    SREDNIA_IMAGENET,
    oblicz_znormalizowane_cechy,
    pobierz_urzadzenie,
    zbuduj_ekstraktor_cech,
)

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
        "--korekta-progu",
        type=float,
        default=0.0,
        help=(
            "Dodatkowa korekta (+/-) do progu podobienstwa skalibrowanego w "
            "skrypcie 03 - przydatne do szybkiego strojenia bez ponownego "
            "liczenia wzorcow (dodatnia wartosc = trudniej o rozpoznanie)."
        ),
    )
    return parser.parse_args()


def wczytaj_wzorce(urzadzenie: torch.device) -> dict[str, dict]:
    """Wczytuje z dysku wzorce tozsamosci (centroidy) i progi zapisane
    przez 03_trenowanie_modelu.py, od razu zamieniajac centroidy z listy
    liczb (format JSON) z powrotem na tensory PyTorch gotowe do liczenia
    podobienstwa kosinusowego."""
    sciezka_wzorcow = FOLDER_MODELI / "wzorce_osob.json"
    if not sciezka_wzorcow.exists():
        raise SystemExit(
            "Brak wzorcow tozsamosci. Najpierw uruchom 03_trenowanie_modelu.py."
        )

    with open(sciezka_wzorcow, "r", encoding="utf-8") as plik:
        dane = json.load(plik)

    wzorce: dict[str, dict] = {}
    for nazwa_osoby, wpis in dane["osoby"].items():
        wzorce[nazwa_osoby] = {
            "centroid": torch.tensor(wpis["centroid"], device=urzadzenie),
            "prog": wpis["prog_podobienstwa"],
        }
    return wzorce


def rozpoznaj_twarz(
    embedding: torch.Tensor, wzorce: dict[str, dict], korekta_progu: float
) -> tuple[str, float]:
    """Porownuje embedding jednej wykrytej twarzy ze wzorcem KAZDEJ znanej
    osoby i zwraca (etykieta, podobienstwo) dla najlepszego dopasowania.

    DLA UCZNIOW: przegladamy wszystkie znane osoby, liczymy podobienstwo
    kosinusowe (zwykly iloczyn skalarny dwoch wektorow jednostkowych) do
    kazdej z nich, i wybieramy NAJLEPSZE dopasowanie. Dopiero to
    najlepsze dopasowanie porownujemy z jego (indywidualnie
    skalibrowanym!) progiem - rozne osoby moga miec rozne progi, jesli ich
    zdjecia treningowe byly mniej lub bardziej jednoznaczne.
    """
    najlepsza_nazwa = None
    najlepsze_podobienstwo = -1.0
    najlepszy_prog = 0.5

    for nazwa_osoby, wpis in wzorce.items():
        podobienstwo = torch.dot(embedding, wpis["centroid"]).item()
        if podobienstwo > najlepsze_podobienstwo:
            najlepsze_podobienstwo = podobienstwo
            najlepsza_nazwa = nazwa_osoby
            najlepszy_prog = wpis["prog"]

    if najlepsza_nazwa is not None and najlepsze_podobienstwo >= najlepszy_prog + korekta_progu:
        return najlepsza_nazwa, najlepsze_podobienstwo
    return "Nieznana osoba", najlepsze_podobienstwo


def main() -> None:
    argumenty = parsuj_argumenty()
    urzadzenie = pobierz_urzadzenie()

    wzorce = wczytaj_wzorce(urzadzenie)
    print(f"[info] Wczytano wzorce dla {len(wzorce)} osob: {list(wzorce.keys())}")

    ekstraktor = zbuduj_ekstraktor_cech().to(urzadzenie)
    detektor = DetektorTwarzy()

    # Przeksztalcenia musza byc IDENTYCZNE jak te uzyte przy liczeniu
    # wzorcow w kroku 3 (bez augmentacji losowych) - inaczej porownywalibysmy
    # jablka z gruszkami.
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

            # Odbicie lustrzane (patrz szczegolowy komentarz w
            # 00_test_detekcji.py) - dzieki niemu podglad na ekranie
            # zachowuje sie tak samo intuicyjnie jak w kazdej aplikacji do
            # wideorozmow (ruch w lewo = obraz w lewo).
            klatka = cv2.flip(klatka, 1)

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

                # oblicz_znormalizowane_cechy() sama w sobie uzywa juz
                # torch.no_grad() (patrz common/model.py) - nie trenujemy
                # niczego podczas rozpoznawania na zywo, wiec gradienty sa
                # tu zupelnie niepotrzebne.
                embedding = oblicz_znormalizowane_cechy(ekstraktor, tensor_wejsciowy)[0]

                etykieta_osoby, podobienstwo = rozpoznaj_twarz(
                    embedding, wzorce, argumenty.korekta_progu
                )
                if etykieta_osoby == "Nieznana osoba":
                    etykieta = f"{etykieta_osoby} ({podobienstwo * 100:.0f}%)"
                    kolor_ramki = KOLOR_NIEZNANY
                else:
                    etykieta = f"{etykieta_osoby} ({podobienstwo * 100:.0f}%)"
                    kolor_ramki = KOLOR_ROZPOZNANY

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
