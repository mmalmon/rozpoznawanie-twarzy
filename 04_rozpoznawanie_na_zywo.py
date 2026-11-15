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
  2. Dwa kadry tej samej twarzy sa porownywane z embeddingami zdjec kazdej
     znanej osoby; wygrywa najlepsze dopasowanie z models/wzorce_osob.json.
  3. Jesli NAJWYZSZE znalezione podobienstwo przekracza wczesniej
     skalibrowany prog danej osoby - twarz zostaje rozpoznana jako ta
     osoba. Jesli nie przekracza progu ZADNEJ znanej osoby - etykieta
     "Nieznana osoba".

Kolory ramek:
  - ZIELONA + imie   -> podobienstwo do wzorca >= skalibrowanego progu,
  - CZERWONA + "Nieznana osoba" -> podobienstwo ponizej progu kazdej osoby.
"""

from __future__ import annotations

# Ta linijka MUSI byc pierwszym kodem wykonywalnym w tym pliku - patrz
# szczegolowe wyjasnienie w common/uruchom_w_venv.py (automatyczne
# przelaczenie na Python z .venv, jesli uruchomiono innym interpreterem).
from common.uruchom_w_venv import przelacz_na_venv_jesli_trzeba

przelacz_na_venv_jesli_trzeba()

# argparse - obsluga argumentow uruchomieniowych (--kamera, --korekta-progu itd.).
import argparse
from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor
# json - do wczytania wzorcow tozsamosci i progow zapisanych
# przez skrypt 03 (wzorce_osob.json).
import json
# sys - sprawdzenie, czy program dziala w interaktywnym terminalu (przed
# pytaniem o usuniecie danych osoby po zakonczeniu podgladu).
import sys
# time - do mierzenia FPS (klatek na sekunde).
import time
# Path - obiektowa reprezentacja sciezek plikow/folderow.
from pathlib import Path

# cv2 - OpenCV: obsluga kamery, rysowanie ramek/tekstu, konwersja kolorow.
import cv2
import numpy as np
# torch - glowny silnik PyTorch, tu uzywany do uruchomienia zamrozonego
# ekstraktora cech na obrazie z kamery (tzw. inferencja).
import torch
from common.camera import dopasuj_do_ekranu, otworz_kamere, wybierz_kamere_interaktywnie
from common.dane_osob import (
    lista_osob,
    przetrenuj_bez_usunietych,
    usun_dane_osoby,
    usun_stare_pliki_modelu,
)
from common.emotions import RozpoznawaczEmocji
from common.face_detector import DetektorTwarzy, WykrytaTwarz
from common.model import (
    WERSJA_PRZYGOTOWANIA_OBRAZU,
    oblicz_znormalizowane_cechy,
    policz_podobienstwa_wzorca,
    przygotuj_przeksztalcenie,
    pobierz_urzadzenie,
    zbuduj_ekstraktor_cech,
)

FOLDER_MODELI = Path(__file__).parent / "models"

# Kolory ramek w formacie BGR (tak przechowuje kolory OpenCV - odwrotnie niz RGB).
KOLOR_ROZPOZNANY = (0, 200, 0)  # zielony
KOLOR_NIEZNANY = (0, 0, 220)  # czerwony
SKALE_KADRU_ROZPOZNAWANIA = (0.82, 0.72)
LICZBA_WATKOW_OPENCV = 8


def parsuj_argumenty() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Rozpoznawanie twarzy na zywo.")
    parser.add_argument(
        "--kamera",
        type=int,
        default=None,
        help="Indeks kamery. Jesli pominiety, program zapyta interaktywnie.",
    )
    parser.add_argument("--szerokosc", type=int, default=3840)
    parser.add_argument("--wysokosc", type=int, default=2160)
    parser.add_argument(
        "--emocje",
        action="store_true",
        help=(
            "wlacza rozpoznawanie emocji (domyslnie WYLACZONE). Tylko do nauki "
            "poza szkola/na wlasnej twarzy - w szkole zabronione przez AI Act "
            "(art. 5 ust. 1 lit. f)."
        ),
    )
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
    """Wczytuje z dysku wzorce tozsamosci i progi zapisane
    przez 03_trenowanie_modelu.py, zamieniajac embeddingi z list
    (format JSON) z powrotem na tensory PyTorch gotowe do liczenia
    podobienstwa kosinusowego."""
    sciezka_wzorcow = FOLDER_MODELI / "wzorce_osob.json"
    if not sciezka_wzorcow.exists():
        raise SystemExit(
            "Brak wzorcow tozsamosci. Najpierw uruchom 03_trenowanie_modelu.py."
        )

    with open(sciezka_wzorcow, "r", encoding="utf-8") as plik:
        try:
            dane = json.load(plik)
        except json.JSONDecodeError as blad:
            raise SystemExit(
                f"Plik {sciezka_wzorcow} jest uszkodzony ({blad}). "
                "Uruchom ponownie 03_trenowanie_modelu.py."
            ) from blad

    if dane.get("wersja_przygotowania_obrazu") != WERSJA_PRZYGOTOWANIA_OBRAZU:
        raise SystemExit(
            "Wzorce twarzy sa nieaktualne i nie pasuja do nowego kadrowania. "
            "Uruchom ponownie 03_trenowanie_modelu.py."
        )

    wzorce: dict[str, dict] = {}
    for nazwa_osoby, wpis in dane.get("osoby", {}).items():
        try:
            prototypy = torch.tensor(wpis["prototypy"], device=urzadzenie)
            prog = float(wpis["prog_podobienstwa"])
        except (KeyError, TypeError, ValueError) as blad:
            raise SystemExit(
                f"Niepoprawny wpis dla osoby '{nazwa_osoby}' w {sciezka_wzorcow}: "
                f"{blad}. Uruchom ponownie 03_trenowanie_modelu.py."
            ) from blad
        wzorce[nazwa_osoby] = {"prototypy": prototypy, "prog": prog}
    return wzorce


def rozpoznaj_twarz(
    embedding: torch.Tensor, wzorce: dict[str, dict], korekta_progu: float
) -> tuple[str, float]:
    """Porownuje jeden lub kilka kadrow tej samej twarzy ze wzorcami osob.

    DLA UCZNIOW: przegladamy wszystkie znane osoby, liczymy podobienstwo
    kosinusowe (zwykly iloczyn skalarny dwoch wektorow jednostkowych) do
    kazdej z nich, i wybieramy NAJLEPSZE dopasowanie sposrod kadrow oraz
    prototypow. Kazda osoba ma WLASNY, skalibrowany prog, dlatego:

      1. najpierw odrzucamy osoby, ktorych prog nie zostal osiagniety,
      2. sposrod pozostalych wybieramy te z najwyzszym podobienstwem,
      3. jesli zadna osoba nie przekroczyla swojego progu - "Nieznana osoba".

    To wazne: samo "najwyzsze podobienstwo" nie wystarcza, bo osoba z bardzo
    ostrym progiem moze miec surowo wyzszy wynik, a mimo to nie zostac
    rozpoznana, podczas gdy inna osoba z nizszym progiem przekracza go.
    """
    if not wzorce:
        return "Nieznana osoba", 0.0

    embeddingi = embedding.unsqueeze(0) if embedding.ndim == 1 else embedding
    najlepsza_nazwa = "Nieznana osoba"
    najlepsze_podobienstwo = -1.0
    najlepsze_podobienstwo_ogolne = -1.0

    for nazwa_osoby, wpis in wzorce.items():
        podobienstwo = (
            policz_podobienstwa_wzorca(embeddingi, wpis["prototypy"]).max().item()
        )
        najlepsze_podobienstwo_ogolne = max(
            najlepsze_podobienstwo_ogolne, podobienstwo
        )

        # Korekta z linii polecen nie moze wypchnac progu poza sensowny
        # zakres podobienstwa kosinusowego [0, 1].
        prog = min(1.0, max(0.0, wpis["prog"] + korekta_progu))
        if podobienstwo >= prog and podobienstwo > najlepsze_podobienstwo:
            najlepsza_nazwa = nazwa_osoby
            najlepsze_podobienstwo = podobienstwo

    if najlepsza_nazwa == "Nieznana osoba":
        return najlepsza_nazwa, max(0.0, najlepsze_podobienstwo_ogolne)
    return najlepsza_nazwa, najlepsze_podobienstwo


def analizuj_klatke(
    klatka: np.ndarray,
    detektor: DetektorTwarzy,
    ekstraktor: torch.nn.Module,
    przygotuj_obrazy: list[Callable[[np.ndarray], torch.Tensor]],
    wzorce: dict[str, dict],
    urzadzenie: torch.device,
    korekta_progu: float,
    emocje: RozpoznawaczEmocji | None = None,
) -> list[tuple[WykrytaTwarz, str, float, str, float]]:
    """Batchuje embeddingi twarzy i odpytuje model dwoma ciasnymi kadrami.
    Zwraca (twarz, osoba, podobienstwo, emocja, pewnosc_emocji)."""
    tensory_wejsciowe = []
    twarze = []
    for twarz in detektor.wykryj(klatka):
        wycinek = twarz.wytnij(klatka, margines=0.2)
        if wycinek.size == 0:
            continue

        wycinek_rgb = cv2.cvtColor(wycinek, cv2.COLOR_BGR2RGB)
        tensory_wejsciowe.extend(
            przygotuj_obraz(wycinek_rgb) for przygotuj_obraz in przygotuj_obrazy
        )
        twarze.append(twarz)

    if not tensory_wejsciowe:
        if emocje is not None:
            emocje.rozpoznaj(klatka, [])
        return []

    wyniki_emocji = (
        emocje.rozpoznaj(klatka, twarze) if emocje is not None else [("", 0.0)] * len(twarze)
    )

    obrazy = torch.stack(tensory_wejsciowe).to(urzadzenie)
    embeddingi = oblicz_znormalizowane_cechy(ekstraktor, obrazy)
    liczba_kadrow = len(przygotuj_obrazy)
    wyniki = []
    for indeks, twarz in enumerate(twarze):
        zakres = slice(indeks * liczba_kadrow, (indeks + 1) * liczba_kadrow)
        etykieta, podobienstwo = rozpoznaj_twarz(
            embeddingi[zakres], wzorce, korekta_progu
        )
        emocja, pewnosc_emocji = wyniki_emocji[indeks]
        wyniki.append((twarz, etykieta, podobienstwo, emocja, pewnosc_emocji))
    return wyniki


def main() -> None:
    argumenty = parsuj_argumenty()
    cv2.setNumThreads(LICZBA_WATKOW_OPENCV)
    urzadzenie = pobierz_urzadzenie()

    wzorce = wczytaj_wzorce(urzadzenie)
    if not wzorce:
        raise SystemExit(
            "Plik models/wzorce_osob.json nie zawiera zadnej osoby. "
            "Uruchom 03_trenowanie_modelu.py po zebraniu danych."
        )
    print(f"[info] Wczytano wzorce dla {len(wzorce)} osob: {list(wzorce.keys())}")

    ekstraktor = zbuduj_ekstraktor_cech().to(urzadzenie)
    detektor = DetektorTwarzy()
    emocje = None
    if argumenty.emocje:
        if RozpoznawaczEmocji.dostepny():
            emocje = RozpoznawaczEmocji()
            print("[info] Rozpoznawanie emocji WLACZONE (--emocje).")
        else:
            print("[info] Brak modelu emocji w common/modele - pomijam emocje.")

    # Kadry korzystaja z tego samego rozmiaru i normalizacji co wzorce;
    # wejscie z OpenCV jest przed nimi konwertowane do PIL.
    przygotuj_obrazy = [
        przygotuj_przeksztalcenie(
            wejscie_z_kamery=True,
            skala_kadru=skala,
        )
        for skala in SKALE_KADRU_ROZPOZNAWANIA
    ]

    print(
        f"[info] Zadana rozdzielczosc kamery: "
        f"{argumenty.szerokosc}x{argumenty.wysokosc} "
        "(domyslnie 3840x2160; mozna zmienic przez --szerokosc i --wysokosc)."
    )
    indeks_kamery = (
        argumenty.kamera if argumenty.kamera is not None else wybierz_kamere_interaktywnie()
    )
    try:
        kamera = otworz_kamere(
            indeks_kamery,
            szerokosc=argumenty.szerokosc,
            wysokosc=argumenty.wysokosc,
        )
    except RuntimeError as blad:
        raise SystemExit(f"[blad] {blad}") from blad

    # cv2.WINDOW_NORMAL (zamiast domyslnego cv2.WINDOW_AUTOSIZE) tworzy okno,
    # ktore uczen moze recznie zmniejszyc/powiekszyc myszka (przeciagajac
    # jego krawedz) - obraz z kamery automatycznie SKALUJE SIE do aktualnego
    # rozmiaru okna, wiec nic nie jest przycinane. Bez tej flagi OpenCV
    # tworzy okno o rozmiarze DOKLADNIE rownym rozmiarowi klatki, ktorego
    # nie da sie zmniejszyc - a jesli klatka jest wieksza niz ekran, taki
    # sztywny rozmiar okna wygladal jak przypadkowe przyciecie obrazu.
    nazwa_okna = "Rozpoznawanie twarzy (q=koniec)"
    cv2.namedWindow(nazwa_okna, cv2.WINDOW_NORMAL)

    print("[info] Nacisnij 'q' lub ESC, aby zakonczyc.")
    poprzedni_czas = time.time()
    ostatnie_wyniki: list[tuple[WykrytaTwarz, str, float, str, float]] = []

    with ThreadPoolExecutor(
        max_workers=1, thread_name_prefix="analiza-twarzy"
    ) as executor:
        przyszla_analiza = None
        czas_zlecenia_analizy = 0.0
        fps_analizy = 0.0
        analiza_wylaczona = False
        try:
            while True:
                czy_odczytano, klatka = kamera.read()
                if not czy_odczytano:
                    print("[blad] Nie udalo sie odczytac klatki z kamery.")
                    break

                klatka = cv2.flip(klatka, 1)
                if przyszla_analiza is not None and przyszla_analiza.done():
                    try:
                        ostatnie_wyniki = przyszla_analiza.result()
                    except Exception as blad:  # noqa: BLE001 - nie przerywamy podgladu
                        print(
                            f"[blad] Analiza klatki nie powiodla sie: {blad}. "
                            "Wylaczam rozpoznawanie, podglad z kamery nadal dziala."
                        )
                        analiza_wylaczona = True
                        ostatnie_wyniki = []
                    else:
                        fps_analizy = 1.0 / max(
                            1e-6, time.perf_counter() - czas_zlecenia_analizy
                        )
                    przyszla_analiza = None

                if przyszla_analiza is None and not analiza_wylaczona:
                    czas_zlecenia_analizy = time.perf_counter()
                    przyszla_analiza = executor.submit(
                        analizuj_klatke,
                        klatka.copy(),
                        detektor,
                        ekstraktor,
                        przygotuj_obrazy,
                        wzorce,
                        urzadzenie,
                        argumenty.korekta_progu,
                        emocje,
                    )

                # Rysujemy na kopii ZMniejszonej do rozmiaru okna, a nie na
                # pelnej klatce 4K. Dzieki temu tekst i ramki maja czytelny
                # rozmiar na ekranie (na 4K bylyby 4x za male), a pelna klatka
                # pozostaje nietknieta na potrzeby analizy.
                podglad = dopasuj_do_ekranu(klatka)
                skala_x = podglad.shape[1] / klatka.shape[1]
                skala_y = podglad.shape[0] / klatka.shape[0]
                skala_tekstu = max(0.6, podglad.shape[1] / 1600)

                for twarz, etykieta_osoby, podobienstwo, emocja, pewnosc_emocji in ostatnie_wyniki:
                    etykieta = f"{etykieta_osoby} ({podobienstwo * 100:.0f}%)"
                    if emocja:
                        etykieta += f" | {emocja} {pewnosc_emocji * 100:.0f}%"
                    kolor_ramki = (
                        KOLOR_NIEZNANY
                        if etykieta_osoby == "Nieznana osoba"
                        else KOLOR_ROZPOZNANY
                    )
                    x1 = int(twarz.x * skala_x)
                    y1 = int(twarz.y * skala_y)
                    x2 = int((twarz.x + twarz.w) * skala_x)
                    y2 = int((twarz.y + twarz.h) * skala_y)
                    cv2.rectangle(podglad, (x1, y1), (x2, y2), kolor_ramki, 2)
                    cv2.putText(
                        podglad,
                        etykieta,
                        (x1, max(0, y1 - int(8 * skala_tekstu))),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.7 * skala_tekstu,
                        kolor_ramki,
                        max(2, int(2 * skala_tekstu)),
                    )

                teraz = time.time()
                fps = 1.0 / max(1e-6, teraz - poprzedni_czas)
                poprzedni_czas = teraz
                cv2.putText(
                    podglad,
                    f"{klatka.shape[1]}x{klatka.shape[0]}  "
                    f"Podglad: {fps:.1f} FPS  Analiza: {fps_analizy:.1f} FPS",
                    (20, int(40 * skala_tekstu)),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.8 * skala_tekstu,
                    (255, 255, 0),
                    max(2, int(2 * skala_tekstu)),
                )

                # Podglad nie czeka na kosztowne wykrywanie i rozpoznawanie.
                cv2.imshow(nazwa_okna, podglad)

                klawisz = cv2.waitKey(1) & 0xFF
                if klawisz in (ord("q"), 27):
                    break
        finally:
            kamera.release()
            cv2.destroyAllWindows()


def zapytaj_o_usuniecie_danych() -> None:
    """Po zakonczeniu pyta, czy usunac dane osob i przetrenowac wzorce od nowa."""
    if not sys.stdin or not sys.stdin.isatty():
        return
    osoby = lista_osob()
    if not osoby:
        return
    print(f"\nOsoby w bazie: {', '.join(osoby)}")
    odpowiedz = input("Czy usunac dane jakichs osob? [t/N]: ").strip().lower()
    if odpowiedz not in ("t", "tak", "y", "yes"):
        return
    wybor = input(
        "Podaj nazwy osob po przecinku lub 'wszystkie' (puste = anuluj): "
    ).strip()
    if not wybor:
        return
    if wybor.lower() == "wszystkie":
        wybrane = osoby
    else:
        wybrane = [n.strip() for n in wybor.split(",") if n.strip()]
        nieznane = [n for n in wybrane if n not in osoby]
        if nieznane:
            print(f"[blad] Nie ma takich osob: {', '.join(nieznane)}. Nic nie usunieto.")
            return
    potwierdzenie = input(
        f"POTWIERDZ: trwale usunac dane osob ({', '.join(wybrane)}) i przetrenowac "
        "wzorce od nowa? Wpisz 'tak': "
    ).strip().lower()
    if potwierdzenie != "tak":
        print("Anulowano - nic nie usunieto.")
        return
    for osoba in wybrane:
        usun_dane_osoby(osoba)
    usun_stare_pliki_modelu()
    if przetrenuj_bez_usunietych():
        print("Gotowe: dane usuniete, wzorce przeliczone bez tych osob.")


if __name__ == "__main__":
    main()
    zapytaj_o_usuniecie_danych()
