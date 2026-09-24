"""
Krok 3: Budowanie "wzorcow tozsamosci" (embeddingow) i kalibracja progu
rozpoznawania.

DLA UCZNIOW - dlaczego ten skrypt NIE trenuje juz klasycznego klasyfikatora?

Pierwsza wersja tego projektu uczyla siec neuronowa odpowiadac na pytanie
"kto z tych N znanych mi osob jest na zdjeciu" (klasyfikacja, softmax).
W praktyce (czyli na zywym podgladzie z kamery) okazalo sie to zawodne:
KAZDA nowa, prawdziwa osoba przed kamera byla rozpoznawana jako znana
osoba. Powod: klasa "nieznajomy" byla zbudowana ze zdjec pobranych z
internetu (inna ostrosc, kompresja, tlo studyjne) - siec nauczyla sie wiec
odrozniac "ostre zdjecie z tego pokoju" od "rozmyte zdjecie z internetu"
zamiast prawdziwych rysow twarzy. To klasyczny przyklad tzw. "uczenia sie
skrotow" (shortcut learning) - siec zawsze znajdzie NAJLATWIEJSZY sposob
na zminimalizowanie bledu na zbiorze treningowym, niekoniecznie ten, o ktory
nam chodzilo.

NOWE PODEJSCIE - rozpoznawanie przez PODOBIENSTWO (verification), a nie
klasyfikacje zamknieta:
  1. Bierzemy siec MobileNetV3-Small z wagami ImageNet, ale BEZ warstwy
     klasyfikujacej na koncu (patrz common/model.py::zbuduj_ekstraktor_cech).
     Ta siec NIGDY nie jest trenowana/douczana na naszych zdjeciach - jest
     tylko "czytnikiem" zamieniajacym kazde zdjecie twarzy na wektor kilkuset
     liczb (tzw. embedding), opisujacy jego wyglad.
  2. Dla kazdej znanej osoby liczymy embeddingi wszystkich jej zdjec ze
     zbioru treningowego i usredniamy je -> to jej "wzorzec" (centroid).
  3. Uzywajac zbioru WALIDACYJNEGO (w tym takze zdjec klasy "nieznajomy",
     jesli sa dostepne) sprawdzamy, jak podobne do wzorca sa: (a) zdjecia
     TEJ SAMEJ osoby (powinny byc bardzo podobne) i (b) zdjecia
     wszystkich INNYCH osob/nieznajomych (powinny byc mniej podobne).
     Na tej podstawie automatycznie DOBIERAMY prog podobienstwa oddzielajacy
     "to ta osoba" od "to ktos inny" - BEZ trenowania jakiejkolwiek warstwy
     na tym porownaniu (uzywamy tych danych tylko do wyliczenia jednej,
     prostej liczby - progu).

Zaleta: poniewaz nigdy nie trenujemy zadnej wagi na parze (moja twarz vs
zdjecia z internetu), nie ma jak "nauczyc sie na skrotow" mylacej roznicy w
stylu zdjec - o dopasowaniu decyduje wylacznie ogolna wiedza sieci o
obrazach, wyniesiona z treningu na milionach zdjec ImageNet. Dodatkowa
zaleta praktyczna: dodanie NOWEJ osoby do systemu nie wymaga juz zadnego
"treningu" (minut oczekiwania) - wystarczy policzyc jej embeddingi (sekundy),
co bedzie kluczowe przy planowanym pozniej automatycznym zapisywaniu nowych
osob przez system (rozpoznawanie mowy + synteza mowy).

Wymaga wczesniejszego uruchomienia:
    01_zbieranie_danych.py (dla kazdej osoby)
    02_podzial_danych.py

Uzycie:
    python 03_trenowanie_modelu.py
    python 03_trenowanie_modelu.py --margines-bezpieczenstwa 0.05

Wynik:
    models/wzorce_osob.json     - wzorce (centroidy) + skalibrowane progi
    models/kalibracja_progu.png - wykres pomagajacy zrozumiec dobor progu
"""

from __future__ import annotations

# Ta linijka MUSI byc pierwszym kodem wykonywalnym w tym pliku - patrz
# szczegolowe wyjasnienie w common/uruchom_w_venv.py (automatyczne
# przelaczenie na Python z .venv, jesli uruchomiono innym interpreterem).
from common.uruchom_w_venv import przelacz_na_venv_jesli_trzeba

przelacz_na_venv_jesli_trzeba()

# argparse - obsluga argumentow uruchomieniowych (--margines-bezpieczenstwa itd.).
import argparse
# json - zapis wzorcow tozsamosci (embeddingow) i progow w czytelnym,
# tekstowym formacie, ktory potem odczyta skrypt 04.
import json
# Path - obiektowa reprezentacja sciezek plikow/folderow.
from pathlib import Path

# torch - glowny silnik PyTorch do obliczen na tensorach (tablicach
# wielowymiarowych), tutaj uzywany wylacznie do "odczytu" (inferencji) -
# nie trenujemy juz zadnej sieci w tym skrypcie.
import torch
# DataLoader automatycznie dzieli caly zbior danych na mniejsze paczki
# (batch) i w tle rownolegle wczytuje/przygotowuje kolejne paczki - tutaj
# przyspiesza to liczenie embeddingow dla setek zdjec naraz.
from torch.utils.data import DataLoader
# datasets.ImageFolder to gotowa klasa z torchvision, ktora automatycznie
# wczytuje zdjecia z folderow (klasa = nazwa podfolderu). transforms to
# zestaw gotowych przeksztalcen obrazu (zmiana rozmiaru, normalizacja).
from torchvision import datasets, transforms

from common.model import (
    ODCHYLENIE_IMAGENET,
    ROZMIAR_OBRAZU,
    SREDNIA_IMAGENET,
    oblicz_znormalizowane_cechy,
    pobierz_urzadzenie,
    zbuduj_ekstraktor_cech,
)

FOLDER_PRZETWORZONYCH_DANYCH = Path(__file__).parent / "data" / "processed"
FOLDER_MODELI = Path(__file__).parent / "models"

# Nazwa folderu klasy "nieznajomy" (patrz pobierz_zdjecia_nieznajomych.py) -
# to NIE jest prawdziwa, rozpoznawana osoba, wiec pomijamy ja przy liczeniu
# wzorcow (centroidow), ale WYKORZYSTUJEMY jej zdjecia jako dodatkowy,
# "trudny" material do kalibracji progu (patrz funkcja skalibruj_progi).
NAZWA_KLASY_NIEZNAJOMY = "nieznajomy"


def parsuj_argumenty() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Budowanie wzorcow tozsamosci (embeddingow) i kalibracja progu rozpoznawania."
    )
    parser.add_argument("--batch-size", type=int, default=32, help="Ile zdjec naraz trafia do sieci przy liczeniu embeddingow")
    parser.add_argument(
        "--margines-bezpieczenstwa",
        type=float,
        default=0.03,
        help=(
            "O ile podniesc automatycznie wyliczony prog podobienstwa "
            "(wieksza wartosc = trudniej o falszywe rozpoznanie obcej "
            "osoby, ale latwiej o odrzucenie znanej osoby w gorszych "
            "warunkach oswietlenia)"
        ),
    )
    return parser.parse_args()


def przygotuj_przeksztalcenie() -> transforms.Compose:
    """Zwraca przeksztalcenie obrazu uzywane przy liczeniu embeddingow.

    DLA UCZNIOW: w odroznieniu od poprzedniej wersji tego skryptu NIE
    uzywamy tu zadnej augmentacji (losowego kadrowania, rozmycia itp.) -
    liczymy wzorzec (centroid) na podstawie mozliwie "czystych",
    powtarzalnych zdjec. Augmentacja miala sens przy TRENOWANIU wag sieci
    (uczyla ja ignorowac drobne roznice) - tutaj niczego juz nie trenujemy,
    wiec chcemy po prostu jak najdokladniejszy, stabilny opis kazdego
    zdjecia.
    """
    return transforms.Compose(
        [
            transforms.Resize((ROZMIAR_OBRAZU, ROZMIAR_OBRAZU)),
            transforms.ToTensor(),
            transforms.Normalize(SREDNIA_IMAGENET, ODCHYLENIE_IMAGENET),
        ]
    )


def policz_embeddingi(
    ekstraktor: torch.nn.Module,
    folder: Path,
    urzadzenie: torch.device,
    rozmiar_batcha: int,
) -> tuple[torch.Tensor, list[int], list[str]]:
    """Liczy znormalizowane embeddingi wszystkich zdjec w podanym folderze
    (train lub val), pogrupowanych na podfoldery-klasy (osoby).

    Zwraca:
      - macierz embeddingow o ksztalcie [liczba_zdjec, wymiar_cech],
      - liste numerow klas (indeksow) odpowiadajacych kolejnym zdjeciom,
      - liste nazw klas (osob) w kolejnosci ustalonej przez ImageFolder.
    """
    zbior = datasets.ImageFolder(folder, transform=przygotuj_przeksztalcenie())
    ladowarka = DataLoader(zbior, batch_size=rozmiar_batcha, shuffle=False, num_workers=2)

    wszystkie_embeddingi = []
    wszystkie_etykiety: list[int] = []
    for obrazy, etykiety in ladowarka:
        obrazy = obrazy.to(urzadzenie)
        embeddingi = oblicz_znormalizowane_cechy(ekstraktor, obrazy)
        wszystkie_embeddingi.append(embeddingi.cpu())
        wszystkie_etykiety.extend(etykiety.tolist())

    return torch.cat(wszystkie_embeddingi, dim=0), wszystkie_etykiety, zbior.classes


def zbuduj_wzorce(
    embeddingi_trening: torch.Tensor, etykiety_trening: list[int], klasy: list[str]
) -> dict[str, torch.Tensor]:
    """Usrednia embeddingi kazdej znanej osoby ze zbioru treningowego,
    tworzac jej "wzorzec" (centroid) - jeden, reprezentatywny wektor.

    Pomija klase NAZWA_KLASY_NIEZNAJOMY - to nie jest prawdziwa osoba do
    rozpoznania, tylko pomocniczy zbior "obcych" twarzy uzywany wylacznie
    do kalibracji progu (patrz skalibruj_progi ponizej).
    """
    wzorce: dict[str, torch.Tensor] = {}
    for indeks_klasy, nazwa_osoby in enumerate(klasy):
        if nazwa_osoby == NAZWA_KLASY_NIEZNAJOMY:
            continue
        maska = torch.tensor([e == indeks_klasy for e in etykiety_trening])
        embeddingi_osoby = embeddingi_trening[maska]
        if embeddingi_osoby.shape[0] == 0:
            print(f"[uwaga] Brak zdjec treningowych dla '{nazwa_osoby}' - pomijam.")
            continue

        # Usredniamy wszystkie embeddingi tej osoby, a nastepnie NORMALIZUJEMY
        # wynik z powrotem do dlugosci 1 (srednia kilku wektorow jednostkowych
        # sama w sobie zwykle nie ma juz dlugosci 1) - dzieki temu centroid
        # nadal mozna porownywac zwyklym iloczynem skalarnym (patrz
        # oblicz_znormalizowane_cechy w common/model.py).
        centroid = embeddingi_osoby.mean(dim=0)
        centroid = centroid / centroid.norm(p=2)
        wzorce[nazwa_osoby] = centroid
        print(f"[info] Wzorzec '{nazwa_osoby}' zbudowany z {embeddingi_osoby.shape[0]} zdjec.")

    return wzorce


def skalibruj_progi(
    wzorce: dict[str, torch.Tensor],
    embeddingi_walidacja: torch.Tensor,
    etykiety_walidacja: list[int],
    klasy: list[str],
    margines_bezpieczenstwa: float,
) -> dict[str, dict]:
    """Dla kazdej znanej osoby wylicza prog podobienstwa kosinusowego
    oddzielajacy jej WLASNE zdjecia walidacyjne od zdjec WSZYSTKICH innych
    osob/nieznajomych.

    DLA UCZNIOW - jak dokladnie dobierany jest prog?
      1. Liczymy podobienstwo (iloczyn skalarny znormalizowanych wektorow =
         cosine similarity) miedzy centroidem tej osoby a KAZDYM zdjeciem
         walidacyjnym NALEZACYM do niej -> zbior "podobienstw wlasnych".
      2. Liczymy to samo dla KAZDEGO zdjecia walidacyjnego NIENALEZACEGO do
         niej (inne osoby + "nieznajomy") -> zbior "podobienstw obcych".
      3. Prog ustawiamy DOKLADNIE POSRODKU miedzy najgorszym (najnizszym)
         wynikiem wlasnym a najlepszym (najwyzszym) wynikiem obcym - to
         najbezpieczniejszy punkt, jaki mozna wybrac na podstawie danych,
         jakie mamy. Jesli te dwa zbiory na siebie "nachodza" (obcy wynik
         wypada wyzej niz wlasny), nie da sie znalezc progu idealnie
         rozdzielajacego obie grupy - wypisujemy wtedy ostrzezenie (patrz
         nizej), bo to znak, ze warto zebrac wiecej/lepszych zdjec.
      4. Dodatkowo podnosimy prog o "margines_bezpieczenstwa" (domyslnie
         0.03) - w praktyce lepiej czasem nieslusznie zapytac znana osobe
         o imie jeszcze raz, niz pomylkowo rozpoznac obca osobe jako znana.
    """
    wyniki: dict[str, dict] = {}

    for nazwa_osoby, centroid in wzorce.items():
        indeks_klasy = klasy.index(nazwa_osoby)
        maska_wlasna = torch.tensor([e == indeks_klasy for e in etykiety_walidacja])
        maska_obca = ~maska_wlasna

        # Mnozenie macierzowe embeddingow (kazdy o dlugosci 1) przez centroid
        # (tez o dlugosci 1) daje wprost podobienstwo kosinusowe dla kazdego
        # zdjecia na raz - bez potrzeby recznej petli.
        podobienstwa = embeddingi_walidacja @ centroid

        podobienstwa_wlasne = podobienstwa[maska_wlasna]
        podobienstwa_obce = podobienstwa[maska_obca]

        if podobienstwa_wlasne.numel() == 0:
            print(f"[uwaga] Brak zdjec walidacyjnych dla '{nazwa_osoby}' - uzywam domyslnego progu 0.5.")
            wyniki[nazwa_osoby] = {"prog": 0.5, "srednie_wlasne": None, "srednie_obce": None}
            continue

        min_wlasne = podobienstwa_wlasne.min().item()
        srednia_wlasne = podobienstwa_wlasne.mean().item()

        if podobienstwa_obce.numel() > 0:
            maks_obce = podobienstwa_obce.max().item()
            srednia_obce = podobienstwa_obce.mean().item()
        else:
            # Brak jakichkolwiek "obcych" zdjec walidacyjnych (np. tylko
            # jedna osoba w bazie i brak klasy "nieznajomy") - nie mamy jak
            # empirycznie ocenic progu, wiec przyjmujemy ostrozna wartosc
            # domyslna.
            maks_obce = min_wlasne - 0.2
            srednia_obce = maks_obce
            print(
                f"[uwaga] Brak 'obcych' zdjec walidacyjnych do porownania z '{nazwa_osoby}' "
                "- prog jest tylko przyblizony. Rozwaz uruchomienie "
                "pobierz_zdjecia_nieznajomych.py, jesli tego jeszcze nie zrobiono."
            )

        if maks_obce >= min_wlasne:
            print(
                f"[uwaga] Slaba separowalnosc dla '{nazwa_osoby}' - podobienstwo obcego zdjecia "
                f"({maks_obce:.2f}) jest wyzsze niz najgorszy wynik wlasny ({min_wlasne:.2f}). "
                "System bedzie dzialac, ale rozwaz dodanie wiecej/lepszych zdjec treningowych "
                "(rozne oswietlenie, kat glowy, itp.)."
            )
            prog_bazowy = (srednia_wlasne + srednia_obce) / 2
        else:
            prog_bazowy = (min_wlasne + maks_obce) / 2

        prog = min(0.95, max(0.2, prog_bazowy + margines_bezpieczenstwa))

        wyniki[nazwa_osoby] = {
            "prog": prog,
            "srednie_wlasne": srednia_wlasne,
            "srednie_obce": srednia_obce,
        }
        print(
            f"[info] '{nazwa_osoby}': podobienstwo wlasne ~{srednia_wlasne:.2f} "
            f"(min {min_wlasne:.2f}), podobienstwo obce ~{srednia_obce:.2f} (maks {maks_obce:.2f}) "
            f"-> wybrany prog = {prog:.2f}"
        )

    return wyniki


def main() -> None:
    argumenty = parsuj_argumenty()
    FOLDER_MODELI.mkdir(parents=True, exist_ok=True)

    if not (FOLDER_PRZETWORZONYCH_DANYCH / "train").exists():
        raise SystemExit(
            f"Brak danych w {FOLDER_PRZETWORZONYCH_DANYCH}. Najpierw uruchom 02_podzial_danych.py."
        )

    urzadzenie = pobierz_urzadzenie()
    # zbuduj_ekstraktor_cech() zwraca siec ZAWSZE z wagami ImageNet - nigdy
    # jej nie trenujemy, wiec to jedyne wagi, jakich kiedykolwiek uzyje.
    ekstraktor = zbuduj_ekstraktor_cech().to(urzadzenie)

    print("[info] Liczenie embeddingow zbioru treningowego...")
    embeddingi_trening, etykiety_trening, klasy = policz_embeddingi(
        ekstraktor, FOLDER_PRZETWORZONYCH_DANYCH / "train", urzadzenie, argumenty.batch_size
    )
    print(f"[info] Klasy ({len(klasy)}): {klasy}")

    print("[info] Liczenie embeddingow zbioru walidacyjnego...")
    embeddingi_walidacja, etykiety_walidacja, klasy_walidacja = policz_embeddingi(
        ekstraktor, FOLDER_PRZETWORZONYCH_DANYCH / "val", urzadzenie, argumenty.batch_size
    )
    if klasy_walidacja != klasy:
        raise SystemExit(
            "Zbior treningowy i walidacyjny maja rozne listy osob - uruchom ponownie "
            "02_podzial_danych.py, aby je zsynchronizowac."
        )

    wzorce = zbuduj_wzorce(embeddingi_trening, etykiety_trening, klasy)
    if not wzorce:
        raise SystemExit(
            "Nie udalo sie zbudowac zadnego wzorca tozsamosci - sprawdz, czy "
            "data/processed/train zawiera podfoldery z prawdziwymi osobami."
        )

    kalibracja = skalibruj_progi(
        wzorce, embeddingi_walidacja, etykiety_walidacja, klasy, argumenty.margines_bezpieczenstwa
    )

    # Zapisujemy wzorce (centroidy) razem ze skalibrowanymi progami do
    # jednego pliku JSON - skrypt 04 wczyta go w calosci przy starcie.
    dane_do_zapisu = {
        "wymiar_cech": next(iter(wzorce.values())).shape[0],
        "osoby": {
            nazwa_osoby: {
                # .tolist() zamienia tensor PyTorch na zwykla liste liczb
                # Pythona - JSON nie potrafi bezposrednio zapisac tensorow.
                "centroid": centroid.tolist(),
                "prog_podobienstwa": kalibracja[nazwa_osoby]["prog"],
            }
            for nazwa_osoby, centroid in wzorce.items()
        },
    }
    with open(FOLDER_MODELI / "wzorce_osob.json", "w", encoding="utf-8") as plik:
        json.dump(dane_do_zapisu, plik, ensure_ascii=False, indent=2)

    print(f"\n[gotowe] Wzorce tozsamosci zapisane w: {FOLDER_MODELI / 'wzorce_osob.json'}")
    print("[gotowe] Mozesz teraz uruchomic: python 04_rozpoznawanie_na_zywo.py")

    # Rysujemy histogram podobienstw "wlasnych" i "obcych" dla kazdej osoby -
    # bardzo przydatne dydaktycznie: uczniowie na wlasne oczy widza, jak
    # daleko od siebie znajduja sie te dwie grupy i dlaczego prog zostal
    # ustawiony akurat w tym miejscu.
    try:
        import matplotlib.pyplot as plt

        liczba_osob = len(wzorce)
        rysunek, osie = plt.subplots(1, liczba_osob, figsize=(5 * liczba_osob, 4), squeeze=False)
        for numer_wykresu, (nazwa_osoby, centroid) in enumerate(wzorce.items()):
            indeks_klasy = klasy.index(nazwa_osoby)
            maska_wlasna = torch.tensor([e == indeks_klasy for e in etykiety_walidacja])
            podobienstwa = embeddingi_walidacja @ centroid

            os_wykresu = osie[0][numer_wykresu]
            os_wykresu.hist(podobienstwa[maska_wlasna].numpy(), bins=15, alpha=0.6, label="wlasne zdjecia")
            os_wykresu.hist(podobienstwa[~maska_wlasna].numpy(), bins=15, alpha=0.6, label="obce zdjecia")
            os_wykresu.axvline(kalibracja[nazwa_osoby]["prog"], color="red", linestyle="--", label="prog")
            os_wykresu.set_title(f"Kalibracja progu: {nazwa_osoby}")
            os_wykresu.set_xlabel("podobienstwo kosinusowe")
            os_wykresu.legend()

        rysunek.tight_layout()
        rysunek.savefig(FOLDER_MODELI / "kalibracja_progu.png")
        print(f"[gotowe] Wykres kalibracji progu: {FOLDER_MODELI / 'kalibracja_progu.png'}")
    except ImportError:
        print("[uwaga] matplotlib niedostepny - pomijam wykres kalibracji progu.")


if __name__ == "__main__":
    main()
