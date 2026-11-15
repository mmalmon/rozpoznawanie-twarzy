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
  2. Dla kazdej znanej osoby zachowujemy embeddingi zdjec ze zbioru
     treningowego jako wiele wzorcow obejmujacych rozne pozy i oswietlenie.
  3. Uzywajac zbioru WALIDACYJNEGO (w tym takze zdjec klasy "nieznajomy",
     jesli sa dostepne) sprawdzamy, jak podobne do wzorca sa: (a) zdjecia
     TEJ SAMEJ osoby (powinny byc bardzo podobne) i (b) zdjecia
     wszystkich INNYCH osob/nieznajomych (powinny byc mniej podobne).
     Na tej podstawie automatycznie DOBIERAMY prog podobienstwa oddzielajacy
     "to ta osoba" od "to ktos inny" - BEZ trenowania jakiejkolwiek warstwy
     na tym porownaniu (uzywamy tych danych tylko do wyliczenia jednej,
     prostej liczby - progu).

Zaleta: nie uczymy osobnej warstwy rozrozniania znanych osob od pobranych z
internetu zdjec "nieznajomych", wiec ograniczamy ryzyko nauczenia sie
roznic w stylu zdjec zamiast tozsamosci. Nie eliminuje to jednak wplywu tla:
MobileNet jest modelem ogolnym, a kadrowanie i przyklady z roznych teł
pomagaja, ale nie zastepuja dobrych danych. Dodanie NOWEJ osoby do systemu
nie wymaga juz zadnego
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
    models/wzorce_osob.json     - wzorce zdjec + skalibrowane progi
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
# datasets.ImageFolder automatycznie wczytuje zdjecia z folderow (klasa =
# nazwa podfolderu). Wspolne przeksztalcenie obrazu pochodzi z common.model.
from torchvision import datasets, transforms

from common.model import (
    WERSJA_PRZYGOTOWANIA_OBRAZU,
    oblicz_znormalizowane_cechy,
    policz_podobienstwa_wzorca,
    przygotuj_przeksztalcenie,
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
WSPOLCZYNNIKI_JASNOSCI = (0.65, 1.0, 1.4)
MAKS_WZORCOW_NA_JASNOSC = 128
MAKS_PROG_PODOBIENSTWA = 0.87


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
    wszystkie_embeddingi = []
    wszystkie_etykiety: list[int] = []
    klasy: list[str] | None = None
    for wspolczynnik_jasnosci in WSPOLCZYNNIKI_JASNOSCI:
        przygotuj_obraz = transforms.Compose(
            [
                transforms.ColorJitter(
                    brightness=(wspolczynnik_jasnosci, wspolczynnik_jasnosci)
                ),
                przygotuj_przeksztalcenie(),
            ]
        )
        zbior = datasets.ImageFolder(folder, transform=przygotuj_obraz)
        if klasy is None:
            klasy = zbior.classes
        elif zbior.classes != klasy:
            raise RuntimeError(f"Lista klas zmienila sie podczas czytania danych z {folder}.")

        ladowarka = DataLoader(zbior, batch_size=rozmiar_batcha, shuffle=False, num_workers=2)
        for obrazy, etykiety in ladowarka:
            obrazy = obrazy.to(urzadzenie)
            embeddingi = oblicz_znormalizowane_cechy(ekstraktor, obrazy)
            wszystkie_embeddingi.append(embeddingi.cpu())
            wszystkie_etykiety.extend(etykiety.tolist())

    if klasy is None or not wszystkie_embeddingi:
        raise SystemExit(f"Brak zdjec do przetworzenia w {folder}.")
    return torch.cat(wszystkie_embeddingi, dim=0), wszystkie_etykiety, klasy


def zbuduj_wzorce(
    embeddingi_trening: torch.Tensor, etykiety_trening: list[int], klasy: list[str]
) -> dict[str, torch.Tensor]:
    """Zachowuje embeddingi zdjec kazdej znanej osoby jako wiele wzorcow
    obejmujacych rozne ujecia twarzy.

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

        prototypy = []
        for embeddingi_wariantu in embeddingi_osoby.chunk(len(WSPOLCZYNNIKI_JASNOSCI)):
            liczba_wzorcow = min(MAKS_WZORCOW_NA_JASNOSC, embeddingi_wariantu.shape[0])
            indeksy = torch.linspace(
                0, embeddingi_wariantu.shape[0] - 1, liczba_wzorcow
            ).round().long()
            prototypy.append(embeddingi_wariantu[indeksy])

        wzorce[nazwa_osoby] = torch.cat(prototypy, dim=0)
        print(
            f"[info] Wzorce '{nazwa_osoby}' zbudowane z {embeddingi_osoby.shape[0]} "
            f"zdjec (zachowano {wzorce[nazwa_osoby].shape[0]} prototypow)."
        )

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
      1. Dla kazdego zdjecia walidacyjnego wybieramy podobienstwo do
         najblizszego wzorca tej osoby -> "podobienstwo wlasne" lub "obce".
      2. Dolny decyl wynikow wlasnych ogranicza wplyw pojedynczych
         nietypowych ujec, a najwyzsze podobienstwo obcego zdjecia pomaga
         unikac falszywych rozpoznan.
      3. Prog ustawiamy miedzy dolnym decylem wynikow wlasnych a najlepszym
         (najwyzszym) wynikiem obcym - to
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

    for nazwa_osoby, wzorzec in wzorce.items():
        indeks_klasy = klasy.index(nazwa_osoby)
        maska_wlasna = torch.tensor([e == indeks_klasy for e in etykiety_walidacja])
        maska_obca = ~maska_wlasna

        podobienstwa = policz_podobienstwa_wzorca(embeddingi_walidacja, wzorzec)

        podobienstwa_wlasne = podobienstwa[maska_wlasna]
        podobienstwa_obce = podobienstwa[maska_obca]

        if podobienstwa_wlasne.numel() == 0:
            print(f"[uwaga] Brak zdjec walidacyjnych dla '{nazwa_osoby}' - uzywam domyslnego progu 0.5.")
            wyniki[nazwa_osoby] = {"prog": 0.5, "srednie_wlasne": None, "srednie_obce": None}
            continue

        min_wlasne = podobienstwa_wlasne.quantile(0.1).item()
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
                f"({maks_obce:.2f}) jest wyzsze niz dolne 10% wynikow wlasnych ({min_wlasne:.2f}). "
                "System bedzie dzialac, ale rozwaz dodanie wiecej/lepszych zdjec treningowych "
                "(rozne oswietlenie, kat glowy, itp.)."
            )
            prog_bazowy = (srednia_wlasne + srednia_obce) / 2
        else:
            prog_bazowy = (min_wlasne + maks_obce) / 2

        prog_skalibrowany = prog_bazowy + margines_bezpieczenstwa
        prog = min(MAKS_PROG_PODOBIENSTWA, max(0.2, prog_skalibrowany))
        if prog_skalibrowany > MAKS_PROG_PODOBIENSTWA:
            print(
                f"[info] '{nazwa_osoby}': ograniczam prog do "
                f"{MAKS_PROG_PODOBIENSTWA:.0%}, zgodnie z ustawionym limitem."
            )
        if prog > min_wlasne:
            print(
                f"[uwaga] '{nazwa_osoby}': wyliczony prog ({prog:.2f}) jest wyzszy "
                f"niz dolne 10% podobienstw wlasnych ({min_wlasne:.2f}) - czesc "
                "zdjec tej osoby moze byc rozpoznawana jako 'Nieznana osoba'. "
                "Zbierz wiecej/roznorodniejszych zdjec albo zmniejsz "
                "--margines-bezpieczenstwa."
            )

        wyniki[nazwa_osoby] = {
            "prog": prog,
            "srednie_wlasne": srednia_wlasne,
            "srednie_obce": srednia_obce,
        }
        print(
            f"[info] '{nazwa_osoby}': podobienstwo wlasne ~{srednia_wlasne:.2f} "
            f"(dolne 10% {min_wlasne:.2f}), podobienstwo obce ~{srednia_obce:.2f} (maks {maks_obce:.2f}) "
            f"-> wybrany prog = {prog:.2f}"
        )

    return wyniki


def main() -> None:
    argumenty = parsuj_argumenty()
    if argumenty.margines_bezpieczenstwa < 0:
        raise SystemExit("--margines-bezpieczenstwa nie moze byc ujemny.")
    if argumenty.batch_size <= 0:
        raise SystemExit("--batch-size musi byc dodatni.")
    FOLDER_MODELI.mkdir(parents=True, exist_ok=True)

    if not (FOLDER_PRZETWORZONYCH_DANYCH / "train").exists():
        raise SystemExit(
            f"Brak danych w {FOLDER_PRZETWORZONYCH_DANYCH}. Najpierw uruchom 02_podzial_danych.py."
        )
    if not (FOLDER_PRZETWORZONYCH_DANYCH / "val").exists():
        raise SystemExit(
            f"Brak zbioru walidacyjnego w {FOLDER_PRZETWORZONYCH_DANYCH}. "
            "Uruchom ponownie 02_podzial_danych.py."
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

    # Zapisujemy wzorce (embeddingi zdjec) razem ze skalibrowanymi progami do
    # jednego pliku JSON - skrypt 04 wczyta go w calosci przy starcie.
    dane_do_zapisu = {
        "wersja_przygotowania_obrazu": WERSJA_PRZYGOTOWANIA_OBRAZU,
        "wymiar_cech": next(iter(wzorce.values())).shape[0],
        "osoby": {
            nazwa_osoby: {
                # .tolist() zamienia tensory PyTorch na listy liczb JSON.
                "prototypy": wzorzec.tolist(),
                "prog_podobienstwa": kalibracja[nazwa_osoby]["prog"],
            }
            for nazwa_osoby, wzorzec in wzorce.items()
        },
    }
    sciezka_wzorcow = FOLDER_MODELI / "wzorce_osob.json"
    # Zapisujemy najpierw do pliku tymczasowego, a potem podmieniamy go
    # jednym ruchem. Dzieki temu przerwanie programu w trakcie zapisu nie
    # zostawi uszkodzonego (nieczytelnego) pliku z wzorcami.
    sciezka_tymczasowa = sciezka_wzorcow.with_suffix(".json.tmp")
    with open(sciezka_tymczasowa, "w", encoding="utf-8") as plik:
        json.dump(dane_do_zapisu, plik, ensure_ascii=False, indent=2)
    sciezka_tymczasowa.replace(sciezka_wzorcow)

    print(f"\n[gotowe] Wzorce tozsamosci zapisane w: {sciezka_wzorcow}")
    print("[gotowe] Mozesz teraz uruchomic: python 04_rozpoznawanie_na_zywo.py")

    # Rysujemy histogram podobienstw "wlasnych" i "obcych" dla kazdej osoby -
    # bardzo przydatne dydaktycznie: uczniowie na wlasne oczy widza, jak
    # daleko od siebie znajduja sie te dwie grupy i dlaczego prog zostal
    # ustawiony akurat w tym miejscu.
    try:
        import matplotlib.pyplot as plt

        liczba_osob = len(wzorce)
        rysunek, osie = plt.subplots(1, liczba_osob, figsize=(5 * liczba_osob, 4), squeeze=False)
        for numer_wykresu, (nazwa_osoby, wzorzec) in enumerate(wzorce.items()):
            indeks_klasy = klasy.index(nazwa_osoby)
            maska_wlasna = torch.tensor([e == indeks_klasy for e in etykiety_walidacja])
            podobienstwa = policz_podobienstwa_wzorca(embeddingi_walidacja, wzorzec)

            os_wykresu = osie[0][numer_wykresu]
            os_wykresu.hist(podobienstwa[maska_wlasna].numpy(), bins=15, alpha=0.6, label="wlasne zdjecia")
            os_wykresu.hist(podobienstwa[~maska_wlasna].numpy(), bins=15, alpha=0.6, label="obce zdjecia")
            os_wykresu.axvline(kalibracja[nazwa_osoby]["prog"], color="red", linestyle="--", label="prog")
            os_wykresu.set_title(f"Kalibracja progu: {nazwa_osoby}")
            os_wykresu.set_xlabel("podobienstwo kosinusowe")
            os_wykresu.legend()

        rysunek.tight_layout()
        rysunek.savefig(FOLDER_MODELI / "kalibracja_progu.png")
        plt.close(rysunek)
        print(f"[gotowe] Wykres kalibracji progu: {FOLDER_MODELI / 'kalibracja_progu.png'}")
    except ImportError:
        print("[uwaga] matplotlib niedostepny - pomijam wykres kalibracji progu.")


if __name__ == "__main__":
    main()
