"""
Modul pomocniczy: automatyczne przelaczanie na wlasciwy interpreter Pythona
(wirtualne srodowisko projektu w folderze .venv).

DLA UCZNIOW - po co to jest?

Ten projekt ma wlasne "wirtualne srodowisko" (venv) w folderze .venv - to
osobna, odizolowana instalacja Pythona razem ze wszystkimi bibliotekami
projektu (opencv-python, torch, torchvision itd.), zeby nie mieszac ich z
innymi projektami czy z systemowym/globalnym Pythonem (np. Anaconda/Miniconda).

Problem: jesli na komputerze jest wiecej niz jeden Python (np. Miniconda w
PATH systemowym ORAZ .venv tego projektu), samo polecenie "python skrypt.py"
w terminalu moze uruchomic ZUPELNIE INNY Python niz ten z .venv - taki,
w ktorym NIE MA zainstalowanego cv2/torch. Wtedy program konczy sie bledem
"ModuleNotFoundError: No module named 'cv2'", mimo ze biblioteka jest
poprawnie zainstalowana - po prostu w innym, "niewlasciwym" Pythonie.

Funkcja przelacz_na_venv_jesli_trzeba() ponizej sprawdza, czy skrypt jest
aktualnie uruchomiony przez Python z folderu .venv tego projektu. Jesli nie
(a .venv istnieje), automatycznie URUCHAMIA TEN SAM SKRYPT PONOWNIE, ale juz
przy uzyciu poprawnego interpretera z .venv - dzieki temu uczniowie moga po
prostu pisac "python 04_rozpoznawanie_na_zywo.py" w dowolnym terminalu, bez
pamietania o recznym aktywowaniu wirtualnego srodowiska (".venv\\Scripts\\
Activate.ps1") za kazdym razem.

WAZNE: ten modul musi byc zaimportowany i wywolany na samej gorze kazdego
skryptu startowego (00../01../02../03../04..), ZANIM zaimportujemy cv2/torch/
torchvision - inaczej program i tak wywroci sie bledem importu, zanim zdazy
sie przelaczyc. Dlatego ten plik celowo uzywa WYLACZNIE modulow
z biblioteki standardowej (os, sys, pathlib) - one sa dostepne w kazdym
Pythonie, wiec ten "kontrolny" import zawsze sie powiedzie.
"""

from __future__ import annotations

# os - potrzebny do odczytania/ustawienia zmiennej srodowiskowej, ktora
# zapobiega nieskonczonej petli przelaczania (patrz ponizej).
import os

# subprocess - do uruchomienia TEGO SAMEGO skryptu ponownie, ale juz przy
# uzyciu interpretera z .venv. Uzywamy subprocess (a nie os.execv/execve),
# bo na Windows funkcje execv* nie sa prawdziwym "podmienieniem" procesu
# (jak na Linuksie) i w praktyce potrafia sie wywalic z bledem dostepu do
# pamieci - subprocess.run() jest tutaj znacznie bardziej niezawodny.
import subprocess

# sys - odczytujemy z niego sciezke aktualnie uzywanego interpretera
# (sys.executable) oraz argumenty uruchomieniowe (sys.argv), ktore trzeba
# przekazac dalej do ponownie uruchamianego procesu.
import sys

# Path - obiektowa reprezentacja sciezek plikow/folderow, uzywana do
# wyliczenia polozenia folderu .venv wzgledem tego pliku.
from pathlib import Path

# Nazwa zmiennej srodowiskowej ustawianej TYLKO wewnatrz procesu potomnego
# (tego uruchomionego juz przez .venv) - jej obecnosc informuje kolejne
# wywolanie tej funkcji "nie przelaczaj sie ponownie, juz tu jestes",
# co zabezpiecza przed przypadkowa nieskonczona petla restartow.
_FLAGA_JUZ_PRZELACZONO = "ROZPOZNAWANIE_TWARZY_VENV_AKTYWNY"


def _sciezka_pythona_w_venv() -> Path | None:
    """Zwraca sciezke do interpretera Pythona wewnatrz folderu .venv tego
    projektu, albo None, jesli taki folder nie istnieje (np. ktos jeszcze
    nie zainstalowal zaleznosci - patrz sekcja instalacji w README.md).
    """
    # __file__ to sciezka do TEGO pliku (common/uruchom_w_venv.py) - .parent
    # daje folder common/, kolejny .parent daje glowny folder projektu
    # (tam, gdzie lezy folder .venv i skrypty 00../01.. itd.).
    folder_projektu = Path(__file__).resolve().parent.parent

    # Na Windows interpreter venv lezy w .venv/Scripts/python.exe, a na
    # Linuksie/macOS w .venv/bin/python - sprawdzamy oba warianty, zeby
    # kod dzialal niezaleznie od systemu operacyjnego.
    kandydaci = [
        folder_projektu / ".venv" / "Scripts" / "python.exe",
        folder_projektu / ".venv" / "bin" / "python",
    ]
    for kandydat in kandydaci:
        if kandydat.exists():
            return kandydat
    return None


def przelacz_na_venv_jesli_trzeba() -> None:
    """Jesli ten skrypt zostal uruchomiony innym Pythonem niz ten z .venv
    projektu (a .venv istnieje), uruchamia go ponownie poprawnym
    interpreterem i konczy biezacy proces.

    Wywolujcie te funkcje jako PIERWSZA linijka kodu w kazdym skrypcie
    startowym - przed importem cv2/torch/numpy itp.
    """
    # Jesli juz jestesmy w procesie potomnym (patrz _FLAGA_JUZ_PRZELACZONO
    # wyzej) - nic wiecej nie robimy, zeby uniknac petli.
    if os.environ.get(_FLAGA_JUZ_PRZELACZONO) == "1":
        return

    sciezka_venv = _sciezka_pythona_w_venv()
    if sciezka_venv is None:
        # Brak folderu .venv - prawdopodobnie ktos uruchamia projekt po raz
        # pierwszy i jeszcze nie wykonal kroków z sekcji instalacji w
        # README.md. Nie przerywamy programu tutaj - pozwalamy, zeby dalsze
        # importy (cv2, torch) same zglosily czytelny blad, jesli czegos
        # brakuje.
        return

    # Porownujemy sciezki "rozwiazane" (resolve()) - eliminuje to roznice
    # takie jak wielkosc liter na Windows czy skroty w stylu "..\" w sciezce,
    # ktore inaczej moglyby oszukac proste porownanie tekstowe.
    if Path(sys.executable).resolve() == sciezka_venv.resolve():
        # Juz uruchomieni poprawnym interpreterem z .venv - nic do zrobienia.
        return

    print(
        f"[info] Uruchomiono przez '{sys.executable}', ktory nie ma "
        "zainstalowanych bibliotek tego projektu - przelaczam na "
        f"wirtualne srodowisko projektu: {sciezka_venv}"
    )

    # Przygotowujemy zmienne srodowiskowe procesu potomnego - kopia
    # obecnych (os.environ) plus nasza flaga zabezpieczajaca przed petla.
    nowe_srodowisko = dict(os.environ)
    nowe_srodowisko[_FLAGA_JUZ_PRZELACZONO] = "1"

    # subprocess.run(...) uruchamia nowy proces z venv, CZEKA az sie
    # zakonczy, i "przezroczyscie" przekazuje mu terminal (stdin/stdout/
    # stderr) - dzieki temu okno podgladu kamery (cv2.imshow), wpisywanie
    # tekstu czy Ctrl+C dzialaja dokladnie tak samo, jakby od razu
    # uruchomiono program poprawnym Pythonem. sys.argv zawiera nazwe
    # skryptu i wszystkie argumenty (np. --kamera 0), ktore trzeba
    # przekazac dalej bez zmian.
    wynik = subprocess.run(
        [str(sciezka_venv), *sys.argv],
        env=nowe_srodowisko,
    )

    # Konczymy TEN (nadrzedny) proces z takim samym kodem wyjscia, jaki
    # zwrocil wlasciwy program - dzieki temu np. skrypty *.bat/*.ps1
    # wywolujace ten plik nadal poprawnie wykryja sukces/blad.
    sys.exit(wynik.returncode)
