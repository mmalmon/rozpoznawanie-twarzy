"""Wyjaśnianie decyzji modelu rozpoznawania twarzy (XAI).

DLA UCZNIÓW: sieć neuronowa nie „myśli” jak człowiek i nie podaje gotowego
powodu. Zamiast zgadywać, co „czuła”, robimy dwa proste eksperymenty:

1. **Najbliższe wzorce** — pokazujemy, do których zdjęć treningowych embedding
   jest najbardziej podobny. To odpowiedź na pytanie „na co model patrzył?”.
2. **Mapa okluzji** — zasłaniamy kolejne fragmenty twarzy szarym kwadratem
   i mierzymy, o ile spada podobieństwo do wybranej osoby. Jeśli zasłonięcie
   oczu bardzo psuje wynik, model naprawdę korzysta z okolic oczu. Jeśli
   wynik nie zmienia się wcale, decyzja opiera się na czymś innym (np. tle).

To nie jest doskonałe wyjaśnienie — sieć nie jest liniowa, a zasłanianie
wprowadza sztuczne zakłócenia. Jest jednak uczciwe: nie zgadujemy, tylko
sprawdzamy eksperymentalnie, co zmienia decyzję. Dlatego XAI uruchamiamy na
żądanie, a nie przy każdej klatce: to dodatkowe dziesiątki przejść przez sieć.
"""

from __future__ import annotations

import json
import time
from datetime import datetime
from pathlib import Path
from urllib.parse import quote

import cv2
import numpy as np
import torch

from common.imgio import zapisz_obraz
from common.model import oblicz_znormalizowane_cechy
from common.rozpoznawanie import NAZWA_NIEZNANEJ, ocen_kandydatow

KORZEN = Path(__file__).resolve().parent.parent
FOLDER_XAI = KORZEN / "models" / "xai"
FOLDER_TRENINGU = KORZEN / "data" / "processed" / "train"
PLIK_WZORCOW = KORZEN / "models" / "wzorce_osob.json"

MARGINES_TWARZY = 0.2
DOMYSLNA_SIATKA = 6
DOMYSLNY_TOP = 5
DOMYSLNY_ROZMIAR_PLAMKI = 0.34


def wczytaj_zrodla_wzorcow(
    sciezka: Path = PLIK_WZORCOW,
) -> dict[str, list[dict]]:
    """Czyta opcjonalne metadane XAI z pliku wzorców.

    Starsze modele nie mają pola ``zrodla``. W takim wypadku zwracamy pusty
    słownik, a panel nadal pokaże mapę okluzji i ranking kandydatów —
    tylko bez miniatur zdjęć treningowych.
    """
    try:
        dane = json.loads(sciezka.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}

    wynik: dict[str, list[dict]] = {}
    for osoba, wpis in dane.get("osoby", {}).items():
        zrodla = wpis.get("zrodla")
        if not isinstance(zrodla, list):
            continue
        poprawne = []
        for pozycja in zrodla:
            if not isinstance(pozycja, dict):
                continue
            plik = str(pozycja.get("plik", "")).strip()
            if not plik:
                continue
            try:
                jasnosc = float(pozycja.get("jasnosc", 1.0))
            except (TypeError, ValueError):
                jasnosc = 1.0
            poprawne.append({"plik": plik, "jasnosc": jasnosc})
        wynik[osoba] = poprawne
    return wynik


def _embeddingi_obrazu(
    ekstraktor: torch.nn.Module,
    obraz_rgb: np.ndarray,
    przygotuj_obrazy: list,
    urzadzenie: torch.device,
) -> torch.Tensor:
    """Liczy embeddingi jednego obrazu dla wszystkich skal kadrowania."""
    tensory = [przygotuj(obraz_rgb) for przygotuj in przygotuj_obrazy]
    obrazy = torch.stack(tensory).to(urzadzenie)
    return oblicz_znormalizowane_cechy(ekstraktor, obrazy)


def podobienstwo_do_osoby(
    embedding: torch.Tensor,
    wzorce: dict[str, dict],
    osoba: str,
) -> float:
    """Najlepsze podobieństwo do dowolnego prototypu wskazanej osoby."""
    wpis = wzorce.get(osoba)
    if wpis is None:
        return 0.0
    with torch.no_grad():
        wartosci = embedding @ wpis["prototypy"].T
        return float(wartosci.max().item())


def najblizsze_prototypy(
    embedding: torch.Tensor,
    wzorce: dict[str, dict],
    zrodla: dict[str, list[dict]] | None = None,
    top: int = DOMYSLNY_TOP,
) -> list[dict]:
    """Zwraca ``top`` najbliższych prototypów spośród wszystkich osób.

    To odpowiednik pytania: „które konkretne zdjęcia treningowe najbardziej
    przypominają tę twarz?”. Dla każdego prototypu bierzemy maksimum po
    skalach kadrowania (tak samo jak przy decyzji), a potem sortujemy globalnie.
    """
    if not wzorce:
        return []
    if embedding.ndim == 1:
        embedding = embedding.unsqueeze(0)

    top = max(1, min(int(top), 50))
    wpisy: list[dict] = []
    with torch.no_grad():
        for osoba, wpis in wzorce.items():
            prototypy = wpis["prototypy"]
            wartosci = (embedding @ prototypy.T).max(dim=0).values
            zrodla_osoby = (zrodla or {}).get(osoba, [])
            for indeks, wartosc in enumerate(wartosci.tolist()):
                zrodlo = zrodla_osoby[indeks] if indeks < len(zrodla_osoby) else {}
                wpisy.append(
                    {
                        "osoba": osoba,
                        "podobienstwo": round(float(wartosc), 4),
                        "plik": zrodlo.get("plik"),
                        "jasnosc": zrodlo.get("jasnosc"),
                        "indeks": indeks,
                    }
                )

    wpisy.sort(
        key=lambda element: (
            -element["podobienstwo"],
            element["osoba"],
            element["indeks"],
        )
    )
    return wpisy[:top]


def _opis_komorki(wiersz: int, kolumna: int, siatka: int) -> str:
    """Krótki opis położenia najważniejszego fragmentu twarzy."""
    if wiersz < siatka / 3:
        pion = "górnej"
    elif wiersz >= 2 * siatka / 3:
        pion = "dolnej"
    else:
        pion = "środkowej"

    if kolumna < siatka / 3:
        poziom = "lewej"
    elif kolumna >= 2 * siatka / 3:
        poziom = "prawej"
    else:
        poziom = "środkowej"

    return f"najważniejszy obszar to {pion} {poziom} część twarzy"


def mapa_okluzji(
    ekstraktor: torch.nn.Module,
    obraz_rgb: np.ndarray,
    przygotuj_obrazy: list,
    wzorce: dict[str, dict],
    osoba: str,
    urzadzenie: torch.device,
    podobienstwo_bazowe: float,
    siatka: int = DOMYSLNA_SIATKA,
    rozmiar_plamki: float = DOMYSLNY_ROZMIAR_PLAMKI,
) -> dict:
    """Liczy mapę ważności metodą okluzji (occlusion sensitivity).

    Dzielimy twarz na siatkę ``siatka x siatka``. Dla każdej komórki
    zasłaniamy odpowiadający jej fragment szarym prostokątem i liczymy, jak
    bardzo spadło podobieństwo do wskazanej osoby. Duży spadek = fragment
    był ważny dla decyzji.

    Wszystkie zasłonięte warianty przepuszczamy przez sieć w JEDNEJ paczce —
    dlatego wyjaśnienie trwa sekundy, a nie minuty.
    """
    if siatka < 2:
        raise ValueError("Siatka musi mieć co najmniej 2x2.")
    if not 0.05 <= rozmiar_plamki <= 0.9:
        raise ValueError("Rozmiar plamki musi być w zakresie 0.05-0.9.")

    obraz = np.ascontiguousarray(obraz_rgb)
    wysokosc, szerokosc = obraz.shape[:2]
    if wysokosc < 8 or szerokosc < 8:
        raise ValueError("Twarz jest za mała, żeby policzyć mapę okluzji.")

    krok_y = wysokosc / siatka
    krok_x = szerokosc / siatka
    rozmiar_y = max(1, int(round(wysokosc * rozmiar_plamki)))
    rozmiar_x = max(1, int(round(szerokosc * rozmiar_plamki)))

    zasloniete: list[np.ndarray] = []
    for wiersz in range(siatka):
        srodek_y = int((wiersz + 0.5) * krok_y)
        y1 = max(0, srodek_y - rozmiar_y // 2)
        y2 = min(wysokosc, y1 + rozmiar_y)
        y1 = max(0, y2 - rozmiar_y)
        for kolumna in range(siatka):
            srodek_x = int((kolumna + 0.5) * krok_x)
            x1 = max(0, srodek_x - rozmiar_x // 2)
            x2 = min(szerokosc, x1 + rozmiar_x)
            x1 = max(0, x2 - rozmiar_x)

            wariant = obraz.copy()
            # Neutralna szarość zamiast czerni: czerń bywa „innym obiektem”,
            # a szary kwadrat po prostu odbiera informację.
            wariant[y1:y2, x1:x2] = 128
            zasloniete.append(wariant)

    liczba_skal = len(przygotuj_obrazy)
    tensory = [
        przygotuj(wariant)
        for wariant in zasloniete
        for przygotuj in przygotuj_obrazy
    ]
    obrazy = torch.stack(tensory).to(urzadzenie)
    embeddingi = oblicz_znormalizowane_cechy(ekstraktor, obrazy)

    mapa = np.zeros((siatka, siatka), dtype=np.float32)
    for indeks in range(len(zasloniete)):
        zakres = slice(indeks * liczba_skal, (indeks + 1) * liczba_skal)
        podobienstwo = podobienstwo_do_osoby(embeddingi[zakres], wzorce, osoba)
        mapa.flat[indeks] = max(0.0, float(podobienstwo_bazowe) - podobienstwo)

    maksymalny_spadek = float(mapa.max())
    if maksymalny_spadek > 1e-9:
        mapa_norm = mapa / maksymalny_spadek
    else:
        mapa_norm = np.zeros_like(mapa)

    mapa_pelna = cv2.resize(
        mapa_norm, (szerokosc, wysokosc), interpolation=cv2.INTER_CUBIC
    )
    mapa_pelna = np.clip(mapa_pelna, 0.0, 1.0)

    wiersz, kolumna = np.unravel_index(int(mapa.argmax()), mapa.shape)
    return {
        "mapa": [[round(float(wartosc), 4) for wartosc in wiersz_] for wiersz_ in mapa],
        "mapa_pelna": mapa_pelna,
        "maks_spadek": round(maksymalny_spadek, 4),
        "komorka": {"wiersz": int(wiersz), "kolumna": int(kolumna)},
        "opis": _opis_komorki(int(wiersz), int(kolumna), siatka),
        "siatka": siatka,
    }


def zapisz_wizualizacje(
    obraz_rgb: np.ndarray,
    mapa_pelna: np.ndarray,
    sciezka: Path,
) -> bool:
    """Zapisuje obok siebie oryginał i nałożoną mapę cieplną."""
    obraz_bgr = cv2.cvtColor(obraz_rgb, cv2.COLOR_RGB2BGR)
    wysokosc, szerokosc = obraz_bgr.shape[:2]

    mapa_255 = np.uint8(np.clip(mapa_pelna, 0.0, 1.0) * 255)
    kolor = cv2.applyColorMap(mapa_255, cv2.COLORMAP_JET)
    nalozone = cv2.addWeighted(obraz_bgr, 0.55, kolor, 0.45, 0.0)

    separator = np.full((wysokosc, 8, 3), 255, dtype=np.uint8)
    panel = np.hstack([obraz_bgr, separator, nalozone])

    # Podpisy po angielsku, bo czcionka OpenCV nie ma polskich znaków.
    cv2.putText(
        panel,
        "oryginal",
        (8, 20),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.5,
        (255, 255, 255),
        1,
        cv2.LINE_AA,
    )
    cv2.putText(
        panel,
        "wazne dla decyzji",
        (szerokosc + 16, 20),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.5,
        (255, 255, 255),
        1,
        cv2.LINE_AA,
    )

    if szerokosc < 180:
        skala = max(1, int(round(180 / szerokosc)))
        panel = cv2.resize(
            panel, None, fx=skala, fy=skala, interpolation=cv2.INTER_NEAREST
        )
    return zapisz_obraz(sciezka, panel)


def wyjasnij_twarz(
    obraz_bgr: np.ndarray,
    ekstraktor: torch.nn.Module,
    przygotuj_obrazy: list,
    wzorce: dict[str, dict],
    urzadzenie: torch.device,
    zrodla: dict[str, list[dict]] | None = None,
    korekta_progu: float = 0.0,
    prostokat: tuple[int, int, int, int] | None = None,
    top: int = DOMYSLNY_TOP,
    siatka: int = DOMYSLNA_SIATKA,
    zapisz: bool = True,
    folder_wynikow: Path = FOLDER_XAI,
) -> dict:
    """Pełne wyjaśnienie decyzji dla jednej twarzy.

    ``prostokat`` to (x, y, szerokość, wysokość) w pikselach obrazu. Gdy go
    brak, cały obraz jest traktowany jak wycinek twarzy (przydatne w testach
    i w skrypcie CLI, gdy zdjęcie jest już przycięte).
    """
    if not wzorce:
        raise RuntimeError(
            "Brak wzorców tożsamości. Najpierw uruchom trening (03)."
        )

    obraz_bgr = np.asarray(obraz_bgr)
    if obraz_bgr.ndim != 3 or obraz_bgr.shape[2] != 3:
        raise ValueError("Obraz musi być kolorowy (trzy kanały BGR).")

    wysokosc_obrazu, szerokosc_obrazu = obraz_bgr.shape[:2]
    if prostokat is None:
        x, y, szerokosc, wysokosc = 0, 0, szerokosc_obrazu, wysokosc_obrazu
    else:
        x, y, szerokosc, wysokosc = (int(wartosc) for wartosc in prostokat)
    if szerokosc <= 0 or wysokosc <= 0:
        raise ValueError("Prostokąt twarzy ma niepoprawne wymiary.")

    margines_x = int(szerokosc * MARGINES_TWARZY)
    margines_y = int(wysokosc * MARGINES_TWARZY)
    x1 = max(0, x - margines_x)
    y1 = max(0, y - margines_y)
    x2 = min(szerokosc_obrazu, x + szerokosc + margines_x)
    y2 = min(wysokosc_obrazu, y + wysokosc + margines_y)
    wycinek = obraz_bgr[y1:y2, x1:x2]
    if wycinek.size == 0:
        raise ValueError("Prostokąt twarzy jest pusty.")

    wycinek_rgb = cv2.cvtColor(wycinek, cv2.COLOR_BGR2RGB)
    start = time.perf_counter()

    embedding = _embeddingi_obrazu(
        ekstraktor, wycinek_rgb, przygotuj_obrazy, urzadzenie
    )
    etykieta, _, kandydaci = ocen_kandydatow(embedding, wzorce, korekta_progu)
    if not kandydaci:
        raise RuntimeError("Model nie zwrócił żadnego kandydata do wyjaśnienia.")

    # Gdy model nikogo nie zaakceptował, wyjaśniamy NAJBLIŻSZEGO kandydata.
    # To nadal uczy: pokazuje, dlaczego model był blisko (albo daleko) progu.
    rozpoznana = etykieta != NAZWA_NIEZNANEJ
    osoba = etykieta if rozpoznana else kandydaci[0]["osoba"]
    wpis_osoby = next(
        (kandydat for kandydat in kandydaci if kandydat["osoba"] == osoba), None
    )
    prog = wpis_osoby["prog"] if wpis_osoby is not None else None

    podobienstwo_bazowe = podobienstwo_do_osoby(embedding, wzorce, osoba)
    najblizsze = najblizsze_prototypy(embedding, wzorce, zrodla, top=top)
    for wpis in najblizsze:
        plik = wpis.get("plik")
        wpis["url"] = (
            f"/api/xai/wzorzec?plik={quote(str(plik), safe='')}" if plik else None
        )

    okluzja = mapa_okluzji(
        ekstraktor,
        wycinek_rgb,
        przygotuj_obrazy,
        wzorce,
        osoba,
        urzadzenie,
        podobienstwo_bazowe=podobienstwo_bazowe,
        siatka=siatka,
    )

    obraz_url = None
    if zapisz:
        folder_wynikow.mkdir(parents=True, exist_ok=True)
        nazwa = f"wyjasnienie_{datetime.now():%Y%m%d_%H%M%S_%f}.png"
        if zapisz_wizualizacje(
            wycinek_rgb, okluzja["mapa_pelna"], folder_wynikow / nazwa
        ):
            obraz_url = f"/api/xai/obraz/{nazwa}"

    return {
        "osoba": osoba,
        "rozpoznana": rozpoznana,
        "etykieta": etykieta,
        "podobienstwo": round(float(podobienstwo_bazowe), 4),
        "prog": prog,
        "liczba_wzorcow": sum(
            int(wpis["prototypy"].shape[0]) for wpis in wzorce.values()
        ),
        "kandydaci": kandydaci[:5],
        "wzorce": najblizsze,
        "mapa": okluzja["mapa"],
        "maks_spadek": okluzja["maks_spadek"],
        "komorka": okluzja["komorka"],
        "opis": okluzja["opis"],
        "siatka": okluzja["siatka"],
        "obraz_url": obraz_url,
        "czas_ms": int((time.perf_counter() - start) * 1000),
    }
