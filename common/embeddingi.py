"""Wczytywanie zdjęć i liczenie embeddingów - wspólne dla ewaluacji i eksperymentów.

DLA UCZNIÓW: dlaczego osobny moduł?

Skrypty `07_ewaluacja.py` i `08_eksperymenty.py` muszą przygotować dane
dokładnie w ten sam sposób: te same rozmiary, te same warianty jasności i ta
sama normalizacja. Gdyby każdy skrypt robił to po swojemu, wyniki nie byłyby
porównywalne. Tutaj jest jedno źródło prawdy - tak jak w prawdziwym projekcie
ML, gdzie pipeline danych jest wspólny dla treningu i testów.

Trzy warianty jasności (0.65, 1.0, 1.4) odpowiadają temu, co robi skrypt 03
przy budowaniu wzorców: model widzi zdjęcie ciemniejsze, oryginalne i jaśniejsze,
dzięki czemu jest mniej wrażliwy na oświetlenie.
"""

from __future__ import annotations

from pathlib import Path
from typing import Protocol

import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision import datasets, transforms

from common.model import oblicz_znormalizowane_cechy, przygotuj_przeksztalcenie

WSPOLCZYNNIKI_JASNOSCI = (0.65, 1.0, 1.4)
ROZSZERZENIA_OBRAZOW = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


class LicznikPostepu(Protocol):
    """Minimalny interfejs licznika postępu (patrz klasa Postep w skryptach)."""

    def dodaj(self, ile: int) -> None: ...


def policz_obrazy(folder: Path) -> int:
    """Liczy pliki-obrazki w folderze (do mianownika paska postępu)."""
    if not folder.exists():
        return 0
    return sum(
        1
        for plik in folder.rglob("*")
        if plik.is_file() and plik.suffix.lower() in ROZSZERZENIA_OBRAZOW
    )


class ZbiorPlikow(Dataset):
    """Prosty zbiór zdjęć z jednego, płaskiego folderu (np. data/raw/nieznajomy).

    ImageFolder wymaga podfolderów-klas. Tutaj wszystkie zdjęcia to jedna
    klasa "obcy", więc wystarczy lista plików.
    """

    def __init__(self, pliki: list[Path], transform) -> None:
        self.pliki = pliki
        self.transform = transform

    def __len__(self) -> int:
        return len(self.pliki)

    def __getitem__(self, indeks: int):
        with Image.open(self.pliki[indeks]) as obraz:
            obraz = obraz.convert("RGB")
            return self.transform(obraz), 0


def transformacja(jasnosc: float = 1.0) -> transforms.Compose:
    """Dokładnie ten sam sposób przygotowania obrazu co w skrypcie 03."""
    kroki = []
    if jasnosc != 1.0:
        kroki.append(transforms.ColorJitter(brightness=(jasnosc, jasnosc)))
    kroki.append(przygotuj_przeksztalcenie())
    return transforms.Compose(kroki)


def ladowarka(zbior: Dataset, rozmiar_batcha: int) -> DataLoader:
    return DataLoader(
        zbior,
        batch_size=rozmiar_batcha,
        shuffle=False,
        num_workers=2 if len(zbior) > 50 else 0,
    )


def policz_embeddingi_folderu(
    ekstraktor: torch.nn.Module,
    folder: Path,
    urzadzenie: torch.device,
    rozmiar_batcha: int,
    transform,
    postep: LicznikPostepu | None = None,
) -> tuple[torch.Tensor, list[int], list[str], list[Path]]:
    """Liczy embeddingi zdjęć z folderu z podfolderami-klasami (train/val/test).

    Zwraca: macierz [N, D], etykiety liczbowe, nazwy klas i ścieżki plików
    w tej samej kolejności co wiersze macierzy.
    """
    zbior = datasets.ImageFolder(folder, transform=transform)
    if len(zbior) == 0:
        raise SystemExit(f"Brak zdjęć w {folder}.")

    ladowarka_zbioru = ladowarka(zbior, rozmiar_batcha)
    embeddingi: list[torch.Tensor] = []
    etykiety: list[int] = []
    for obrazy, etykiety_batch in ladowarka_zbioru:
        embeddingi.append(
            oblicz_znormalizowane_cechy(ekstraktor, obrazy.to(urzadzenie)).cpu()
        )
        etykiety.extend(etykiety_batch.tolist())
        if postep is not None:
            postep.dodaj(obrazy.shape[0])

    return (
        torch.cat(embeddingi, dim=0),
        etykiety,
        list(zbior.classes),
        [Path(plik) for plik, _ in zbior.samples],
    )


def policz_embeddingi_plaskie(
    ekstraktor: torch.nn.Module,
    folder: Path,
    urzadzenie: torch.device,
    rozmiar_batcha: int,
    transform,
    postep: LicznikPostepu | None = None,
) -> tuple[torch.Tensor, list[Path]]:
    """Liczy embeddingi wszystkich zdjęć z płaskiego folderu (bez podfolderów)."""
    pliki = sorted(
        plik
        for plik in folder.iterdir()
        if plik.is_file() and plik.suffix.lower() in ROZSZERZENIA_OBRAZOW
    )
    if not pliki:
        return torch.empty((0, 0)), []

    zbior = ZbiorPlikow(pliki, transform)
    ladowarka_zbioru = ladowarka(zbior, rozmiar_batcha)
    embeddingi: list[torch.Tensor] = []
    for obrazy, _ in ladowarka_zbioru:
        embeddingi.append(
            oblicz_znormalizowane_cechy(ekstraktor, obrazy.to(urzadzenie)).cpu()
        )
        if postep is not None:
            postep.dodaj(obrazy.shape[0])
    return torch.cat(embeddingi, dim=0), pliki
