"""
Krok 2: Podzial zebranych danych na zbior treningowy i walidacyjny.

Wczytuje zdjecia z data/raw/<osoba>/*.jpg i kopiuje je do:
    data/processed/train/<osoba>/...
    data/processed/val/<osoba>/...

wg proporcji ustawionej parametrem --val-split (domyslnie 20% do walidacji).

Uzycie:
    python 02_podzial_danych.py
    python 02_podzial_danych.py --val-split 0.15
"""

from __future__ import annotations

import argparse
import random
import shutil
from pathlib import Path

RAW_DIR = Path(__file__).parent / "data" / "raw"
PROCESSED_DIR = Path(__file__).parent / "data" / "processed"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Podzial danych na train/val.")
    parser.add_argument(
        "--val-split", type=float, default=0.2, help="Ulamek danych na zbior walidacyjny"
    )
    parser.add_argument("--seed", type=int, default=42, help="Ziarno losowosci podzialu")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    random.seed(args.seed)

    people = [p for p in RAW_DIR.iterdir() if p.is_dir()]
    if not people:
        raise SystemExit(
            f"Brak danych w {RAW_DIR}. Najpierw uruchom 01_zbieranie_danych.py dla kazdej osoby."
        )

    # Czyscimy poprzedni podzial, zeby uniknac mieszania starych i nowych danych.
    if PROCESSED_DIR.exists():
        shutil.rmtree(PROCESSED_DIR)

    summary = []
    for person_dir in sorted(people):
        images = sorted(person_dir.glob("*.jpg"))
        if len(images) < 10:
            print(
                f"[uwaga] Osoba '{person_dir.name}' ma tylko {len(images)} zdjec - "
                "to bardzo malo, zalecane min. 100-150."
            )

        random.shuffle(images)
        n_val = max(1, int(len(images) * args.val_split))
        val_images = images[:n_val]
        train_images = images[n_val:]

        for split_name, split_images in (("train", train_images), ("val", val_images)):
            out_dir = PROCESSED_DIR / split_name / person_dir.name
            out_dir.mkdir(parents=True, exist_ok=True)
            for img_path in split_images:
                shutil.copy2(img_path, out_dir / img_path.name)

        summary.append((person_dir.name, len(train_images), len(val_images)))

    print("\nPodsumowanie podzialu danych:")
    print(f"{'Osoba':<25}{'Train':>8}{'Val':>8}")
    for name, n_train, n_val in summary:
        print(f"{name:<25}{n_train:>8}{n_val:>8}")

    print(f"\nDane gotowe w: {PROCESSED_DIR}")


if __name__ == "__main__":
    main()
