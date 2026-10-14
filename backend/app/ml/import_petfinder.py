"""Converte um download autorizado do PetFinder.my em fotos e rótulos locais.

O usuário deve obter train.csv, color_labels.csv e train_images pela página oficial
do concurso Kaggle e aceitar suas regras. Este script não baixa nem publica o dataset.
Uso: python -m app.ml.import_petfinder --input-dir dataset/petfinder-raw
"""

import argparse
import csv
import hashlib
import re
import shutil
from pathlib import Path

from PIL import Image, UnidentifiedImageError

from app.ml.export_reviewed import FIELDS


COLOR_NAMES = {
    'black': 'black', 'brown': 'brown', 'cream': 'cream',
    'gray': 'gray', 'grey': 'gray', 'white': 'white',
}
FUR_LENGTHS = {'1': 'short', '2': 'medium', '3': 'long'}


def import_dataset(input_dir: Path, output_dir: Path) -> dict[str, int]:
    for name in ('train.csv', 'color_labels.csv'):
        if not (input_dir / name).is_file():
            raise FileNotFoundError(input_dir / name)
    image_root = input_dir / 'train_images'
    if not image_root.is_dir():
        raise FileNotFoundError(image_root)
    with (input_dir / 'color_labels.csv').open(encoding='utf-8-sig', newline='') as stream:
        color_ids = {row['ColorID']: row['ColorName'].strip().lower()
                     for row in csv.DictReader(stream)}

    output_images = output_dir / 'images'
    output_images.mkdir(parents=True, exist_ok=True)
    records = []
    counts = {'cat_profiles': 0, 'imported': 0, 'imported_photos': 0,
              'unsupported_colors': 0, 'missing_images': 0}
    with (input_dir / 'train.csv').open(encoding='utf-8-sig', newline='') as stream:
        for row in csv.DictReader(stream):
            if row['Type'] != '2' or row['Quantity'] != '1':
                continue
            counts['cat_profiles'] += 1
            color_names = [color_ids.get(row[key], '') for key in ('Color1', 'Color2', 'Color3')
                           if row[key] not in ('', '0')]
            if not color_names or any(name not in COLOR_NAMES for name in color_names):
                counts['unsupported_colors'] += 1
                continue
            length = FUR_LENGTHS.get(row['FurLength'], '')
            if not length:
                continue
            pet_id = row['PetID']
            if not re.fullmatch(r'[A-Za-z0-9]+', pet_id):
                counts['missing_images'] += 1
                continue
            photos = sorted(image_root.glob(f'{pet_id}-*.jpg'),
                            key=lambda path: int(path.stem.rsplit('-', 1)[-1])
                            if path.stem.rsplit('-', 1)[-1].isdigit() else 9999)
            if not photos:
                counts['missing_images'] += 1
                continue
            imported_for_cat = 0
            for source in photos:
                try:
                    with Image.open(source) as image:
                        image.verify()
                except (OSError, UnidentifiedImageError, ValueError):
                    counts['missing_images'] += 1
                    continue
                relative = Path('images') / source.name
                destination = output_dir / relative
                if not destination.exists():
                    shutil.copy2(source, destination)
                records.append({
                    'path': relative.as_posix(), 'cat_id': pet_id, 'image_id': source.stem,
                    'sha256': hashlib.sha256(source.read_bytes()).hexdigest(),
                    'breed': '', 'colors': '|'.join(dict.fromkeys(COLOR_NAMES[name] for name in color_names)),
                    'coat_length': length, 'source': 'petfinder_my',
                })
                imported_for_cat += 1
            counts['imported_photos'] += imported_for_cat
            counts['imported'] += bool(imported_for_cat)
    with (output_dir / 'labels.csv').open('w', encoding='utf-8', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=FIELDS)
        writer.writeheader()
        writer.writerows(records)
    return counts


def run() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument('--input-dir', type=Path, required=True)
    parser.add_argument('--output-dir', type=Path, default=Path('dataset/petfinder'))
    args = parser.parse_args()
    counts = import_dataset(args.input_dir, args.output_dir)
    print(f"Perfis de gato único: {counts['cat_profiles']}; importados: {counts['imported']} "
          f"({counts['imported_photos']} fotos); "
          f"cores fora da taxonomia: {counts['unsupported_colors']}; sem foto: {counts['missing_images']}")
    print(f"Rótulos: {args.output_dir / 'labels.csv'}")


if __name__ == '__main__':
    run()
