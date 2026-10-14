"""Importa fotos e rótulos de gatos diretamente do ZIP oficial do PetFinder.

Uso: python -m app.ml.import_petfinder_zip --archive caminho/do/arquivo.zip
Somente perfis de um gato com cores e comprimento mapeáveis são copiados.
O ZIP original e as fotos importadas devem permanecer fora do Git.
"""

import argparse
import csv
import hashlib
import io
import re
from collections import defaultdict
from pathlib import Path
from zipfile import ZipFile

from PIL import Image, UnidentifiedImageError

from app.ml.export_reviewed import FIELDS
from app.ml.import_petfinder import COLOR_NAMES, FUR_LENGTHS


IMAGE_NAME = re.compile(r'^(train_images|test_images)/([A-Za-z0-9]+)-([0-9]+)\.jpg$')


def import_archive(archive: Path, output_dir: Path) -> dict[str, int]:
    counts = {'cat_profiles': 0, 'imported': 0, 'imported_photos': 0,
              'unsupported_colors': 0, 'missing_images': 0}
    output_images = output_dir / 'images'
    output_images.mkdir(parents=True, exist_ok=True)
    records = []
    with ZipFile(archive) as source:
        color_ids = {row['ColorID']: row['ColorName'].strip().lower()
                     for row in csv.DictReader(io.TextIOWrapper(source.open('color_labels.csv'),
                                                                encoding='utf-8-sig'))}
        photos = defaultdict(list)
        for member in source.namelist():
            match = IMAGE_NAME.fullmatch(member)
            if match:
                photos[(match.group(1), match.group(2))].append((int(match.group(3)), member))

        for split, csv_name, folder in (
            ('train', 'train/train.csv', 'train_images'),
            ('test', 'test/test.csv', 'test_images'),
        ):
            with source.open(csv_name) as handle:
                rows = csv.DictReader(io.TextIOWrapper(handle, encoding='utf-8-sig'))
                for row in rows:
                    if row['Type'] != '2' or row['Quantity'] != '1':
                        continue
                    counts['cat_profiles'] += 1
                    names = [color_ids.get(row[key], '') for key in ('Color1', 'Color2', 'Color3')
                             if row[key] not in ('', '0')]
                    if not names or any(name not in COLOR_NAMES for name in names):
                        counts['unsupported_colors'] += 1
                        continue
                    length = FUR_LENGTHS.get(row['FurLength'], '')
                    if not length:
                        continue
                    pet_id = row['PetID']
                    if not re.fullmatch(r'[A-Za-z0-9]+', pet_id):
                        counts['missing_images'] += 1
                        continue
                    matching = sorted(photos.get((folder, pet_id), []))
                    if not matching:
                        counts['missing_images'] += 1
                        continue
                    saved = 0
                    for number, member in matching:
                        raw = source.read(member)
                        try:
                            with Image.open(io.BytesIO(raw)) as image:
                                image.verify()
                        except (OSError, UnidentifiedImageError, ValueError):
                            counts['missing_images'] += 1
                            continue
                        relative = Path('images') / split / f'{pet_id}-{number}.jpg'
                        destination = output_dir / relative
                        destination.parent.mkdir(parents=True, exist_ok=True)
                        if not destination.exists() or hashlib.sha256(destination.read_bytes()).digest() != hashlib.sha256(raw).digest():
                            destination.write_bytes(raw)
                        records.append({
                            'path': relative.as_posix(), 'cat_id': f'{split}-{pet_id}',
                            'image_id': f'{split}-{pet_id}-{number}',
                            'sha256': hashlib.sha256(raw).hexdigest(), 'breed': '',
                            'colors': '|'.join(dict.fromkeys(COLOR_NAMES[name] for name in names)),
                            'coat_length': length, 'source': 'petfinder_my',
                        })
                        saved += 1
                    counts['imported_photos'] += saved
                    counts['imported'] += bool(saved)
                    if counts['imported'] and counts['imported'] % 500 == 0:
                        print(f"Importados {counts['imported']} gatos / {counts['imported_photos']} fotos", flush=True)
    with (output_dir / 'labels.csv').open('w', encoding='utf-8', newline='') as handle:
        writer = csv.DictWriter(handle, fieldnames=FIELDS)
        writer.writeheader()
        writer.writerows(records)
    return counts


def run() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument('--archive', type=Path, required=True)
    parser.add_argument('--output-dir', type=Path, default=Path('dataset/petfinder'))
    args = parser.parse_args()
    counts = import_archive(args.archive, args.output_dir)
    print(f"Perfis únicos: {counts['cat_profiles']}; importados: {counts['imported']} "
          f"({counts['imported_photos']} fotos); cores incompatíveis: "
          f"{counts['unsupported_colors']}; sem foto/foto inválida: {counts['missing_images']}")
    print(f"Rótulos: {args.output_dir / 'labels.csv'}")


if __name__ == '__main__':
    run()
