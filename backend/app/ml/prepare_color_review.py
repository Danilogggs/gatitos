"""Cria uma fila local de fotos Oxford para revisão humana de laranja/creme.

As sugestões são heurísticas, nunca rótulos de treino. Preencha reviewed_color
com orange, cream ou neither depois de inspecionar cada foto.
"""

import argparse
import csv
import io
import random
from collections import defaultdict
from pathlib import Path
from zipfile import ZipFile

import numpy as np
from PIL import Image
from torchvision import datasets

from app.ml.import_petfinder_zip import IMAGE_NAME
from app.ml.train_oxford_colors import infer_color_labels


def run() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument('--data-dir', type=Path, default=Path('dataset/oxford'))
    parser.add_argument('--output-dir', type=Path, default=Path('dataset/orange-cream-review'))
    parser.add_argument('--per-color', type=int, default=60)
    parser.add_argument('--petfinder-archive', type=Path,
                        help='ZIP baixado após aceitar as regras; inclui Golden/Yellow só para revisão')
    args = parser.parse_args()
    if args.per_color < 1:
        parser.error('--per-color deve ser positivo')
    source = datasets.OxfordIIITPet(args.data_dir, split='trainval',
                                    target_types=['category', 'binary-category', 'segmentation'],
                                    download=False)
    buckets = {'orange': [], 'cream': []}
    for index in range(len(source)):
        image, (_, species, trimap) = source[index]
        try:
            if species != source.bin_class_to_idx['Cat'] or not (np.asarray(trimap) == 1).any():
                continue
            labels = infer_color_labels(image, trimap)
            matches = [color for color in buckets if color in labels]
            if len(matches) == 1:
                buckets[matches[0]].append(index)
        finally:
            image.close()
            trimap.close()
    rng = random.Random(43)
    output = args.output_dir.resolve()
    output.mkdir(parents=True, exist_ok=True)
    rows = []
    for color, indices in buckets.items():
        rng.shuffle(indices)
        for index in indices[:args.per_color]:
            image, (_, _, trimap) = source[index]
            try:
                filename = f'{color}-{index:05d}.jpg'
                image.convert('RGB').save(output / filename, quality=90)
                rows.append({'file': filename, 'suggested_color': color, 'reviewed_color': ''})
            finally:
                image.close()
                trimap.close()
    if args.petfinder_archive:
        with ZipFile(args.petfinder_archive) as archive:
            color_ids = {row['ColorID']: row['ColorName'].strip().lower()
                         for row in csv.DictReader(io.TextIOWrapper(archive.open('color_labels.csv'), encoding='utf-8-sig'))}
            photos = {}
            for member in archive.namelist():
                match = IMAGE_NAME.fullmatch(member)
                if match and match.group(3) == '1':
                    photos[(match.group(1), match.group(2))] = member
            candidates = defaultdict(list)
            for split in ('train', 'test'):
                with archive.open(f'{split}/{split}.csv') as handle:
                    for row in csv.DictReader(io.TextIOWrapper(handle, encoding='utf-8-sig')):
                        if row['Type'] != '2' or row['Quantity'] != '1':
                            continue
                        names = {color_ids.get(row[key], '') for key in ('Color1', 'Color2', 'Color3')
                                 if row[key] not in ('', '0')}
                        for name in ('golden', 'yellow'):
                            if name in names and (f'{split}_images', row['PetID']) in photos:
                                candidates[name].append((split, row['PetID']))
            for name in ('golden', 'yellow'):
                rng.shuffle(candidates[name])
                for split, pet_id in candidates[name][:args.per_color]:
                    member = photos[(f'{split}_images', pet_id)]
                    try:
                        with Image.open(io.BytesIO(archive.read(member))) as image:
                            filename = f'petfinder-{name}-{split}-{pet_id}.jpg'
                            image.convert('RGB').save(output / filename, quality=90)
                    except (OSError, ValueError):
                        continue
                    rows.append({'file': filename, 'suggested_color': f'petfinder:{name}', 'reviewed_color': ''})
    with (output / 'review_queue.csv').open('w', encoding='utf-8', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=['file', 'suggested_color', 'reviewed_color'])
        writer.writeheader()
        writer.writerows(rows)
    print(f'Fila criada em {output}: {len(rows)} fotos; sugestões não são rótulos confirmados', flush=True)


if __name__ == '__main__':
    run()
