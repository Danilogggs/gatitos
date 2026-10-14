"""Prepara fotos novas para um teste visual independente de laranja/creme.

Exclui os PetIDs e conteúdos da revisão anterior. Sugestões automáticas não são
rótulos; reviewed_color fica vazio até inspeção humana. Não modifica o modelo.
"""

import argparse
import csv
import hashlib
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
    parser.add_argument('--archive', type=Path, required=True)
    parser.add_argument('--previous', type=Path, default=Path('dataset/orange-cream-review/review_queue.csv'))
    parser.add_argument('--output-dir', type=Path, default=Path('dataset/orange-cream-validation'))
    parser.add_argument('--oxford-dir', type=Path, default=Path('dataset/oxford'))
    parser.add_argument('--per-petfinder-color', type=int, default=60)
    parser.add_argument('--oxford-cream', type=int, default=40)
    args = parser.parse_args()
    rng = random.Random(20260925)
    output = args.output_dir.resolve()
    output.mkdir(parents=True, exist_ok=True)
    with args.previous.open(encoding='utf-8', newline='') as stream:
        previous = list(csv.DictReader(stream))
    previous_ids = set()
    previous_hashes = set()
    for row in previous:
        path = args.previous.parent / row['file']
        previous_hashes.add(hashlib.sha256(path.read_bytes()).hexdigest())
        if row['file'].startswith('petfinder-'):
            previous_ids.add(row['file'].split('-', 3)[-1].removesuffix('.jpg'))
    rows = []
    chosen_hashes = set(previous_hashes)
    chosen_ids = set(previous_ids)

    with ZipFile(args.archive) as archive:
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
                    pet_id = row['PetID']
                    if pet_id in chosen_ids or (f'{split}_images', pet_id) not in photos:
                        continue
                    names = {color_ids.get(row[key], '') for key in ('Color1', 'Color2', 'Color3')
                             if row[key] not in ('', '0')}
                    for name in ('golden', 'yellow'):
                        if name in names:
                            candidates[name].append((split, pet_id))
        for name in ('golden', 'yellow'):
            rng.shuffle(candidates[name])
            count = 0
            for split, pet_id in candidates[name]:
                if count >= args.per_petfinder_color:
                    break
                if pet_id in chosen_ids:
                    continue
                raw = archive.read(photos[(f'{split}_images', pet_id)])
                try:
                    with Image.open(io.BytesIO(raw)) as image:
                        prepared = image.convert('RGB')
                except (OSError, ValueError):
                    continue
                digest = hashlib.sha256(prepared.tobytes()).hexdigest()
                if digest in chosen_hashes:
                    continue
                filename = f'petfinder-{name}-{split}-{pet_id}.jpg'
                prepared.save(output / filename, quality=90)
                rows.append({'file': filename, 'source': 'petfinder', 'suggested_color': name,
                             'reviewed_color': '', 'pet_id': pet_id})
                chosen_hashes.add(digest)
                chosen_ids.add(pet_id)
                count += 1

    source = datasets.OxfordIIITPet(args.oxford_dir, split='test',
                                    target_types=['category', 'binary-category', 'segmentation'],
                                    download=False)
    cream_indices = []
    for index in range(len(source)):
        image, (_, species, trimap) = source[index]
        try:
            if species == source.bin_class_to_idx['Cat'] and (np.asarray(trimap) == 1).any():
                if 'cream' in infer_color_labels(image, trimap):
                    cream_indices.append(index)
        finally:
            image.close()
            trimap.close()
    rng.shuffle(cream_indices)
    for index in cream_indices[:args.oxford_cream]:
        image, (_, _, trimap) = source[index]
        try:
            filename = f'oxford-test-{index:05d}.jpg'
            image.convert('RGB').save(output / filename, quality=90)
            rows.append({'file': filename, 'source': 'oxford_test', 'suggested_color': 'cream',
                         'reviewed_color': '', 'pet_id': ''})
        finally:
            image.close()
            trimap.close()
    manifest = output / 'validation_queue.csv'
    with manifest.open('w', encoding='utf-8', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=['file', 'source', 'suggested_color', 'reviewed_color', 'pet_id'])
        writer.writeheader()
        writer.writerows(rows)
    print(f'Novas fotos: {len(rows)}; PetFinder IDs sem interseção com fila anterior; {manifest}')


if __name__ == '__main__':
    run()
