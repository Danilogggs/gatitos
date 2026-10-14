"""Compara checkpoints em fotos visualmente revisadas e nunca usadas no treino."""

import argparse
import csv
import hashlib
from pathlib import Path

import torch
from PIL import Image
from torch import nn
from torchvision import models, transforms

from app.ml.taxonomy import COLORS


TARGETS = ('orange', 'cream')


def run() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument('--review', type=Path,
                        default=Path('dataset/orange-cream-validation/visually_reviewed_validation.csv'))
    parser.add_argument('--previous', type=Path,
                        default=Path('dataset/orange-cream-review/visually_reviewed.csv'))
    parser.add_argument('--base', type=Path, default=Path('models/catcare-oxford-colors-length.pt'))
    parser.add_argument('--candidate', type=Path, default=Path('models/catcare-reviewed-colors-candidate.pt'))
    args = parser.parse_args()
    previous_hashes = set()
    previous_pet_ids = set()
    with args.previous.open(encoding='utf-8', newline='') as stream:
        for row in csv.DictReader(stream):
            if row['reviewed_color'] in (*TARGETS, 'neither'):
                previous_hashes.add(hashlib.sha256((args.previous.parent / row['file']).read_bytes()).hexdigest())
                if row['file'].startswith('petfinder-'):
                    previous_pet_ids.add(row['file'].split('-', 3)[-1].removesuffix('.jpg'))
    with args.review.open(encoding='utf-8', newline='') as stream:
        rows = [row for row in csv.DictReader(stream)
                if row['reviewed_color'] in (*TARGETS, 'neither')]
    hashes = [hashlib.sha256((args.review.parent / row['file']).read_bytes()).hexdigest() for row in rows]
    if len(hashes) != len(set(hashes)) or set(hashes) & previous_hashes:
        parser.error('Há foto duplicada dentro do teste ou usada na revisão de treino')
    if any(row['pet_id'] in previous_pet_ids for row in rows if row['pet_id']):
        parser.error('Há PetID no teste que foi usado na revisão de treino')
    base = torch.load(args.base, map_location='cpu', weights_only=True)
    candidate = torch.load(args.candidate, map_location='cpu', weights_only=True)
    if any(not torch.equal(base['backbone'][key], candidate['backbone'][key]) for key in base['backbone']):
        parser.error('Backbones diferentes; avaliação direta inválida')
    for name in ('breed_head', 'length_head'):
        if any(not torch.equal(base[name][key], candidate[name][key]) for key in base[name]):
            parser.error(f'{name} foi alterado indevidamente no candidato')
    untouched = [index for index, color in enumerate(COLORS) if color not in TARGETS]
    for key in ('weight', 'bias'):
        if not torch.equal(base['color_head'][key][untouched], candidate['color_head'][key][untouched]):
            parser.error(f'Outras cores foram alteradas indevidamente: {key}')
    backbone = models.efficientnet_b0(weights=None)
    size = backbone.classifier[1].in_features
    backbone.classifier = nn.Identity()
    backbone.load_state_dict(base['backbone'])
    backbone.eval()
    transform = transforms.Compose([
        transforms.Resize((224, 224)), transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])
    vectors = []
    with torch.no_grad():
        for row in rows:
            with Image.open(args.review.parent / row['file']) as image:
                vectors.append(backbone(transform(image.convert('RGB')).unsqueeze(0))[0])
    vectors = torch.stack(vectors)
    print(f'Fotos novas revisadas: {len(rows)}', flush=True)
    for model_name, checkpoint in (('base', base), ('candidato', candidate)):
        head = nn.Linear(size, len(COLORS))
        head.load_state_dict(checkpoint['color_head'])
        with torch.no_grad():
            scores = torch.sigmoid(head(vectors))
        thresholds = [float(checkpoint.get('color_thresholds', {}).get(color, 0.5)) for color in COLORS]
        for source in ('todos', 'petfinder', 'oxford_test'):
            indices = [i for i, row in enumerate(rows) if source == 'todos' or row['source'] == source]
            report = {}
            for color in TARGETS:
                column = COLORS.index(color)
                tp = fp = fn = 0
                for i in indices:
                    predicted = float(scores[i, column]) >= thresholds[column]
                    truth = rows[i]['reviewed_color'] == color
                    tp += predicted and truth
                    fp += predicted and not truth
                    fn += not predicted and truth
                precision = tp / max(1, tp + fp)
                recall = tp / max(1, tp + fn)
                report[color] = {'F1': round(2 * tp / max(1, 2 * tp + fp + fn), 3),
                                 'precision': round(precision, 3), 'recall': round(recall, 3),
                                 'positives': sum(rows[i]['reviewed_color'] == color for i in indices)}
            print(f'{model_name} {source}: n={len(indices)} {report}', flush=True)


if __name__ == '__main__':
    run()
