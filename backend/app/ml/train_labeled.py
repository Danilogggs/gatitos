"""Treina candidato com fotos rotuladas, sem trocar automaticamente o modelo ativo.

Uso: python -m app.ml.train_labeled --manifest dataset/petfinder/labels.csv
Adicione --manifest dataset/reviewed/labels.csv quando houver revisões reais.
"""

import argparse
import csv
import hashlib
import random
from collections import Counter
from pathlib import Path

import torch
from PIL import Image
from torch import nn
from torchvision import models, transforms

from app.ml.taxonomy import COAT_LENGTHS, COLORS


def load_examples(manifests: list[Path]) -> list[dict]:
    grouped = {}
    hashes = set()
    for manifest in manifests:
        root = manifest.resolve().parent
        with manifest.open(encoding='utf-8', newline='') as stream:
            for row in csv.DictReader(stream):
                path = (root / row['path']).resolve()
                if not path.is_relative_to(root) or not path.is_file():
                    raise ValueError(f'Caminho inválido no manifesto: {row["path"]}')
                digest = hashlib.sha256(path.read_bytes()).hexdigest()
                if digest != row['sha256']:
                    raise ValueError(f'Foto alterada desde a exportação: {path}')
                if digest in hashes:
                    continue
                hashes.add(digest)
                colors = [color for color in row['colors'].split('|') if color]
                if any(color not in COLORS for color in colors):
                    raise ValueError(f'Cor fora da taxonomia: {colors}')
                length = row['coat_length']
                if length and length not in COAT_LENGTHS:
                    raise ValueError(f'Comprimento fora da taxonomia: {length}')
                if not colors and not length:
                    continue
                key = (row['source'], row['cat_id'])
                if key not in grouped:
                    grouped[key] = {'id': key, 'images': [], 'colors': colors, 'length': length}
                elif grouped[key]['colors'] != colors or grouped[key]['length'] != length:
                    raise ValueError(f'Rótulos conflitantes para o gato {key}')
                grouped[key]['images'].append(path)
    return list(grouped.values())


def split_examples(examples: list[dict]) -> tuple[list[dict], list[dict], list[dict]]:
    rng = random.Random(42)
    buckets = {}
    for item in examples:
        buckets.setdefault(item['id'][0], []).append(item)
    training, validation, test = [], [], []
    for bucket in buckets.values():
        rng.shuffle(bucket)
        test_size = max(1, round(len(bucket) * 0.15))
        val_size = max(1, round(len(bucket) * 0.15))
        test.extend(bucket[:test_size])
        validation.extend(bucket[test_size:test_size + val_size])
        training.extend(bucket[test_size + val_size:])
    return training, validation, test


def run() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument('--manifest', type=Path, action='append', required=True)
    parser.add_argument('--checkpoint', type=Path, default=Path('models/catcare-oxford-colors-length.pt'))
    parser.add_argument('--output', type=Path, default=Path('models/catcare-labeled-candidate.pt'))
    parser.add_argument('--max-cats', type=int, default=0,
                        help='Limite opcional de gatos; 0 usa todos os gatos válidos')
    parser.add_argument('--oxford-repeat', type=int, default=1,
                        help='Peso dos exemplos Oxford aproximados no treino, sem repetir validação/teste')
    parser.add_argument('--epochs', type=int, default=80)
    args = parser.parse_args()
    if (args.max_cats != 0 and args.max_cats < 50) or args.epochs < 1 or args.oxford_repeat < 1:
        parser.error('--max-cats deve ser 0 ou pelo menos 50; --epochs e --oxford-repeat devem ser positivos')
    examples = load_examples(args.manifest)
    if len(examples) < 50:
        parser.error(f'São necessários ao menos 50 gatos distintos rotulados; disponíveis: {len(examples)}')
    random.Random(42).shuffle(examples)
    if args.max_cats:
        examples = examples[:args.max_cats]
    training, validation, test = split_examples(examples)
    print(f'Gatos distintos: treino={len(training)}, validação={len(validation)}, teste={len(test)}', flush=True)

    checkpoint = torch.load(args.checkpoint, map_location='cpu', weights_only=True)
    backbone = models.efficientnet_b0(weights=None)
    size = backbone.classifier[1].in_features
    backbone.classifier = nn.Identity()
    backbone.load_state_dict(checkpoint['backbone'])
    backbone.eval()
    color_head = nn.Linear(size, len(COLORS))
    color_head.load_state_dict(checkpoint['color_head'])
    length_head = nn.Linear(size, len(COAT_LENGTHS))
    length_head.load_state_dict(checkpoint['length_head'])
    transform = transforms.Compose([
        transforms.Resize((224, 224)), transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])

    def embed(items):
        vectors, colors, lengths, has_colors, has_lengths = [], [], [], [], []
        with torch.no_grad():
            for index, item in enumerate(items, 1):
                photos = []
                for path in item['images']:
                    with Image.open(path) as opened:
                        photos.append(transform(opened.convert('RGB')))
                vectors.append(backbone(torch.stack(photos)).mean(dim=0))
                colors.append([float(color in item['colors']) for color in COLORS])
                lengths.append(COAT_LENGTHS.index(item['length']) if item['length'] else 0)
                has_colors.append(bool(item['colors']))
                has_lengths.append(bool(item['length']))
                if index % 200 == 0:
                    print(f'Embeddings: {index}/{len(items)}', flush=True)
        return (torch.stack(vectors), torch.tensor(colors), torch.tensor(lengths),
                torch.tensor(has_colors), torch.tensor(has_lengths))

    print('Extraindo embeddings uma vez por gato...', flush=True)
    train_data, val_data, test_data = embed(training), embed(validation), embed(test)
    if args.oxford_repeat > 1:
        oxford_indices = torch.tensor([index for index, item in enumerate(training)
                                       if item['id'][0] == 'oxford_pseudo'])
        if len(oxford_indices):
            train_data = tuple(torch.cat([part] + [part[oxford_indices]] * (args.oxford_repeat - 1))
                               for part in train_data)
            print(f'Peso Oxford no treino: {args.oxford_repeat}x; '
                  f'exemplos ponderados: {len(train_data[0])}', flush=True)
    positive = train_data[1][train_data[3]].sum(dim=0)
    active_colors = positive >= 5
    length_counts = Counter(COAT_LENGTHS[index] for index in train_data[2][train_data[4]].tolist())
    active_lengths = torch.tensor([length_counts.get(label, 0) >= 5 for label in COAT_LENGTHS])
    train_length = int(active_lengths.sum()) >= 2
    if not active_colors.any() and not train_length:
        parser.error('Ainda não há exemplos suficientes em nenhuma cor ou em todas as classes de pelo')
    print('Cores disponíveis:', dict(zip(COLORS, positive.int().tolist())), flush=True)
    print('Comprimento disponível:', dict(length_counts), '; treinar pelo:', train_length,
          '; classes sem exemplos preservadas:', [label for label, active in zip(COAT_LENGTHS, active_lengths)
                                                   if not active], flush=True)
    initial_color = {key: value.detach().clone() for key, value in color_head.state_dict().items()}
    initial_length = {key: value.detach().clone() for key, value in length_head.state_dict().items()}
    parameters = list(color_head.parameters()) + (list(length_head.parameters()) if train_length else [])
    optimizer = torch.optim.Adam(parameters, lr=2e-4)
    pos_weight = ((train_data[3].sum() - positive) / positive.clamp_min(1)).clamp(1, 8)
    bce = nn.BCEWithLogitsLoss(pos_weight=pos_weight, reduction='none')
    ce = nn.CrossEntropyLoss()
    thresholds = torch.tensor([float(checkpoint.get('color_thresholds', {}).get(color, 0.5)) for color in COLORS])

    def metrics(data):
        with torch.no_grad():
            vectors, color_targets, length_targets, color_mask, length_mask = data
            predictions = torch.sigmoid(color_head(vectors)) >= thresholds
            truth = color_targets.bool()
            scores = []
            for index in range(len(COLORS)):
                if not active_colors[index] or not color_mask.any():
                    continue
                predicted, expected = predictions[color_mask, index], truth[color_mask, index]
                if not expected.any():
                    continue
                tp = (predicted & expected).sum().item()
                fp = (predicted & ~expected).sum().item()
                fn = (~predicted & expected).sum().item()
                scores.append(2 * tp / max(1, 2 * tp + fp + fn))
            color_f1 = sum(scores) / len(scores) if scores else 0.0
            length_accuracy = float((length_head(vectors[length_mask]).argmax(dim=1) == length_targets[length_mask]).float().mean()) if length_mask.any() else 0.0
            return color_f1, length_accuracy

    def color_report(data):
        with torch.no_grad():
            vectors, targets, _, mask, _ = data
            predicted = torch.sigmoid(color_head(vectors[mask])) >= thresholds
            expected = targets[mask].bool()
            report = {}
            for index, color in enumerate(COLORS):
                support = int(expected[:, index].sum())
                if not support:
                    continue
                tp = (predicted[:, index] & expected[:, index]).sum().item()
                fp = (predicted[:, index] & ~expected[:, index]).sum().item()
                fn = (~predicted[:, index] & expected[:, index]).sum().item()
                report[color] = {'support': support, 'f1': 2 * tp / max(1, 2 * tp + fp + fn)}
            return report

    baseline_val = metrics(val_data)
    baseline_test = metrics(test_data)
    baseline_colors = color_report(test_data)
    best_score = -1.0
    best_color, best_length = None, None
    generator = torch.Generator().manual_seed(42)
    for epoch in range(1, args.epochs + 1):
        order = torch.randperm(len(train_data[0]), generator=generator)
        for start in range(0, len(order), 32):
            selected = order[start:start + 32]
            vectors, targets, lengths, has_color, has_length = [part[selected] for part in train_data]
            losses = []
            if has_color.any() and active_colors.any():
                loss = bce(color_head(vectors[has_color]), targets[has_color])[:, active_colors].mean()
                losses.append(loss)
            if train_length and has_length.any():
                losses.append(ce(length_head(vectors[has_length]), lengths[has_length]))
            if not losses:
                continue
            optimizer.zero_grad()
            sum(losses).backward()
            optimizer.step()
            # Preserve classes with no trustworthy positive examples.
            with torch.no_grad():
                color_head.weight[~active_colors] = initial_color['weight'][~active_colors]
                color_head.bias[~active_colors] = initial_color['bias'][~active_colors]
                if train_length:
                    length_head.weight[~active_lengths] = initial_length['weight'][~active_lengths]
                    length_head.bias[~active_lengths] = initial_length['bias'][~active_lengths]
        val = metrics(val_data)
        score = val[0] + (val[1] if train_length else 0)
        if score > best_score:
            best_score = score
            best_color = {key: value.detach().clone() for key, value in color_head.state_dict().items()}
            best_length = {key: value.detach().clone() for key, value in length_head.state_dict().items()}
        if epoch == 1 or epoch % 10 == 0 or epoch == args.epochs:
            print(f'Época {epoch}: validação cor F1={val[0]:.1%}, pelo acc={val[1]:.1%}', flush=True)
    color_head.load_state_dict(best_color)
    length_head.load_state_dict(best_length)
    # Ajuste de limiar usa apenas a validação; o teste abaixo permanece intocado.
    with torch.no_grad():
        probabilities = torch.sigmoid(color_head(val_data[0][val_data[3]]))
        expected = val_data[1][val_data[3]].bool()
        for index, color in enumerate(COLORS):
            if not active_colors[index] or expected[:, index].sum() < 3 or (~expected[:, index]).sum() < 3:
                continue
            best_threshold, best_f1 = float(thresholds[index]), -1.0
            for candidate in torch.arange(0.20, 0.81, 0.05):
                predicted = probabilities[:, index] >= candidate
                truth = expected[:, index]
                tp = (predicted & truth).sum().item()
                fp = (predicted & ~truth).sum().item()
                fn = (~predicted & truth).sum().item()
                f1 = 2 * tp / max(1, 2 * tp + fp + fn)
                if f1 > best_f1:
                    best_threshold, best_f1 = float(candidate), f1
            thresholds[index] = best_threshold
    candidate_test = metrics(test_data)
    candidate_colors = color_report(test_data)
    print(f'Teste real reservado: base cor F1={baseline_test[0]:.1%}, candidato={candidate_test[0]:.1%}; '
          f'base pelo acc={baseline_test[1]:.1%}, candidato={candidate_test[1]:.1%}', flush=True)
    print(f'Validação: base={baseline_val}, melhor candidato={best_score:.3f}', flush=True)
    print('F1 por cor no teste reservado (base -> candidato):', {
        color: f"{baseline_colors[color]['f1']:.1%} -> {result['f1']:.1%} (n={result['support']})"
        for color, result in candidate_colors.items()
    }, flush=True)
    output = dict(checkpoint)
    output['color_head'] = {key: value.detach().cpu() for key, value in color_head.state_dict().items()}
    output['color_thresholds'] = {color: float(value) for color, value in zip(COLORS, thresholds)}
    if train_length:
        output['length_head'] = {key: value.detach().cpu() for key, value in length_head.state_dict().items()}
    output['version'] = f'{checkpoint["version"]}-labeled-candidate'
    output['labeled_test_metrics'] = {'baseline_color_f1': baseline_test[0],
                                      'candidate_color_f1': candidate_test[0],
                                      'baseline_length_accuracy': baseline_test[1],
                                      'candidate_length_accuracy': candidate_test[1]}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    torch.save(output, args.output)
    print(f'Candidato salvo em {args.output}; o modelo ativo NÃO foi alterado', flush=True)


if __name__ == '__main__':
    run()
