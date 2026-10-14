"""Treina somente laranja/creme com fotos revisadas visualmente.

Mantém outras quatro cores, raça e pelo intactos. Divide por conteúdo de imagem,
e não usa fotos de teste ou validação no treino. Salva apenas um candidato.
"""

import argparse
import csv
import hashlib
import random
from collections import defaultdict
from pathlib import Path

import torch
from PIL import Image
from torch import nn
from torchvision import models, transforms

from app.ml.taxonomy import COLORS


TARGETS = [COLORS.index('orange'), COLORS.index('cream')]


def load_review(path: Path) -> list[dict]:
    grouped = defaultdict(list)
    with path.open(encoding='utf-8', newline='') as stream:
        for row in csv.DictReader(stream):
            label = row['reviewed_color']
            if label not in ('orange', 'cream', 'neither'):
                continue
            image_path = (path.parent / row['file']).resolve()
            if not image_path.is_relative_to(path.parent.resolve()):
                raise ValueError('Caminho inválido na revisão')
            digest = hashlib.sha256(image_path.read_bytes()).hexdigest()
            grouped[digest].append((image_path, label))
    result = []
    for records in grouped.values():
        labels = {label for _, label in records}
        if len(labels) == 1:
            image_path = records[0][0]
            result.append({'path': image_path, 'label': labels.pop(),
                           'source': 'petfinder' if image_path.name.startswith('petfinder-') else 'oxford'})
    return result


def split(examples: list[dict]) -> tuple[list[dict], list[dict], list[dict]]:
    buckets = defaultdict(list)
    for item in examples:
        buckets[item['label']].append(item)
    rng = random.Random(42)
    training, validation, test = [], [], []
    for label in ('orange', 'cream', 'neither'):
        items = buckets[label]
        rng.shuffle(items)
        if len(items) < 10:
            raise ValueError(f'Poucos exemplos visuais para {label}: {len(items)}')
        n = max(2, round(len(items) * 0.2))
        test.extend(items[:n])
        validation.extend(items[n:2 * n])
        training.extend(items[2 * n:])
    return training, validation, test


def f1(predicted: torch.Tensor, truth: torch.Tensor) -> list[float]:
    result = []
    for index in range(2):
        tp = (predicted[:, index] & truth[:, index]).sum().item()
        fp = (predicted[:, index] & ~truth[:, index]).sum().item()
        fn = (~predicted[:, index] & truth[:, index]).sum().item()
        result.append(2 * tp / max(1, 2 * tp + fp + fn))
    return result


def run() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument('--review', type=Path, default=Path('dataset/orange-cream-review/visually_reviewed.csv'))
    parser.add_argument('--checkpoint', type=Path, default=Path('models/catcare-oxford-colors-length.pt'))
    parser.add_argument('--output', type=Path, default=Path('models/catcare-reviewed-colors-candidate.pt'))
    parser.add_argument('--epochs', type=int, default=120)
    args = parser.parse_args()
    torch.manual_seed(42)
    examples = load_review(args.review)
    training, validation, test = split(examples)
    print(f'Fotos únicas: treino={len(training)}, validação={len(validation)}, teste={len(test)}', flush=True)
    checkpoint = torch.load(args.checkpoint, map_location='cpu', weights_only=True)
    backbone = models.efficientnet_b0(weights=None)
    size = backbone.classifier[1].in_features
    backbone.classifier = nn.Identity()
    backbone.load_state_dict(checkpoint['backbone'])
    backbone.eval()
    head = nn.Linear(size, len(COLORS))
    head.load_state_dict(checkpoint['color_head'])
    original_weight = head.weight.detach().clone()
    original_bias = head.bias.detach().clone()
    transform = transforms.Compose([
        transforms.Resize((224, 224)), transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])

    def embed(items):
        vectors, labels = [], []
        with torch.no_grad():
            for item in items:
                with Image.open(item['path']) as image:
                    vectors.append(backbone(transform(image.convert('RGB')).unsqueeze(0))[0])
                labels.append([float(item['label'] == c) for c in ('orange', 'cream')])
        return torch.stack(vectors), torch.tensor(labels)

    print('Extraindo embeddings...', flush=True)
    train_x, train_y = embed(training)
    val_x, val_y = embed(validation)
    test_x, test_y = embed(test)
    base_thresholds = torch.tensor([float(checkpoint.get('color_thresholds', {}).get(COLORS[i], 0.5))
                                    for i in TARGETS])

    def probabilities(x):
        with torch.no_grad():
            return torch.sigmoid(head(x)[:, TARGETS])

    base_test = f1(probabilities(test_x) >= base_thresholds, test_y.bool())
    positive = train_y.sum(dim=0)
    pos_weight = ((len(train_y) - positive) / positive.clamp_min(1)).clamp(1, 8)
    optimizer = torch.optim.AdamW(head.parameters(), lr=2e-4, weight_decay=1e-3)
    best, best_state = -1.0, None
    generator = torch.Generator().manual_seed(42)
    other = [index for index in range(len(COLORS)) if index not in TARGETS]
    for epoch in range(args.epochs):
        for indices in torch.randperm(len(train_x), generator=generator).split(16):
            loss = nn.functional.binary_cross_entropy_with_logits(
                head(train_x[indices])[:, TARGETS], train_y[indices], pos_weight=pos_weight)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            with torch.no_grad():
                head.weight[other] = original_weight[other]
                head.bias[other] = original_bias[other]
        scores = f1(probabilities(val_x) >= base_thresholds, val_y.bool())
        quality = sum(scores) / 2
        if quality > best:
            best, best_state = quality, {k: v.detach().clone() for k, v in head.state_dict().items()}
        if epoch == 0 or (epoch + 1) % 20 == 0:
            print(f'Época {epoch + 1}: validação laranja={scores[0]:.1%}, creme={scores[1]:.1%}', flush=True)
    head.load_state_dict(best_state)
    val_prob = probabilities(val_x)
    thresholds = base_thresholds.clone()
    for index in range(2):
        selected, best_f1 = float(thresholds[index]), -1.0
        for threshold in torch.arange(0.2, 0.81, 0.05):
            score = f1(val_prob >= torch.tensor([threshold, threshold]), val_y.bool())[index]
            if score > best_f1:
                selected, best_f1 = float(threshold), score
        thresholds[index] = selected
    candidate_test = f1(probabilities(test_x) >= thresholds, test_y.bool())
    print('Teste visual reservado:', {
        color: f'{before:.1%} -> {after:.1%}'
        for color, before, after in zip(('orange', 'cream'), base_test, candidate_test)
    }, flush=True)
    for source in ('oxford', 'petfinder'):
        selected = torch.tensor([item['source'] == source for item in test])
        if not selected.any():
            continue
        # Compara com a cabeça original sem alterar o checkpoint.
        original_logits = test_x[selected] @ original_weight[TARGETS].T + original_bias[TARGETS]
        source_base = f1(torch.sigmoid(original_logits) >= base_thresholds, test_y[selected].bool())
        source_candidate = f1(probabilities(test_x[selected]) >= thresholds, test_y[selected].bool())
        print(f'{source}: n={int(selected.sum())}; ' + str({
            color: f'{before:.1%} -> {after:.1%}'
            for color, before, after in zip(('orange', 'cream'), source_base, source_candidate)
        }), flush=True)
    output = dict(checkpoint)
    output['color_head'] = {key: value.detach().cpu() for key, value in head.state_dict().items()}
    color_thresholds = dict(checkpoint.get('color_thresholds', {}))
    color_thresholds.update({color: float(value) for color, value in zip(('orange', 'cream'), thresholds)})
    output['color_thresholds'] = color_thresholds
    output['version'] = f'{checkpoint["version"]}-reviewed-colors-candidate'
    output['visual_review_test'] = {'base': base_test, 'candidate': candidate_test,
                                    'test_images': len(test)}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    torch.save(output, args.output)
    print(f'Candidato: {args.output}; modelo ativo intacto', flush=True)


if __name__ == '__main__':
    run()
