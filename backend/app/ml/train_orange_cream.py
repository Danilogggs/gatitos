"""Experimento isolado para laranja/creme; nunca altera o checkpoint ativo.

Usa pseudo-rótulos Oxford e variantes sintéticas apenas no treino. Teste Oxford
original permanece intocado; os rótulos ainda exigem auditoria humana.
"""

import argparse
import random
from pathlib import Path

import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision import datasets, models, transforms

from app.ml.taxonomy import COLORS
from app.ml.train_oxford_colors import OxfordColorDataset, _f1_scores


TARGETS = [COLORS.index('orange'), COLORS.index('cream')]


def run() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument('--checkpoint', type=Path, default=Path('models/catcare-oxford-colors-length.pt'))
    parser.add_argument('--output', type=Path, default=Path('models/catcare-orange-cream-candidate.pt'))
    parser.add_argument('--data-dir', type=Path, default=Path('dataset/oxford'))
    parser.add_argument('--epochs', type=int, default=80)
    args = parser.parse_args()
    torch.manual_seed(42)
    checkpoint = torch.load(args.checkpoint, map_location='cpu', weights_only=True)
    backbone = models.efficientnet_b0(weights=None)
    size = backbone.classifier[1].in_features
    backbone.classifier = nn.Identity()
    backbone.load_state_dict(checkpoint['backbone'])
    backbone.eval()
    transform = transforms.Compose([
        transforms.Resize((224, 224)), transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])
    types = ['category', 'binary-category', 'segmentation']
    train_source = datasets.OxfordIIITPet(args.data_dir, split='trainval', target_types=types, download=False)
    test_source = datasets.OxfordIIITPet(args.data_dir, split='test', target_types=types, download=False)
    train_all = OxfordColorDataset(train_source, transform)
    samples = list(train_all.samples)
    random.Random(42).shuffle(samples)
    cut = round(len(samples) * 0.2)
    train = OxfordColorDataset(train_source, transform, samples=samples[cut:], synthetic_variants=True)
    validation = OxfordColorDataset(train_source, transform, samples=samples[:cut])
    test = OxfordColorDataset(test_source, transform)

    def embed(dataset):
        vectors, labels = [], []
        with torch.no_grad():
            for images, targets in DataLoader(dataset, batch_size=32, num_workers=0):
                vectors.append(backbone(images))
                labels.append(targets)
        return torch.cat(vectors), torch.cat(labels)

    print('Extraindo embeddings de treino, validação e teste...', flush=True)
    train_x, train_y = embed(train)
    val_x, val_y = embed(validation)
    test_x, test_y = embed(test)
    print(f'Fotos: treino={len(train_x)}, validação={len(val_x)}, teste={len(test_x)}', flush=True)
    head = nn.Linear(size, len(COLORS))
    head.load_state_dict(checkpoint['color_head'])
    base_weight = head.weight.detach().clone()
    base_bias = head.bias.detach().clone()
    base_thresholds = torch.tensor([float(checkpoint.get('color_thresholds', {}).get(c, 0.5)) for c in COLORS])

    def scores(x, y, thresholds):
        with torch.no_grad():
            return _f1_scores(torch.sigmoid(head(x)), y, thresholds)

    base_test = scores(test_x, test_y, base_thresholds)
    best, best_state = -1.0, None
    positives = train_y[:, TARGETS].sum(dim=0)
    weights = ((len(train_y) - positives) / positives.clamp_min(1)).clamp(1, 12)
    optimizer = torch.optim.AdamW(head.parameters(), lr=5e-4, weight_decay=1e-3)
    generator = torch.Generator().manual_seed(42)
    for epoch in range(args.epochs):
        for indices in torch.randperm(len(train_x), generator=generator).split(32):
            logits = head(train_x[indices])[:, TARGETS]
            loss = nn.functional.binary_cross_entropy_with_logits(
                logits, train_y[indices][:, TARGETS], pos_weight=weights)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            with torch.no_grad():
                others = [i for i in range(len(COLORS)) if i not in TARGETS]
                head.weight[others] = base_weight[others]
                head.bias[others] = base_bias[others]
        val = scores(val_x, val_y, base_thresholds)
        quality = float(val[TARGETS].mean())
        if quality > best:
            best, best_state = quality, {k: v.detach().clone() for k, v in head.state_dict().items()}
        if epoch == 0 or (epoch + 1) % 10 == 0:
            print(f'Época {epoch + 1}: validação laranja/creme F1={quality:.1%}', flush=True)
    head.load_state_dict(best_state)
    # Limiares ajustados apenas na validação.
    thresholds = base_thresholds.clone()
    with torch.no_grad():
        probabilities = torch.sigmoid(head(val_x))
    for index in TARGETS:
        best_threshold, best_f1 = float(thresholds[index]), -1.0
        for threshold in torch.arange(0.2, 0.81, 0.05):
            predicted = probabilities[:, index] >= threshold
            truth = val_y[:, index].bool()
            tp = (predicted & truth).sum().item()
            fp = (predicted & ~truth).sum().item()
            fn = (~predicted & truth).sum().item()
            f1 = 2 * tp / max(1, 2 * tp + fp + fn)
            if f1 > best_f1:
                best_f1, best_threshold = f1, float(threshold)
        thresholds[index] = best_threshold
    candidate_test = scores(test_x, test_y, thresholds)
    print('Teste Oxford por cor (base -> candidato):', {
        color: f'{float(base_test[i]):.1%} -> {float(candidate_test[i]):.1%}'
        for i, color in enumerate(COLORS)
    }, flush=True)
    output = dict(checkpoint)
    output['color_head'] = {k: v.detach().cpu() for k, v in head.state_dict().items()}
    output['color_thresholds'] = {c: float(t) for c, t in zip(COLORS, thresholds)}
    output['version'] = f'{checkpoint["version"]}-orange-cream-candidate'
    output['orange_cream_oxford_test_f1'] = {
        c: {'base': float(base_test[i]), 'candidate': float(candidate_test[i])}
        for i, c in enumerate(COLORS) if i in TARGETS
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    torch.save(output, args.output)
    print(f'Candidato salvo em {args.output}; modelo ativo não alterado', flush=True)


if __name__ == '__main__':
    run()
