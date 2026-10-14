"""Treina uma cabeça local de cores sobre o checkpoint Oxford existente.

O Oxford-IIIT Pet fornece máscaras de segmentação, mas não rótulos de cor. Este
treino cria rótulos aproximados somente a partir dos pixels segmentados do gato,
extrai uma vez os embeddings do backbone existente e treina uma cabeça
multirrótulo. Os rótulos precisam de auditoria humana antes de uso em produção.
"""

import argparse
import random
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from torch import nn
from torch.utils.data import DataLoader, Dataset
from torchvision import datasets, models, transforms

from app.ml.taxonomy import COLORS


COLOR_PROTOTYPES = np.asarray([
    [25, 25, 25],      # black
    [235, 235, 228],   # white
    [128, 132, 136],   # gray
    [220, 105, 30],    # orange
    [225, 195, 145],   # cream
    [105, 65, 38],     # brown
], dtype=np.float32)


def _rgb_to_lab(rgb: np.ndarray) -> np.ndarray:
    """Convert sRGB values in [0, 255] to CIE Lab (D65), without extra deps."""
    values = rgb.astype(np.float32) / 255.0
    values = np.where(values <= 0.04045, values / 12.92, ((values + 0.055) / 1.055) ** 2.4)
    xyz = values @ np.asarray([
        [0.4124564, 0.2126729, 0.0193339],
        [0.3575761, 0.7151522, 0.1191920],
        [0.1804375, 0.0721750, 0.9503041],
    ], dtype=np.float32)
    xyz /= np.asarray([0.95047, 1.0, 1.08883], dtype=np.float32)
    delta = 6 / 29
    converted = np.where(xyz > delta ** 3, np.cbrt(xyz), xyz / (3 * delta ** 2) + 4 / 29)
    x, y, z = converted[..., 0], converted[..., 1], converted[..., 2]
    return np.stack([116 * y - 16, 500 * (x - y), 200 * (y - z)], axis=-1)


PROTOTYPES_LAB = _rgb_to_lab(COLOR_PROTOTYPES)


def infer_color_labels(image: Image.Image, trimap: Image.Image,
                       minimum_fraction: float = 0.08, maximum_colors: int = 3) -> list[str]:
    """Assign approximate coat colors using only foreground pixels (trimap value 1)."""
    rgb = np.asarray(image.convert('RGB'))
    mask_image = trimap.resize(image.size, Image.Resampling.NEAREST)
    foreground = np.asarray(mask_image) == 1
    pixels = rgb[foreground]
    if not len(pixels):
        raise ValueError('Máscara Oxford sem pixels do animal')
    if len(pixels) > 50_000:
        step = max(1, len(pixels) // 50_000)
        pixels = pixels[::step]
    lab = _rgb_to_lab(pixels)
    distances = ((lab[:, None, :] - PROTOTYPES_LAB[None, :, :]) ** 2).sum(axis=2)
    assignments = distances.argmin(axis=1)
    proportions = np.bincount(assignments, minlength=len(COLORS)) / len(assignments)
    ranked = np.argsort(-proportions)
    selected = [int(index) for index in ranked if proportions[index] >= minimum_fraction][:maximum_colors]
    if not selected:
        selected = [int(ranked[0])]
    return [COLORS[index] for index in selected]


def recolor_foreground(image: Image.Image, trimap: Image.Image, color: str) -> Image.Image:
    """Create a texture-preserving color variant only inside the pet mask."""
    hsv = np.asarray(image.convert('HSV')).copy()
    mask = np.asarray(trimap.resize(image.size, Image.Resampling.NEAREST)) == 1
    targets = {
        'black': (0, 20, lambda value: np.clip(value * 0.25, 12, 70)),
        'white': (30, 12, lambda value: np.clip(205 + value * 0.18, 205, 250)),
        'gray': (0, 18, lambda value: np.clip(75 + value * 0.45, 75, 185)),
        'orange': (20, 190, lambda value: np.clip(125 + value * 0.48, 125, 245)),
        'cream': (29, 85, lambda value: np.clip(175 + value * 0.27, 175, 242)),
        'brown': (15, 165, lambda value: np.clip(55 + value * 0.40, 55, 155)),
    }
    hue, saturation, value_transform = targets[color]
    original_value = hsv[..., 2].astype(np.float32)
    hsv[..., 0][mask] = hue
    hsv[..., 1][mask] = saturation
    hsv[..., 2][mask] = value_transform(original_value[mask]).astype(np.uint8)
    return Image.fromarray(hsv, mode='HSV').convert('RGB')


class OxfordColorDataset(Dataset):
    def __init__(self, source: datasets.OxfordIIITPet, transform,
                 samples: list[int] | None = None, synthetic_variants: bool = False) -> None:
        self.source = source
        self.transform = transform
        self.synthetic_variants = synthetic_variants
        self.samples = samples if samples is not None else []
        if samples is None:
            for index in range(len(source)):
                image, (_, species, trimap) = source[index]
                has_foreground = bool((np.asarray(trimap) == 1).any())
                image.close()
                trimap.close()
                if species == source.bin_class_to_idx['Cat'] and has_foreground:
                    self.samples.append(index)

    def __len__(self) -> int:
        return len(self.samples) * (2 if self.synthetic_variants else 1)

    def __getitem__(self, index: int):
        synthetic = self.synthetic_variants and index >= len(self.samples)
        sample_index = index % len(self.samples)
        image, (_, _, trimap) = self.source[self.samples[sample_index]]
        try:
            if synthetic:
                forced_color = COLORS[sample_index % len(COLORS)]
                prepared = recolor_foreground(image, trimap, forced_color)
                labels = {forced_color}
            else:
                prepared = image.convert('RGB')
                labels = set(infer_color_labels(image, trimap))
            target = torch.tensor([float(color in labels) for color in COLORS])
            return self.transform(prepared), target
        finally:
            image.close()
            trimap.close()


def _f1_scores(probabilities: torch.Tensor, targets: torch.Tensor,
               thresholds: torch.Tensor) -> torch.Tensor:
    predicted = probabilities >= thresholds
    expected = targets.bool()
    true_positive = (predicted & expected).sum(dim=0).float()
    false_positive = (predicted & ~expected).sum(dim=0).float()
    false_negative = (~predicted & expected).sum(dim=0).float()
    return 2 * true_positive / (2 * true_positive + false_positive + false_negative).clamp_min(1)


def _best_thresholds(probabilities: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
    thresholds = []
    for index in range(len(COLORS)):
        best_threshold, best_f1 = 0.5, -1.0
        for candidate in torch.arange(0.20, 0.81, 0.05):
            score = _f1_scores(probabilities[:, index:index + 1], targets[:, index:index + 1], candidate)[0]
            if float(score) > best_f1:
                best_threshold, best_f1 = float(candidate), float(score)
        thresholds.append(best_threshold)
    return torch.tensor(thresholds)


def run() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument('--data-dir', type=Path, default=Path('dataset/oxford'))
    parser.add_argument('--checkpoint', type=Path, default=Path('models/catcare-oxford.pt'))
    parser.add_argument('--output', type=Path, default=Path('models/catcare-oxford-colors.pt'))
    parser.add_argument('--epochs', type=int, default=80)
    parser.add_argument('--batch-size', type=int, default=32)
    args = parser.parse_args()
    if args.epochs < 1 or args.batch_size < 1:
        parser.error('Épocas e tamanho do lote devem ser positivos')

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])
    target_types = ['category', 'binary-category', 'segmentation']
    train_source = datasets.OxfordIIITPet(args.data_dir, split='trainval', target_types=target_types, download=True)
    test_source = datasets.OxfordIIITPet(args.data_dir, split='test', target_types=target_types, download=True)
    all_train = OxfordColorDataset(train_source, transform)
    test_dataset = OxfordColorDataset(test_source, transform)
    samples = list(all_train.samples)
    random.Random(42).shuffle(samples)
    validation_size = max(1, round(len(samples) * 0.2))
    validation_dataset = OxfordColorDataset(train_source, transform, samples=samples[:validation_size])
    train_dataset = OxfordColorDataset(train_source, transform, samples=samples[validation_size:],
                                       synthetic_variants=True)
    loader_options = {'batch_size': args.batch_size, 'num_workers': 0,
                      'pin_memory': device.type == 'cuda'}
    train_loader = DataLoader(train_dataset, shuffle=False, **loader_options)
    validation_loader = DataLoader(validation_dataset, shuffle=False, **loader_options)
    test_loader = DataLoader(test_dataset, shuffle=False, **loader_options)
    print(f'Dispositivo: {device}; cores: {", ".join(COLORS)}; treino: {len(train_dataset)}; '
          f'validação: {len(validation_dataset)}; teste: {len(test_dataset)}', flush=True)

    checkpoint = torch.load(args.checkpoint, map_location='cpu', weights_only=True)
    backbone = models.efficientnet_b0(weights=None)
    embedding_size = backbone.classifier[1].in_features
    backbone.classifier = nn.Identity()
    backbone.load_state_dict(checkpoint['backbone'])
    backbone.to(device).eval()
    for parameter in backbone.parameters():
        parameter.requires_grad = False

    def embeddings(loader: DataLoader) -> tuple[torch.Tensor, torch.Tensor]:
        vectors, labels = [], []
        with torch.no_grad():
            for batch, targets in loader:
                vectors.append(backbone(batch.to(device)).cpu())
                labels.append(targets)
        return torch.cat(vectors), torch.cat(labels)

    print('Extraindo embeddings uma única vez...', flush=True)
    train_vectors, train_targets = embeddings(train_loader)
    validation_vectors, validation_targets = embeddings(validation_loader)
    test_vectors, test_targets = embeddings(test_loader)
    positives = train_targets.sum(dim=0)
    pos_weight = ((len(train_targets) - positives) / positives.clamp_min(1)).clamp(1, 12)
    print('Rótulos positivos no treino:', dict(zip(COLORS, positives.int().tolist())), flush=True)

    color_head = nn.Linear(embedding_size, len(COLORS))
    optimizer = torch.optim.AdamW(color_head.parameters(), lr=2e-3, weight_decay=1e-3)
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    best_f1 = -1.0
    best_state = None
    generator = torch.Generator().manual_seed(42)

    for epoch in range(1, args.epochs + 1):
        color_head.train()
        order = torch.randperm(len(train_vectors), generator=generator)
        total_loss = 0.0
        for start in range(0, len(order), args.batch_size):
            selected = order[start:start + args.batch_size]
            logits = color_head(train_vectors[selected])
            loss = criterion(logits, train_targets[selected])
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item() * len(selected)
        color_head.eval()
        with torch.no_grad():
            probabilities = torch.sigmoid(color_head(validation_vectors))
            macro_f1 = float(_f1_scores(probabilities, validation_targets, torch.tensor(0.5)).mean())
        if macro_f1 > best_f1:
            best_f1 = macro_f1
            best_state = {key: value.detach().clone() for key, value in color_head.state_dict().items()}
        if epoch == 1 or epoch % 10 == 0 or epoch == args.epochs:
            print(f'Época {epoch}: loss={total_loss / len(train_vectors):.4f}; '
                  f'val macro-F1@0.5={macro_f1:.1%}', flush=True)

    if best_state is None:
        raise RuntimeError('Treinamento não produziu uma cabeça de cores')
    color_head.load_state_dict(best_state)
    color_head.eval()
    with torch.no_grad():
        validation_probabilities = torch.sigmoid(color_head(validation_vectors))
        thresholds = _best_thresholds(validation_probabilities, validation_targets)
        test_probabilities = torch.sigmoid(color_head(test_vectors))
        test_f1 = _f1_scores(test_probabilities, test_targets, thresholds)
    print('Limiares calibrados:', dict(zip(COLORS, thresholds.tolist())), flush=True)
    print('F1 de teste por cor:', {color: f'{float(score):.1%}' for color, score in zip(COLORS, test_f1)}, flush=True)
    print(f'Macro-F1 de teste: {float(test_f1.mean()):.1%}', flush=True)

    trained_heads = list(dict.fromkeys([*checkpoint.get('trained_heads', []), 'color']))
    output = dict(checkpoint)
    output.update({
        'color_head': {key: value.detach().cpu() for key, value in color_head.state_dict().items()},
        'color_thresholds': {color: float(value) for color, value in zip(COLORS, thresholds)},
        'trained_heads': trained_heads,
        'version': f'{checkpoint.get("version", "efficientnet-b0-oxford")}-colors-v1',
        'color_source': 'Oxford-IIIT Pet pseudo-labels from foreground segmentation pixels',
        'color_test_macro_f1': float(test_f1.mean()),
    })
    args.output.parent.mkdir(parents=True, exist_ok=True)
    torch.save(output, args.output)
    print('Checkpoint salvo em:', args.output, flush=True)


if __name__ == '__main__':
    run()
