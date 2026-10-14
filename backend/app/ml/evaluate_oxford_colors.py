"""Compara duas cabeças de cor no teste Oxford com rótulos aproximados por máscara.

Os rótulos são derivados de pixels, não revisões humanas; use como verificação de
regressão, nunca como medida definitiva de qualidade de cor.
"""

import argparse
from pathlib import Path

import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision import datasets, models, transforms

from app.ml.taxonomy import COLORS, COAT_LENGTHS
from app.ml.train_oxford_length import BREED_TO_LENGTH
from app.ml.train_oxford_colors import OxfordColorDataset, _f1_scores


def run() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument('--base', type=Path, required=True)
    parser.add_argument('--candidate', type=Path, required=True)
    parser.add_argument('--data-dir', type=Path, default=Path('dataset/oxford'))
    args = parser.parse_args()
    base = torch.load(args.base, map_location='cpu', weights_only=True)
    candidate = torch.load(args.candidate, map_location='cpu', weights_only=True)
    if any(not torch.equal(base['backbone'][key], candidate['backbone'][key])
           for key in base['backbone']):
        parser.error('Os checkpoints não compartilham o mesmo backbone')
    backbone = models.efficientnet_b0(weights=None)
    size = backbone.classifier[1].in_features
    backbone.classifier = nn.Identity()
    backbone.load_state_dict(base['backbone'])
    backbone.eval()
    transform = transforms.Compose([
        transforms.Resize((224, 224)), transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])
    source = datasets.OxfordIIITPet(args.data_dir, split='test',
                                    target_types=['category', 'binary-category', 'segmentation'],
                                    download=False)
    dataset = OxfordColorDataset(source, transform)
    loader = DataLoader(dataset, batch_size=32, num_workers=0)
    vectors, targets = [], []
    with torch.no_grad():
        for batch, labels in loader:
            vectors.append(backbone(batch))
            targets.append(labels)
    vectors = torch.cat(vectors)
    targets = torch.cat(targets)
    # Comprimento é um proxy da raça Oxford, não anotação visual por foto.
    length_targets = []
    for source_index in dataset.samples:
        image, (category, _, trimap) = source[source_index]
        image.close()
        trimap.close()
        breed = source.classes[category].lower().replace(' ', '_')
        length_targets.append(COAT_LENGTHS.index(BREED_TO_LENGTH[breed]))
    length_targets = torch.tensor(length_targets)
    print(f'Fotos Oxford de gato no teste: {len(targets)}', flush=True)
    for name, checkpoint in (('base', base), ('candidato', candidate)):
        head = nn.Linear(size, len(COLORS))
        head.load_state_dict(checkpoint['color_head'])
        thresholds = torch.tensor([float(checkpoint.get('color_thresholds', {}).get(color, 0.5))
                                   for color in COLORS])
        with torch.no_grad():
            scores = _f1_scores(torch.sigmoid(head(vectors)), targets, thresholds)
        print(f'{name}: macro-F1={scores.mean():.1%}; por cor=' + str({
            color: f'{float(score):.1%}' for color, score in zip(COLORS, scores)
        }), flush=True)
        length_head = nn.Linear(size, len(COAT_LENGTHS))
        length_head.load_state_dict(checkpoint['length_head'])
        with torch.no_grad():
            predicted = length_head(vectors).argmax(dim=1)
        accuracy = float((predicted == length_targets).float().mean())
        recalls = {label: float((predicted[length_targets == index] == index).float().mean())
                   for index, label in enumerate(COAT_LENGTHS)}
        print(f'{name}: pelo accuracy={accuracy:.1%}; recall por classe=' +
              str({label: f'{recall:.1%}' for label, recall in recalls.items()}), flush=True)


if __name__ == '__main__':
    run()
