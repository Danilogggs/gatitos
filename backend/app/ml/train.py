"""Treino separado da aplicação: python -m app.ml.train --csv dataset/labels.csv --output models/catcare.pt.

CSV: path,breed,features,coat_pattern,colors,coat_length,split
Listas multilabel usam | (ex.: frajola|bicolor). split = train ou val.
Imagens e CSV devem ficar fora do repositório e do Supabase Storage.
"""
import argparse
import csv
from pathlib import Path

import torch
from PIL import Image
from torch import nn
from torch.utils.data import DataLoader, Dataset
from torchvision import models, transforms

from app.ml.taxonomy import BREEDS, COAT_LENGTHS, COLORS, FEATURES, PATTERNS


class CatsDataset(Dataset):
    def __init__(self, csv_path: Path, split: str, transform) -> None:
        with csv_path.open(newline='', encoding='utf-8') as file:
            self.rows = [row for row in csv.DictReader(file) if row['split'] == split]
        self.root = csv_path.parent
        self.transform = transform

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, index: int):
        row = self.rows[index]
        path = (self.root / row['path']).resolve()
        if not path.is_relative_to(self.root.resolve()):
            raise ValueError('Caminho de imagem fora do dataset')
        with Image.open(path) as image:
            tensor = self.transform(image.convert('RGB'))
        features = set(filter(None, row['features'].split('|')))
        colors = set(filter(None, row['colors'].split('|')))
        return (tensor, BREEDS.index(row['breed']),
                torch.tensor([float(item in features) for item in FEATURES]),
                PATTERNS.index(row['coat_pattern']),
                torch.tensor([float(item in colors) for item in COLORS]),
                COAT_LENGTHS.index(row['coat_length']))


def run() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument('--csv', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--epochs', type=int, default=10)
    parser.add_argument('--batch-size', type=int, default=16)
    args = parser.parse_args()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    augment = transforms.Compose([transforms.Resize((256, 256)), transforms.RandomResizedCrop(224),
        transforms.RandomHorizontalFlip(), transforms.ToTensor(),
        transforms.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225])])
    clean = transforms.Compose([transforms.Resize((224,224)), transforms.ToTensor(),
        transforms.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225])])
    train = DataLoader(CatsDataset(args.csv, 'train', augment), batch_size=args.batch_size, shuffle=True)
    val = DataLoader(CatsDataset(args.csv, 'val', clean), batch_size=args.batch_size)
    backbone = models.efficientnet_b0(weights=models.EfficientNet_B0_Weights.DEFAULT)
    size = backbone.classifier[1].in_features
    backbone.classifier = nn.Identity()
    heads = nn.ModuleDict({'breed_head':nn.Linear(size,len(BREEDS)), 'feature_head':nn.Linear(size,len(FEATURES)),
        'pattern_head':nn.Linear(size,len(PATTERNS)), 'color_head':nn.Linear(size,len(COLORS)),
        'length_head':nn.Linear(size,len(COAT_LENGTHS))})
    backbone.to(device); heads.to(device)
    optimizer = torch.optim.AdamW(list(backbone.parameters()) + list(heads.parameters()), lr=1e-4)
    ce = nn.CrossEntropyLoss(); bce = nn.BCEWithLogitsLoss()
    best = float('inf')

    def evaluate(loader: DataLoader, training: bool) -> float:
        backbone.train(training); heads.train(training)
        total = 0.0
        for images, breeds, features, patterns, colors, lengths in loader:
            images = images.to(device); breeds = breeds.to(device); features = features.to(device)
            patterns = patterns.to(device); colors = colors.to(device); lengths = lengths.to(device)
            with torch.set_grad_enabled(training):
                vectors = backbone(images)
                loss = (ce(heads['breed_head'](vectors), breeds) + bce(heads['feature_head'](vectors), features)
                    + ce(heads['pattern_head'](vectors), patterns) + bce(heads['color_head'](vectors), colors)
                    + ce(heads['length_head'](vectors), lengths))
                if training:
                    optimizer.zero_grad(); loss.backward(); optimizer.step()
            total += float(loss) * len(images)
        return total / len(loader.dataset)

    for epoch in range(args.epochs):
        train_loss = evaluate(train, True)
        val_loss = evaluate(val, False)
        print(f'Epoch {epoch+1}: treino={train_loss:.4f}, validação={val_loss:.4f}')
        if val_loss < best:
            best = val_loss
            args.output.parent.mkdir(parents=True, exist_ok=True)
            torch.save({'backbone':{key:value.detach().cpu() for key,value in backbone.state_dict().items()},
                **{name:{key:value.detach().cpu() for key,value in head.state_dict().items()} for name,head in heads.items()},
                'version':f'efficientnet-b0-epoch-{epoch+1}'}, args.output)


if __name__ == '__main__':
    run()
