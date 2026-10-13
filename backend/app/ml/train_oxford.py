"""Treina apenas a classificação de raças de gatos com Oxford-IIIT Pet.

Uso no Colab: python -m app.ml.train_oxford --data-dir /content/data --output /content/catcare-oxford.pt
O dataset é baixado automaticamente pelo torchvision. Oxford não tem rótulos de SRD
nem da taxonomia de características, cores, padrões e comprimento do CatCare AI.
O fallback SRD por baixa confiança acontece na inferência do backend.
"""

import argparse
import random
from pathlib import Path

import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset, Subset
from torchvision import datasets, models, transforms

from app.ml.taxonomy import BREEDS


OXFORD_BREEDS = [breed for breed in BREEDS if breed != 'srd']


def select_cat_samples(source: datasets.OxfordIIITPet) -> list[tuple[int, int]]:
    """Retorna (índice da imagem, índice da raça) somente para gatos."""
    samples = []
    for source_index in range(len(source)):
        image, (category, species) = source[source_index]
        image.close()
        if species != source.bin_class_to_idx['Cat']:
            continue
        breed = source.classes[category].lower().replace(' ', '_')
        if breed not in OXFORD_BREEDS:
            raise ValueError(f'Raça Oxford não mapeada: {breed}')
        samples.append((source_index, OXFORD_BREEDS.index(breed)))

    found = {OXFORD_BREEDS[label] for _, label in samples}
    if found != set(OXFORD_BREEDS):
        raise ValueError(f'Raças Oxford ausentes: {sorted(set(OXFORD_BREEDS) - found)}')
    return samples


def stratified_split(samples: list[tuple[int, int]], validation_ratio: float = 0.2,
                     seed: int = 42) -> tuple[list[int], list[int]]:
    """Separa índices por raça para preservar todas as classes nos dois conjuntos."""
    buckets = [[index for index, (_, label) in enumerate(samples) if label == breed]
               for breed in range(len(OXFORD_BREEDS))]
    rng = random.Random(seed)
    training, validation = [], []
    for bucket in buckets:
        rng.shuffle(bucket)
        validation_size = max(1, round(len(bucket) * validation_ratio))
        validation.extend(bucket[:validation_size])
        training.extend(bucket[validation_size:])
    return training, validation


class OxfordCats(Dataset):
    def __init__(self, source: datasets.OxfordIIITPet, transform,
                 samples: list[tuple[int, int]] | None = None) -> None:
        self.source = source
        self.transform = transform
        self.samples = samples if samples is not None else select_cat_samples(source)

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int):
        source_index, breed = self.samples[index]
        image, _ = self.source[source_index]
        try:
            return self.transform(image), breed
        finally:
            image.close()


def run() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument('--data-dir', type=Path, default=Path('/content/data'))
    parser.add_argument('--output', type=Path, default=Path('/content/catcare-oxford.pt'))
    parser.add_argument('--epochs', type=int, default=8)
    parser.add_argument('--batch-size', type=int, default=16)
    args = parser.parse_args()
    if args.epochs < 1 or args.batch_size < 1:
        parser.error('Épocas e tamanho do lote devem ser positivos')

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    train_transform = transforms.Compose([
        transforms.Resize((256, 256)), transforms.RandomResizedCrop(224),
        transforms.RandomHorizontalFlip(), transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])
    evaluation_transform = transforms.Compose([
        transforms.Resize((224, 224)), transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])

    # 1. Baixa/carrega os splits oficiais completos do Oxford-IIIT Pet.
    targets = ['category', 'binary-category']
    trainval_source = datasets.OxfordIIITPet(
        args.data_dir, split='trainval', target_types=targets, download=True,
    )
    test_source = datasets.OxfordIIITPet(
        args.data_dir, split='test', target_types=targets, download=True,
    )

    # 2. Remove os cães antes de criar os conjuntos usados pelo modelo.
    trainval_samples = select_cat_samples(trainval_source)
    test_samples = select_cat_samples(test_source)

    # 3. Divide trainval por raça; o split oficial test continua independente.
    train_indices, validation_indices = stratified_split(trainval_samples)
    training_images = OxfordCats(trainval_source, train_transform, trainval_samples)
    evaluation_images = OxfordCats(trainval_source, evaluation_transform, trainval_samples)
    train_dataset = Subset(training_images, train_indices)
    validation_dataset = Subset(evaluation_images, validation_indices)
    test_dataset = OxfordCats(test_source, evaluation_transform, test_samples)

    # 4. Cria os lotes depois que os três conjuntos estão definidos.
    loader_options = {'batch_size': args.batch_size, 'num_workers': 2, 'pin_memory': device.type == 'cuda'}
    train_loader = DataLoader(train_dataset, shuffle=True, **loader_options)
    validation_loader = DataLoader(validation_dataset, **loader_options)
    test_loader = DataLoader(test_dataset, **loader_options)
    print(f'Dispositivo: {device}; treino: {len(train_dataset)}; validação: {len(validation_dataset)}; '
          f'teste: {len(test_dataset)}', flush=True)
    print('Raças:', ', '.join(OXFORD_BREEDS), flush=True)

    # 5. Monta a EfficientNet-B0 e substitui o classificador pelas 12 raças.
    backbone = models.efficientnet_b0(weights=models.EfficientNet_B0_Weights.DEFAULT)
    size = backbone.classifier[1].in_features
    backbone.classifier = nn.Identity()
    breed_head = nn.Linear(size, len(OXFORD_BREEDS))
    backbone.to(device)
    breed_head.to(device)
    optimizer = torch.optim.AdamW(list(backbone.parameters()) + list(breed_head.parameters()), lr=1e-4)
    criterion = nn.CrossEntropyLoss()
    best_validation_accuracy = -1.0

    def run_epoch(loader: DataLoader, training: bool) -> tuple[float, float]:
        backbone.train(training)
        breed_head.train(training)
        total_loss = 0.0
        correct = 0
        for images, labels in loader:
            images, labels = images.to(device), labels.to(device)
            with torch.set_grad_enabled(training):
                logits = breed_head(backbone(images))
                loss = criterion(logits, labels)
                if training:
                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()
            total_loss += loss.item() * len(images)
            correct += (logits.argmax(dim=1) == labels).sum().item()
        return total_loss / len(loader.dataset), correct / len(loader.dataset)

    # 6. Treina e salva somente o checkpoint com melhor validação.
    for epoch in range(1, args.epochs + 1):
        train_loss, train_accuracy = run_epoch(train_loader, True)
        validation_loss, validation_accuracy = run_epoch(validation_loader, False)
        print(f'Época {epoch}: treino loss={train_loss:.4f} acc={train_accuracy:.1%}; '
              f'validação loss={validation_loss:.4f} acc={validation_accuracy:.1%}', flush=True)
        if validation_accuracy > best_validation_accuracy:
            best_validation_accuracy = validation_accuracy
            args.output.parent.mkdir(parents=True, exist_ok=True)
            torch.save({
                'backbone': {key: value.detach().cpu() for key, value in backbone.state_dict().items()},
                'breed_head': {key: value.detach().cpu() for key, value in breed_head.state_dict().items()},
                'breeds': OXFORD_BREEDS,
                'trained_heads': ['breed'],
                'version': f'efficientnet-b0-oxford-epoch-{epoch}',
                'source': 'Oxford-IIIT Pet',
            }, args.output)

    # 7. Restaura o melhor checkpoint e só então mede o split oficial de teste.
    checkpoint = torch.load(args.output, map_location=device, weights_only=True)
    backbone.load_state_dict(checkpoint['backbone'])
    breed_head.load_state_dict(checkpoint['breed_head'])
    test_loss, test_accuracy = run_epoch(test_loader, False)
    print(f'Melhor acurácia de validação: {best_validation_accuracy:.1%}; '
          f'teste final loss={test_loss:.4f} acc={test_accuracy:.1%}; checkpoint: {args.output}', flush=True)


if __name__ == '__main__':
    run()
