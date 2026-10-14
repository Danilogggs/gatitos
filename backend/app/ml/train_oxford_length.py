"""Treina comprimento do pelo com rótulos aproximados derivados das raças Oxford.

Uso a partir de backend/: python -m app.ml.train_oxford_length
Os rótulos são protótipos de raça, não anotações individuais de cada foto.
"""

import argparse
import random
from collections import Counter
from pathlib import Path

import torch
from torch import nn
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, models, transforms

from app.ml.taxonomy import COAT_LENGTHS
from app.ml.train_oxford import OXFORD_BREEDS, OxfordCats


BREED_TO_LENGTH = {
    'abyssinian': 'short',
    'bengal': 'short',
    'birman': 'medium',
    'bombay': 'short',
    'british_shorthair': 'short',
    'egyptian_mau': 'short',
    'maine_coon': 'long',
    'persian': 'long',
    'ragdoll': 'medium',
    'russian_blue': 'short',
    'siamese': 'short',
    'sphynx': 'hairless',
}
if set(BREED_TO_LENGTH) != set(OXFORD_BREEDS):
    raise ValueError('Mapeamento de raças Oxford incompleto')
BREED_LABELS = torch.tensor([COAT_LENGTHS.index(BREED_TO_LENGTH[breed]) for breed in OXFORD_BREEDS])


def run() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument('--data-dir', type=Path, default=Path('dataset/oxford'))
    parser.add_argument('--checkpoint', type=Path, default=Path('models/catcare-oxford-colors.pt'))
    parser.add_argument('--output', type=Path, default=Path('models/catcare-oxford-colors-length.pt'))
    parser.add_argument('--epochs', type=int, default=100)
    parser.add_argument('--batch-size', type=int, default=32)
    args = parser.parse_args()
    if args.epochs < 1 or args.batch_size < 1:
        parser.error('Épocas e tamanho do lote devem ser positivos')

    torch.manual_seed(42)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    transform = transforms.Compose([
        transforms.Resize((224, 224)), transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])
    targets = ['category', 'binary-category']
    train_source = datasets.OxfordIIITPet(args.data_dir, split='trainval', target_types=targets, download=True)
    test_source = datasets.OxfordIIITPet(args.data_dir, split='test', target_types=targets, download=True)
    all_train = OxfordCats(train_source, transform)
    test_dataset = OxfordCats(test_source, transform)
    buckets = [[index for index, (_, breed) in enumerate(all_train.samples) if breed == breed_index]
               for breed_index in range(len(OXFORD_BREEDS))]
    rng = random.Random(42)
    training_indices, validation_indices = [], []
    for bucket in buckets:
        rng.shuffle(bucket)
        validation_size = max(1, round(len(bucket) * 0.2))
        validation_indices.extend(bucket[:validation_size])
        training_indices.extend(bucket[validation_size:])
    options = {'batch_size': args.batch_size, 'num_workers': 0,
               'pin_memory': device.type == 'cuda'}
    train_loader = DataLoader(Subset(all_train, training_indices), **options)
    validation_loader = DataLoader(Subset(all_train, validation_indices), **options)
    test_loader = DataLoader(test_dataset, **options)

    checkpoint = torch.load(args.checkpoint, map_location='cpu', weights_only=True)
    backbone = models.efficientnet_b0(weights=None)
    embedding_size = backbone.classifier[1].in_features
    backbone.classifier = nn.Identity()
    backbone.load_state_dict(checkpoint['backbone'])
    backbone.to(device).eval()
    for parameter in backbone.parameters():
        parameter.requires_grad = False

    def embed(loader):
        vectors, labels, breeds = [], [], []
        with torch.no_grad():
            for images, breed_indices in loader:
                vectors.append(backbone(images.to(device)).cpu())
                labels.append(BREED_LABELS[breed_indices])
                breeds.append(breed_indices)
        return torch.cat(vectors), torch.cat(labels), torch.cat(breeds)

    print(f'Dispositivo: {device}; extraindo embeddings...', flush=True)
    train_vectors, train_labels, _ = embed(train_loader)
    val_vectors, val_labels, _ = embed(validation_loader)
    test_vectors, test_labels, test_breeds = embed(test_loader)
    print(f'Treino: {len(train_labels)}; validação: {len(val_labels)}; '
          f'teste: {len(test_labels)}; classes: {dict(Counter(COAT_LENGTHS[i] for i in train_labels.tolist()))}',
          flush=True)

    counts = torch.bincount(train_labels, minlength=len(COAT_LENGTHS)).float()
    weights = (counts.sum() / (len(COAT_LENGTHS) * counts.clamp_min(1))).sqrt()
    head = nn.Linear(embedding_size, len(COAT_LENGTHS))
    optimizer = torch.optim.AdamW(head.parameters(), lr=1e-3, weight_decay=1e-3)
    criterion = nn.CrossEntropyLoss(weight=weights)
    best_score, best_state = -1.0, None
    generator = torch.Generator().manual_seed(42)

    def recall_by_class(predicted, expected):
        return [float((predicted[expected == index] == index).float().mean())
                for index in range(len(COAT_LENGTHS))]

    for epoch in range(1, args.epochs + 1):
        head.train()
        order = torch.randperm(len(train_labels), generator=generator)
        for start in range(0, len(order), args.batch_size):
            selected = order[start:start + args.batch_size]
            loss = criterion(head(train_vectors[selected]), train_labels[selected])
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        head.eval()
        with torch.no_grad():
            prediction = head(val_vectors).argmax(dim=1)
            recalls = recall_by_class(prediction, val_labels)
            score = sum(recalls) / len(recalls)
        if score > best_score:
            best_score = score
            best_state = {key: value.detach().clone() for key, value in head.state_dict().items()}
        if epoch == 1 or epoch % 10 == 0 or epoch == args.epochs:
            print(f'Época {epoch}: macro-recall validação={score:.1%}', flush=True)

    head.load_state_dict(best_state)
    head.eval()
    with torch.no_grad():
        probabilities = torch.softmax(head(test_vectors), dim=1)
        confidence, prediction = probabilities.max(dim=1)
        recalls = recall_by_class(prediction, test_labels)
        accuracy = float((prediction == test_labels).float().mean())
        high_confidence_accuracy = float((prediction[confidence >= 0.5] == test_labels[confidence >= 0.5]).float().mean())
    print('Recall de teste:', {label: f'{value:.1%}' for label, value in zip(COAT_LENGTHS, recalls)}, flush=True)
    print(f'Acurácia de teste: {accuracy:.1%}; acurácia com confiança >= 0.5: '
          f'{high_confidence_accuracy:.1%}; cobertura: {(confidence >= 0.5).float().mean():.1%}', flush=True)
    print('Sphynx no teste:', dict(Counter(COAT_LENGTHS[i] for i in prediction[test_breeds == OXFORD_BREEDS.index('sphynx')].tolist())), flush=True)

    output = dict(checkpoint)
    output.update({
        'length_head': {key: value.detach().cpu() for key, value in head.state_dict().items()},
        'coat_lengths': COAT_LENGTHS,
        'length_min_confidence': 0.5,
        'trained_heads': list(dict.fromkeys([*checkpoint.get('trained_heads', []), 'length'])),
        'version': f'{checkpoint.get("version", "efficientnet-b0-oxford")}-length-v1',
        'length_source': 'Oxford-IIIT Pet breed-derived proxy labels, not per-image coat annotations',
        'length_test_accuracy': accuracy,
        'length_test_recall': dict(zip(COAT_LENGTHS, recalls)),
    })
    args.output.parent.mkdir(parents=True, exist_ok=True)
    torch.save(output, args.output)
    print(f'Checkpoint salvo em: {args.output}', flush=True)


if __name__ == '__main__':
    run()
