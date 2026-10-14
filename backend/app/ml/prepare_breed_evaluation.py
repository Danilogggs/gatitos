"""Cria e avalia uma galeria balanceada do split test Oxford-IIIT Pet."""

import argparse
import csv
import hashlib
from collections import Counter, defaultdict
from pathlib import Path

from PIL import ImageOps
from torchvision import datasets

from app.ml.service import RealMLService
from app.ml.taxonomy import BREEDS


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument('--oxford-dir', type=Path, default=Path('dataset/oxford'))
    parser.add_argument('--old-gallery', type=Path, default=Path('../dataset/avaliacao-24-gatos'))
    parser.add_argument('--output-dir', type=Path, default=Path('../dataset/avaliacao-oxford-240'))
    parser.add_argument('--per-breed', type=int, default=20)
    args = parser.parse_args()
    output = args.output_dir.resolve()
    if output.exists() and any(output.iterdir()):
        parser.error(f'Destino não vazio: {output}. Nada foi sobrescrito.')
    excluded = set()
    old_manifest = args.old_gallery / 'manifesto.csv'
    if old_manifest.exists():
        with old_manifest.open(encoding='utf-8', newline='') as handle:
            excluded = {int(row['id_fonte']) for row in csv.DictReader(handle)
                        if row['fonte'] == 'Oxford-IIIT Pet'}
    source = datasets.OxfordIIITPet(args.oxford_dir, split='test',
                                    target_types=['category', 'binary-category'], download=False)
    buckets = defaultdict(list)
    for index in range(len(source)):
        if index in excluded:
            continue
        image, (category, species) = source[index]
        try:
            if species != source.bin_class_to_idx['Cat']:
                continue
            breed = source.classes[category].lower().replace(' ', '_')
            if breed in BREEDS and min(image.size) >= 250:
                rank = hashlib.sha256(f'catcare-evaluation-2026:{index}'.encode()).hexdigest()
                buckets[breed].append((rank, index))
        finally:
            image.close()
    breed_names = [breed for breed in BREEDS if breed != 'srd']
    for breed in breed_names:
        if len(buckets[breed]) < args.per_breed:
            raise ValueError(f'{breed}: apenas {len(buckets[breed])} fotos elegíveis')
    output.mkdir(parents=True)
    service = RealMLService()
    rows = []
    for breed in breed_names:
        directory = output / breed
        directory.mkdir()
        for _, index in sorted(buckets[breed])[:args.per_breed]:
            image, _ = source[index]
            path = directory / f'oxford-test-{index:04d}.jpg'
            try:
                prepared = ImageOps.exif_transpose(image).convert('RGB')
                prepared.save(path, 'JPEG', quality=92, optimize=True)
            finally:
                image.close()
            prediction = service.predict([path.read_bytes()])
            ranked = sorted(prediction.breed_scores.items(), key=lambda pair: pair[1], reverse=True)
            rows.append({
                'arquivo': path.relative_to(output).as_posix(),
                'oxford_test_index': index,
                'raca_rotulada': breed,
                'raca_exibida_no_app': prediction.breed,
                'acertou': prediction.breed == breed,
                'top1': ranked[0][0], 'top1_pct': round(ranked[0][1] * 100, 2),
                'top2': ranked[1][0], 'top2_pct': round(ranked[1][1] * 100, 2),
                'top3': ranked[2][0], 'top3_pct': round(ranked[2][1] * 100, 2),
                'raca_rotulada_pct': round(prediction.breed_scores[breed] * 100, 2),
                'versao_modelo': prediction.model_version,
            })
        hits = sum(row['acertou'] for row in rows if row['raca_rotulada'] == breed)
        print(f'{breed}: {hits}/{args.per_breed}', flush=True)
    with (output / 'resultados.csv').open('w', encoding='utf-8', newline='') as handle:
        writer = csv.DictWriter(handle, fieldnames=rows[0])
        writer.writeheader()
        writer.writerows(rows)
    totals = Counter(row['raca_rotulada'] for row in rows if row['acertou'])
    count = len(rows)
    overall = sum(row['acertou'] for row in rows)
    lines = [
        '# Avaliação de raças — Oxford-IIIT Pet', '',
        f'{count} imagens originais: {args.per_breed} por raça, cada foto avaliada separadamente.',
        f'Acertos exibidos pelo app: {overall}/{count} ({overall/count:.1%}).',
        f'Modelo: {service.version}. Limiar SRD: {service.breed_min_confidence:.0%}.',
        f'Excluídas {len(excluded)} imagens Oxford da galeria anterior.', '',
        '| Raça | Acertos |', '|---|---:|',
    ]
    for breed in breed_names:
        lines.append(f'| {breed} | {totals[breed]}/{args.per_breed} |')
    lines.extend([
        '', 'O rótulo de raça é o do dataset Oxford-IIIT Pet, não comprovação individual de pedigree.',
        'O split test não foi usado no treino dos pesos de raça; porém o treino original',
        'consultou sua métrica final, e outros experimentos podem tê-lo consultado.',
        'Portanto, este é um teste de regressão com rótulos melhores que PetFinder,',
        'não um holdout totalmente inédito ou teste de desempenho em gatos SRD.',
        'Cada pasta contém gatos distintos; não há três fotos do mesmo animal.',
        'Não use estas imagens em treino futuro se quiser preservar a avaliação.',
        'Fonte e licença: https://www.robots.ox.ac.uk/~vgg/data/pets/',
        '',
    ])
    (output / 'LEIA-ME.md').write_text('\n'.join(lines), encoding='utf-8')
    print(f'Total: {overall}/{count}; {output}', flush=True)


if __name__ == '__main__':
    main()
