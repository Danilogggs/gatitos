"""Cria manifesto local Oxford trainval com rótulos aproximados de cor e pelo.

Nunca inclui o split test: ele permanece reservado para checagem de regressão.
Cor vem de pixels segmentados; comprimento vem do protótipo de raça.
"""

import argparse
import csv
import hashlib
from pathlib import Path

from torchvision import datasets

from app.ml.export_reviewed import FIELDS
from app.ml.train_oxford_colors import infer_color_labels
from app.ml.train_oxford_length import BREED_TO_LENGTH


def export(data_dir: Path, output: Path) -> int:
    source = datasets.OxfordIIITPet(data_dir, split='trainval',
                                    target_types=['category', 'binary-category', 'segmentation'],
                                    download=False)
    root = output.resolve().parent
    records = []
    for index in range(len(source)):
        image, (category, species, trimap) = source[index]
        try:
            if species != source.bin_class_to_idx['Cat']:
                continue
            breed = source.classes[category].lower().replace(' ', '_')
            path = Path(source._images[index]).resolve()
            if not path.is_relative_to(root):
                raise ValueError(f'Imagem Oxford fora do diretório do manifesto: {path}')
            try:
                colors = infer_color_labels(image, trimap)
            except ValueError:
                continue
            records.append({
                'path': path.relative_to(root).as_posix(),
                'cat_id': path.stem, 'image_id': path.stem,
                'sha256': hashlib.sha256(path.read_bytes()).hexdigest(),
                'breed': breed, 'colors': '|'.join(colors),
                'coat_length': BREED_TO_LENGTH[breed], 'source': 'oxford_pseudo',
            })
        finally:
            image.close()
            trimap.close()
        if len(records) and len(records) % 500 == 0:
            print(f'Rotulados aproximadamente: {len(records)}', flush=True)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open('w', encoding='utf-8', newline='') as handle:
        writer = csv.DictWriter(handle, fieldnames=FIELDS)
        writer.writeheader()
        writer.writerows(records)
    return len(records)


def run() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument('--data-dir', type=Path, default=Path('dataset/oxford'))
    parser.add_argument('--output', type=Path, default=Path('dataset/oxford_labels.csv'))
    args = parser.parse_args()
    count = export(args.data_dir, args.output)
    print(f'Fotos Oxford trainval rotuladas aproximadamente: {count}; manifesto: {args.output}')


if __name__ == '__main__':
    run()
