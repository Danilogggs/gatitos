"""Exporta fotos e rótulos de gatos publicados após revisão administrativa.

Uso a partir de backend/: python -m app.ml.export_reviewed
O diretório dataset/ é ignorado pelo Git. Não publique as fotos exportadas.
"""

import argparse
import csv
import hashlib
from pathlib import Path

from PIL import Image, UnidentifiedImageError

from app.core.config import settings
from app.core.db import db
from app.ml.taxonomy import BREEDS, COAT_LENGTHS, COLORS


FIELDS = ['path', 'cat_id', 'image_id', 'sha256', 'breed', 'colors', 'coat_length', 'source']


def reviewed_cat_ids(client) -> list[str]:
    ids = set()
    start = 0
    while True:
        rows = client.table('prediction_feedback').select('cat_id').range(start, start + 999).execute().data
        ids.update(row['cat_id'] for row in rows)
        if len(rows) < 1000:
            return sorted(ids)
        start += 1000


def export(client, output_dir: Path) -> dict[str, int]:
    output_dir.mkdir(parents=True, exist_ok=True)
    images_dir = output_dir / 'images'
    images_dir.mkdir(exist_ok=True)
    rows = []
    counts = {'reviewed': 0, 'exported_cats': 0, 'exported_photos': 0, 'skipped': 0}
    bucket = client.storage.from_(settings().supabase_storage_bucket)

    for cat_id in reviewed_cat_ids(client):
        counts['reviewed'] += 1
        cats = client.table('cats').select(
            'id,status,breed,primary_color,secondary_color,coat_length'
        ).eq('id', cat_id).limit(1).execute().data
        if not cats or cats[0]['status'] not in {'PUBLISHED', 'ADOPTED'}:
            counts['skipped'] += 1
            continue
        cat = cats[0]
        breed = cat['breed'] if cat['breed'] in BREEDS else ''
        colors = list(dict.fromkeys(color for color in [cat['primary_color'], cat['secondary_color']]
                                    if color in COLORS))
        length = cat['coat_length'] if cat['coat_length'] in COAT_LENGTHS else ''
        if not (breed or colors or length):
            counts['skipped'] += 1
            continue
        photos = client.table('cat_images').select('id,storage_path').eq('cat_id', cat_id).order('position').execute().data
        exported = 0
        for photo in photos:
            try:
                raw = bucket.download(photo['storage_path'])
                from io import BytesIO
                with Image.open(BytesIO(raw)) as image:
                    image.verify()
            except (OSError, UnidentifiedImageError, ValueError):
                counts['skipped'] += 1
                continue
            relative = Path('images') / f"{photo['id']}.webp"
            destination = output_dir / relative
            if not destination.exists() or hashlib.sha256(destination.read_bytes()).digest() != hashlib.sha256(raw).digest():
                destination.write_bytes(raw)
            rows.append({
                'path': relative.as_posix(), 'cat_id': cat_id, 'image_id': photo['id'],
                'sha256': hashlib.sha256(raw).hexdigest(), 'breed': breed,
                'colors': '|'.join(colors), 'coat_length': length, 'source': 'admin_review',
            })
            exported += 1
        if exported:
            counts['exported_cats'] += 1
            counts['exported_photos'] += exported

    with (output_dir / 'labels.csv').open('w', encoding='utf-8', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=FIELDS)
        writer.writeheader()
        writer.writerows(rows)
    return counts


def run() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument('--output-dir', type=Path, default=Path('dataset/reviewed'))
    args = parser.parse_args()
    if not settings().supabase_service_role_key:
        parser.error('SUPABASE_SERVICE_ROLE_KEY é necessária para exportar fotos revisadas')
    counts = export(db(), args.output_dir)
    print(f"Revisados: {counts['reviewed']}; gatos exportados: {counts['exported_cats']}; "
          f"fotos: {counts['exported_photos']}; ignorados: {counts['skipped']}")
    print(f"Rótulos: {args.output_dir / 'labels.csv'}")


if __name__ == '__main__':
    run()
