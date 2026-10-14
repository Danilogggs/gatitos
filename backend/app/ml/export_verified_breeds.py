"""Exporta apenas raças com procedência confirmada pelo administrador.

Uso em backend/: python -m app.ml.export_verified_breeds
O manifesto resultante é para um futuro candidato de raça; não inicia treino.
"""

import argparse
import csv
import hashlib
from io import BytesIO
from pathlib import Path

from PIL import Image, UnidentifiedImageError

from app.core.config import settings
from app.core.db import db
from app.ml.taxonomy import BREEDS


def export(client, output_dir: Path) -> dict[str, int]:
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / 'images').mkdir(exist_ok=True)
    bucket = client.storage.from_(settings().supabase_storage_bucket)
    counts = {'verified_cats': 0, 'exported_cats': 0, 'exported_photos': 0, 'skipped': 0}
    rows = []
    offset = 0
    while True:
        cats = client.table('cats').select(
            'id,status,breed,breed_training_verified,breed_verification_note'
        ).eq('breed_training_verified', True).range(offset, offset + 999).execute().data
        for cat in cats:
            counts['verified_cats'] += 1
            if (cat['status'] not in {'PUBLISHED', 'ADOPTED'} or cat['breed'] not in BREEDS
                    or cat['breed'] == 'srd' or not (cat['breed_verification_note'] or '').strip()):
                counts['skipped'] += 1
                continue
            photos = client.table('cat_images').select('id,storage_path').eq('cat_id', cat['id']).order('position').execute().data
            exported = 0
            for photo in photos:
                try:
                    raw = bucket.download(photo['storage_path'])
                    with Image.open(BytesIO(raw)) as image:
                        image.verify()
                except (OSError, UnidentifiedImageError, ValueError):
                    counts['skipped'] += 1
                    continue
                relative = Path('images') / f"{photo['id']}.webp"
                destination = output_dir / relative
                if not destination.exists() or hashlib.sha256(destination.read_bytes()).digest() != hashlib.sha256(raw).digest():
                    destination.write_bytes(raw)
                rows.append({'path': relative.as_posix(), 'cat_id': cat['id'],
                             'image_id': photo['id'], 'sha256': hashlib.sha256(raw).hexdigest(),
                             'breed': cat['breed'], 'verification_note': cat['breed_verification_note']})
                exported += 1
            if exported:
                counts['exported_cats'] += 1
                counts['exported_photos'] += exported
        if len(cats) < 1000:
            break
        offset += 1000
    with (output_dir / 'labels.csv').open('w', encoding='utf-8', newline='') as handle:
        writer = csv.DictWriter(handle, fieldnames=['path', 'cat_id', 'image_id', 'sha256', 'breed', 'verification_note'])
        writer.writeheader()
        writer.writerows(rows)
    return counts


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument('--output-dir', type=Path, default=Path('dataset/verified-breeds'))
    args = parser.parse_args()
    if not settings().supabase_service_role_key:
        parser.error('SUPABASE_SERVICE_ROLE_KEY é necessária para exportar')
    counts = export(db(), args.output_dir)
    print(f"Gatos marcados: {counts['verified_cats']}; exportados: {counts['exported_cats']}; "
          f"fotos: {counts['exported_photos']}; ignorados: {counts['skipped']}")
    print(f"Manifesto: {args.output_dir / 'labels.csv'}")


if __name__ == '__main__':
    main()
