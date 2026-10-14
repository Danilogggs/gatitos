"""Audita a galeria local com o mesmo serviço de ML usado pela API."""

import argparse
import csv
from pathlib import Path

from app.ml.service import RealMLService


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument('--gallery', type=Path, default=Path('../dataset/avaliacao-24-gatos'))
    args = parser.parse_args()
    service = RealMLService()
    rows = []
    for cat in sorted(args.gallery.glob('*/gato-*')):
        photos = sorted(cat.glob('foto-*.jpg'))
        prediction = service.predict([photo.read_bytes() for photo in photos])
        ranked = sorted(prediction.breed_scores.items(), key=lambda pair: pair[1], reverse=True)
        row = {
            'raca_da_pasta': cat.parent.name,
            'gato': cat.name,
            'previsao': prediction.breed,
            'acertou': prediction.breed == cat.parent.name,
            'top1': ranked[0][0], 'top1_pct': round(ranked[0][1] * 100, 1),
            'top2': ranked[1][0], 'top2_pct': round(ranked[1][1] * 100, 1),
            'top3': ranked[2][0], 'top3_pct': round(ranked[2][1] * 100, 1),
            'raca_esperada_pct': round(prediction.breed_scores[cat.parent.name] * 100, 1),
            'versao_modelo': prediction.model_version,
        }
        rows.append(row)
        print(f"{row['raca_da_pasta']}/{row['gato']}: {row['previsao']} "
              f"({row['top1_pct']}%) | {row['top2']} {row['top2_pct']}% | "
              f"{row['top3']} {row['top3_pct']}%", flush=True)
    destination = args.gallery / 'auditoria-modelo.csv'
    with destination.open('w', encoding='utf-8', newline='') as handle:
        writer = csv.DictWriter(handle, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)
    print(f"Acertos: {sum(row['acertou'] for row in rows)}/{len(rows)}. Relatório: {destination}")


if __name__ == '__main__':
    main()
