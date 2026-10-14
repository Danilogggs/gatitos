"""Gera pranchas numeradas para inspecionar a fila de laranja/creme."""

import argparse
import csv
from pathlib import Path

from PIL import Image, ImageDraw


def run() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument('--directory', type=Path, default=Path('dataset/orange-cream-review'))
    parser.add_argument('--manifest', default='review_queue.csv')
    parser.add_argument('--per-sheet', type=int, default=24)
    args = parser.parse_args()
    with (args.directory / args.manifest).open(encoding='utf-8', newline='') as stream:
        rows = list(csv.DictReader(stream))
    columns = 4
    cell_width, cell_height = 240, 225
    for offset in range(0, len(rows), args.per_sheet):
        chunk = rows[offset:offset + args.per_sheet]
        lines = (len(chunk) + columns - 1) // columns
        sheet = Image.new('RGB', (columns * cell_width, lines * cell_height), 'white')
        draw = ImageDraw.Draw(sheet)
        for local_index, row in enumerate(chunk):
            x = local_index % columns * cell_width
            y = local_index // columns * cell_height
            with Image.open(args.directory / row['file']) as opened:
                thumb = opened.convert('RGB')
                thumb.thumbnail((230, 190))
                sheet.paste(thumb, (x + (cell_width - thumb.width) // 2, y))
            draw.text((x + 5, y + 192), f'{offset + local_index + 1}: {row["suggested_color"]}', fill='black')
        path = args.directory / f'sheet-{offset // args.per_sheet + 1:02d}.png'
        sheet.save(path)
        print(path)


if __name__ == '__main__':
    run()
