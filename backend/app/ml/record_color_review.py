"""Registra revisão visual conservadora das pranchas de 25/09/2026.

Índices são posições 1-based das pranchas, não IDs de raça. Itens omitidos
permanecem sem revisão. Esta anotação é visual, não confirmação de pedigree.
"""

import csv
from collections import Counter
from pathlib import Path


ORANGE = {
    1, 2, 4, 17, 19, 38, 45, 49, 54, 58, 60, 66,
    70, 79, 80, 81, 82, 83, 86, 89, 96, 100, 101, 102,
    106, 111, 112, 119, 126, 129, 138, 142, 150,
    151, 168, 170, 171, 173, 175, 176, 178, 180,
}
CREAM = {
    7, 9, 12, 14, 23, 25, 35, 46, 47, 48, 59, 64,
    71, 75, 94, 97, 104, 115, 120, 139, 140, 144,
    147, 155, 158, 163, 164, 166, 174, 184,
}
NEITHER = {
    3, 5, 6, 8, 10, 11, 13, 15, 16, 18, 20, 21, 22, 24,
    26, 27, 28, 29, 30, 31, 32, 34, 36, 37, 39, 40,
    41, 42, 43, 44, 51, 52, 55, 56, 57, 61, 62, 63,
    67, 68, 76, 84, 85, 88, 90, 95, 98, 103, 105,
    107, 108, 109, 114, 116, 123, 125, 127, 130,
    131, 132, 134, 135, 136, 137, 141, 143, 146,
    148, 149, 152, 153, 156, 157, 159, 160, 161,
    162, 165, 167, 169, 172, 177, 179, 181, 183,
}


def run() -> None:
    sets = [ORANGE, CREAM, NEITHER]
    if any(a & b for i, a in enumerate(sets) for b in sets[i + 1:]):
        raise ValueError('Revisão contém categorias conflitantes')
    directory = Path('dataset/orange-cream-review')
    source = directory / 'review_queue.csv'
    rows = list(csv.DictReader(source.open(encoding='utf-8', newline='')))
    if len(rows) != 184:
        raise ValueError('Fila mudou; índices visuais precisam ser revisados novamente')
    for index, row in enumerate(rows, 1):
        row['reviewed_color'] = ('orange' if index in ORANGE else
                                 'cream' if index in CREAM else
                                 'neither' if index in NEITHER else '')
    destination = directory / 'visually_reviewed.csv'
    with destination.open('w', encoding='utf-8', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=['file', 'suggested_color', 'reviewed_color'])
        writer.writeheader()
        writer.writerows(rows)
    print(f'{destination}: {dict(Counter(row["reviewed_color"] or "unreviewed" for row in rows))}')


if __name__ == '__main__':
    run()
