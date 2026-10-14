"""Registra a inspeção visual conservadora da fila independente de 160 fotos.

Índices 1-based correspondem às pranchas em orange-cream-validation. Nenhum
destes rótulos entra no treino; apenas em avaliação independente.
"""

import csv
from collections import Counter
from pathlib import Path


ORANGE = {
    1, 3, 5, 7, 9, 12, 13, 15, 17, 18, 23,
    26, 27, 31, 33, 34, 35, 36, 43, 45, 47, 48,
    49, 52, 65, 66, 67, 69, 72, 74, 75, 77, 78,
    84, 88, 91, 92, 93, 94, 99, 103, 104, 110,
    112, 114, 115, 118, 120, 126, 140, 145, 148,
    151, 158,
}
CREAM = {
    10, 20, 24, 28, 32, 37, 41, 44, 57, 58,
    60, 64, 86, 116, 119, 121, 122, 123, 127,
    129, 131, 136, 138, 143, 146, 159, 160,
}
NEITHER = {
    2, 4, 6, 8, 11, 14, 16, 19, 21, 22, 25, 30,
    38, 39, 40, 42, 46, 50, 51, 53, 54, 55, 56,
    59, 61, 62, 68, 70, 71, 76, 79, 81, 82,
    83, 85, 87, 89, 90, 95, 96, 97, 100, 101,
    102, 105, 107, 108, 109, 111, 113, 117, 124,
    125, 128, 130, 132, 134, 135, 137, 139, 141,
    142, 147, 149, 150, 152, 153, 154, 155, 156,
    157,
}


def run() -> None:
    sets = [ORANGE, CREAM, NEITHER]
    if any(a & b for i, a in enumerate(sets) for b in sets[i + 1:]):
        raise ValueError('Categorias conflitantes')
    directory = Path('dataset/orange-cream-validation')
    with (directory / 'validation_queue.csv').open(encoding='utf-8', newline='') as stream:
        rows = list(csv.DictReader(stream))
    if len(rows) != 160:
        raise ValueError('Fila alterada; inspecione novamente antes de rotular')
    for index, row in enumerate(rows, 1):
        row['reviewed_color'] = ('orange' if index in ORANGE else
                                 'cream' if index in CREAM else
                                 'neither' if index in NEITHER else '')
    destination = directory / 'visually_reviewed_validation.csv'
    with destination.open('w', encoding='utf-8', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=rows[0])
        writer.writeheader()
        writer.writerows(rows)
    print(f'{destination}: {dict(Counter(row["reviewed_color"] or "unreviewed" for row in rows))}')


if __name__ == '__main__':
    run()
