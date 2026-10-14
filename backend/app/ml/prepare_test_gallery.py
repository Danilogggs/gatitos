"""Cria 24 perfis de teste (2 por raça) com 3 imagens por perfil.

Para 9 raças usa perfis PetFinder com Quantity=1 e três fotos distintas.
Para Birman, Egyptian Mau e Sphynx usa duas fotos Oxford por raça, criando
três enquadramentos da MESMA foto para cada gato. Esses enquadramentos testam
upload/múltiplas imagens; não constituem fotos independentes para avaliar ML.

Uso em backend/:
  python -m app.ml.prepare_test_gallery --archive C:/.../petfinder-adoption-prediction.zip
"""

import argparse
import csv
import io
from collections import defaultdict
from pathlib import Path
from zipfile import ZipFile

from PIL import Image, ImageEnhance, ImageOps
from torchvision import datasets

from app.ml.import_petfinder_zip import IMAGE_NAME


PETFINDER_BREEDS = {
    'abyssinian': 'Abyssinian',
    'bengal': 'Bengal',
    'bombay': 'Bombay',
    'british_shorthair': 'British Shorthair',
    'maine_coon': 'Maine Coon',
    'persian': 'Persian',
    'ragdoll': 'Ragdoll',
    'russian_blue': 'Russian Blue',
    'siamese': 'Siamese',
}
OXFORD_BREEDS = ('birman', 'egyptian_mau', 'sphynx')


def save_jpeg(image: Image.Image, destination: Path) -> None:
    prepared = ImageOps.exif_transpose(image).convert('RGB')
    prepared.thumbnail((1600, 1600), Image.Resampling.LANCZOS)
    prepared.save(destination, format='JPEG', quality=88, optimize=True)
    if destination.stat().st_size > 5 * 1024 * 1024:
        raise ValueError(f'Foto acima do limite de upload: {destination}')


def petfinder_profiles(archive: ZipFile) -> dict[str, list[tuple[str, str, list[str]]]]:
    breed_names = {row['BreedID']: row['BreedName'].strip()
                   for row in csv.DictReader(io.TextIOWrapper(archive.open('breed_labels.csv'), encoding='utf-8-sig'))}
    photos = defaultdict(list)
    for member in archive.namelist():
        match = IMAGE_NAME.fullmatch(member)
        if match:
            photos[(match.group(1), match.group(2))].append((int(match.group(3)), member))
    profiles = defaultdict(list)
    wanted = set(PETFINDER_BREEDS.values())
    for split in ('train', 'test'):
        with archive.open(f'{split}/{split}.csv') as handle:
            for row in csv.DictReader(io.TextIOWrapper(handle, encoding='utf-8-sig')):
                breed = breed_names.get(row['Breed1'])
                if (row['Type'] != '2' or row['Quantity'] != '1' or row['Breed2'] != '0'
                        or breed not in wanted):
                    continue
                members = [member for _, member in sorted(photos[(f'{split}_images', row['PetID'])])]
                if len(members) >= 3:
                    profiles[breed].append((split, row['PetID'], members))
    return profiles


def choose_photos(archive: ZipFile, profiles: list[tuple[str, str, list[str]]]) -> list[tuple[str, str, list[bytes]]]:
    eligible = []
    for split, pet_id, members in profiles:
        images = []
        score = 0
        for member in members:
            try:
                raw = archive.read(member)
                with Image.open(io.BytesIO(raw)) as image:
                    width, height = image.size
                    image.verify()
                if min(width, height) < 200:
                    continue
                images.append(raw)
                score += min(width * height, 2_000_000)
            except (OSError, ValueError):
                continue
            if len(images) == 3:
                break
        if len(images) == 3:
            eligible.append((score, split, pet_id, images))
    eligible.sort(key=lambda item: (-item[0], item[1], item[2]))
    return [(split, pet_id, images) for _, split, pet_id, images in eligible[:2]]


def add_oxford_breed(output: Path, breed: str, source: datasets.OxfordIIITPet, records: list[dict]) -> None:
    matches = []
    for index in range(len(source)):
        image, (category, species) = source[index]
        try:
            if species != source.bin_class_to_idx['Cat']:
                continue
            found = source.classes[category].lower().replace(' ', '_')
            if found == breed and min(image.size) >= 250:
                matches.append((index, image.width * image.height))
        finally:
            image.close()
    for ordinal, (index, _) in enumerate(sorted(matches, key=lambda item: -item[1])[:2], 1):
        directory = output / breed / f'gato-{ordinal:02d}'
        directory.mkdir(parents=True)
        image, _ = source[index]
        try:
            original = ImageOps.exif_transpose(image).convert('RGB')
            save_jpeg(original, directory / 'foto-01.jpg')
            save_jpeg(ImageOps.mirror(original), directory / 'foto-02.jpg')
            margin_x, margin_y = round(original.width * 0.06), round(original.height * 0.06)
            crop = original.crop((margin_x, margin_y, original.width - margin_x, original.height - margin_y))
            save_jpeg(ImageEnhance.Contrast(crop).enhance(1.08), directory / 'foto-03.jpg')
        finally:
            image.close()
        records.append({'raca': breed, 'gato': f'gato-{ordinal:02d}', 'fonte': 'Oxford-IIIT Pet',
                        'id_fonte': str(index), 'tipo_de_fotos': '1 foto original + 2 variações da mesma foto'})


def run() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument('--archive', type=Path, required=True)
    parser.add_argument('--oxford-dir', type=Path, default=Path('dataset/oxford'))
    parser.add_argument('--output-dir', type=Path, default=Path('../dataset/avaliacao-24-gatos'))
    args = parser.parse_args()
    output = args.output_dir.resolve()
    if output.exists() and any(output.iterdir()):
        parser.error(f'Destino não vazio: {output}. Não sobrescrevo fotos existentes.')
    output.mkdir(parents=True, exist_ok=True)
    records = []
    with ZipFile(args.archive) as archive:
        profiles = petfinder_profiles(archive)
        for breed, petfinder_label in PETFINDER_BREEDS.items():
            selected = choose_photos(archive, profiles[petfinder_label])
            if len(selected) != 2:
                raise ValueError(f'Não há dois perfis adequados para {breed}')
            for ordinal, (split, pet_id, raws) in enumerate(selected, 1):
                directory = output / breed / f'gato-{ordinal:02d}'
                directory.mkdir(parents=True)
                for photo_number, raw in enumerate(raws, 1):
                    with Image.open(io.BytesIO(raw)) as image:
                        save_jpeg(image, directory / f'foto-{photo_number:02d}.jpg')
                records.append({'raca': breed, 'gato': f'gato-{ordinal:02d}', 'fonte': 'PetFinder.my',
                                'id_fonte': f'{split}-{pet_id}', 'tipo_de_fotos': '3 fotos do mesmo perfil'})
            print(f'{breed}: 2 perfis, 6 fotos', flush=True)
    oxford = datasets.OxfordIIITPet(args.oxford_dir, split='test',
                                    target_types=['category', 'binary-category'], download=False)
    for breed in OXFORD_BREEDS:
        add_oxford_breed(output, breed, oxford, records)
        print(f'{breed}: 2 perfis, 6 variações', flush=True)
    with (output / 'manifesto.csv').open('w', encoding='utf-8', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=['raca', 'gato', 'fonte', 'id_fonte', 'tipo_de_fotos'])
        writer.writeheader()
        writer.writerows(records)
    (output / 'LEIA-ME.txt').write_text(
        'Conjunto LOCAL para testar cadastro, upload e classificação no CatCare AI.\n'
        '24 perfis: 2 por raça; 72 arquivos JPG: 3 por perfil.\n'
        'PetFinder: 18 perfis com 3 fotos do mesmo anúncio, Quantity=1; raça declarada pelo anunciante, NÃO pedigree verificado.\n'
        'Oxford: 6 perfis em Birman, Egyptian Mau e Sphynx; 1 foto original + espelho + recorte do mesmo gato.\n'
        'As três variações Oxford NÃO são poses/fotos independentes e não devem medir precisão estatística.\n'
        'Não publique nem redistribua as fotos PetFinder. Este diretório é ignorado pelo Git.\n'
        'Fontes: https://www.kaggle.com/competitions/petfinder-adoption-prediction/data\n'
        'https://www.robots.ox.ac.uk/~vgg/data/pets/\n',
        encoding='utf-8',
    )
    image_count = len(list(output.glob('*/*/foto-*.jpg')))
    print(f'Criados {len(records)} gatos e {image_count} fotos em {output}', flush=True)
    if len(records) != 24 or image_count != 72:
        raise RuntimeError('Contagem final incorreta')


if __name__ == '__main__':
    run()
