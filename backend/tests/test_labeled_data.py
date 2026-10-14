import csv
import hashlib
import tempfile
import unittest
from io import BytesIO
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch
from zipfile import ZipFile

from PIL import Image

from app.ml.import_petfinder import import_dataset
from app.ml.import_petfinder_zip import import_archive
from app.ml.export_reviewed import export
from app.ml.train_labeled import load_examples, split_examples
from app.ml.train_reviewed_colors import load_review, split as split_color_review


class LabeledDataTest(unittest.TestCase):
    def test_visual_color_review_deduplicates_and_keeps_splits_disjoint(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            rows = []
            for label in ('orange', 'cream', 'neither'):
                for index in range(10):
                    name = f'{label}-{index}.jpg'
                    (root / name).write_bytes(f'{label}-{index}'.encode())
                    rows.append({'file': name, 'suggested_color': '', 'reviewed_color': label})
            rows.append({'file': 'orange-0.jpg', 'suggested_color': '', 'reviewed_color': 'orange'})
            rows.append({'file': 'cream-0.jpg', 'suggested_color': '', 'reviewed_color': 'neither'})
            manifest = root / 'review.csv'
            with manifest.open('w', newline='', encoding='utf-8') as stream:
                writer = csv.DictWriter(stream, fieldnames=rows[0])
                writer.writeheader()
                writer.writerows(rows)
            examples = load_review(manifest)
            self.assertEqual(len(examples), 29)
            with self.assertRaises(ValueError):
                split_color_review(examples)
            rows.pop()
            with manifest.open('w', newline='', encoding='utf-8') as stream:
                writer = csv.DictWriter(stream, fieldnames=rows[0])
                writer.writeheader()
                writer.writerows(rows)
            training, validation, test = split_color_review(load_review(manifest))
            paths = [item['path'] for group in (training, validation, test) for item in group]
            self.assertEqual(len(paths), len(set(paths)))
            self.assertEqual(len(paths), 30)

    @patch('app.ml.export_reviewed.settings')
    def test_export_uses_only_published_reviewed_cats(self, mock_settings):
        mock_settings.return_value = SimpleNamespace(supabase_storage_bucket='test')
        image = BytesIO()
        Image.new('RGB', (24, 24), 'black').save(image, format='WEBP')

        class Table:
            def __init__(self, rows):
                self.rows = rows[:]

            def select(self, *_): return self
            def eq(self, key, value):
                self.rows = [row for row in self.rows if row[key] == value]
                return self
            def order(self, *_): return self
            def limit(self, count):
                self.rows = self.rows[:count]
                return self
            def range(self, start, end):
                self.rows = self.rows[start:end + 1]
                return self
            def execute(self): return SimpleNamespace(data=self.rows)

        class Client:
            storage = SimpleNamespace(from_=lambda *_: SimpleNamespace(download=lambda *_: image.getvalue()))
            data = {
                'prediction_feedback': [{'cat_id': 'approved'}, {'cat_id': 'pending'}],
                'cats': [
                    {'id': 'approved', 'status': 'PUBLISHED', 'breed': 'srd', 'primary_color': 'black',
                     'secondary_color': None, 'coat_length': 'short'},
                    {'id': 'pending', 'status': 'PENDING_REVIEW', 'breed': 'srd', 'primary_color': 'white',
                     'secondary_color': None, 'coat_length': 'long'},
                ],
                'cat_images': [{'id': 'photo-1', 'cat_id': 'approved', 'storage_path': 'approved/photo.webp'}],
            }

            def table(self, name): return Table(self.data[name])

        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            counts = export(Client(), root)
            with (root / 'labels.csv').open(encoding='utf-8', newline='') as stream:
                rows = list(csv.DictReader(stream))
            self.assertEqual(counts['exported_cats'], 1)
            self.assertEqual(counts['exported_photos'], 1)
            self.assertEqual(rows[0]['colors'], 'black')
            self.assertEqual(rows[0]['coat_length'], 'short')
            self.assertEqual(len(rows), 1)

    def test_petfinder_import_keeps_only_single_cats_with_supported_colors(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            source = root / 'raw'
            images = source / 'train_images'
            images.mkdir(parents=True)
            with (source / 'color_labels.csv').open('w', newline='', encoding='utf-8') as stream:
                writer = csv.DictWriter(stream, fieldnames=['ColorID', 'ColorName'])
                writer.writeheader()
                writer.writerows([{'ColorID': '1', 'ColorName': 'Black'},
                                  {'ColorID': '2', 'ColorName': 'Orange'},
                                  {'ColorID': '3', 'ColorName': 'White'}])
            fields = ['Type', 'Quantity', 'PetID', 'Color1', 'Color2', 'Color3', 'FurLength']
            with (source / 'train.csv').open('w', newline='', encoding='utf-8') as stream:
                writer = csv.DictWriter(stream, fieldnames=fields)
                writer.writeheader()
                writer.writerows([
                    {'Type': '2', 'Quantity': '1', 'PetID': 'good', 'Color1': '1', 'Color2': '3', 'Color3': '0', 'FurLength': '2'},
                    {'Type': '2', 'Quantity': '1', 'PetID': 'unknown', 'Color1': '2', 'Color2': '0', 'Color3': '0', 'FurLength': '1'},
                    {'Type': '2', 'Quantity': '2', 'PetID': 'group', 'Color1': '1', 'Color2': '0', 'Color3': '0', 'FurLength': '1'},
                ])
            Image.new('RGB', (24, 24), 'black').save(images / 'good-1.jpg')
            Image.new('RGB', (24, 24), 'white').save(images / 'good-2.jpg')
            output = root / 'converted'

            counts = import_dataset(source, output)
            examples = load_examples([output / 'labels.csv'])

            self.assertEqual(counts['imported'], 1)
            self.assertEqual(counts['imported_photos'], 2)
            self.assertEqual(counts['unsupported_colors'], 1)
            self.assertEqual(examples[0]['colors'], ['black', 'white'])
            self.assertEqual(examples[0]['length'], 'medium')
            self.assertEqual(len(examples[0]['images']), 2)

    def test_split_keeps_cat_images_together(self):
        examples = [{'id': ('review', str(index)), 'images': [index, index],
                     'colors': ['black'], 'length': 'short'} for index in range(30)]
        training, validation, test = split_examples(examples)
        self.assertEqual(len(training) + len(validation) + len(test), 30)
        self.assertEqual(len({item['id'] for item in training + validation + test}), 30)

    def test_zip_import_uses_train_and_test_photos(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            archive = root / 'petfinder.zip'
            photo = BytesIO()
            Image.new('RGB', (24, 24), 'black').save(photo, format='JPEG')
            fields = ['Type', 'Quantity', 'PetID', 'Color1', 'Color2', 'Color3', 'FurLength']
            with ZipFile(archive, 'w') as zipped:
                zipped.writestr('color_labels.csv', 'ColorID,ColorName\n1,Black\n')
                for split in ('train', 'test'):
                    buffer = BytesIO()
                    import io
                    text = io.TextIOWrapper(buffer, encoding='utf-8', newline='', write_through=True)
                    writer = csv.DictWriter(text, fieldnames=fields)
                    writer.writeheader()
                    writer.writerow({'Type': '2', 'Quantity': '1', 'PetID': split,
                                     'Color1': '1', 'Color2': '0', 'Color3': '0', 'FurLength': '1'})
                    text.flush()
                    zipped.writestr(f'{split}/{split}.csv', buffer.getvalue())
                    zipped.writestr(f'{split}_images/{split}-1.jpg', photo.getvalue())
                    zipped.writestr(f'{split}_images/{split}-2.jpg', photo.getvalue())
            output = root / 'imported'
            counts = import_archive(archive, output)
            self.assertEqual(counts['imported'], 2)
            self.assertEqual(counts['imported_photos'], 4)
            with (output / 'labels.csv').open(encoding='utf-8', newline='') as handle:
                self.assertEqual(len(list(csv.DictReader(handle))), 4)

    def test_changed_photo_is_rejected(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            photo = root / 'cat.jpg'
            photo.write_bytes(b'changed')
            with (root / 'labels.csv').open('w', newline='', encoding='utf-8') as stream:
                writer = csv.DictWriter(stream, fieldnames=['path', 'cat_id', 'source', 'colors', 'coat_length', 'sha256'])
                writer.writeheader()
                writer.writerow({'path': 'cat.jpg', 'cat_id': '1', 'source': 'review',
                                 'colors': 'black', 'coat_length': 'short',
                                 'sha256': hashlib.sha256(b'original').hexdigest()})
            with self.assertRaisesRegex(ValueError, 'Foto alterada'):
                load_examples([root / 'labels.csv'])
