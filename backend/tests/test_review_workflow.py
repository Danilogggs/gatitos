import unittest
import csv
import tempfile
from io import BytesIO
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from PIL import Image

from app.ml.export_verified_breeds import export
from app.services.cats import feedback


class Table:
    def __init__(self, rows):
        self.rows = rows
        self.saved = None
        self.mode = None

    def select(self, *_): return self
    def eq(self, *_): return self
    def order(self, *_, **__): return self
    def limit(self, *_): return self
    def insert(self, values):
        self.saved, self.mode = values, 'insert'
        return self
    def update(self, values):
        self.saved, self.mode = values, 'update'
        return self
    def execute(self): return SimpleNamespace(data=self.rows)


class Client:
    def __init__(self, existing=False):
        self.tables = {
            'ml_predictions': Table([{'id': 'prediction', 'predicted_breed': 'srd',
                                      'predicted_coat_pattern': None}]),
            'prediction_feedback': Table([{'id': 'feedback'}] if existing else []),
        }

    def table(self, name): return self.tables[name]


class ReviewWorkflowTest(unittest.TestCase):
    def test_srd_kept_as_srd_is_not_a_trained_breed_hit(self):
        client = Client()
        feedback(client, {'id': 'cat', 'breed': 'srd', 'coat_pattern': None}, 'admin')
        self.assertIsNone(client.tables['prediction_feedback'].saved['breed_correct'])

    def test_admin_change_from_srd_counts_as_correction(self):
        client = Client(existing=True)
        feedback(client, {'id': 'cat', 'breed': 'persian', 'coat_pattern': None}, 'admin')
        table = client.tables['prediction_feedback']
        self.assertEqual(table.mode, 'update')
        self.assertFalse(table.saved['breed_correct'])
        self.assertEqual(table.saved['corrected_breed'], 'persian')

    @patch('app.ml.export_verified_breeds.settings')
    def test_breed_export_requires_verified_published_label(self, mock_settings):
        mock_settings.return_value = SimpleNamespace(supabase_storage_bucket='test')
        image = BytesIO()
        Image.new('RGB', (20, 20), 'black').save(image, format='WEBP')

        class Rows:
            def __init__(self, rows): self.rows = rows
            def select(self, *_): return self
            def eq(self, key, value):
                self.rows = [row for row in self.rows if row[key] == value]
                return self
            def range(self, start, end):
                self.rows = self.rows[start:end + 1]
                return self
            def order(self, *_): return self
            def execute(self): return SimpleNamespace(data=self.rows)

        class ExportClient:
            storage = SimpleNamespace(from_=lambda *_: SimpleNamespace(download=lambda *_: image.getvalue()))
            data = {
                'cats': [
                    {'id': 'good', 'status': 'PUBLISHED', 'breed': 'persian',
                     'breed_training_verified': True, 'breed_verification_note': 'pedigree conferido'},
                    {'id': 'srd', 'status': 'PUBLISHED', 'breed': 'srd',
                     'breed_training_verified': True, 'breed_verification_note': 'incerto'},
                    {'id': 'pending', 'status': 'PENDING_REVIEW', 'breed': 'persian',
                     'breed_training_verified': True, 'breed_verification_note': 'pedigree conferido'},
                ],
                'cat_images': [{'id': 'image', 'cat_id': 'good', 'storage_path': 'good/image.webp'}],
            }
            def table(self, name): return Rows(self.data[name])

        with tempfile.TemporaryDirectory() as temp:
            counts = export(ExportClient(), Path(temp))
            with (Path(temp) / 'labels.csv').open(encoding='utf-8', newline='') as handle:
                rows = list(csv.DictReader(handle))
        self.assertEqual(counts['exported_cats'], 1)
        self.assertEqual(len(rows), 1)
        self.assertEqual(rows[0]['breed'], 'persian')


if __name__ == '__main__':
    unittest.main()
