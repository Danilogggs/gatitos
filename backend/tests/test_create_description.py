import json
import unittest
from types import SimpleNamespace
from unittest.mock import AsyncMock, patch

from app.api.routes import create_cat


class Table:
    def __init__(self, name):
        self.name = name
        self.inserted = []

    def insert(self, data):
        self.inserted.append(data)
        return self

    def execute(self):
        if self.name == 'cats':
            row = self.inserted[-1]
            return SimpleNamespace(data=[{'id': row['id'], 'status': row['status']}])
        return SimpleNamespace(data=[])


class Client:
    def __init__(self):
        self.tables = {name: Table(name) for name in ('cats', 'cat_images', 'ml_predictions')}

    def table(self, name):
        return self.tables[name]


class CreateDescriptionTest(unittest.IsolatedAsyncioTestCase):
    async def test_new_registration_saves_initial_description(self):
        client = Client()
        prediction = SimpleNamespace(
            breed='birman', breed_confidence=0.91,
            breed_scores={'birman': 0.91}, colors=['gray', 'white'],
            coat_length='medium', model_version='test-model',
        )
        optional = {'behavior': 'tranquilo', 'location': 'Curitiba'}

        with patch('app.api.routes.read_images', new_callable=AsyncMock,
                   return_value=[b'image']), \
                patch('app.api.routes.ml_service', return_value=SimpleNamespace(
                    predict=lambda _images: prediction)), \
                patch('app.api.routes.describe_confirmed', new_callable=AsyncMock,
                      return_value='Descrição inicial.') as generator, \
                patch('app.api.routes.upload', return_value={
                    'id': 'image-id', 'cat_id': 'cat-id', 'storage_path': 'cats/image.jpg',
                    'position': 0, 'is_primary': True,
                }), \
                patch('app.api.routes.settings', return_value=SimpleNamespace(
                    ml_mode='real', supabase_storage_bucket='cat-images')):
            result = await create_cat('Mimi', [SimpleNamespace()], json.dumps(optional), client)

        inserted = client.tables['cats'].inserted[0]
        self.assertEqual(inserted['description'], 'Descrição inicial.')
        self.assertEqual(inserted['status'], 'PENDING_REVIEW')
        self.assertEqual(inserted['breed'], 'birman')
        self.assertEqual(inserted['primary_color'], 'gray')
        self.assertEqual(inserted['secondary_color'], 'white')
        self.assertEqual(result['status'], 'PENDING_REVIEW')
        generated_from = generator.await_args.args[0]
        self.assertEqual(generated_from['name'], 'Mimi')
        self.assertEqual(generated_from['behavior'], 'tranquilo')
        self.assertEqual(generated_from['location'], 'Curitiba')
