import unittest
from types import SimpleNamespace
from unittest.mock import AsyncMock, patch
from uuid import uuid4

from app.api.routes import edit_cat
from app.schemas.cats import CatEdit


class Table:
    def __init__(self):
        self.updated = None

    def update(self, data):
        self.updated = data
        return self

    def delete(self):
        return self

    def eq(self, *_):
        return self

    def execute(self):
        return SimpleNamespace(data=[])


class Client:
    def __init__(self):
        self.cats = Table()
        self.features = Table()

    def table(self, name):
        return self.cats if name == 'cats' else self.features


class AdminDescriptionTest(unittest.IsolatedAsyncioTestCase):
    def setUp(self):
        self.cat_id = str(uuid4())
        self.previous = {
            'name': 'Felix', 'breed': 'ragdoll', 'coat_length': 'medium',
            'primary_color': 'black', 'secondary_color': 'brown',
            'description': 'Felix: ragdoll, preto e marrom.',
        }
        self.client = Client()

    async def test_corrections_regenerate_untouched_description(self):
        changes = CatEdit(name='Felix', breed='srd', coat_length='medium',
                          primary_color='black', secondary_color='white',
                          description=self.previous['description'])
        with patch('app.api.routes.one', side_effect=[self.previous, {'id': self.cat_id}]), \
                patch('app.api.routes.review_event'), \
                patch('app.api.routes.describe_confirmed', new_callable=AsyncMock,
                      return_value='Felix: SRD, preto e branco.') as generator:
            await edit_cat(self.cat_id, changes, self.client, 'admin')
        self.assertEqual(self.client.cats.updated['description'], 'Felix: SRD, preto e branco.')
        self.assertNotIn('regenerate_description', self.client.cats.updated)
        generator.assert_awaited_once()

    async def test_manual_description_is_preserved(self):
        changes = CatEdit(name='Felix', breed='srd', coat_length='medium',
                          primary_color='black', secondary_color='white',
                          description='Texto escrito pelo administrador.')
        with patch('app.api.routes.one', side_effect=[self.previous, {'id': self.cat_id}]), \
                patch('app.api.routes.review_event'), \
                patch('app.api.routes.describe_confirmed', new_callable=AsyncMock) as generator:
            await edit_cat(self.cat_id, changes, self.client, 'admin')
        self.assertEqual(self.client.cats.updated['description'], changes.description)
        generator.assert_not_awaited()

    async def test_explicit_refresh_regenerates_existing_description(self):
        changes = CatEdit(name='Felix', breed='ragdoll', coat_length='medium',
                          primary_color='black', secondary_color='brown',
                          description=self.previous['description'], regenerate_description=True)
        with patch('app.api.routes.one', side_effect=[self.previous, {'id': self.cat_id}]), \
                patch('app.api.routes.review_event'), \
                patch('app.api.routes.describe_confirmed', new_callable=AsyncMock,
                      return_value='Descrição refeita.') as generator:
            await edit_cat(self.cat_id, changes, self.client, 'admin')
        self.assertEqual(self.client.cats.updated['description'], 'Descrição refeita.')
        generator.assert_awaited_once()
