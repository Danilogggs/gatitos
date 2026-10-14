import unittest
from types import SimpleNamespace
from unittest.mock import AsyncMock, patch

from app.llm.service import describe, describe_confirmed
from app.ml.service import Prediction


class DescriptionTest(unittest.IsolatedAsyncioTestCase):
    @patch('app.llm.service.settings')
    async def test_fallback_uses_portuguese_labels_and_omits_removed_fields(self, mock_settings):
        mock_settings.return_value = SimpleNamespace(gemini_api_key='', gemini_model='')
        prediction = Prediction('russian_blue', 0.9, ['cinza'], 'solid',
                                ['gray', 'white'], 'short', None, 'local-model')

        result = await describe('Mimi', prediction, None)

        self.assertIn('azul russo', result)
        self.assertIn('pelo curto', result)
        self.assertIn('cinza', result)
        self.assertIn('branco', result)
        self.assertNotIn('russian_blue', result)
        self.assertNotIn('solid', result)

    @patch('app.llm.service.settings')
    async def test_confirmed_description_uses_admin_fields_not_old_prediction(self, mock_settings):
        mock_settings.return_value = SimpleNamespace(gemini_api_key='', gemini_model='')
        result = await describe_confirmed({
            'name': 'Felix', 'breed': 'srd', 'coat_length': 'short',
            'primary_color': 'black', 'secondary_color': 'white',
            'behavior': 'Gato muito tranquilo', 'health_information': 'Ótima',
            'additional_notes': 'Muito comilão',
        })
        self.assertIn('SRD', result)
        self.assertIn('preto e branco', result)
        self.assertIn('pelo curto', result)
        self.assertIn('Gato muito tranquilo', result)
        self.assertIn('Muito comilão', result)
        self.assertNotIn('ragdoll', result.lower())
        self.assertNotIn('marrom', result.lower())

    async def test_sensitive_admin_fields_stay_out_of_gemini_prompt(self):
        data = {
            'name': 'Gato Teste', 'breed': 'srd', 'coat_length': 'short',
            'primary_color': 'black', 'secondary_color': 'white',
            'health_information': 'INFORMAÇÃO_PRIVADA_DE_SAÚDE',
            'location': 'LOCALIZAÇÃO_PRIVADA',
            'additional_notes': 'OBSERVAÇÃO_PRIVADA',
        }
        with patch('app.llm.service._generate', new_callable=AsyncMock,
                   return_value='Gato Teste é SRD, de pelo curto, preto e branco.') as generator:
            result = await describe_confirmed(data)
        prompt = generator.await_args.args[0]
        for private in ('INFORMAÇÃO_PRIVADA_DE_SAÚDE', 'LOCALIZAÇÃO_PRIVADA', 'OBSERVAÇÃO_PRIVADA'):
            self.assertNotIn(private, prompt)
            self.assertIn(private, result)
