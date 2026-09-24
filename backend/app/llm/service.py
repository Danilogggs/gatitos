import httpx

from app.core.config import settings
from app.ml.service import Prediction


async def describe(name: str, prediction: Prediction, behavior: str | None) -> str:
    visual = ', '.join(filter(None, [prediction.breed, prediction.coat_pattern, prediction.coat_length, *prediction.features, *prediction.colors]))
    fallback = f'{name} está disponível para adoção.' + (f' Características visuais: {visual}.' if visual else '')
    if behavior:
        fallback += f' Comportamento informado pelo responsável: {behavior}.'
    config = settings()
    if not config.gemini_api_key or not config.gemini_model:
        return fallback
    prompt = (
        'Escreva em português uma descrição curta e factual para adoção. Use SOMENTE os dados a seguir. '
        'Não invente personalidade, saúde, raça genética ou dados ausentes. '
        f'Nome: {name}. Dados visuais do classificador: {visual or "nenhum"}. '
        f'Comportamento informado pela pessoa: {behavior or "não informado"}.'
    )
    url = f'https://generativelanguage.googleapis.com/v1beta/models/{config.gemini_model}:generateContent'
    try:
        async with httpx.AsyncClient(timeout=15) as client:
            response = await client.post(url, headers={'x-goog-api-key': config.gemini_api_key}, json={'contents': [{'parts': [{'text': prompt}]}]})
            response.raise_for_status()
            return response.json()['candidates'][0]['content']['parts'][0]['text'].strip()[:1200]
    except (httpx.HTTPError, KeyError, IndexError, TypeError):
        return fallback

