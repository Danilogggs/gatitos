import json
import re
from uuid import UUID, uuid4

from fastapi import APIRouter, Depends, File, Form, HTTPException, Query, UploadFile
from supabase import Client

from app.core.auth import admin_user
from app.core.config import settings
from app.core.db import db
from app.llm.service import describe
from app.ml.service import ml_service
from app.ml.taxonomy import BREEDS, FEATURES
from app.schemas.cats import CatEdit, CatOptional, ImageOrder
from app.services.cats import ADMIN_FIELDS, PUBLIC_FIELDS, enrich, feedback, one
from app.storage.images import read_images, upload

router = APIRouter(prefix='/api')


def valid_id(value: str) -> str:
    try:
        return str(UUID(value))
    except ValueError as exc:
        raise HTTPException(422, 'ID inválido') from exc


@router.get('/config')
def public_config() -> dict:
    config = settings()
    return {'ml_mode': config.ml_mode, 'features': FEATURES, 'breeds': BREEDS}


@router.post('/ml/predict')
async def predict(files: list[UploadFile] = File(...), name: str = Form(''), behavior: str = Form('')) -> dict:
    images = await read_images(files)
    prediction = ml_service().predict(images)
    return {**prediction.as_dict(), 'description': await describe(name.strip()[:100] or 'Este gato', prediction, behavior.strip()[:1000] or None)}


@router.post('/cats')
async def create_cat(name: str = Form(...), files: list[UploadFile] = File(...), optional: str = Form('{}'), client: Client = Depends(db)) -> dict:
    name = name.strip()
    if not name or len(name) > 100:
        raise HTTPException(422, 'Nome obrigatório (até 100 caracteres)')
    try:
        extra = CatOptional.model_validate(json.loads(optional)).model_dump()
    except (ValueError, TypeError) as exc:
        raise HTTPException(422, 'Informações opcionais inválidas') from exc
    images = await read_images(files)
    prediction = ml_service().predict(images)
    description = await describe(name, prediction, extra.get('behavior'))
    cat_id = str(uuid4())
    uploaded = []
    try:
        cat = client.table('cats').insert({'id': cat_id, 'name': name, **extra,
            'breed': prediction.breed, 'coat_pattern': prediction.coat_pattern,
            'coat_length': prediction.coat_length, 'primary_color': prediction.colors[0] if prediction.colors else None,
            'secondary_color': prediction.colors[1] if len(prediction.colors) > 1 else None,
            'description': description, 'status': 'PENDING_REVIEW'}).execute().data[0]
        for position, image in enumerate(images):
            row = upload(client, cat_id, image, position)
            uploaded.append(row['storage_path'])
            client.table('cat_images').insert(row).execute()
        for feature in prediction.features:
            client.table('cat_features').insert({'cat_id': cat_id, 'feature': feature}).execute()
        client.table('ml_predictions').insert({'cat_id': cat_id, 'predicted_breed': prediction.breed,
            'breed_confidence': prediction.breed_confidence, 'predicted_features': prediction.features,
            'predicted_coat_pattern': prediction.coat_pattern, 'coat_confidence': prediction.coat_confidence,
            'predicted_colors': prediction.colors, 'predicted_coat_length': prediction.coat_length,
            'model_version': prediction.model_version}).execute()
    except Exception:
        if uploaded:
            client.storage.from_(settings().supabase_storage_bucket).remove(uploaded)
        client.table('cats').delete().eq('id', cat_id).execute()
        raise
    return {'id': cat['id'], 'status': cat['status'], 'ml_mode': settings().ml_mode}


@router.get('/cats')
def list_cats(name: str | None = Query(None, max_length=100), breed: str | None = None,
              feature: str | None = None, color: str | None = None, sex: str | None = None,
              size: str | None = None, age: str | None = None, coat_length: str | None = None,
              page: int = Query(1, ge=1), client: Client = Depends(db)) -> list[dict]:
    query = client.table('cats').select(PUBLIC_FIELDS).eq('status', 'PUBLISHED')
    if name:
        query = query.ilike('name', '%' + name.replace('%', '\\%').replace('_', '\\_') + '%')
    for key, value in [('breed', breed), ('sex', sex), ('size', size), ('approximate_age', age), ('coat_length', coat_length)]:
        if value:
            query = query.eq(key, value)
    if color:
        if not re.fullmatch(r'[\w -]{1,50}', color):
            raise HTTPException(422, 'Cor inválida')
        query = query.or_(f'primary_color.eq.{color},secondary_color.eq.{color}')
    if feature:
        ids = [row['cat_id'] for row in client.table('cat_features').select('cat_id').eq('feature', feature).execute().data]
        if not ids:
            return []
        query = query.in_('id', ids)
    rows = query.order('published_at', desc=True).range((page - 1) * 24, page * 24 - 1).execute().data
    return [enrich(client, row) for row in rows]


@router.get('/cats/{cat_id}')
def get_cat(cat_id: str, client: Client = Depends(db)) -> dict:
    return one(client, valid_id(cat_id))


@router.get('/cats/{cat_id}/similar')
def similar(cat_id: str, client: Client = Depends(db)) -> list[dict]:
    one(client, valid_id(cat_id))
    return []  # pgvector e índice de similaridade poderão ser adicionados após treinamento.


@router.get('/admin/cats')
def admin_cats(status: str = 'PENDING_REVIEW', client: Client = Depends(db), _: str = Depends(admin_user)) -> list[dict]:
    allowed = {'PENDING_REVIEW', 'PUBLISHED', 'ADOPTED', 'ARCHIVED', 'REJECTED'}
    if status not in allowed:
        raise HTTPException(422, 'Status inválido')
    rows = client.table('cats').select(ADMIN_FIELDS).eq('status', status).order('created_at', desc=True).limit(100).execute().data
    return [enrich(client, row, True) for row in rows]


@router.get('/admin/cats/{cat_id}')
def admin_cat(cat_id: str, client: Client = Depends(db), _: str = Depends(admin_user)) -> dict:
    return one(client, valid_id(cat_id), True)


@router.put('/cats/{cat_id}')
def edit_cat(cat_id: str, changes: CatEdit, client: Client = Depends(db), _: str = Depends(admin_user)) -> dict:
    cat_id = valid_id(cat_id)
    one(client, cat_id, True)
    features = list(dict.fromkeys(changes.features))
    if len(features) > len(FEATURES) or any(feature not in FEATURES for feature in features):
        raise HTTPException(422, 'Característica inválida')
    data = changes.model_dump(exclude={'features'})
    data['name'] = data['name'].strip()
    if not data['name']:
        raise HTTPException(422, 'Nome obrigatório')
    client.table('cats').update(data).eq('id', cat_id).execute()
    client.table('cat_features').delete().eq('cat_id', cat_id).execute()
    if features:
        client.table('cat_features').insert([{'cat_id': cat_id, 'feature': feature} for feature in features]).execute()
    return one(client, cat_id, True)


def transition(cat_id: str, statuses: tuple[str, ...], target: str, client: Client, reviewer: str) -> dict:
    cat_id = valid_id(cat_id)
    cat = one(client, cat_id, True)
    if cat['status'] not in statuses:
        raise HTTPException(409, 'Transição de status inválida')
    if not cat['images']:
        raise HTTPException(409, 'O gato precisa ter uma foto')
    if target in ('PUBLISHED', 'REJECTED'):
        feedback(client, cat, reviewer)
    from datetime import datetime, timezone
    now = datetime.now(timezone.utc).isoformat()
    data = {'status': target}
    if target == 'PUBLISHED': data['published_at'] = now
    if target == 'ADOPTED': data['adopted_at'] = now
    client.table('cats').update(data).eq('id', cat_id).execute()
    return one(client, cat_id, True)


@router.post('/cats/{cat_id}/submit')
def submit(cat_id: str, client: Client = Depends(db), reviewer: str = Depends(admin_user)) -> dict:
    return transition(cat_id, ('DRAFT',), 'PENDING_REVIEW', client, reviewer)


@router.post('/cats/{cat_id}/publish')
def publish(cat_id: str, client: Client = Depends(db), reviewer: str = Depends(admin_user)) -> dict:
    return transition(cat_id, ('PENDING_REVIEW',), 'PUBLISHED', client, reviewer)


@router.post('/cats/{cat_id}/reject')
def reject(cat_id: str, client: Client = Depends(db), reviewer: str = Depends(admin_user)) -> dict:
    return transition(cat_id, ('PENDING_REVIEW',), 'REJECTED', client, reviewer)


@router.post('/cats/{cat_id}/adopt')
def adopt(cat_id: str, client: Client = Depends(db), reviewer: str = Depends(admin_user)) -> dict:
    return transition(cat_id, ('PUBLISHED',), 'ADOPTED', client, reviewer)


@router.post('/cats/{cat_id}/archive')
def archive(cat_id: str, client: Client = Depends(db), reviewer: str = Depends(admin_user)) -> dict:
    return transition(cat_id, ('PUBLISHED', 'ADOPTED'), 'ARCHIVED', client, reviewer)


@router.post('/cats/{cat_id}/images')
async def add_images(cat_id: str, files: list[UploadFile] = File(...), client: Client = Depends(db), reviewer: str = Depends(admin_user)) -> dict:
    cat_id = valid_id(cat_id)
    cat = one(client, cat_id, True)
    images = await read_images(files)
    if len(cat['images']) + len(images) > 8:
        raise HTTPException(422, 'Máximo de 8 fotos')
    for index, image in enumerate(images):
        row = upload(client, cat_id, image, len(cat['images']) + index, reviewer)
        client.table('cat_images').insert(row).execute()
    return one(client, cat_id, True)


@router.delete('/cats/{cat_id}/images/{image_id}')
def delete_image(cat_id: str, image_id: str, client: Client = Depends(db), _: str = Depends(admin_user)) -> dict:
    cat_id = valid_id(cat_id); image_id = valid_id(image_id)
    cat = one(client, cat_id, True)
    if len(cat['images']) <= 1:
        raise HTTPException(409, 'O gato precisa manter uma foto')
    image = next((item for item in cat['images'] if item['id'] == image_id), None)
    if not image:
        raise HTTPException(404, 'Foto não encontrada')
    client.table('cat_images').delete().eq('id', image_id).eq('cat_id', cat_id).execute()
    if image['is_primary']:
        client.table('cat_images').update({'is_primary': True}).eq('id', next(item['id'] for item in cat['images'] if item['id'] != image_id)).execute()
    client.storage.from_(settings().supabase_storage_bucket).remove([image['storage_path']])
    return one(client, cat_id, True)


@router.put('/cats/{cat_id}/images/{image_id}/primary')
def primary_image(cat_id: str, image_id: str, client: Client = Depends(db), _: str = Depends(admin_user)) -> dict:
    cat_id = valid_id(cat_id); image_id = valid_id(image_id)
    cat = one(client, cat_id, True)
    if not any(image['id'] == image_id for image in cat['images']):
        raise HTTPException(404, 'Foto não encontrada')
    client.table('cat_images').update({'is_primary': False}).eq('cat_id', cat_id).execute()
    client.table('cat_images').update({'is_primary': True}).eq('id', image_id).execute()
    return one(client, cat_id, True)


@router.put('/cats/{cat_id}/images/order')
def reorder(cat_id: str, order: ImageOrder, client: Client = Depends(db), _: str = Depends(admin_user)) -> dict:
    cat_id = valid_id(cat_id)
    cat = one(client, cat_id, True)
    if set(order.image_ids) != {image['id'] for image in cat['images']} or len(order.image_ids) != len(cat['images']):
        raise HTTPException(422, 'Ordem incompleta')
    for position, image_id in enumerate(order.image_ids):
        client.table('cat_images').update({'position': position}).eq('id', image_id).execute()
    return one(client, cat_id, True)


@router.get('/admin/ml/stats')
def ml_stats(client: Client = Depends(db), _: str = Depends(admin_user)) -> dict:
    predictions = client.table('ml_predictions').select('id', count='exact').limit(1).execute()
    rows = client.table('prediction_feedback').select('breed_correct,coat_correct').execute().data
    return {'total_predictions': predictions.count or 0, 'total_reviews': len(rows),
            'breed_correct': sum(row['breed_correct'] is True for row in rows),
            'breed_errors': sum(row['breed_correct'] is False for row in rows),
            'coat_correct': sum(row['coat_correct'] is True for row in rows),
            'coat_errors': sum(row['coat_correct'] is False for row in rows)}
