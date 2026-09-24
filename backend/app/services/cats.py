from fastapi import HTTPException
from supabase import Client

from app.storage.images import public_url

PUBLIC_FIELDS = 'id,name,sex,approximate_age,size,breed,coat_pattern,coat_length,primary_color,secondary_color,behavior,health_information,location,additional_notes,description,status,created_at,published_at'
ADMIN_FIELDS = PUBLIC_FIELDS + ',created_by,updated_at,adopted_at'


def one(client: Client, cat_id: str, admin: bool = False) -> dict:
    fields = ADMIN_FIELDS if admin else PUBLIC_FIELDS
    query = client.table('cats').select(fields).eq('id', cat_id)
    if not admin:
        query = query.eq('status', 'PUBLISHED')
    rows = query.limit(1).execute().data
    if not rows:
        raise HTTPException(404, 'Gato não encontrado')
    return enrich(client, rows[0], admin)


def enrich(client: Client, cat: dict, admin: bool = False) -> dict:
    images = client.table('cat_images').select('id,storage_path,position,is_primary,mime_type,file_size').eq('cat_id', cat['id']).order('position').execute().data
    cat['images'] = [{**image, 'url': public_url(client, image['storage_path'])} for image in images]
    cat['features'] = [item['feature'] for item in client.table('cat_features').select('feature').eq('cat_id', cat['id']).execute().data]
    if admin:
        predictions = client.table('ml_predictions').select('*').eq('cat_id', cat['id']).order('created_at', desc=True).limit(1).execute().data
        cat['prediction'] = predictions[0] if predictions else None
    return cat


def feedback(client: Client, cat: dict, reviewer: str) -> None:
    prediction = client.table('ml_predictions').select('*').eq('cat_id', cat['id']).order('created_at', desc=True).limit(1).execute().data
    if not prediction:
        return
    original = prediction[0]
    already = client.table('prediction_feedback').select('id').eq('prediction_id', original['id']).limit(1).execute().data
    if already:
        return
    client.table('prediction_feedback').insert({
        'prediction_id': original['id'], 'cat_id': cat['id'],
        'predicted_breed': original['predicted_breed'], 'corrected_breed': cat['breed'],
        'breed_correct': original['predicted_breed'] == cat['breed'] if original['predicted_breed'] else None,
        'predicted_coat_pattern': original['predicted_coat_pattern'],
        'corrected_coat_pattern': cat['coat_pattern'],
        'coat_correct': original['predicted_coat_pattern'] == cat['coat_pattern'] if original['predicted_coat_pattern'] else None,
        'corrected_by': reviewer,
    }).execute()

