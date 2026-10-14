from fastapi import HTTPException
from supabase import Client

from app.storage.images import public_url

PUBLIC_FIELDS = 'id,name,sex,approximate_age,size,breed,coat_pattern,coat_length,primary_color,secondary_color,behavior,health_information,location,additional_notes,description,status,created_at,published_at'
ADMIN_FIELDS = PUBLIC_FIELDS + ',created_by,updated_at,adopted_at,breed_training_verified,breed_verification_note'


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
    predicted_breed = original['predicted_breed']
    corrected_breed = cat['breed']
    # SRD é rejeição por baixa confiança, não uma classe treinada: SRD→SRD não é acerto.
    breed_correct = (predicted_breed == corrected_breed if predicted_breed and corrected_breed
                     and not (predicted_breed == 'srd' and corrected_breed == 'srd') else None)
    values = {
        'prediction_id': original['id'], 'cat_id': cat['id'],
        'predicted_breed': predicted_breed, 'corrected_breed': corrected_breed,
        'breed_correct': breed_correct,
        'predicted_coat_pattern': original['predicted_coat_pattern'],
        'corrected_coat_pattern': cat['coat_pattern'],
        'coat_correct': original['predicted_coat_pattern'] == cat['coat_pattern'] if original['predicted_coat_pattern'] else None,
        'corrected_by': reviewer,
    }
    if already:
        client.table('prediction_feedback').update(values).eq('id', already[0]['id']).execute()
    else:
        client.table('prediction_feedback').insert(values).execute()


def review_event(client: Client, cat: dict, reviewer: str, event_type: str,
                 previous_breed: str | None = None) -> None:
    prediction = client.table('ml_predictions').select('id').eq('cat_id', cat['id']).order('created_at', desc=True).limit(1).execute().data
    client.table('cat_review_events').insert({
        'cat_id': cat['id'], 'prediction_id': prediction[0]['id'] if prediction else None,
        'event_type': event_type, 'previous_breed': previous_breed,
        'reviewed_breed': cat.get('breed'),
        'breed_training_verified': cat.get('breed_training_verified', False),
        'breed_verification_note': cat.get('breed_verification_note'),
        'reviewed_by': reviewer,
    }).execute()

