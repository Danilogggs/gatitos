from io import BytesIO
from uuid import uuid4

from fastapi import HTTPException, UploadFile
from PIL import Image, UnidentifiedImageError
from supabase import Client

from app.core.config import settings

MAX_IMAGE_BYTES = 5 * 1024 * 1024
ALLOWED = {'image/jpeg': 'JPEG', 'image/png': 'PNG', 'image/webp': 'WEBP'}
EXTENSIONS = {'.jpg': 'image/jpeg', '.jpeg': 'image/jpeg', '.png': 'image/png', '.webp': 'image/webp'}


async def read_images(files: list[UploadFile]) -> list[bytes]:
    if not 1 <= len(files) <= 8:
        raise HTTPException(422, 'Envie entre 1 e 8 fotos')
    result = []
    for file in files:
        suffix = '.' + (file.filename or '').rsplit('.', 1)[-1].lower()
        if suffix not in EXTENSIONS or file.content_type != EXTENSIONS[suffix]:
            raise HTTPException(422, 'Formato de foto inválido')
        raw = await file.read(MAX_IMAGE_BYTES + 1)
        if not raw or len(raw) > MAX_IMAGE_BYTES:
            raise HTTPException(422, 'Cada foto deve ter até 5 MB')
        try:
            with Image.open(BytesIO(raw)) as image:
                if image.format != ALLOWED[file.content_type]:
                    raise ValueError('MIME incompatível')
                image.verify()
            with Image.open(BytesIO(raw)) as image:
                image.thumbnail((2200, 2200))
                output = BytesIO()
                image.convert('RGB').save(output, format='WEBP', quality=85)
                converted = output.getvalue()
                if len(converted) > MAX_IMAGE_BYTES:
                    raise HTTPException(422, 'Foto convertida excede 5 MB')
                result.append(converted)
        except (UnidentifiedImageError, OSError, ValueError) as exc:
            raise HTTPException(422, 'Arquivo de imagem inválido') from exc
    return result


def upload(client: Client, cat_id: str, image: bytes, position: int, uploaded_by: str | None = None) -> dict:
    path = f'{cat_id}/{uuid4()}.webp'
    bucket = settings().supabase_storage_bucket
    client.storage.from_(bucket).upload(path, image, {'content-type': 'image/webp', 'upsert': 'false'})
    return {'cat_id': cat_id, 'storage_path': path, 'mime_type': 'image/webp', 'file_size': len(image), 'position': position, 'is_primary': position == 0, 'uploaded_by': uploaded_by}


def public_url(client: Client, path: str) -> str:
    return client.storage.from_(settings().supabase_storage_bucket).get_public_url(path)
