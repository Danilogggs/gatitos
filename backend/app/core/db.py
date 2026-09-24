from functools import lru_cache

from supabase import Client, create_client

from app.core.config import settings


@lru_cache
def db() -> Client:
    config = settings()
    return create_client(config.supabase_url, config.supabase_service_role_key)

