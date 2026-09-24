from functools import lru_cache
from pathlib import Path

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=Path(__file__).resolve().parents[3] / '.env', extra='ignore')
    supabase_url: str
    supabase_anon_key: str
    supabase_service_role_key: str
    supabase_storage_bucket: str = 'cat-images'
    gemini_api_key: str = ''
    gemini_model: str = ''
    ml_mode: str = 'mock'
    ml_model_path: str = ''
    frontend_origin: str = 'http://localhost:5173'


@lru_cache
def settings() -> Settings:
    return Settings()
