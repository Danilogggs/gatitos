from urllib.parse import urlsplit

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.api.routes import router
from app.core.config import settings

def frontend_origins(configured: str) -> list[str]:
    """Accept both loopback hostnames for a locally configured frontend."""
    origins = [configured.rstrip('/')]
    parsed = urlsplit(origins[0])
    if parsed.scheme in ('http', 'https') and parsed.hostname in ('localhost', '127.0.0.1'):
        alternate = '127.0.0.1' if parsed.hostname == 'localhost' else 'localhost'
        port = f':{parsed.port}' if parsed.port is not None else ''
        origins.append(f'{parsed.scheme}://{alternate}{port}')
    return origins


app = FastAPI(title='CatCare AI API')
app.add_middleware(CORSMiddleware, allow_origins=frontend_origins(settings().frontend_origin), allow_credentials=False,
                   allow_methods=['GET', 'POST', 'PUT', 'DELETE'], allow_headers=['Authorization', 'Content-Type'])
app.include_router(router)

