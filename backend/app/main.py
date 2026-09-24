from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.api.routes import router
from app.core.config import settings

app = FastAPI(title='CatCare AI API')
app.add_middleware(CORSMiddleware, allow_origins=[settings().frontend_origin], allow_credentials=False,
                   allow_methods=['GET', 'POST', 'PUT', 'DELETE'], allow_headers=['Authorization', 'Content-Type'])
app.include_router(router)

