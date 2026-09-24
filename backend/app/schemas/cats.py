from enum import Enum

from pydantic import BaseModel, Field

from app.ml.taxonomy import FEATURES


class CatStatus(str, Enum):
    DRAFT = 'DRAFT'
    PENDING_REVIEW = 'PENDING_REVIEW'
    PUBLISHED = 'PUBLISHED'
    REJECTED = 'REJECTED'
    ADOPTED = 'ADOPTED'
    ARCHIVED = 'ARCHIVED'


class CatOptional(BaseModel):
    sex: str | None = Field(None, max_length=30)
    approximate_age: str | None = Field(None, max_length=80)
    size: str | None = Field(None, max_length=30)
    behavior: str | None = Field(None, max_length=1000)
    health_information: str | None = Field(None, max_length=1000)
    location: str | None = Field(None, max_length=150)
    additional_notes: str | None = Field(None, max_length=2000)


class CatEdit(CatOptional):
    name: str = Field(min_length=1, max_length=100)
    breed: str | None = Field(None, max_length=100)
    coat_pattern: str | None = Field(None, max_length=100)
    coat_length: str | None = Field(None, max_length=30)
    primary_color: str | None = Field(None, max_length=50)
    secondary_color: str | None = Field(None, max_length=50)
    description: str | None = Field(None, max_length=2000)
    features: list[str] = Field(default_factory=list)


class ImageOrder(BaseModel):
    image_ids: list[str]

