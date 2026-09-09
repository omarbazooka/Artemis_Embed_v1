from pydantic import BaseModel, Field, field_validator
from typing import List, Optional

ALLOWED_DIMENSIONS = {768, 512, 256, 128}


class EmbedRequest(BaseModel):
    texts: List[str] = Field(min_length=1, max_length=64)
    dimension: int = 768

    @field_validator("dimension")
    @classmethod
    def validate_dimension(cls, value: int) -> int:
        if value not in ALLOWED_DIMENSIONS:
            raise ValueError(f"dimension must be one of {sorted(ALLOWED_DIMENSIONS, reverse=True)}")
        return value

    @field_validator("texts")
    @classmethod
    def validate_texts(cls, values: List[str]) -> List[str]:
        cleaned = [v.strip() for v in values]
        if any(not v for v in cleaned):
            raise ValueError("texts cannot contain empty strings")
        return cleaned


class SimilarityRequest(BaseModel):
    text_a: str = Field(min_length=1, max_length=10_000)
    text_b: str = Field(min_length=1, max_length=10_000)
    dimension: int = 768

    @field_validator("dimension")
    @classmethod
    def validate_dimension(cls, value: int) -> int:
        if value not in ALLOWED_DIMENSIONS:
            raise ValueError(f"dimension must be one of {sorted(ALLOWED_DIMENSIONS, reverse=True)}")
        return value


class SearchRequest(BaseModel):
    query: str = Field(min_length=1, max_length=10_000)
    documents: List[str] = Field(min_length=1, max_length=64)
    dimension: int = 256
    top_k: int = Field(default=5, ge=1, le=20)

    @field_validator("dimension")
    @classmethod
    def validate_dimension(cls, value: int) -> int:
        if value not in ALLOWED_DIMENSIONS:
            raise ValueError(f"dimension must be one of {sorted(ALLOWED_DIMENSIONS, reverse=True)}")
        return value


class DocumentQueryRequest(BaseModel):
    document_id: Optional[str] = None
    query: str = Field(min_length=1, max_length=10_000)
    top_k: int = Field(default=5, ge=1, le=20)
