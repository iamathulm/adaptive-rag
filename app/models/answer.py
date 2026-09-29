from typing import Literal

from pydantic import BaseModel, Field

from app.models.retrieval import RetrievalResult


class AnswerRequest(BaseModel):
    query: str = Field(min_length=1)
    top_k: int = Field(default=5, ge=1, le=50)
    backend: Literal["bm25", "dense", "hybrid"] = "hybrid"
    document_id: str | None = Field(default=None, min_length=1)
    source: str | None = Field(default=None, min_length=1)


class AnswerCitation(BaseModel):
    chunk_id: str
    document_id: str
    source: str
    page: int | None = None
    content_hash: str | None = None


class AnswerResponse(BaseModel):
    answer: str
    citations: list[AnswerCitation]
    grounded: bool
    sources: list[RetrievalResult]