from typing import Literal

from pydantic import BaseModel, Field


class RetrievalRequest(BaseModel):
    query: str = Field(min_length=1)
    top_k: int | None = Field(default=None, ge=1, le=50)
    backend: Literal["bm25", "dense", "hybrid"] = "hybrid"
    document_id: str | None = Field(default=None, min_length=1)
    source: str | None = Field(default=None, min_length=1)


class RetrievalResponseItem(BaseModel):
    document_id: str
    content: str
    score: float
    source: str


class RetrievalResponse(BaseModel):
    results: list[RetrievalResponseItem]