from pydantic import BaseModel, Field

from app.models.retrieval import RetrievalResult


class RetrievalRequest(BaseModel):
    query: str = Field(min_length=1)
    top_k: int | None = Field(default=None, ge=1, le=50)


class RetrievalResponse(BaseModel):
    results: list[RetrievalResult]