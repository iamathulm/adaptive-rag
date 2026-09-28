from pydantic import BaseModel, Field

from app.models.retrieval import RetrievalResult


class AnswerRequest(BaseModel):
    query: str = Field(min_length=1)
    top_k: int = Field(default=5, ge=1, le=50)


class AnswerResponse(BaseModel):
    answer: str
    sources: list[RetrievalResult]