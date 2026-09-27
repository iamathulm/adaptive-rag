from pydantic import BaseModel, Field


class RetrievalRequest(BaseModel):
    query: str = Field(min_length=1)
    top_k: int | None = Field(default=None, ge=1, le=50)


class RetrievalResponseItem(BaseModel):
    document_id: str
    content: str
    score: float
    source: str


class RetrievalResponse(BaseModel):
    results: list[RetrievalResponseItem]