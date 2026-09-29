from pydantic import BaseModel


class DocumentChunk(BaseModel):
    document_id: str
    chunk_id: str
    content: str
    source: str
    page: int | None = None
    content_hash: str | None = None