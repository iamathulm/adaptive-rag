import hashlib
from uuid import NAMESPACE_URL, uuid5

from app.models.chunk import DocumentChunk
from app.retrieval.embeddings import Embedder
from app.retrieval.vector_store import VectorStore


class DocumentIndexer:
    def __init__(self) -> None:
        self.embedder = Embedder()
        self.vector_store = VectorStore()

    def index(
        self,
        chunks: list[DocumentChunk],
        collection_name: str = "documents",
    ) -> None:
        self.vector_store.ensure_collection(collection_name)
        for document_id in {chunk.document_id for chunk in chunks}:
            self.vector_store.delete_document(document_id, collection_name)

        points = []

        for chunk in chunks:
            vector = self.embedder.embed(chunk.content)

            payload = chunk.model_dump(exclude_none=True)
            payload["content_hash"] = payload.get("content_hash") or hashlib.sha256(
                chunk.content.encode("utf-8")
            ).hexdigest()

            points.append(
                {
                    "id": str(uuid5(NAMESPACE_URL, chunk.chunk_id)),
                    "vector": vector,
                    "payload": payload,
                }
            )

        self.vector_store.client.upsert(
            collection_name=collection_name,
            points=points,
        )