from uuid import uuid5, NAMESPACE_URL

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

        points = []

        for chunk in chunks:
            vector = self.embedder.embed(chunk.content)

            points.append(
                {
                    "id": str(uuid5(NAMESPACE_URL, chunk.chunk_id)),
                    "vector": vector,
                    "payload": chunk.model_dump(),
                }
            )

        self.vector_store.client.upsert(
            collection_name=collection_name,
            points=points,
        )