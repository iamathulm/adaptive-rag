from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams

from app.core.config import settings


class VectorStore:
    def __init__(self) -> None:
        self.client = QdrantClient(url=settings.qdrant_url)

    def ensure_collection(
        self,
        collection_name: str = "documents",
        vector_size: int = 768,
    ) -> None:
        if not self.client.collection_exists(collection_name):
            self.client.create_collection(
                collection_name=collection_name,
                vectors_config=VectorParams(
                    size=vector_size,
                    distance=Distance.COSINE,
                ),
            )

    def search(
        self,
        vector: list[float],
        collection_name: str = "documents",
        top_k: int = 5,
    ):
        return self.client.query_points(
            collection_name=collection_name,
            query=vector,
            limit=top_k,
        ).points

    def get_documents(
        self,
        collection_name: str = "documents",
        batch_size: int = 256,
    ) -> list[dict[str, str]]:
        documents: list[dict[str, str]] = []
        offset = None

        while True:
            points, offset = self.client.scroll(
                collection_name=collection_name,
                limit=batch_size,
                offset=offset,
                with_payload=True,
                with_vectors=False,
            )

            for point in points:
                payload = point.payload or {}
                content = payload.get("content")
                document_id = payload.get("document_id")

                if content is None or document_id is None:
                    continue

                documents.append(
                    {
                        "document_id": str(document_id),
                        "chunk_id": str(payload.get("chunk_id", point.id)),
                        "content": str(content),
                        "source": str(payload.get("source", "")),
                    }
                )

            if offset is None:
                break

        return documents