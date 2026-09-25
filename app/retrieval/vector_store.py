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