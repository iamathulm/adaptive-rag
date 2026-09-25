from app.retrieval.embeddings import Embedder
from app.retrieval.vector_store import VectorStore


class DocumentIndexer:
    def __init__(self) -> None:
        self.embedder = Embedder()
        self.vector_store = VectorStore()

    def index(
        self,
        documents: list[dict[str, str]],
        collection_name: str = "documents",
    ) -> None:
        self.vector_store.ensure_collection(collection_name)

        points = []

        for index, document in enumerate(documents):
            vector = self.embedder.embed(document["content"])
            points.append(
                {
                    "id": index,
                    "vector": vector,
                    "payload": document,
                }
            )

        self.vector_store.client.upsert(
            collection_name=collection_name,
            points=points,
        )