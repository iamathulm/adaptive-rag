from rank_bm25 import BM25Okapi

from app.models.retrieval import RetrievalResult


class BM25Retriever:
    def __init__(self, documents: list[dict[str, str]]) -> None:
        self.documents = documents
        tokenized = [
            document["content"].lower().split()
            for document in documents
        ]
        self._index = BM25Okapi(tokenized) if tokenized else None

    def search(
        self,
        query: str,
        top_k: int = 5,
    ) -> list[RetrievalResult]:
        if self._index is None:
            return []

        scores = self._index.get_scores(query.lower().split())
        ranked = sorted(
            zip(self.documents, scores, strict=True),
            key=lambda item: item[1],
            reverse=True,
        )

        return [
            RetrievalResult(
                document_id=document["document_id"],
                chunk_id=document.get("chunk_id", document["document_id"]),
                content=document["content"],
                score=float(score),
                source="bm25",
            )
            for document, score in ranked[:top_k]
        ]