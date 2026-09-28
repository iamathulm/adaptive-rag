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
        document_id: str | None = None,
        source: str | None = None,
    ) -> list[RetrievalResult]:
        if self._index is None:
            return []

        scores = self._index.get_scores(query.lower().split())
        ranked = sorted(
            zip(self.documents, scores, strict=True),
            key=lambda item: item[1],
            reverse=True,
        )
        ranked = [
            (document, score)
            for document, score in ranked
            if (document_id is None or document.get("document_id") == document_id)
            and (source is None or document.get("source") == source)
        ]

        return [
            RetrievalResult(
    document_id=document["document_id"],
    chunk_id=document["chunk_id"],
    content=document["content"],
    score=float(score),
    source="bm25",
)
            for document, score in ranked[:top_k]
        ]