from rank_bm25 import BM25Okapi


class BM25Retriever:
    def __init__(self, documents: list[str]) -> None:
        self.documents = documents
        tokenized = [document.lower().split() for document in documents]
        self._index = BM25Okapi(tokenized)

    def search(self, query: str, top_k: int = 5) -> list[tuple[str, float]]:
        scores = self._index.get_scores(query.lower().split())
        ranked = sorted(
            zip(self.documents, scores, strict=True),
            key=lambda item: item[1],
            reverse=True,
        )
        return ranked[:top_k]