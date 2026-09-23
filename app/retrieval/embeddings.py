from sentence_transformers import SentenceTransformer


class Embedder:
    def __init__(self) -> None:
        self._model = SentenceTransformer("BAAI/bge-base-en")

    def embed(self, text: str) -> list[float]:
        return self._model.encode(text).tolist()