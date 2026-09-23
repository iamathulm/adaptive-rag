from FlagEmbedding import BGEM3FlagModel


class Embedder:
    def __init__(self) -> None:
        self._model = BGEM3FlagModel("BAAI/bge-m3", use_fp16=True)

    def embed(self, text: str) -> list[float]:
        result = self._model.encode([text])
        return result["dense_vecs"][0].tolist()