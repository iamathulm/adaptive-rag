from app.retrieval.embeddings import Embedder


def test_embedder_class_exists() -> None:
    assert Embedder is not None