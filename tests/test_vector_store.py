from app.retrieval import VectorStore


def test_vector_store_initializes() -> None:
    store = VectorStore()
    assert store.client is not None