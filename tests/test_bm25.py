from app.retrieval.bm25 import BM25Retriever


def test_bm25_retrieves_relevant_document() -> None:
    documents = [
        "Python is a programming language.",
        "Docker packages applications into containers.",
        "Retrieval augmented generation combines search with language models.",
    ]

    retriever = BM25Retriever(documents)
    results = retriever.search("language model retrieval", top_k=1)

    assert results[0][0] == documents[2]