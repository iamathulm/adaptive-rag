from app.retrieval.bm25 import BM25Retriever


def test_bm25_retrieves_relevant_document() -> None:
    documents = [
        {
            "document_id": "1",
            "chunk_id": "1:0",
            "content": "Python is a programming language.",
            "source": "test",
        },
        {
            "document_id": "2",
            "chunk_id": "2:0",
            "content": "Docker packages applications into containers.",
            "source": "test",
        },
        {
            "document_id": "3",
            "chunk_id": "3:0",
            "content": "Retrieval augmented generation combines search with language models.",
            "source": "test",
        },
    ]

    retriever = BM25Retriever(documents)
    results = retriever.search("language model retrieval", top_k=1)

    assert results[0].document_id == "3"
    assert results[0].chunk_id == "3:0"
    assert results[0].content == documents[2]["content"]