import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app.evaluation import RetrievalEvaluator
from app.ingestion import DocumentChunker, DocumentProcessor
from app.models.evaluation import EvaluationCase
from app.retrieval.hybrid import HybridRetriever


PDF_PATH = "tests/fixtures/sample.pdf"
DOCUMENT_ID = "sample-pdf"
TOP_K = 3


def main() -> None:
    processor = DocumentProcessor()
    document = processor.convert(PDF_PATH).document

    chunker = DocumentChunker()
    chunks = chunker.chunk(
        document,
        document_id=DOCUMENT_ID,
        source=PDF_PATH,
    )

    retriever = HybridRetriever(
        [chunk.model_dump() for chunk in chunks]
    )

    cases = [
        EvaluationCase(
            query="What is this document about?",
            relevant_chunk_ids={chunk.chunk_id for chunk in chunks},
        ),
    ]

    evaluator = RetrievalEvaluator(retriever)
    results = evaluator.evaluate(cases, top_k=TOP_K)

    for result in results:
        print(f"\nQuery: {result.query}")
        print(f"Retrieved: {result.retrieved_chunk_ids}")
        print(f"Recall@{TOP_K}: {result.recall_at_k:.3f}")
        print(f"MRR@{TOP_K}: {result.reciprocal_rank:.3f}")


if __name__ == "__main__":
    main()