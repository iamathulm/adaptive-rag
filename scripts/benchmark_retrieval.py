import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app.evaluation import benchmark_retrieval, load_evaluation_cases
from app.ingestion import DocumentChunker, DocumentProcessor
from app.retrieval.hybrid import HybridRetriever

PDF_PATH = Path("tests/fixtures/sample.pdf")
DATASET_PATH = Path("tests/fixtures/evaluation_cases_v1.json")


def main() -> None:
    parser = argparse.ArgumentParser(description="Benchmark retrieval backends.")
    parser.add_argument("--top-k", type=int, default=3)
    parser.add_argument("--output", type=Path, default=Path("benchmark_results.json"))
    args = parser.parse_args()

    document = DocumentProcessor().convert(PDF_PATH).document
    chunks = DocumentChunker().chunk(
        document,
        document_id="sample-pdf",
        source=PDF_PATH,
    )
    retriever = HybridRetriever([chunk.model_dump() for chunk in chunks])
    cases = load_evaluation_cases(DATASET_PATH)
    results = benchmark_retrieval(retriever, cases, top_k=args.top_k)

    payload = [result.model_dump() for result in results]
    args.output.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()