from pathlib import Path

from app.ingestion import DocumentChunker, DocumentProcessor


def test_document_chunker_preserves_content_and_provenance() -> None:
    pdf_path = Path("tests/fixtures/sample.pdf")

    processor = DocumentProcessor()
    document = processor.convert(pdf_path).document

    chunker = DocumentChunker()
    chunks = chunker.chunk(
        document,
        document_id="test-document",
        source=pdf_path,
    )

    assert chunks
    assert all(chunk.content.strip() for chunk in chunks)
    assert all(chunk.document_id == "test-document" for chunk in chunks)
    assert all(chunk.chunk_id for chunk in chunks)
    assert all(chunk.source == str(pdf_path) for chunk in chunks)
    assert all(chunk.page is not None for chunk in chunks)