from app.ingestion import DocumentProcessor


def test_document_processor_initializes() -> None:
    processor = DocumentProcessor()
    assert processor is not None