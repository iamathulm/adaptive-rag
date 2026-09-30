from pathlib import Path
from unittest.mock import MagicMock

from app.ingestion import IngestionService
from app.models.chunk import DocumentChunk


def test_ingest_converts_chunks_and_indexes() -> None:
    processor = MagicMock()
    chunker = MagicMock()
    indexer = MagicMock()
    retriever = MagicMock()

    document = MagicMock()
    processor.convert.return_value.document = document

    chunks = [
        DocumentChunk(
            document_id="doc-1",
            chunk_id="doc-1:0",
            content="first chunk",
            source="tests/fixtures/sample.pdf",
            page=1,
        ),
        DocumentChunk(
            document_id="doc-1",
            chunk_id="doc-1:1",
            content="second chunk",
            source="tests/fixtures/sample.pdf",
            page=1,
        ),
    ]

    chunker.chunk.return_value = chunks

    service = IngestionService(
        processor=processor,
        chunker=chunker,
        indexer=indexer,
        retriever=retriever,
    )

    count = service.ingest(
        Path("tests/fixtures/sample.pdf"),
        document_id="doc-1",
    )

    assert count == 2

    processor.convert.assert_called_once_with(
        Path("tests/fixtures/sample.pdf")
    )

    chunker.chunk.assert_called_once_with(
        document,
        document_id="doc-1",
        source=Path("tests/fixtures/sample.pdf"),
    )

    indexer.index.assert_called_once_with(chunks)
    retriever.refresh.assert_called_once_with()