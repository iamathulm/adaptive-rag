from pathlib import Path

from app.ingestion.chunker import DocumentChunker
from app.ingestion.converter import DocumentProcessor
from app.retrieval.indexer import DocumentIndexer


class IngestionService:
    def __init__(
        self,
        processor: DocumentProcessor | None = None,
        chunker: DocumentChunker | None = None,
        indexer: DocumentIndexer | None = None,
    ) -> None:
        self.processor = processor or DocumentProcessor()
        self.chunker = chunker or DocumentChunker()
        self.indexer = indexer or DocumentIndexer()

    def ingest(
        self,
        file_path: str | Path,
        document_id: str,
    ) -> int:
        path = Path(file_path)

        result = self.processor.convert(path)

        chunks = self.chunker.chunk(
            result.document,
            document_id=document_id,
            source=path,
        )

        self.indexer.index(chunks)

        return len(chunks)