from pathlib import Path

from app.ingestion.chunker import DocumentChunker
from app.ingestion.converter import DocumentProcessor
from app.retrieval.hybrid import HybridRetriever
from app.retrieval.indexer import DocumentIndexer


class IngestionService:
    def __init__(
        self,
        processor: DocumentProcessor | None = None,
        chunker: DocumentChunker | None = None,
        indexer: DocumentIndexer | None = None,
        retriever: HybridRetriever | None = None,
    ) -> None:
        self.processor = processor or DocumentProcessor()
        self.chunker = chunker or DocumentChunker()
        self.indexer = indexer or DocumentIndexer()
        self.retriever = retriever

    def ingest(
        self,
        file_path: str | Path,
        document_id: str,
        source: str | Path | None = None,
    ) -> int:
        path = Path(file_path)

        result = self.processor.convert(path)

        chunks = self.chunker.chunk(
            result.document,
            document_id=document_id,
            source=source or path,
        )

        self.indexer.index(chunks)
        if self.retriever is not None:
            self.retriever.refresh()

        return len(chunks)