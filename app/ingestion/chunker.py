from pathlib import Path

from docling.chunking import HybridChunker
from docling_core.transforms.chunker.tokenizer.huggingface import (
    HuggingFaceTokenizer,
)
from transformers import AutoTokenizer

from app.models.chunk import DocumentChunk


class DocumentChunker:
    def __init__(
        self,
        model_name: str = "BAAI/bge-base-en",
    ) -> None:
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        hf_tokenizer = HuggingFaceTokenizer(tokenizer=tokenizer)

        self._chunker = HybridChunker(
            tokenizer=hf_tokenizer,
        )

    def chunk(
        self,
        document,
        document_id: str,
        source: str | Path,
    ) -> list[DocumentChunk]:
        chunks: list[DocumentChunk] = []

        for index, chunk in enumerate(self._chunker.chunk(document)):
            page = None

            if chunk.meta.doc_items:
                pages = [
                    provenance.page_no
                    for item in chunk.meta.doc_items
                    for provenance in item.prov
                ]
                if pages:
                    page = min(pages)

            chunks.append(
                DocumentChunk(
                    document_id=document_id,
                    chunk_id=f"{document_id}:{index}",
                    content=chunk.text,
                    source=str(source),
                    page=page,
                )
            )

        return chunks