from pathlib import Path

from docling.document_converter import DocumentConverter


class DocumentProcessor:
    def __init__(self) -> None:
        self._converter = DocumentConverter()

    def convert(self, file_path: str | Path):
        return self._converter.convert(Path(file_path))