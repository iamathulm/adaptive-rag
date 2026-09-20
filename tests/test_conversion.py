from pathlib import Path

from app.ingestion import DocumentProcessor


def test_convert_pdf() -> None:
    pdf_path = Path("tests/fixtures/Unit -2 - TC.pdf")

    processor = DocumentProcessor()
    result = processor.convert(pdf_path)

    assert result.document is not None
    assert len(result.document.pages) > 0