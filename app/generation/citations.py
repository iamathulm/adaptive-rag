import re

from app.models.answer import AnswerCitation
from app.models.retrieval import RetrievalResult

_CITATION_PATTERN = re.compile(r"\[([^\[\]\s]+)\]")


def validate_citations(
    answer: str,
    results: list[RetrievalResult],
) -> tuple[list[AnswerCitation], bool]:
    results_by_chunk = {result.chunk_id: result for result in results}
    citation_ids = _CITATION_PATTERN.findall(answer)
    citations: list[AnswerCitation] = []
    seen: set[str] = set()
    has_invalid_citation = False

    for citation_id in citation_ids:
        result = results_by_chunk.get(citation_id)
        if result is None:
            has_invalid_citation = True
            continue
        if citation_id in seen:
            continue

        seen.add(citation_id)
        citations.append(
            AnswerCitation(
                chunk_id=result.chunk_id,
                document_id=result.document_id,
                source=result.source,
            )
        )

    is_grounded = bool(results) and bool(citations) and not has_invalid_citation
    return citations, is_grounded