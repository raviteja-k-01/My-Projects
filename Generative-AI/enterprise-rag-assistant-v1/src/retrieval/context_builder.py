import re
from typing import Any


DEFAULT_MAX_CHUNKS = 8
DEFAULT_MAX_CONTEXT_TOKENS = 4000
DEFAULT_DUPLICATE_THRESHOLD = 0.78

TOKEN_PATTERN = re.compile(
    r"[A-Za-z0-9]+(?:-[A-Za-z0-9]+)*"
)


class ContextBuilder:
    """
    Builds a clean evidence packet from reranked
    retrieval results for the LLM.
    """

    def __init__(
        self,
        max_chunks: int = DEFAULT_MAX_CHUNKS,
        max_context_tokens: int = DEFAULT_MAX_CONTEXT_TOKENS,
        duplicate_threshold: float = DEFAULT_DUPLICATE_THRESHOLD,
    ) -> None:
        if max_chunks <= 0:
            raise ValueError(
                "max_chunks must be greater than zero"
            )

        if max_context_tokens <= 0:
            raise ValueError(
                "max_context_tokens must be greater than zero"
            )

        if not 0.0 <= duplicate_threshold <= 1.0:
            raise ValueError(
                "duplicate_threshold must be between 0 and 1"
            )

        self.max_chunks = max_chunks
        self.max_context_tokens = max_context_tokens
        self.duplicate_threshold = duplicate_threshold

    @staticmethod
    def _estimate_tokens(text: str) -> int:
        return max(1, len(text) // 4)

    @staticmethod
    def _token_set(text: str) -> set[str]:
        """
        Convert text into a normalized set of words
        for duplicate comparison.
        """

        return {
            token.lower()
            for token in TOKEN_PATTERN.findall(text)
        }

    @classmethod
    def _text_similarity(
        cls,
        text_a: str,
        text_b: str,
    ) -> float:
        """
        Calculate Jaccard similarity between the
        unique normalized words of two chunks.
        """

        tokens_a = cls._token_set(text_a)
        tokens_b = cls._token_set(text_b)

        if not tokens_a or not tokens_b:
            return 0.0

        intersection = tokens_a & tokens_b
        union = tokens_a | tokens_b

        return len(intersection) / len(union)

    def _is_near_duplicate(
        self,
        text: str,
        selected_texts: list[str],
    ) -> bool:
        """
        Check whether this chunk is too similar to
        any evidence chunk already selected.
        """

        for selected_text in selected_texts:
            similarity = self._text_similarity(
                text,
                selected_text,
            )

            if similarity >= self.duplicate_threshold:
                return True

        return False

    @staticmethod
    def _format_section(
        chunk: dict[str, Any],
    ) -> str:
        section_parts = []

        section_number = chunk.get(
            "section_number"
        )
        section_title = chunk.get(
            "section_title"
        )

        subsection_number = chunk.get(
            "subsection_number"
        )
        subsection_title = chunk.get(
            "subsection_title"
        )

        if section_number:
            section_parts.append(
                str(section_number)
            )

        if section_title:
            section_parts.append(
                str(section_title)
            )

        if subsection_number:
            section_parts.append(
                str(subsection_number)
            )

        if subsection_title:
            section_parts.append(
                str(subsection_title)
            )

        if not section_parts:
            return "Not specified"

        return " | ".join(section_parts)

    def build(
        self,
        candidates: list[dict[str, Any]],
    ) -> dict[str, Any]:
        """
        Select strong, non-duplicate evidence within
        the chunk and token budgets.
        """

        if not candidates:
            return {
                "context": "",
                "evidence": [],
                "evidence_count": 0,
                "total_tokens": 0,
            }

        evidence_items = []
        context_blocks = []

        seen_chunk_ids: set[str] = set()
        selected_texts: list[str] = []

        total_tokens = 0

        for candidate in candidates:
            if len(evidence_items) >= self.max_chunks:
                break

            chunk = candidate.get("chunk")

            if not chunk:
                continue

            chunk_id = chunk.get("chunk_id")

            if not chunk_id:
                raise ValueError(
                    "Candidate chunk has no chunk_id"
                )

            if chunk_id in seen_chunk_ids:
                continue

            text = str(
                chunk.get("text", "")
            ).strip()

            if not text:
                continue

            # Skip chunks whose text is highly similar
            # to evidence already selected.
            if self._is_near_duplicate(
                text=text,
                selected_texts=selected_texts,
            ):
                continue

            token_count = chunk.get(
                "token_count"
            )

            if not isinstance(token_count, int):
                token_count = (
                    self._estimate_tokens(text)
                )

            if (
                total_tokens + token_count
                > self.max_context_tokens
            ):
                continue

            evidence_number = (
                len(evidence_items) + 1
            )

            page_start = chunk.get(
                "page_start"
            )
            page_end = chunk.get(
                "page_end"
            )

            evidence = {
                "evidence_id": evidence_number,
                "chunk_id": chunk_id,
                "document_id": chunk.get(
                    "document_id"
                ),
                "document_name": chunk.get(
                    "document_name"
                ),
                "page_start": page_start,
                "page_end": page_end,
                "section_number": chunk.get(
                    "section_number"
                ),
                "section_title": chunk.get(
                    "section_title"
                ),
                "subsection_number": chunk.get(
                    "subsection_number"
                ),
                "subsection_title": chunk.get(
                    "subsection_title"
                ),
                "content_type": chunk.get(
                    "content_type"
                ),
                "text": text,
                "token_count": token_count,
                "rrf_rank": candidate.get(
                    "rank"
                ),
                "reranker_rank": candidate.get(
                    "reranker_rank"
                ),
                "reranker_score": candidate.get(
                    "reranker_score"
                ),
            }

            evidence_items.append(
                evidence
            )

            seen_chunk_ids.add(
                chunk_id
            )

            selected_texts.append(
                text
            )

            total_tokens += token_count

            section_label = (
                self._format_section(chunk)
            )

            context_block = (
                f"[Evidence {evidence_number}]\n"
                f"Chunk ID: {chunk_id}\n"
                f"Document: "
                f"{chunk.get('document_id')}\n"
                f"Page: {page_start}\n"
                f"Section: {section_label}\n"
                f"Text:\n{text}"
            )

            context_blocks.append(
                context_block
            )

        context = "\n\n".join(
            context_blocks
        )

        return {
            "context": context,
            "evidence": evidence_items,
            "evidence_count": len(
                evidence_items
            ),
            "total_tokens": total_tokens,
        }