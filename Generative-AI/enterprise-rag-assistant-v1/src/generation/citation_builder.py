import re
from typing import Any


# Matches complete citation groups such as:
# [Evidence 1]
# [Evidence 1, Evidence 2]
# [Evidence 1, Evidence 2, Evidence 4]
# [Evidence 1 and Evidence 2]
EVIDENCE_GROUP_PATTERN = re.compile(
    r"\["
    r"Evidence\s+\d+"
    r"(?:\s*(?:,|and)\s*Evidence\s+\d+)*"
    r"\]",
    re.IGNORECASE,
)

# Extracts each individual evidence number
# from a valid citation group.
EVIDENCE_ID_PATTERN = re.compile(
    r"Evidence\s+(\d+)",
    re.IGNORECASE,
)


class CitationBuilder:
    """
    Maps evidence references in the generated answer
    back to the real source metadata produced by the
    Context Builder.

    Supported examples:
        [Evidence 1]
        [Evidence 1, Evidence 2]
        [Evidence 1, Evidence 2, Evidence 4]
    """

    def build(
        self,
        answer: str,
        evidence: list[dict[str, Any]],
    ) -> dict[str, Any]:

        answer = answer.strip()

        if not answer:
            raise ValueError(
                "Answer cannot be empty"
            )

        if not evidence:
            raise ValueError(
                "Evidence cannot be empty"
            )

        # Maps:
        #
        # Evidence 1 -> real chunk metadata
        # Evidence 2 -> real chunk metadata
        #
        # This allows us to convert Gemini's citation
        # references into deterministic source metadata.
        evidence_map = {
            item["evidence_id"]: item
            for item in evidence
        }

        citation_ids: list[int] = []
        seen_ids: set[int] = set()

        # First find complete citation groups.
        #
        # Example:
        # [Evidence 1, Evidence 2, Evidence 4]
        for group_match in EVIDENCE_GROUP_PATTERN.finditer(
            answer
        ):
            group_text = group_match.group(0)

            # Then extract every Evidence N
            # inside that group.
            for evidence_match in EVIDENCE_ID_PATTERN.finditer(
                group_text
            ):
                evidence_id = int(
                    evidence_match.group(1)
                )

                # Gemini must never reference evidence
                # that was not supplied in the context.
                if evidence_id not in evidence_map:
                    raise ValueError(
                        f"Answer cites Evidence "
                        f"{evidence_id}, but that "
                        "evidence does not exist"
                    )

                # Preserve citation order based on the
                # first time each evidence item appears.
                #
                # Repeated citations are included only once
                # in the final citation metadata.
                if evidence_id not in seen_ids:
                    citation_ids.append(
                        evidence_id
                    )
                    seen_ids.add(
                        evidence_id
                    )

        citations: list[dict[str, Any]] = []

        for evidence_id in citation_ids:
            source = evidence_map[
                evidence_id
            ]

            page_start = source.get(
                "page_start"
            )
            page_end = source.get(
                "page_end"
            )

            if (
                page_end is not None
                and page_end != page_start
            ):
                page_label = (
                    f"{page_start}-{page_end}"
                )
            else:
                page_label = str(
                    page_start
                )

            section_number = source.get(
                "section_number"
            )
            section_title = source.get(
                "section_title"
            )
            subsection_number = source.get(
                "subsection_number"
            )
            subsection_title = source.get(
                "subsection_title"
            )

            section_parts: list[str] = []

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

            section_label = (
                " | ".join(section_parts)
                if section_parts
                else "Not specified"
            )

            citation = {
                "evidence_id": evidence_id,
                "chunk_id": source.get(
                    "chunk_id"
                ),
                "document_id": source.get(
                    "document_id"
                ),
                "document_name": source.get(
                    "document_name"
                ),
                "page_start": page_start,
                "page_end": page_end,
                "page_label": page_label,
                "section_number": section_number,
                "section_title": section_title,
                "subsection_number": (
                    subsection_number
                ),
                "subsection_title": (
                    subsection_title
                ),
                "section_label": section_label,
            }

            citations.append(
                citation
            )

        return {
            "answer": answer,
            "citations": citations,
            "citation_count": len(
                citations
            ),
        }