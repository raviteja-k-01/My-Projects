"""Normalize explicit Groq citation notation without adding inferred citations.

Only an explicit "Evidence N" reference can become a citation. This is a
formatting repair, not factual grounding. CitationBuilder still validates
every reference against the retrieved evidence packet and rejects unknown IDs.
"""
from __future__ import annotations

import re

# Matches bracket pairs containing a single-line fragment. We also preserve
# *invalid* bracket groups unchanged, so a bare-reference repair cannot
# silently fix only part of a malformed group.
_BRACKETS = re.compile(
    r"\[([^\[\]\n]{1,160})\]"
    r"|\(([^()\n]{1,160})\)"
    r"|【([^【】\n]{1,160})】"
)

# Explicit evidence groups, allowing harmless punctuation and abbreviated
# numeric lists: "[Evidence 1, 2]", "(Evidence 1 & Evidence 2)".
# Ranges and arbitrary unlabelled numbers are deliberately not accepted.
_GROUP = re.compile(
    r"""
    \s*Evidence\s+\#?\s*\d+
    (?:\s*(?:,|;|&|\band\b)\s*(?:Evidence\s+\#?\s*)?\d+)*
    \s*
    """,
    re.IGNORECASE | re.VERBOSE,
)

# Explicit unbracketed references are made canonical: "Evidence 2" ->
# "[Evidence 2]". Do not reinterpret ranges like "Evidence 1-3".
_BARE = re.compile(
    r"(?<![A-Za-z0-9])Evidence\s+\#?\s*(\d+)(?!\d)(?!\s*[-–—]\s*\d)",
    re.IGNORECASE,
)


def normalize_explicit_citations(answer: str) -> str:
    """Repair notation ONLY; preserve all original cited evidence IDs.

    Do not repair range citations ("Evidence 1–3"), bare "[1]" references,
    unrelated square brackets or unknown evidence IDs. The downstream
    CitationBuilder rejects unknown IDs and remaining malformed references.
    """
    parts: list[str] = []
    position = 0
    for match in _BRACKETS.finditer(answer):
        before = answer[position:match.start()]
        parts.append(_BARE.sub(lambda m: f"[Evidence {int(m.group(1))}]", before))

        inner = next(group for group in match.groups() if group is not None)
        if _GROUP.fullmatch(inner):
            ids = re.findall(r"\d+", inner)
            parts.append("[" + ", ".join(f"Evidence {int(i)}" for i in ids) + "]")
        else:
            parts.append(match.group())
        position = match.end()

    parts.append(_BARE.sub(lambda m: f"[Evidence {int(m.group(1))}]", answer[position:]))
    return "".join(parts)
