import re
from collections.abc import Iterable


CANONICAL_BULLET = "\u2022"
PDF_SYMBOL_BULLET = "\uf0b7"

_WHITESPACE_PATTERN = re.compile(r"\s+")


def normalize_bullet_characters(text: str) -> str:
    """Map the corpus' private-use Word bullet to a standard bullet."""

    return text.replace(PDF_SYMBOL_BULLET, CANONICAL_BULLET)


def normalize_inline_text(text: str) -> str:
    """Apply formatting-only normalization without rewriting content."""

    text = normalize_bullet_characters(text)
    return _WHITESPACE_PATTERN.sub(" ", text).strip()


def split_bullet_prefix(text: str) -> tuple[bool, str]:
    """Return whether a line starts with a bullet and its remaining text."""

    normalized = normalize_inline_text(text)

    if normalized == CANONICAL_BULLET:
        return True, ""

    if normalized.startswith(CANONICAL_BULLET):
        return True, normalized[len(CANONICAL_BULLET):].strip()

    return False, normalized


def join_wrapped_lines(lines: Iterable[str]) -> str:
    """
    Remove line-wrap boundaries inside one known paragraph or bullet.

    A visible trailing hyphen is retained. Only the line break after it is
    removed, so ``identity-`` + ``level`` becomes ``identity-level`` rather
    than silently changing the source spelling.
    """

    normalized_lines = [
        normalized
        for line in lines
        if (normalized := normalize_inline_text(line))
    ]

    if not normalized_lines:
        return ""

    joined = normalized_lines[0]

    for line in normalized_lines[1:]:
        has_attached_hyphen = (
            len(joined) >= 2
            and joined[-1] in {"-", "\u2010", "\u2011"}
            and not joined[-2].isspace()
        )
        separator = "" if has_attached_hyphen else " "
        joined = f"{joined}{separator}{line}"

    return joined
