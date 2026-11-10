from __future__ import annotations


def canonical_class_name(class_name: str, aliases: dict[str, str] | None = None) -> str:
    value = class_name.strip().lower().replace("-", "_").replace(" ", "_")
    mapping = aliases or {}
    return mapping.get(value, value)
