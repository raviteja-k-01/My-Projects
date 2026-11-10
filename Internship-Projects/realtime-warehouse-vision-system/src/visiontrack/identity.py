from __future__ import annotations

import re


_PALLET_CODE = re.compile(r"(?:PALLET|PLT|PAL|P)_?0*(\d+)$")
_BOX_CODE = re.compile(r"(?:BOX|CARTON)_?0*(\d+)$")
_PACKAGE_BOX_CODE = re.compile(r"(?:PKG|PACKAGE)_?[A-Z]*?0*(\d+)$")
_ZONE_CODE = re.compile(r"ZONE_?0*(\d+)$")
_DOCK_CODE = re.compile(r"DOCK_?0*(\d+)$")
_CONVEYOR_CODE = re.compile(r"(?:CONVEYOR|BELT)_?0*(\d+)$")


def canonicalize_identity(value: str | None) -> str | None:
    """Convert readable QR identities to the database's canonical codes."""
    if value is None:
        return None

    normalized = value.strip().upper().replace("-", "_").replace(" ", "_")
    normalized = re.sub(r"_+", "_", normalized)
    if not normalized:
        return None

    for pattern, prefix, width in (
        (_PALLET_CODE, "PALLET", 2),
        (_BOX_CODE, "BOX", 3),
        (_PACKAGE_BOX_CODE, "BOX", 3),
        (_ZONE_CODE, "ZONE", 2),
        (_DOCK_CODE, "DOCK", 2),
        (_CONVEYOR_CODE, "CONVEYOR", 2),
    ):
        match = pattern.fullmatch(normalized)
        if match:
            return f"{prefix}_{int(match.group(1)):0{width}d}"

    return {
        "FLOOR": "floor",
        "DOCKING_STATION": "docking_station",
        "CONVEYOR_BELT": "conveyor_belt",
        "STOCKING_AREA": "stocking_area",
    }.get(normalized, normalized)


def is_package_identity(value: str | None) -> bool:
    """Return true only for box/package QR values, never region QR values."""
    if value is None:
        return False

    normalized = value.strip().upper().replace("-", "_").replace(" ", "_")
    normalized = re.sub(r"_+", "_", normalized)
    return bool(
        _BOX_CODE.fullmatch(normalized)
        or _PACKAGE_BOX_CODE.fullmatch(normalized)
    )
