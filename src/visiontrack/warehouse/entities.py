from __future__ import annotations

from dataclasses import dataclass
from typing import Optional


@dataclass
class BoxState:
    box_code: str
    current_pallet_code: Optional[str] = None
    current_zone_code: Optional[str] = None
    status: str = "DETECTED_UNLOCATED"
    missing_frames: int = 0
    slot_no: Optional[int] = None
    row_no: Optional[int] = None
    column_no: Optional[int] = None
    layer_no: Optional[int] = None
    current_location_confidence_score: Optional[float] = None
    identity_from_qr: bool = False


@dataclass
class PalletState:
    pallet_code: str
    current_zone_code: Optional[str] = None
    status: str = "EMPTY"
    visible: bool = True
    max_boxes: int = 18
    current_location_confidence_score: Optional[float] = None
