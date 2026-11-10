from __future__ import annotations

from dataclasses import dataclass, field
from datetime import UTC, datetime
from enum import Enum
from typing import Optional


class EventType(str, Enum):
    CONVEYOR_ENTRY = "CONVEYOR_ENTRY"
    ZONE_TRANSFER = "ZONE_TRANSFER"
    DOCK_ARRIVAL = "DOCK_ARRIVAL"
    ADD = "ADD"
    MOVE = "MOVE"
    SWAP = "SWAP"
    STACK = "STACK"
    REMOVE = "REMOVE"
    RE_ADD = "RE_ADD"
    DISPATCH = "DISPATCH"


@dataclass(frozen=True)
class WarehouseEvent:
    event_type: EventType
    box_code: str
    from_pallet_code: Optional[str] = None
    to_pallet_code: Optional[str] = None
    from_zone_code: Optional[str] = None
    to_zone_code: Optional[str] = None
    remarks: str = ""
    timestamp: datetime = field(
        default_factory=lambda: datetime.now(UTC).replace(tzinfo=None)
    )
    slot_no: Optional[int] = None
    row_no: Optional[int] = None
    column_no: Optional[int] = None
    layer_no: Optional[int] = None
    # Probability in the inclusive 0..1 range, derived from the real detection
    # evidence supporting the confirmed location change. The repository alone
    # converts this to the database's 0..100 percentage representation.
    confidence_score: Optional[float] = None
    # True only when the box identity originated from a successfully decoded
    # QR value. None is reserved for callers that do not supply provenance.
    qr_read_success: Optional[bool] = None
