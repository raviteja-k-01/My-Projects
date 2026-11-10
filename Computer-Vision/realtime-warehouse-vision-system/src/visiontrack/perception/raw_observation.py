from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

@dataclass(frozen=True)
class RawObservation:
    class_name: str
    display_class_name: str
    confidence: float
    bbox: tuple[int, int, int, int]
    track_id: Optional[int] = None
    qr_value: Optional[str] = None
    zone_code: Optional[str] = None
    qr_read_success: bool = False

    @property
    def identity_hint(self) -> str:
        if self.qr_value:
            return self.qr_value
        if self.track_id is not None:
            return f"{self.class_name}_track_{self.track_id}"
        x1, y1, x2, y2 = self.bbox
        return f"{self.class_name}_{x1}_{y1}_{x2}_{y2}"
