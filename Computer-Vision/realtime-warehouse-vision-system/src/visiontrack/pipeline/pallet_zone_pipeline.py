from __future__ import annotations

from dataclasses import dataclass

from visiontrack.config.model_registry import ModelProfile
from visiontrack.perception.detector import Detector
from visiontrack.perception.pipeline import PerceptionPipeline, conflicting_identities
from visiontrack.perception.raw_observation import RawObservation
from visiontrack.warehouse.events import WarehouseEvent
from visiontrack.warehouse.pallet_zone_engine import PalletZoneEngine


@dataclass(frozen=True)
class FrameResult:
    frame_number: int
    observations: list[RawObservation]
    display_observations: list[RawObservation]
    qrs: list[dict]
    events: list[WarehouseEvent]


class PalletZonePipeline:
    def __init__(
        self,
        app_config: dict,
        model_profile: ModelProfile,
        detector: Detector | None = None,
    ):
        self.perception = PerceptionPipeline(model_profile, app_config, detector)
        self.engine = PalletZoneEngine(app_config)

    def process_frame(self, frame, frame_number: int) -> FrameResult:
        observations = self.perception.process_frame(frame, frame_number)
        if getattr(self.perception, "guard_qr_identity", False):
            conflicts = self.perception.identity_conflicts
            claims = []
            retained = []
            for item in observations:
                code = self.engine.track_identities.get((item.class_name.upper(), item.track_id))
                if item.class_name in {"box", "pallet"}:
                    if code and not self.engine._is_temporary_identity(code):
                        if item.qr_value and item.qr_value != code:
                            conflicts.add(code)
                            conflicts.add(item.qr_value)
                            # Require a clean track or a revalidated original identity.
                            continue
                    identity = item.qr_value or code
                    if identity:
                        claims.append((identity, item.bbox))
                retained.append((item, code))
            conflicts.update(conflicting_identities(claims))
            observations = [
                item for item, code in retained
                if item.qr_value not in conflicts and code not in conflicts
            ]
            events = self.engine.update(observations, blocked_identities=conflicts)
        else:
            events = self.engine.update(observations)

        return FrameResult(
            frame_number=frame_number,
            observations=observations,
            display_observations=self.perception.last_display_observations,
            qrs=self.perception.last_qrs,
            events=events,
        )
