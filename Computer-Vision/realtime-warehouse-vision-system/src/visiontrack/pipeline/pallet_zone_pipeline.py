from __future__ import annotations

from dataclasses import dataclass

from visiontrack.config.model_registry import ModelProfile
from visiontrack.perception.pipeline import PerceptionPipeline
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
    ):
        self.perception = PerceptionPipeline(model_profile, app_config)
        self.engine = PalletZoneEngine(app_config)

    def process_frame(self, frame, frame_number: int) -> FrameResult:
        observations = self.perception.process_frame(frame, frame_number)
        events = self.engine.update(observations)

        return FrameResult(
            frame_number=frame_number,
            observations=observations,
            display_observations=self.perception.last_display_observations,
            qrs=self.perception.last_qrs,
            events=events,
        )
