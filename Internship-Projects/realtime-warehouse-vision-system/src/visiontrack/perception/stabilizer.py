from __future__ import annotations

from dataclasses import dataclass, replace

from visiontrack.geometry.polygon import bbox_center
from visiontrack.perception.raw_observation import RawObservation


@dataclass
class _TrackRecord:
    observation: RawObservation
    seen_frames: int
    last_frame: int


class DetectionStabilizer:
    def __init__(
        self,
        enabled: bool = True,
        min_confirmed_frames: int = 2,
        stale_frames: int = 8,
        smoothing_alpha: float = 0.65,
    ):
        self.enabled = enabled
        self.min_confirmed_frames = max(1, min_confirmed_frames)
        self.stale_frames = max(1, stale_frames)
        self.smoothing_alpha = min(max(smoothing_alpha, 0.0), 1.0)
        self.records: dict[str, _TrackRecord] = {}

    def update(self, observations: list[RawObservation], frame_number: int) -> list[RawObservation]:
        if not self.enabled:
            return observations

        active_keys: set[str] = set()
        for observation in observations:
            key = self._key(observation)
            active_keys.add(key)
            existing = self.records.get(key)
            if existing is None:
                self.records[key] = _TrackRecord(
                    observation=observation,
                    seen_frames=1,
                    last_frame=frame_number,
                )
                continue

            smoothed = replace(
                observation,
                bbox=(
                    self._smooth_bbox(existing.observation.bbox, observation.bbox)
                    if self._should_smooth(existing.observation.bbox, observation.bbox)
                    else observation.bbox
                ),
                qr_value=observation.qr_value or existing.observation.qr_value,
                zone_code=observation.zone_code or existing.observation.zone_code,
                qr_read_success=(
                    observation.qr_read_success
                    or existing.observation.qr_read_success
                ),
            )
            existing.observation = smoothed
            existing.seen_frames += 1
            existing.last_frame = frame_number

        for key in list(self.records):
            if key not in active_keys and frame_number - self.records[key].last_frame > self.stale_frames:
                del self.records[key]

        stable = []
        for key in active_keys:
            record = self.records.get(key)
            if record is None or record.last_frame != frame_number:
                continue
            if (
                record.seen_frames >= self.min_confirmed_frames
                or record.observation.qr_value
                or record.observation.class_name == "conveyor_belt"
            ):
                stable.append(record.observation)

        stable = self._dedupe_observations(stable)
        stable.sort(key=lambda item: (item.class_name, item.track_id or 0, item.bbox))
        return stable

    def _smooth_bbox(
        self,
        previous: tuple[int, int, int, int],
        current: tuple[int, int, int, int],
    ) -> tuple[int, int, int, int]:
        alpha = self.smoothing_alpha
        return tuple(
            int(round(alpha * current_value + (1.0 - alpha) * previous_value))
            for previous_value, current_value in zip(previous, current)
        )

    @staticmethod
    def _should_smooth(
        previous: tuple[int, int, int, int],
        current: tuple[int, int, int, int],
    ) -> bool:
        previous_center = bbox_center(previous)
        current_center = bbox_center(current)
        distance = (
            (previous_center[0] - current_center[0]) ** 2
            + (previous_center[1] - current_center[1]) ** 2
        ) ** 0.5
        max_side = max(
            previous[2] - previous[0],
            previous[3] - previous[1],
            current[2] - current[0],
            current[3] - current[1],
            1,
        )
        return distance <= max(180, max_side * 1.35)

    @staticmethod
    def _key(observation: RawObservation) -> str:
        if observation.qr_value and observation.class_name in {"box", "pallet", "qr_code"}:
            return f"{observation.class_name}:qr:{observation.qr_value}"
        if observation.track_id is not None:
            return f"{observation.class_name}:track:{observation.track_id}"
        if observation.class_name in {"box", "pallet", "conveyor_belt"}:
            return DetectionStabilizer._grid_key(observation)
        if observation.qr_value:
            return f"{observation.class_name}:qr:{observation.qr_value}"
        return DetectionStabilizer._grid_key(observation)

    @staticmethod
    def _grid_key(observation: RawObservation) -> str:
        center_x, center_y = bbox_center(observation.bbox)
        return (
            f"{observation.class_name}:grid:"
            f"{int(center_x // 48)}:{int(center_y // 48)}"
        )

    @staticmethod
    def _dedupe_observations(observations: list[RawObservation]) -> list[RawObservation]:
        best_by_key: dict[str, RawObservation] = {}
        for observation in observations:
            key = DetectionStabilizer._key(observation)
            current = best_by_key.get(key)
            if current is None or observation.confidence > current.confidence:
                best_by_key[key] = observation
        return list(best_by_key.values())
