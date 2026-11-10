from __future__ import annotations

from dataclasses import dataclass, replace
import math
from typing import Any

from visiontrack.config.model_registry import ModelProfile
from visiontrack.geometry.polygon import bbox_center, point_in_polygon
from visiontrack.identity import is_package_identity
from visiontrack.perception.detector import Detector
from visiontrack.perception.qr_decoder import QRDecoder
from visiontrack.perception.raw_observation import RawObservation
from visiontrack.perception.stabilizer import DetectionStabilizer


class PerceptionPipeline:
    def __init__(
        self,
        model_profile: ModelProfile,
        app_config: dict | None = None,
        detector: Detector | None = None,
    ):
        self.model_profile = model_profile
        self.detector = detector if detector is not None else Detector(model_profile)
        perception_config = (app_config or {}).get("perception", {})
        self.guard_qr_identity = perception_config.get("guard_qr_identity", False)
        self.identity_conflicts: set[str] = set()
        self.track_continuity = TrackContinuity()
        qr_config = dict(model_profile.qr)
        if self.guard_qr_identity:
            qr_config["preserve_duplicate_reads"] = True
        self.qr_decoder = QRDecoder(qr_config)
        self.last_qrs: list[dict] = []
        self.last_display_observations: list[RawObservation] = []
        self.synthesize_missing_boxes_from_qr = bool(
            perception_config.get("synthesize_missing_boxes_from_qr", False)
        )
        stability = (app_config or {}).get("detection_stability", {})
        stale_frames = int(stability.get("stale_frames", 8))
        smoothing_alpha = float(stability.get("smoothing_alpha", 0.65))
        qr_cache_stale_frames = int(
            model_profile.qr.get("track_cache_stale_frames", max(30, stale_frames * 4))
        )
        self.stabilizer = DetectionStabilizer(
            enabled=bool(stability.get("enabled", True)),
            min_confirmed_frames=int(stability.get("min_confirmed_frames", 2)),
            stale_frames=stale_frames,
            smoothing_alpha=smoothing_alpha,
        )
        self.display_stabilizer = DetectionStabilizer(
            enabled=bool(stability.get("enabled", True)),
            min_confirmed_frames=int(stability.get("min_confirmed_frames", 2)),
            stale_frames=stale_frames,
            smoothing_alpha=smoothing_alpha,
        )
        self.qr_identity_cache = QRTrackIdentityCache(
            stale_frames=qr_cache_stale_frames,
            recheck=self.guard_qr_identity,
        )
        self.qr_stabilizer = QRStabilizer(
            stale_frames=max(3, min(6, stale_frames)),
        )

    def process_frame(self, frame, frame_number: int) -> list[RawObservation]:
        self.identity_conflicts.clear()
        observations = self.detector.track_objects(frame)
        if self.guard_qr_identity:
            observations = self.track_continuity.update(observations, frame_number)
        qr_observations = [observation for observation in observations if observation.class_name == "qr_code"]
        cached_qrs, unresolved_qr_observations = self.qr_identity_cache.partition(
            qr_observations,
            frame_number,
        )
        decoded_qrs = self.qr_decoder.decode_frame(
            frame,
            [observation.bbox for observation in unresolved_qr_observations],
        )
        decoded_qrs = self._attach_qr_tracks(decoded_qrs, unresolved_qr_observations)
        self.qr_identity_cache.remember(decoded_qrs, frame_number)
        decoded_qrs = [*cached_qrs, *decoded_qrs]
        if self.guard_qr_identity:
            self.identity_conflicts.update(conflicting_identities(
                (qr["value"], qr["bbox"]) for qr in decoded_qrs
            ))
            # The cache already covers brief decode failures on continuous tracks.
            # Avoid transferring an old read to a different track by proximity.
            qr_values = [qr for qr in decoded_qrs if qr["value"] not in self.identity_conflicts]
        else:
            qr_values = self.qr_stabilizer.update(decoded_qrs, qr_observations, frame_number)
        self.last_qrs = qr_values

        enriched = []
        for observation in observations:
            matching_qr = self._matching_qr_item(
                observation.bbox,
                qr_values,
                class_name=observation.class_name,
            )
            qr_value = observation.qr_value or (
                str(matching_qr["value"])
                if matching_qr is not None
                else None
            )
            enriched.append(
                replace(
                    observation,
                    qr_value=qr_value,
                    zone_code=None,
                    qr_read_success=(
                        observation.qr_read_success
                        or matching_qr is not None
                    ),
                )
            )

        # Preserve model-native geometry exclusively for drawing. QR-derived
        # box geometry below is an internal association aid and must never be
        # presented as a model bounding box.
        self.last_display_observations = self.display_stabilizer.update(
            enriched,
            frame_number,
        )

        if self.synthesize_missing_boxes_from_qr:
            enriched = self._add_missing_boxes_from_package_qrs(
                frame,
                enriched,
                qr_values,
            )

        if self.guard_qr_identity:
            for kind in ("box", "pallet"):
                self.identity_conflicts.update(conflicting_identities(
                    (item.qr_value, item.bbox) for item in enriched
                    if item.class_name == kind and item.qr_value
                ))
            enriched = [
                item for item in enriched
                if item.qr_value not in self.identity_conflicts
                and not (
                    item.class_name in {"box", "pallet"}
                    and any(
                        qr["value"] in self.identity_conflicts
                        and point_in_polygon(bbox_center(qr["bbox"]), bbox_polygon(item.bbox))
                        for qr in decoded_qrs
                    )
                )
            ]
        return self.stabilizer.update(enriched, frame_number)

    def _add_missing_boxes_from_package_qrs(
        self,
        frame,
        observations: list[RawObservation],
        qrs: list[dict],
    ) -> list[RawObservation]:
        package_qrs: list[dict] = []
        seen_values: set[str] = set()

        for qr in qrs:
            if not is_package_qr(str(qr["value"])):
                continue
            try:
                confidence = float(qr.get("confidence"))
            except (TypeError, ValueError):
                continue
            if not math.isfinite(confidence) or not 0.0 <= confidence <= 1.0:
                continue
            value_key = str(qr["value"]).strip().upper()
            if value_key in seen_values:
                continue
            seen_values.add(value_key)
            package_qr = dict(qr)
            package_qr["confidence"] = confidence
            package_qrs.append(package_qr)

        # A model box and a QR-derived box must never compete for the same
        # canonical identity. Replace any QR-labelled model observation with
        # exactly one observation anchored to that QR's current geometry.
        kept = [
            observation
            for observation in observations
            if not (
                observation.class_name == "box"
                and str(observation.qr_value or "").strip().upper()
                in seen_values
            )
        ]

        for qr in package_qrs:
            bbox = estimate_carton_bbox_from_qr(qr["bbox"], frame.shape)
            kept.append(
                RawObservation(
                    class_name="box",
                    display_class_name="box_from_qr",
                    confidence=qr["confidence"],
                    bbox=bbox,
                    # QR detections and physical box detections use different
                    # tracker lifecycles. Canonical QR identity is sufficient
                    # here and avoids colliding with a reused box track ID.
                    track_id=None,
                    qr_value=str(qr["value"]),
                    zone_code=None,
                    qr_read_success=True,
                )
            )

        return kept

    @staticmethod
    def _matching_qr_item(
        bbox: tuple[int, int, int, int],
        qrs: list[dict],
        class_name: str | None = None,
    ) -> dict | None:
        polygon = [(bbox[0], bbox[1]), (bbox[2], bbox[1]), (bbox[2], bbox[3]), (bbox[0], bbox[3])]
        best = None
        best_score = 0.0
        for qr in qrs:
            value = str(qr["value"])
            upper_value = value.upper()

            if class_name == "box" and not is_package_qr(value):
                continue
            if class_name == "pallet" and not upper_value.startswith(
                ("PALLET_", "P_")
            ):
                continue
            if class_name == "dock" and not upper_value.startswith("DOCK_"):
                continue
            if class_name == "zone" and not upper_value.startswith("ZONE_"):
                continue
            if class_name == "conveyor_belt" and not upper_value.startswith(
                ("CONVEYOR_", "BELT_")
            ):
                continue
            score = bbox_overlap_ratio(qr["bbox"], bbox)
            if point_in_polygon(bbox_center(qr["bbox"]), polygon):
                score = max(score, 1.0)
            if score > best_score:
                best = qr
                best_score = score
        return best if best_score > 0.20 else None

    @staticmethod
    def _attach_qr_tracks(qrs: list[dict], observations: list[RawObservation]) -> list[dict]:
        enriched = []
        for qr in qrs:
            item = dict(qr)
            match = max(
                observations,
                key=lambda observation: bbox_overlap_ratio(qr["bbox"], observation.bbox),
                default=None,
            )
            if match is not None and bbox_overlap_ratio(qr["bbox"], match.bbox) > 0.15:
                item["track_id"] = match.track_id
                item["confidence"] = float(match.confidence)
            enriched.append(item)
        return enriched


@dataclass
class _QRIdentityRecord:
    value: str
    last_seen_frame: int
    decoded_frame: int
    bbox: tuple[int, int, int, int]


class TrackContinuity:
    """Give a reused tracker ID a fresh run-local identity after a long gap."""

    def __init__(self, max_gap=30):
        self.max_gap = max_gap
        self.tracks = {}
        self.next_id = 0

    def update(self, observations, frame_number):
        for key, (_, last_seen, _) in list(self.tracks.items()):
            if frame_number - last_seen > self.max_gap:
                del self.tracks[key]
        current = []
        for item in observations:
            if item.track_id is None:
                current.append(item)
                continue
            key = (item.class_name, item.track_id)
            previous = self.tracks.get(key)
            if previous is None or (
                bbox_overlap_ratio(previous[2], item.bbox) == 0
                and center_distance_score(previous[2], item.bbox) == 0
            ):
                self.next_id += 1
                identity = self.next_id
            else:
                identity = previous[0]
            self.tracks[key] = (identity, frame_number, item.bbox)
            current.append(replace(item, track_id=identity))
        return current


class QRTrackIdentityCache:
    def __init__(self, stale_frames: int = 30, recheck: bool = False):
        self.stale_frames = max(1, stale_frames)
        self.recheck = recheck
        self.records: dict[int, _QRIdentityRecord] = {}

    def partition(
        self,
        observations: list[RawObservation],
        frame_number: int,
    ) -> tuple[list[dict[str, Any]], list[RawObservation]]:
        self._expire(frame_number)
        cached: list[dict[str, Any]] = []
        unresolved: list[RawObservation] = []

        for observation in observations:
            track_id = observation.track_id
            record = self.records.get(track_id) if track_id is not None else None
            if record is None or (
                self.recheck
                and (
                    frame_number - record.last_seen_frame > 2
                    or frame_number - record.decoded_frame >= self.stale_frames
                    or bbox_overlap_ratio(record.bbox, observation.bbox) < 0.15
                )
            ):
                unresolved.append(observation)
                continue

            record.last_seen_frame = frame_number
            record.bbox = observation.bbox
            cached.append(
                {
                    "value": record.value,
                    "bbox": observation.bbox,
                    "track_id": track_id,
                    "confidence": float(observation.confidence),
                }
            )

        return cached, unresolved

    def remember(self, qrs: list[dict[str, Any]], frame_number: int) -> None:
        for qr in qrs:
            track_id = qr.get("track_id")
            value = qr.get("value")
            if track_id is None or not value:
                continue
            self.records[int(track_id)] = _QRIdentityRecord(
                value=str(value),
                last_seen_frame=frame_number,
                decoded_frame=frame_number,
                bbox=qr["bbox"],
            )

    def _expire(self, frame_number: int) -> None:
        for track_id, record in list(self.records.items()):
            if frame_number - record.last_seen_frame > self.stale_frames:
                del self.records[track_id]


@dataclass
class _QRRecord:
    item: dict[str, Any]
    seen_frames: int
    last_frame: int


class QRStabilizer:
    def __init__(self, stale_frames: int = 16):
        self.stale_frames = max(1, stale_frames)
        self.records: dict[str, _QRRecord] = {}

    def update(
        self,
        qrs: list[dict[str, Any]],
        qr_observations: list[RawObservation],
        frame_number: int,
    ) -> list[dict[str, Any]]:
        active_values: set[str] = set()
        used_observations: set[int] = set()

        for qr in qrs:
            value = qr["value"]
            active_values.add(value)
            existing = self.records.get(value)
            if existing is None:
                self.records[value] = _QRRecord(item=dict(qr), seen_frames=1, last_frame=frame_number)
            else:
                existing.item = dict(qr)
                existing.seen_frames += 1
                existing.last_frame = frame_number
            matched_index = self._matching_observation_index(qr, qr_observations, used_observations)
            if matched_index is not None:
                used_observations.add(matched_index)

        for value, record in list(self.records.items()):
            if value in active_values:
                continue
            matched_index = self._matching_observation_index(
                record.item,
                qr_observations,
                used_observations,
            )
            if matched_index is None:
                continue

            observation = qr_observations[matched_index]
            used_observations.add(matched_index)
            current = dict(record.item)
            current["bbox"] = observation.bbox
            current["track_id"] = observation.track_id
            current["confidence"] = float(observation.confidence)
            current.pop("points", None)
            record.item = current
            record.last_frame = frame_number

        for value in list(self.records):
            if frame_number - self.records[value].last_frame > self.stale_frames:
                del self.records[value]

        return [
            dict(record.item)
            for record in sorted(
                (record for record in self.records.values() if record.last_frame == frame_number),
                key=lambda record: (record.item["bbox"][1], record.item["bbox"][0], record.item["value"]),
            )
        ]

    @staticmethod
    def _matching_observation_index(
        qr: dict[str, Any],
        observations: list[RawObservation],
        used_indexes: set[int],
    ) -> int | None:
        track_id = qr.get("track_id")
        if track_id is not None:
            for index, observation in enumerate(observations):
                if index not in used_indexes and observation.track_id == track_id:
                    return index

        best_index = None
        best_score = 0.0
        for index, observation in enumerate(observations):
            if index in used_indexes:
                continue
            overlap = bbox_overlap_ratio(qr["bbox"], observation.bbox)
            distance_score = center_distance_score(qr["bbox"], observation.bbox)
            score = max(overlap, distance_score)
            if score > best_score:
                best_score = score
                best_index = index

        return best_index if best_score > 0.22 else None


def conflicting_identities(claims):
    locations = {}
    conflicts = set()
    for code, bbox in claims:
        previous = locations.setdefault(code, [])
        if any(bbox_overlap_ratio(bbox, other) < 0.15 for other in previous):
            conflicts.add(code)
        previous.append(bbox)
    return conflicts


def bbox_polygon(bbox):
    x1, y1, x2, y2 = bbox
    return [(x1, y1), (x2, y1), (x2, y2), (x1, y2)]


def bbox_overlap_ratio(a: tuple[int, int, int, int], b: tuple[int, int, int, int]) -> float:
    left = max(a[0], b[0])
    top = max(a[1], b[1])
    right = min(a[2], b[2])
    bottom = min(a[3], b[3])
    intersection = max(0, right - left) * max(0, bottom - top)
    area_a = max(1, (a[2] - a[0]) * (a[3] - a[1]))
    area_b = max(1, (b[2] - b[0]) * (b[3] - b[1]))
    return intersection / min(area_a, area_b)


def center_distance_score(a: tuple[int, int, int, int], b: tuple[int, int, int, int]) -> float:
    ax, ay = bbox_center(a)
    bx, by = bbox_center(b)
    distance = ((ax - bx) ** 2 + (ay - by) ** 2) ** 0.5
    max_side = max(a[2] - a[0], a[3] - a[1], b[2] - b[0], b[3] - b[1], 1)
    if distance > max(120, max_side * 1.5):
        return 0.0
    return max(0.0, 1.0 - distance / max(120, max_side * 1.5))


def is_package_qr(value: str) -> bool:
    return is_package_identity(value)


def estimate_carton_bbox_from_qr(
    qr_bbox: tuple[int, int, int, int],
    frame_shape,
) -> tuple[int, int, int, int]:
    return expand_carton_bbox_from_qr(qr_bbox, frame_shape)


def expand_carton_bbox_from_qr(
    qr_bbox: tuple[int, int, int, int],
    frame_shape,
) -> tuple[int, int, int, int]:
    frame_height, frame_width = frame_shape[:2]
    qx1, qy1, qx2, qy2 = clip_bbox(qr_bbox, frame_width, frame_height)
    qr_width = max(1, qx2 - qx1)
    qr_height = max(1, qy2 - qy1)
    target_width = max(int(qr_width * 1.55), qr_width + 36, 70)
    target_height = max(int(qr_height * 2.05), qr_height + 52, 95)
    center_x = (qx1 + qx2) // 2
    x1 = center_x - target_width // 2
    y1 = qy1 - int(qr_height * 0.28)
    return clip_bbox((x1, y1, x1 + target_width, y1 + target_height), frame_width, frame_height)


def clip_bbox(
    bbox: tuple[int, int, int, int],
    frame_width: int,
    frame_height: int,
) -> tuple[int, int, int, int]:
    x1, y1, x2, y2 = bbox
    x1 = max(0, min(frame_width - 1, int(x1)))
    y1 = max(0, min(frame_height - 1, int(y1)))
    x2 = max(x1 + 1, min(frame_width, int(x2)))
    y2 = max(y1 + 1, min(frame_height, int(y2)))
    return x1, y1, x2, y2
