from __future__ import annotations

import unittest

import numpy as np

from visiontrack.perception.pipeline import (
    PerceptionPipeline,
    QRStabilizer,
    QRTrackIdentityCache,
)
from visiontrack.perception.raw_observation import RawObservation
from visiontrack.pipeline.pallet_zone_pipeline import PalletZonePipeline


def box_observation(code: str) -> RawObservation:
    return RawObservation(
        class_name="box",
        display_class_name="box",
        confidence=0.90,
        bbox=(10, 10, 60, 60),
        track_id=40,
        qr_value=code,
        qr_read_success=True,
    )


class QRBoxSynthesisTests(unittest.TestCase):
    def setUp(self) -> None:
        self.pipeline = object.__new__(PerceptionPipeline)
        self.frame = np.zeros((100, 100, 3), dtype=np.uint8)

    def test_different_overlapping_qr_creates_new_stacked_box(self) -> None:
        existing = box_observation("BOX_468")
        qrs = [
            {
                "value": "BOX_468",
                "bbox": (18, 18, 32, 32),
                "confidence": 0.81,
            },
            {
                "value": "BOX_500",
                "bbox": (36, 18, 50, 32),
                "confidence": 0.76,
            },
        ]

        boxes = self.pipeline._add_missing_boxes_from_package_qrs(
            self.frame,
            [existing],
            qrs,
        )

        self.assertEqual([box.qr_value for box in boxes], ["BOX_468", "BOX_500"])
        self.assertTrue(
            all(box.display_class_name == "box_from_qr" for box in boxes)
        )
        self.assertTrue(all(box.track_id is None for box in boxes))
        self.assertIsNone(boxes[1].track_id)
        self.assertEqual(boxes[1].display_class_name, "box_from_qr")
        self.assertEqual(boxes[1].confidence, 0.76)
        self.assertTrue(boxes[1].qr_read_success)

    def test_same_overlapping_qr_does_not_duplicate_box(self) -> None:
        existing = box_observation("BOX_468")
        boxes = self.pipeline._add_missing_boxes_from_package_qrs(
            self.frame,
            [existing],
            [
                {
                    "value": "BOX_468",
                    "bbox": (18, 18, 32, 32),
                    "confidence": 0.81,
                }
            ],
        )

        self.assertEqual(len(boxes), 1)
        self.assertEqual(boxes[0].qr_value, "BOX_468")
        self.assertEqual(boxes[0].display_class_name, "box_from_qr")
        self.assertIsNone(boxes[0].track_id)

    def test_non_package_qr_never_creates_box(self) -> None:
        boxes = self.pipeline._add_missing_boxes_from_package_qrs(
            self.frame,
            [],
            [
                {
                    "value": "PALLET_04",
                    "bbox": (18, 18, 32, 32),
                    "confidence": 0.81,
                }
            ],
        )

        self.assertEqual(boxes, [])

    def test_missing_detector_confidence_is_not_fabricated(self) -> None:
        existing = box_observation("BOX_468")
        boxes = self.pipeline._add_missing_boxes_from_package_qrs(
            self.frame,
            [existing],
            [{"value": "BOX_468", "bbox": (18, 18, 32, 32)}],
        )

        self.assertEqual(boxes, [existing])

    def test_qr_detector_confidence_survives_decode_cache_and_stability(self) -> None:
        observation = RawObservation(
            class_name="qr_code",
            display_class_name="qr_code",
            confidence=0.87654,
            bbox=(18, 18, 32, 32),
            track_id=9,
        )
        attached = PerceptionPipeline._attach_qr_tracks(
            [{"value": "BOX_468", "bbox": observation.bbox}],
            [observation],
        )

        self.assertEqual(attached[0]["confidence"], 0.87654)

        cache = QRTrackIdentityCache(stale_frames=5)
        cache.remember(attached, frame_number=1)
        current = RawObservation(
            class_name="qr_code",
            display_class_name="qr_code",
            confidence=0.76543,
            bbox=(19, 19, 33, 33),
            track_id=9,
        )
        cached, unresolved = cache.partition([current], frame_number=2)

        self.assertEqual(unresolved, [])
        self.assertEqual(cached[0]["confidence"], 0.76543)

        stabilizer = QRStabilizer(stale_frames=5)
        first = stabilizer.update(attached, [observation], frame_number=1)
        refreshed = stabilizer.update([], [current], frame_number=2)

        self.assertEqual(first[0]["confidence"], 0.87654)
        self.assertEqual(refreshed[0]["confidence"], 0.76543)


class _FakePerception:
    def __init__(
        self,
        logical: list[RawObservation],
        display: list[RawObservation],
    ) -> None:
        self.logical = logical
        self.last_display_observations = display
        self.last_qrs: list[dict] = []

    def process_frame(self, frame, frame_number: int) -> list[RawObservation]:
        return self.logical


class _FakeEngine:
    def __init__(self) -> None:
        self.received: list[RawObservation] = []

    def update(self, observations: list[RawObservation]) -> list:
        self.received = observations
        return []


class DisplayObservationSeparationTests(unittest.TestCase):
    def test_synthetic_logic_geometry_is_not_returned_for_drawing(self) -> None:
        model_box = box_observation("BOX_468")
        synthetic_box = RawObservation(
            class_name="box",
            display_class_name="box_from_qr",
            confidence=0.52,
            bbox=(0, 0, 90, 90),
            track_id=None,
            qr_value="BOX_468",
            qr_read_success=True,
        )
        pipeline = object.__new__(PalletZonePipeline)
        pipeline.perception = _FakePerception([synthetic_box], [model_box])
        pipeline.engine = _FakeEngine()

        result = pipeline.process_frame(None, 1)

        self.assertEqual(result.observations, [synthetic_box])
        self.assertEqual(pipeline.engine.received, [synthetic_box])
        self.assertEqual(result.display_observations, [model_box])
        self.assertNotIn(synthetic_box, result.display_observations)


if __name__ == "__main__":
    unittest.main()
