from __future__ import annotations

import unittest

from visiontrack.perception.raw_observation import RawObservation
from visiontrack.warehouse.events import EventType
from visiontrack.warehouse.pallet_zone_engine import PalletZoneEngine


def observation(
    class_name: str,
    bbox: tuple[int, int, int, int],
    *,
    track_id: int,
    qr_value: str | None,
    confidence: float = 0.99,
) -> RawObservation:
    return RawObservation(
        class_name=class_name,
        display_class_name=class_name,
        confidence=confidence,
        bbox=bbox,
        track_id=track_id,
        qr_value=qr_value,
        qr_read_success=qr_value is not None,
    )


class PendingIdentityEventTests(unittest.TestCase):
    def setUp(self) -> None:
        self.engine = PalletZoneEngine(
            {
                "dynamic_location": {
                    "confirmation_frames": 1,
                    "startup_baseline_frames": 1,
                    "identity_resolution_timeout_frames": 5,
                }
            }
        )
        self.zone = observation(
            "zone",
            (0, 0, 100, 100),
            track_id=4,
            qr_value="ZONE_04",
        )
        self.pallet = observation(
            "pallet",
            (10, 10, 90, 90),
            track_id=4,
            qr_value="PALLET_04",
        )

    def test_temporary_add_waits_for_canonical_box_identity(self) -> None:
        self.engine.update([self.zone, self.pallet])

        temporary_box = observation(
            "box",
            (20, 20, 40, 40),
            track_id=40,
            qr_value=None,
            confidence=0.73,
        )
        self.assertEqual(
            self.engine.update([self.zone, self.pallet, temporary_box]),
            [],
        )
        self.assertEqual(len(self.engine.pending_events), 1)

        resolved_box = observation(
            "box",
            (20, 20, 40, 40),
            track_id=40,
            qr_value="BOX_468",
            confidence=0.91,
        )
        events = self.engine.update(
            [self.zone, self.pallet, resolved_box]
        )

        self.assertEqual(len(events), 1)
        self.assertEqual(events[0].event_type, EventType.STACK)
        self.assertEqual(events[0].box_code, "BOX_468")
        self.assertEqual(events[0].confidence_score, 0.73)
        self.assertTrue(events[0].qr_read_success)
        self.assertNotIn("BOX_TRACK_40", events[0].remarks)
        self.assertEqual(self.engine.pending_events, [])
        self.assertIn("BOX_468", self.engine.boxes)
        self.assertNotIn("BOX_TRACK_40", self.engine.boxes)

    def test_unresolved_temporary_event_expires_without_emission(self) -> None:
        self.engine.update([self.zone, self.pallet])
        temporary_box = observation(
            "box",
            (20, 20, 40, 40),
            track_id=46,
            qr_value=None,
        )
        self.assertEqual(
            self.engine.update([self.zone, self.pallet, temporary_box]),
            [],
        )

        emitted = []
        for _ in range(6):
            emitted.extend(self.engine.update([]))

        self.assertEqual(emitted, [])
        self.assertEqual(self.engine.pending_events, [])
        self.assertEqual(self.engine.dropped_unresolved_event_count, 1)

    def test_yolo_conveyor_source_uses_stable_master_zone(self) -> None:
        conveyor = observation(
            "conveyor_belt",
            (100, 0, 200, 100),
            track_id=5,
            qr_value=None,
        )
        conveyor_box = observation(
            "box",
            (120, 20, 140, 40),
            track_id=50,
            qr_value="BOX_500",
        )
        self.engine.update([self.zone, self.pallet, conveyor, conveyor_box])

        pallet_box = observation(
            "box",
            (20, 20, 40, 40),
            track_id=50,
            qr_value="BOX_500",
            confidence=0.87654,
        )
        events = self.engine.update([self.zone, self.pallet, pallet_box])

        self.assertEqual(len(events), 1)
        self.assertEqual(events[0].event_type, EventType.STACK)
        self.assertEqual(events[0].box_code, "BOX_500")
        self.assertEqual(events[0].from_zone_code, "conveyor_belt")
        self.assertEqual(events[0].to_zone_code, "ZONE_04")
        self.assertEqual(events[0].confidence_score, 0.87654)
        self.assertTrue(events[0].qr_read_success)
        self.assertNotIn("TRACK", events[0].remarks)
        self.assertEqual(
            events[0].remarks,
            "BOX_500: conveyor_belt -> PALLET_04 in ZONE_04",
        )
        self.assertEqual(self.engine.pending_events, [])


class StackedBoxIdentityTests(unittest.TestCase):
    def setUp(self) -> None:
        self.engine = PalletZoneEngine(
            {
                "dynamic_location": {
                    "confirmation_frames": 1,
                    "startup_baseline_frames": 1,
                    "identity_resolution_timeout_frames": 5,
                    "identity_change_confirmation_frames": 2,
                }
            }
        )
        self.zone = observation(
            "zone",
            (0, 0, 100, 100),
            track_id=4,
            qr_value="ZONE_04",
        )
        self.pallet = observation(
            "pallet",
            (10, 10, 90, 90),
            track_id=4,
            qr_value="PALLET_04",
        )
        self.engine.update([self.zone, self.pallet])

    def _stack_box(self, code: str, *, first: bool = False) -> list:
        box = observation(
            "box",
            (20, 20, 40, 40),
            track_id=40,
            qr_value=code,
        )
        events = self.engine.update([self.zone, self.pallet, box])
        if not first:
            events.extend(self.engine.update([self.zone, self.pallet, box]))
        return events

    def test_new_qr_on_reused_track_preserves_boxes_underneath(self) -> None:
        events = []
        events.extend(self._stack_box("BOX_468", first=True))
        events.extend(self._stack_box("BOX_500"))
        events.extend(self._stack_box("BOX_120"))
        events.extend(self._stack_box("BOX_305"))

        stacked_codes = [
            event.box_code
            for event in events
            if event.event_type == EventType.STACK
        ]
        self.assertEqual(
            stacked_codes,
            ["BOX_468", "BOX_500", "BOX_120", "BOX_305"],
        )
        self.assertEqual(
            set(self.engine.boxes),
            {"BOX_468", "BOX_500", "BOX_120", "BOX_305"},
        )
        self.assertTrue(
            all(
                box.current_pallet_code == "PALLET_04"
                for box in self.engine.boxes.values()
            )
        )

        dock = observation(
            "dock",
            (0, 0, 100, 100),
            track_id=2,
            qr_value="DOCK_02",
        )
        top_box = observation(
            "box",
            (20, 20, 40, 40),
            track_id=40,
            qr_value="BOX_305",
        )
        dock_arrival_events = self.engine.update([dock, self.pallet, top_box])

        self.assertEqual(
            {
                event.box_code
                for event in dock_arrival_events
                if event.event_type == EventType.DOCK_ARRIVAL
            },
            {"BOX_468", "BOX_500", "BOX_120", "BOX_305"},
        )
        self.assertTrue(
            all(event.qr_read_success for event in dock_arrival_events)
        )
        self.assertTrue(
            all(event.confidence_score == 0.99 for event in dock_arrival_events)
        )

    def test_single_conflicting_qr_does_not_create_false_box(self) -> None:
        self._stack_box("BOX_468", first=True)

        conflicting = observation(
            "box",
            (20, 20, 40, 40),
            track_id=40,
            qr_value="BOX_500",
        )
        original = observation(
            "box",
            (20, 20, 40, 40),
            track_id=40,
            qr_value="BOX_468",
        )

        self.assertEqual(
            self.engine.update([self.zone, self.pallet, conflicting]),
            [],
        )
        self.assertEqual(
            self.engine.update([self.zone, self.pallet, original]),
            [],
        )
        self.assertEqual(set(self.engine.boxes), {"BOX_468"})


class EventEvidenceTests(unittest.TestCase):
    def test_confirmation_window_averages_weakest_real_evidence(self) -> None:
        engine = PalletZoneEngine(
            {
                "dynamic_location": {
                    "confirmation_frames": 2,
                    "startup_baseline_frames": 2,
                }
            }
        )
        conveyor = observation(
            "conveyor_belt",
            (100, 0, 200, 100),
            track_id=5,
            qr_value=None,
            confidence=0.95,
        )
        conveyor_box = observation(
            "box",
            (120, 20, 140, 40),
            track_id=50,
            qr_value="BOX_500",
            confidence=0.80,
        )
        engine.update([conveyor, conveyor_box])
        engine.update([conveyor, conveyor_box])

        target_frames = (
            (0.95, 0.85, 0.90),
            (0.75, 0.65, 0.70),
        )
        events = []
        for box_confidence, pallet_confidence, zone_confidence in target_frames:
            zone = observation(
                "zone",
                (0, 0, 100, 100),
                track_id=4,
                qr_value="ZONE_04",
                confidence=zone_confidence,
            )
            pallet = observation(
                "pallet",
                (10, 10, 90, 90),
                track_id=4,
                qr_value="PALLET_04",
                confidence=pallet_confidence,
            )
            box = observation(
                "box",
                (20, 20, 40, 40),
                track_id=50,
                qr_value="BOX_500",
                confidence=box_confidence,
            )
            events.extend(engine.update([zone, pallet, box]))

        self.assertEqual(len(events), 1)
        # Per-frame weakest evidence is 0.85 then 0.65; stabilized mean=0.75.
        self.assertEqual(events[0].confidence_score, 0.75)
        self.assertTrue(events[0].qr_read_success)

    def test_hidden_box_dock_arrival_uses_current_pallet_and_dock_evidence(self) -> None:
        engine = PalletZoneEngine(
            {
                "dynamic_location": {
                    "confirmation_frames": 1,
                    "startup_baseline_frames": 1,
                }
            }
        )
        zone = observation(
            "zone",
            (0, 0, 100, 100),
            track_id=4,
            qr_value="ZONE_04",
            confidence=0.95,
        )
        pallet = observation(
            "pallet",
            (10, 10, 90, 90),
            track_id=4,
            qr_value="PALLET_04",
            confidence=0.90,
        )
        lower_box = observation(
            "box",
            (20, 20, 40, 40),
            track_id=40,
            qr_value="BOX_468",
            confidence=0.40,
        )
        upper_box = observation(
            "box",
            (45, 20, 65, 40),
            track_id=41,
            qr_value="BOX_500",
            confidence=0.90,
        )
        engine.update([zone, pallet, lower_box, upper_box])

        dock = observation(
            "dock",
            (0, 0, 100, 100),
            track_id=2,
            qr_value="DOCK_02",
            confidence=0.60,
        )
        moving_pallet = observation(
            "pallet",
            (10, 10, 90, 90),
            track_id=4,
            qr_value="PALLET_04",
            confidence=0.80,
        )
        visible_upper_box = observation(
            "box",
            (45, 20, 65, 40),
            track_id=41,
            qr_value="BOX_500",
            confidence=0.90,
        )
        events = engine.update([dock, moving_pallet, visible_upper_box])

        dock_arrival_events = [
            event
            for event in events
            if event.event_type == EventType.DOCK_ARRIVAL
        ]
        self.assertEqual(
            {event.box_code for event in dock_arrival_events},
            {"BOX_468", "BOX_500"},
        )
        self.assertTrue(
            all(
                event.confidence_score == 0.60
                for event in dock_arrival_events
            )
        )
        self.assertTrue(
            all(event.qr_read_success for event in dock_arrival_events)
        )


if __name__ == "__main__":
    unittest.main()
