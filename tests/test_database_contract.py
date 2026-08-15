from __future__ import annotations

from datetime import datetime
import unittest

from visiontrack.database.repository import VisionTrackRepository
from visiontrack.identity import canonicalize_identity, is_package_identity
from visiontrack.warehouse.events import EventType, WarehouseEvent
from visiontrack.warehouse.pallet_zone_engine import PalletZoneEngine


class IdentityContractTests(unittest.TestCase):
    def test_database_codes_are_canonicalized(self) -> None:
        cases = {
            "BOX-1": "BOX_001",
            "CARTON_004": "BOX_004",
            "PKG_F468": "BOX_468",
            "PKG_B120": "BOX_120",
            "P_4": "PALLET_04",
            "PALLET-04": "PALLET_04",
            "ZONE_4": "ZONE_04",
            "DOCK-2": "DOCK_02",
            "BELT_1": "CONVEYOR_01",
            "CONVEYOR_02": "CONVEYOR_02",
        }

        for raw_value, expected in cases.items():
            with self.subTest(raw_value=raw_value):
                self.assertEqual(canonicalize_identity(raw_value), expected)

    def test_named_database_regions_keep_their_schema_codes(self) -> None:
        self.assertEqual(canonicalize_identity("floor"), "floor")
        self.assertEqual(
            canonicalize_identity("conveyor belt"),
            "conveyor_belt",
        )
        self.assertEqual(
            canonicalize_identity("stocking-area"),
            "stocking_area",
        )

    def test_only_package_codes_can_synthesize_boxes(self) -> None:
        self.assertTrue(is_package_identity("BOX_001"))
        self.assertTrue(is_package_identity("PKG_F468"))
        self.assertFalse(is_package_identity("PALLET_01"))
        self.assertFalse(is_package_identity("ZONE_04"))
        self.assertFalse(is_package_identity("DOCK_02"))


class PersistenceContractTests(unittest.TestCase):
    def test_event_locations_are_persisted_as_structured_codes(self) -> None:
        event = WarehouseEvent(
            event_type=EventType.DISPATCH,
            box_code="BOX_305",
            from_zone_code="ZONE_04",
            to_zone_code="DOCK-2",
        )

        self.assertEqual(
            VisionTrackRepository._structured_event_locations(event),
            (None, "ZONE_04", "DOCK_02", "DOCK_02"),
        )

    def test_conveyor_code_is_derived_only_from_a_database_code(self) -> None:
        event = WarehouseEvent(
            event_type=EventType.MOVE,
            box_code="BOX_305",
            from_zone_code="CONVEYOR-1",
            to_zone_code="ZONE_04",
        )

        self.assertEqual(
            VisionTrackRepository._structured_event_locations(event),
            ("CONVEYOR_01", "CONVEYOR_01", "ZONE_04", None),
        )

    def test_real_event_quality_is_written_to_video_events(self) -> None:
        cursor = _RecordingCursor()
        connection = _RecordingConnection(cursor)
        repository = VisionTrackRepository(
            _RecordingConnectionFactory(connection)
        )
        event_values = (
            (0.8735, True),
            (0.642, False),
            (None, None),
        )
        events = [
            WarehouseEvent(
                event_type=EventType.DISPATCH,
                box_code="BOX_305",
                from_pallet_code="PALLET_04",
                to_pallet_code="PALLET_04",
                from_zone_code="ZONE_04",
                to_zone_code="DOCK_02",
                timestamp=datetime(2026, 8, 14, 12, 0, index),
                confidence_score=confidence_score,
                qr_read_success=qr_read_success,
            )
            for index, (confidence_score, qr_read_success) in enumerate(
                event_values
            )
        ]

        repository.persist_events(
            run_id=7,
            processed_index=917,
            timestamp_sec=30.534,
            events=events,
        )

        self.assertTrue(connection.committed)
        self.assertIsNotNone(cursor.insert_rows)
        self.assertEqual(
            [(row[14], row[15], row[16]) for row in cursor.insert_rows],
            [
                (87.35, True, "NEW"),
                (64.2, False, "NEW"),
                (None, None, "NEW"),
            ],
        )

    def test_detector_confidence_boundaries_and_database_conversion(self) -> None:
        self.assertEqual(PalletZoneEngine._validated_confidence(0.0), 0.0)
        self.assertEqual(PalletZoneEngine._validated_confidence(1.0), 1.0)
        self.assertIsNone(PalletZoneEngine._validated_confidence(-0.01))
        self.assertIsNone(PalletZoneEngine._validated_confidence(1.01))
        self.assertIsNone(
            PalletZoneEngine._validated_confidence(float("nan"))
        )
        self.assertIsNone(
            PalletZoneEngine._validated_confidence(float("inf"))
        )
        self.assertEqual(
            VisionTrackRepository._event_confidence_percent(0.87654),
            87.65,
        )

    def test_capacity_percentage_is_safe(self) -> None:
        self.assertEqual(VisionTrackRepository._percent(250, 1000), 25.0)
        self.assertIsNone(VisionTrackRepository._percent(10, None))
        self.assertIsNone(VisionTrackRepository._percent(10, 0))

    def test_event_severity_is_database_compatible(self) -> None:
        self.assertEqual(
            VisionTrackRepository._event_severity("ADD"),
            "NORMAL",
        )
        self.assertEqual(
            VisionTrackRepository._event_severity("REMOVE"),
            "WARNING",
        )

    def test_pallet_capacity_comes_from_configuration(self) -> None:
        engine = PalletZoneEngine(
            {"capacity": {"default_pallet": {"max_boxes": 24}}}
        )
        self.assertEqual(engine.default_pallet_max_boxes, 24)


class _RecordingCursor:
    def __init__(self) -> None:
        self._rows: list[tuple[str]] = []
        self.insert_rows: list[tuple] | None = None

    def execute(self, query, parameters=None) -> None:
        if "FROM boxes" in query:
            self._rows = [("BOX_305",)]
        elif "FROM pallets" in query:
            self._rows = [("PALLET_04",)]
        else:
            self._rows = []

    def executemany(self, query, rows) -> None:
        self.insert_rows = list(rows)

    def fetchall(self):
        return list(self._rows)

    def close(self) -> None:
        return None


class _RecordingConnection:
    def __init__(self, cursor: _RecordingCursor) -> None:
        self.cursor_instance = cursor
        self.committed = False

    def cursor(self) -> _RecordingCursor:
        return self.cursor_instance

    def commit(self) -> None:
        self.committed = True

    def rollback(self) -> None:
        return None

    def close(self) -> None:
        return None


class _RecordingConnectionFactory:
    def __init__(self, connection: _RecordingConnection) -> None:
        self.connection = connection

    def connect(self) -> _RecordingConnection:
        return self.connection


if __name__ == "__main__":
    unittest.main()
