from __future__ import annotations

import json
from collections import Counter
from contextlib import closing
from typing import Any, Mapping, cast

from visiontrack.database.connection import DatabaseConnectionFactory
from visiontrack.identity import canonicalize_identity
from visiontrack.warehouse.entities import BoxState, PalletState
from visiontrack.warehouse.events import WarehouseEvent


class VisionTrackRepository:
    """Persistence adapter for one video-processing run."""

    def __init__(self, connection_factory: DatabaseConnectionFactory):
        self.connection_factory = connection_factory

    def create_run(
        self,
        *,
        video_name: str,
        input_video_path: str,
        model_path: str,
        device: str,
        gpu_name: str | None,
        source_width: int,
        source_height: int,
        source_fps: float,
        video_duration_sec: float | None,
        frame_skip: int,
        process_width: int,
        process_height: int,
        yolo_imgsz: int,
    ) -> int:
        with closing(self.connection_factory.connect()) as connection:
            with closing(connection.cursor()) as cursor:
                try:
                    cursor.execute(
                        """
                        INSERT INTO video_processing_runs (
                            video_name, input_video_path, model_path, device, gpu_name,
                            source_width, source_height, source_fps, video_duration_sec,
                            frame_skip, process_width, process_height, yolo_imgsz, status
                        )
                        VALUES (
                            %s, %s, %s, %s, %s,
                            %s, %s, %s, %s,
                            %s, %s, %s, %s, 'RUNNING'
                        )
                        """,
                        (
                            video_name,
                            input_video_path,
                            model_path,
                            device,
                            gpu_name,
                            source_width,
                            source_height,
                            source_fps,
                            video_duration_sec,
                            frame_skip,
                            process_width,
                            process_height,
                            yolo_imgsz,
                        ),
                    )
                    lastrowid = cursor.lastrowid
                    if lastrowid is None:
                        raise RuntimeError("Failed to retrieve last row ID from video processing runs insert")
                    run_id = int(lastrowid)
                    connection.commit()
                    return run_id
                except Exception:
                    connection.rollback()
                    raise

    def persist_events(
        self,
        *,
        run_id: int,
        processed_index: int,
        timestamp_sec: float,
        events: list[WarehouseEvent],
    ) -> None:
        if not events:
            return

        video_time = self._format_video_time(timestamp_sec)
        candidate_rows = []

        for event in events:
            box_code = canonicalize_identity(event.box_code)
            from_pallet_code = canonicalize_identity(
                event.from_pallet_code
            )
            to_pallet_code = canonicalize_identity(
                event.to_pallet_code
            )
            (
                conveyor_code,
                from_zone_code,
                to_zone_code,
                dock_code,
            ) = self._structured_event_locations(event)
            identity_values = (
                box_code,
                from_pallet_code,
                to_pallet_code,
            )

            if any(self._is_temporary_identity(value) for value in identity_values):
                continue

            candidate_rows.append(
                (
                    run_id,
                    processed_index,
                    video_time,
                    timestamp_sec,
                    event.event_type.value,
                    self._event_severity(event.event_type.value),
                    box_code,
                    to_pallet_code or from_pallet_code,
                    from_pallet_code,
                    to_pallet_code,
                    conveyor_code,
                    from_zone_code,
                    to_zone_code,
                    dock_code,
                    self._event_confidence_percent(
                        event.confidence_score
                    ),
                    event.qr_read_success,
                    "NEW",
                    self._event_description(event),
                    event.timestamp,
                )
            )
        if not candidate_rows:
            return

        with closing(self.connection_factory.connect()) as connection:
            with closing(connection.cursor()) as cursor:
                try:
                    known_box_codes = self._existing_codes(
                        cursor,
                        table="boxes",
                        column="box_code",
                        values={row[6] for row in candidate_rows if row[6]},
                    )
                    requested_pallet_codes = {
                        value
                        for row in candidate_rows
                        for value in row[7:10]
                        if value
                    }
                    known_pallet_codes = self._existing_codes(
                        cursor,
                        table="pallets",
                        column="pallet_code",
                        values=requested_pallet_codes,
                    )
                    rows = [
                        row
                        for row in candidate_rows
                        if row[6] in known_box_codes
                        and all(
                            value is None or value in known_pallet_codes
                            for value in row[7:10]
                        )
                    ]
                    if not rows:
                        return

                    cursor.executemany(
                        """
                        INSERT INTO video_events (
                            run_id, processed_index, video_time, timestamp_sec,
                            event_type, severity, box_code, pallet_code,
                            from_pallet_code, to_pallet_code, conveyor_code,
                            from_zone_code, to_zone_code, dock_code,
                            confidence_score, qr_read_success,
                            processing_status, description, event_created_at
                        )
                        VALUES (
                            %s, %s, %s, %s, %s, %s,
                            %s, %s, %s, %s, %s, %s,
                            %s, %s, %s, %s, %s, %s, %s
                        )
                        """,
                        rows,
                    )
                    connection.commit()
                except Exception:
                    connection.rollback()
                    raise

    def complete_run(
        self,
        *,
        run_id: int,
        boxes: dict[str, BoxState],
        pallets: dict[str, PalletState],
        processed_frames: int,
        source_frames_covered: int,
        video_duration_sec: float | None,
        total_processing_time_sec: float,
        end_to_end_time_sec: float,
        metrics: Mapping[str, Any] | None = None,
    ) -> None:
        speed_vs_realtime = None
        realtime_factor = None

        if (
            video_duration_sec is not None
            and video_duration_sec > 0
            and end_to_end_time_sec > 0
        ):
            speed_vs_realtime = video_duration_sec / end_to_end_time_sec
            realtime_factor = end_to_end_time_sec / video_duration_sec
        permanent_boxes = [
            box
            for box in boxes.values()
            if not self._is_temporary_identity(box.box_code)
            and not self._is_temporary_identity(box.current_pallet_code or "")
        ]

        permanent_pallets = [
            pallet
            for pallet in pallets.values()
            if not self._is_temporary_identity(pallet.pallet_code)
        ]
        with closing(self.connection_factory.connect()) as connection:
            with closing(connection.cursor()) as cursor:
                try:
                    canonical_boxes = [
                        (
                            box,
                            canonicalize_identity(box.box_code),
                            canonicalize_identity(box.current_pallet_code),
                        )
                        for box in permanent_boxes
                    ]
                    canonical_pallets = [
                        (pallet, canonicalize_identity(pallet.pallet_code))
                        for pallet in permanent_pallets
                    ]
                    box_master = self._load_box_master(
                        cursor,
                        {code for _, code, _ in canonical_boxes if code},
                    )
                    pallet_master = self._load_pallet_master(
                        cursor,
                        {code for _, code in canonical_pallets if code},
                    )
                    known_pallet_codes = set(pallet_master)
                    unmatched_box_codes = sorted(
                        code
                        for _, code, _ in canonical_boxes
                        if code and code not in box_master
                    )
                    unmatched_pallet_codes = sorted(
                        {
                            code
                            for _, code in canonical_pallets
                            if code and code not in pallet_master
                        }
                        | {
                            pallet_code
                            for _, _, pallet_code in canonical_boxes
                            if pallet_code
                            and pallet_code not in known_pallet_codes
                        }
                    )
                    matched_boxes = [
                        (box, code, pallet_code, box_master[code])
                        for box, code, pallet_code in canonical_boxes
                        if code in box_master
                        and (
                            pallet_code is None
                            or pallet_code in known_pallet_codes
                        )
                    ]
                    matched_pallets = [
                        (pallet, code, pallet_master[code])
                        for pallet, code in canonical_pallets
                        if code in pallet_master
                    ]
                    pallet_box_counts = Counter(
                        pallet_code
                        for _, _, pallet_code, _ in matched_boxes
                        if pallet_code is not None
                    )

                    cursor.execute(
                        "DELETE FROM video_final_boxes WHERE run_id = %s",
                        (run_id,),
                    )
                    cursor.execute(
                        "DELETE FROM video_final_pallets WHERE run_id = %s",
                        (run_id,),
                    )
                    cursor.execute(
                        "DELETE FROM video_processing_metrics WHERE run_id = %s",
                        (run_id,),
                    )

                    box_rows = [
                        (
                            run_id,
                            code,
                            box.status,
                            pallet_code,
                            box.slot_no,
                            box.row_no,
                            box.column_no,
                            box.layer_no,
                            master["length_cm"],
                            master["breadth_cm"],
                            master["height_cm"],
                            master["weight_kg"],
                            master["volume_cm3"],
                        )
                        for box, code, pallet_code, master in matched_boxes
                    ]

                    if box_rows:
                        cursor.executemany(
                            """
                            INSERT INTO video_final_boxes (
                                run_id, box_code, status, pallet_code,
                                slot_no, row_no, column_no, layer_no,
                                length_cm, width_cm, height_cm, weight_kg,
                                volume_cm3
                            )
                            VALUES (
                                %s, %s, %s, %s, %s, %s, %s, %s,
                                %s, %s, %s, %s, %s
                            )
                            """,
                            box_rows,
                        )

                    pallet_rows = []
                    for pallet, code, master in matched_pallets:
                        assigned_boxes = [
                            box_master_row
                            for _, _, pallet_code, box_master_row in matched_boxes
                            if pallet_code == code
                        ]
                        total_weight = sum(
                            float(item["weight_kg"] or 0)
                            for item in assigned_boxes
                        )
                        total_volume = sum(
                            float(item["volume_cm3"] or 0)
                            for item in assigned_boxes
                        )
                        max_weight = self._float_or_none(
                            master["max_weight_kg"]
                        )
                        max_volume = self._float_or_none(
                            master["max_volume_cm3"]
                        )
                        max_height = self._float_or_none(
                            master["max_height_cm"]
                        )
                        weight_utilization = self._percent(
                            total_weight,
                            max_weight,
                        )
                        volume_utilization = self._percent(
                            total_volume,
                            max_volume,
                        )
                        available_utilizations = [
                            value
                            for value in (
                                weight_utilization,
                                volume_utilization,
                            )
                            if value is not None
                        ]
                        bottleneck = (
                            max(available_utilizations)
                            if available_utilizations
                            else None
                        )
                        box_count = int(pallet_box_counts.get(code, 0))
                        pallet_rows.append(
                            (
                                run_id,
                                code,
                                int(pallet.visible),
                                pallet.status,
                                box_count,
                                pallet.max_boxes,
                                total_weight,
                                max_weight,
                                weight_utilization,
                                total_volume,
                                max_volume,
                                volume_utilization,
                                None,
                                max_height,
                                None,
                                bottleneck,
                                (
                                    f"{box_count} database-matched box(es); "
                                    "physical slot/layer placement is not inferred."
                                ),
                            )
                        )

                    if pallet_rows:
                        cursor.executemany(
                            """
                            INSERT INTO video_final_pallets (
                                run_id, pallet_code, active, status, box_count,
                                max_boxes, total_weight_kg, max_weight_kg,
                                weight_utilization_percent, total_volume_cm3,
                                max_volume_cm3, volume_efficiency_percent,
                                max_stack_height_cm, max_height_cm,
                                height_utilization_percent,
                                bottleneck_utilization_percent, reasons
                            )
                            VALUES (
                                %s, %s, %s, %s, %s, %s, %s, %s, %s,
                                %s, %s, %s, %s, %s, %s, %s, %s
                            )
                            """,
                            pallet_rows,
                        )

                    cursor.execute(
                        "SELECT COUNT(*) FROM video_events WHERE run_id = %s",
                        (run_id,),
                    )
                    row = cursor.fetchone()
                    if row is None:
                        total_events = 0
                    else:
                        total_events = int(cast(int, row[0]))  # type: ignore[index]

                    metric_values: dict[str, Any] = dict(metrics or {})
                    metric_values.update(
                        {
                            "final_box_count": len(matched_boxes),
                            "final_pallet_count": len(matched_pallets),
                            "total_event_count": total_events,
                            "database_matched_box_count": len(matched_boxes),
                            "database_matched_pallet_count": len(matched_pallets),
                            "database_unmatched_box_codes": unmatched_box_codes,
                            "database_unmatched_pallet_codes": unmatched_pallet_codes,
                        }
                    )

                    cursor.execute(
                        """
                        SELECT event_type, COUNT(*)
                        FROM video_events
                        WHERE run_id = %s
                        GROUP BY event_type
                        """,
                        (run_id,),
                    )

                    for row in cursor.fetchall():
                        event_type = cast(str, row[0])  # type: ignore[index]
                        event_count = cast(int, row[1])  # type: ignore[index]
                        metric_values[f"event_count_{event_type}"] = int(event_count)

                    self._insert_metrics(cursor, run_id, metric_values)

                    cursor.execute(
                        """
                        UPDATE video_processing_runs
                        SET processed_frames = %s,
                            source_frames_covered = %s,
                            video_duration_sec = %s,
                            total_processing_time_sec = %s,
                            end_to_end_time_sec = %s,
                            speed_vs_realtime = %s,
                            realtime_factor = %s,
                            status = 'COMPLETED',
                            completed_at = CURRENT_TIMESTAMP
                        WHERE run_id = %s
                        """,
                        (
                            processed_frames,
                            source_frames_covered,
                            video_duration_sec,
                            total_processing_time_sec,
                            end_to_end_time_sec,
                            speed_vs_realtime,
                            realtime_factor,
                            run_id,
                        ),
                    )

                    if cursor.rowcount != 1:
                        raise RuntimeError(
                            f"Video processing run {run_id} does not exist."
                        )

                    connection.commit()
                except Exception:
                    connection.rollback()
                    raise

    def fail_run(
        self,
        *,
        run_id: int,
        error_message: str,
        processed_frames: int,
        source_frames_covered: int,
        elapsed_sec: float,
    ) -> None:
        with closing(self.connection_factory.connect()) as connection:
            with closing(connection.cursor()) as cursor:
                try:
                    cursor.execute(
                        """
                        UPDATE video_processing_runs
                        SET processed_frames = %s,
                            source_frames_covered = %s,
                            total_processing_time_sec = %s,
                            end_to_end_time_sec = %s,
                            status = 'FAILED',
                            completed_at = CURRENT_TIMESTAMP
                        WHERE run_id = %s
                        """,
                        (
                            processed_frames,
                            source_frames_covered,
                            elapsed_sec,
                            elapsed_sec,
                            run_id,
                        ),
                    )

                    self._insert_metrics(
                        cursor,
                        run_id,
                        {"failure_message": error_message},
                    )

                    connection.commit()
                except Exception:
                    connection.rollback()
                    raise

    @staticmethod
    def _is_temporary_identity(value: str | None) -> bool:
        return bool(value and "_TRACK_" in value.upper())

    @staticmethod
    def _existing_codes(
        cursor,
        *,
        table: str,
        column: str,
        values: set[str],
    ) -> set[str]:
        if not values:
            return set()

        allowed: dict[str, str] = {
            "boxes": "box_code",
            "pallets": "pallet_code",
        }
        if table not in allowed or allowed[table] != column:
            raise ValueError("Unsupported database identity lookup")

        # Column names are controlled by whitelist, safe for dynamic SQL
        safe_column: str = column
        safe_table: str = table
        placeholders = ", ".join(["%s"] * len(values))
        cursor.execute(
            f"SELECT {safe_column} FROM {safe_table} WHERE {safe_column} IN ({placeholders})",
            tuple(sorted(values)),
        )
        return {str(row[0]) for row in cursor.fetchall()}

    @staticmethod
    def _load_box_master(cursor, values: set[str]) -> dict[str, dict[str, Any]]:
        if not values:
            return {}
        placeholders = ", ".join(["%s"] * len(values))
        # Column and table names are database constants, safe for dynamic SQL
        query = f"""
            SELECT box_code, length_cm, breadth_cm, height_cm,
                   weight_kg, volume_cm3
            FROM boxes
            WHERE box_code IN ({placeholders})
            """
        cursor.execute(query, tuple(sorted(values)))
        return {
            str(row[0]): {
                "length_cm": row[1],
                "breadth_cm": row[2],
                "height_cm": row[3],
                "weight_kg": row[4],
                "volume_cm3": row[5],
            }
            for row in cursor.fetchall()
        }

    @staticmethod
    def _load_pallet_master(cursor, values: set[str]) -> dict[str, dict[str, Any]]:
        if not values:
            return {}
        placeholders = ", ".join(["%s"] * len(values))
        # Column and table names are database constants, safe for dynamic SQL
        query = f"""
            SELECT pallet_code, max_weight_kg, max_volume_cm3, max_height_cm
            FROM pallets
            WHERE pallet_code IN ({placeholders})
            """
        cursor.execute(query, tuple(sorted(values)))
        return {
            str(row[0]): {
                "max_weight_kg": row[1],
                "max_volume_cm3": row[2],
                "max_height_cm": row[3],
            }
            for row in cursor.fetchall()
        }

    @staticmethod
    def _float_or_none(value: Any) -> float | None:
        return None if value is None else float(value)

    @staticmethod
    def _percent(value: float, maximum: float | None) -> float | None:
        if maximum is None or maximum <= 0:
            return None
        return round((float(value) / maximum) * 100.0, 2)

    @staticmethod
    def _event_severity(event_type: str) -> str:
        return "WARNING" if event_type in {"REMOVE", "RE_ADD"} else "NORMAL"

    @staticmethod
    def _insert_metrics(
        cursor,
        run_id: int,
        metrics: Mapping[str, Any],
    ) -> None:
        rows = [
            (
                run_id,
                str(name),
                VisionTrackRepository._metric_text(value),
            )
            for name, value in sorted(metrics.items())
        ]

        if not rows:
            return

        cursor.executemany(
            """
            INSERT INTO video_processing_metrics (
                run_id, metric_name, metric_value
            )
            VALUES (%s, %s, %s)
            """,
            rows,
        )

    @staticmethod
    def _metric_text(value: Any) -> str | None:
        if value is None:
            return None

        if isinstance(value, (dict, list, tuple)):
            return json.dumps(value, sort_keys=True)

        return str(value)

    @staticmethod
    def _event_confidence_percent(value: float | None) -> float | None:
        if value is None:
            return None
        confidence = float(value)
        if not 0.0 <= confidence <= 1.0:
            return None
        return round(confidence * 100.0, 2)

    @classmethod
    def _structured_event_locations(
        cls,
        event: WarehouseEvent,
    ) -> tuple[str | None, str | None, str | None, str | None]:
        from_zone_code = canonicalize_identity(event.from_zone_code)
        to_zone_code = canonicalize_identity(event.to_zone_code)

        if cls._is_temporary_identity(from_zone_code):
            from_zone_code = None
        if cls._is_temporary_identity(to_zone_code):
            to_zone_code = None

        conveyor_code = next(
            (
                value
                for value in (to_zone_code, from_zone_code)
                if value and value.startswith("CONVEYOR_")
            ),
            None,
        )
        dock_code = next(
            (
                value
                for value in (to_zone_code, from_zone_code)
                if value and value.startswith("DOCK_")
            ),
            None,
        )

        return conveyor_code, from_zone_code, to_zone_code, dock_code

    @staticmethod
    def _event_description(event: WarehouseEvent) -> str | None:
        parts: list[str] = []

        remarks = event.remarks.strip()
        for identity in (
            event.from_zone_code,
            event.to_zone_code,
            event.from_pallet_code,
            event.to_pallet_code,
        ):
            if (
                identity
                and VisionTrackRepository._is_temporary_identity(identity)
            ):
                remarks = remarks.replace(identity, "UNRESOLVED")

        if remarks:
            parts.append(remarks)

        if (
            event.from_zone_code is not None
            or event.to_zone_code is not None
        ):
            from_zone = canonicalize_identity(event.from_zone_code)
            to_zone = canonicalize_identity(event.to_zone_code)
            if VisionTrackRepository._is_temporary_identity(from_zone):
                from_zone = None
            if VisionTrackRepository._is_temporary_identity(to_zone):
                to_zone = None
            parts.append(
                f"zone: {from_zone or 'NONE'} -> {to_zone or 'NONE'}"
            )

        placement = []

        for name, value in (
            ("slot", event.slot_no),
            ("row", event.row_no),
            ("column", event.column_no),
            ("layer", event.layer_no),
        ):
            if value is not None:
                placement.append(f"{name}={value}")

        if placement:
            parts.append(", ".join(placement))

        return " | ".join(parts) or None

    @staticmethod
    def _format_video_time(timestamp_sec: float) -> str:
        total_ms = max(0, int(round(timestamp_sec * 1000)))
        hours, remainder = divmod(total_ms, 3_600_000)
        minutes, remainder = divmod(remainder, 60_000)
        seconds, milliseconds = divmod(remainder, 1_000)

        return (
            f"{hours:02d}:{minutes:02d}:"
            f"{seconds:02d}.{milliseconds:03d}"
        )
