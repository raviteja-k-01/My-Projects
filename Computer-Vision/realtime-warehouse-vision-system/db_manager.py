from __future__ import annotations

from contextlib import closing
from dataclasses import dataclass
from datetime import datetime
from typing import Any, Iterable, Optional

try:
    import mysql.connector as mysql_connector
except Exception:
    mysql_connector = None


@dataclass
class DatabaseConfig:
    host: str = "localhost"
    port: int = 3306
    user: str = "root"
    password: str = ""
    database: str = "warehouse"


class DatabaseManager:
    """
    Database layer matched to guide schema.

    Uses existing guide tables:
      boxes, pallets, events, pallet_box_assignments

    Does NOT create/change/use:
      zones, orders, order_items, customers, products structure

    It only reads one existing zone_id because pallets.current_zone_id is NOT NULL.
    If zones table has no row, pallet/assignment sync is skipped safely.
    """

    def __init__(self, config: DatabaseConfig) -> None:
        self.config = config
        self.default_user_id: Optional[int] = None
        self.default_zone_id: Optional[int] = None

    def connect(self):
        if mysql_connector is None:
            raise RuntimeError(
                "mysql-connector-python is not installed. Run: pip install mysql-connector-python"
            )
        return mysql_connector.connect(
            host=self.config.host,
            port=self.config.port,
            user=self.config.user,
            password=self.config.password,
            database=self.config.database,
            autocommit=False,
        )

    @staticmethod
    def _execute_statements(cursor, statements: Iterable[str]) -> None:
        for statement in statements:
            cleaned = statement.strip()
            if cleaned:
                cursor.execute(cleaned)

    def initialize(self) -> None:
        """Create only video/dashboard tables and cache existing user/zone IDs."""
        with closing(self.connect()) as connection:
            cursor = connection.cursor(dictionary=True)
            self._execute_statements(cursor, EXTENSION_DDL)

            cursor.execute("SELECT MIN(user_id) AS user_id FROM team_users")
            row = cursor.fetchone() or {}
            self.default_user_id = row.get("user_id")

            cursor.execute("SELECT MIN(zone_id) AS zone_id FROM zones")
            row = cursor.fetchone() or {}
            self.default_zone_id = row.get("zone_id")

            connection.commit()

    def save_master_data(
        self,
        box_master: dict[str, dict[str, float]],
        pallet_master: dict[str, dict[str, float]],
    ) -> None:
        """
        Intentionally no-op.

        This avoids inserting BOX_001 to BOX_030 blindly.
        Only boxes/pallets detected in the processed video are written later.
        """
        return None

    def create_completed_run(self, metrics: dict[str, Any]) -> int:
        with closing(self.connect()) as connection:
            cursor = connection.cursor()
            cursor.execute(
                """
                INSERT INTO video_processing_runs (
                    video_name, input_video_path, model_path, device, gpu_name,
                    source_width, source_height, source_fps, video_duration_sec,
                    frame_skip, process_width, process_height, yolo_imgsz,
                    processed_frames, source_frames_covered,
                    total_processing_time_sec, end_to_end_time_sec,
                    speed_vs_realtime, realtime_factor,
                    status, completed_at
                )
                VALUES (
                    %s, %s, %s, %s, %s,
                    %s, %s, %s, %s,
                    %s, %s, %s, %s,
                    %s, %s,
                    %s, %s,
                    %s, %s,
                    'COMPLETED', NOW()
                )
                """,
                (
                    str(metrics.get("video_argument") or metrics.get("input_video") or ""),
                    str(metrics.get("input_video") or ""),
                    str(metrics.get("model") or ""),
                    str(metrics.get("device") or ""),
                    str(metrics.get("gpu_name") or ""),
                    self._int_or_none(metrics.get("source_width")),
                    self._int_or_none(metrics.get("source_height")),
                    self._float_or_none(metrics.get("source_fps")),
                    self._float_or_none(metrics.get("video_duration_sec")),
                    self._int_or_none(metrics.get("frame_skip")),
                    self._int_or_none(metrics.get("process_width")),
                    self._int_or_none(metrics.get("process_height")),
                    self._int_or_none(metrics.get("yolo_imgsz")),
                    self._int_or_none(metrics.get("processed_frames")),
                    self._int_or_none(metrics.get("source_frames_covered")),
                    self._float_or_none(metrics.get("total_processing_time_sec")),
                    self._float_or_none(metrics.get("end_to_end_time_sec")),
                    self._float_or_none(metrics.get("speed_vs_realtime")),
                    self._float_or_none(metrics.get("realtime_factor")),
                ),
            )
            run_id = int(cursor.lastrowid)
            connection.commit()
            return run_id

    def save_run_outputs(
        self,
        run_id: int,
        metrics: dict[str, Any],
        events: list[dict[str, Any]],
        final_boxes: list[dict[str, Any]],
        final_pallets: list[dict[str, Any]],
    ) -> None:
        with closing(self.connect()) as connection:
            cursor = connection.cursor(dictionary=True)

            if self.default_user_id is None or self.default_zone_id is None:
                self._cache_existing_ids(cursor)

            self._save_metrics(cursor, run_id, metrics)
            self._save_video_events(cursor, run_id, events)
            self._save_final_pallets(cursor, run_id, final_pallets)
            self._save_final_boxes(cursor, run_id, final_boxes)

            self._sync_guide_pallets(cursor, final_pallets)
            self._sync_guide_boxes(cursor, final_boxes)
            self._sync_guide_assignments(cursor, final_boxes)
            self._sync_guide_events(cursor, events)

            connection.commit()

    def _cache_existing_ids(self, cursor) -> None:
        cursor.execute("SELECT MIN(user_id) AS user_id FROM team_users")
        row = cursor.fetchone() or {}
        self.default_user_id = row.get("user_id")

        cursor.execute("SELECT MIN(zone_id) AS zone_id FROM zones")
        row = cursor.fetchone() or {}
        self.default_zone_id = row.get("zone_id")

    def _save_metrics(self, cursor, run_id: int, metrics: dict[str, Any]) -> None:
        rows = [(run_id, str(k), "" if v is None else str(v)) for k, v in metrics.items()]
        if rows:
            cursor.executemany(
                """
                INSERT INTO video_processing_metrics (run_id, metric_name, metric_value)
                VALUES (%s, %s, %s)
                """,
                rows,
            )

    def _save_video_events(self, cursor, run_id: int, events: list[dict[str, Any]]) -> None:
        rows = []
        for event in events:
            rows.append(
                (
                    run_id,
                    self._int_or_none(event.get("processed_index")),
                    event.get("video_time") or "",
                    self._float_or_none(event.get("timestamp_sec")),
                    event.get("event_type") or "",
                    event.get("severity") or "INFO",
                    event.get("box_id") or None,
                    event.get("pallet_id") or None,
                    event.get("from_pallet") or None,
                    event.get("to_pallet") or None,
                    event.get("description") or "",
                    self._datetime_or_none(event.get("created_at")),
                )
            )
        if rows:
            cursor.executemany(
                """
                INSERT INTO video_events (
                    run_id, processed_index, video_time, timestamp_sec,
                    event_type, severity, box_code, pallet_code,
                    from_pallet_code, to_pallet_code, description, event_created_at
                )
                VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
                """,
                rows,
            )

    def _save_final_boxes(self, cursor, run_id: int, boxes: list[dict[str, Any]]) -> None:
        rows = []
        for row in boxes:
            rows.append(
                (
                    run_id,
                    row.get("box_id") or row.get("box_code") or "",
                    row.get("status") or "",
                    row.get("pallet_id") or row.get("pallet_code") or None,
                    self._int_or_none(row.get("slot_no")),
                    self._slot_row(row.get("slot_no")),
                    self._slot_col(row.get("slot_no")),
                    self._int_or_none(row.get("layer_no")),
                    self._float_or_none(row.get("length_cm")),
                    self._float_or_none(row.get("width_cm")),
                    self._float_or_none(row.get("height_cm")),
                    self._float_or_none(row.get("weight_kg")),
                    self._float_or_none(row.get("volume_cm3")),
                )
            )
        if rows:
            cursor.executemany(
                """
                INSERT INTO video_final_boxes (
                    run_id, box_code, status, pallet_code,
                    slot_no, row_no, column_no, layer_no,
                    length_cm, width_cm, height_cm, weight_kg, volume_cm3
                )
                VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
                """,
                rows,
            )

    def _save_final_pallets(self, cursor, run_id: int, pallets: list[dict[str, Any]]) -> None:
        rows = []
        for row in pallets:
            rows.append(
                (
                    run_id,
                    row.get("pallet_id") or row.get("pallet_code") or "",
                    1 if self._truthy(row.get("active")) else 0,
                    row.get("status") or "",
                    self._int_or_none(row.get("box_count")),
                    self._int_or_none(row.get("max_boxes")),
                    self._float_or_none(row.get("total_weight_kg")),
                    self._float_or_none(row.get("max_weight_kg")),
                    self._float_or_none(row.get("weight_utilization_percent")),
                    self._float_or_none(row.get("total_volume_cm3")),
                    self._float_or_none(row.get("max_volume_cm3")),
                    self._float_or_none(row.get("volume_efficiency_percent")),
                    self._float_or_none(row.get("max_stack_height_cm")),
                    self._float_or_none(row.get("max_height_cm")),
                    self._float_or_none(row.get("height_utilization_percent")),
                    self._float_or_none(row.get("bottleneck_utilization_percent")),
                    row.get("reasons") or "",
                )
            )
        if rows:
            cursor.executemany(
                """
                INSERT INTO video_final_pallets (
                    run_id, pallet_code, active, status, box_count, max_boxes,
                    total_weight_kg, max_weight_kg, weight_utilization_percent,
                    total_volume_cm3, max_volume_cm3, volume_efficiency_percent,
                    max_stack_height_cm, max_height_cm, height_utilization_percent,
                    bottleneck_utilization_percent, reasons
                )
                VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
                """,
                rows,
            )

    def _sync_guide_pallets(self, cursor, final_pallets: list[dict[str, Any]]) -> None:
        if self.default_zone_id is None:
            return
        for row in final_pallets:
            pallet_code = row.get("pallet_id") or row.get("pallet_code")
            if not pallet_code:
                continue
            box_count = self._int_or_none(row.get("box_count")) or 0
            max_boxes = self._int_or_none(row.get("max_boxes")) or 0
            dashboard_status = str(row.get("status") or "").upper()
            if dashboard_status in {"OVERLOAD", "CRITICAL"} or (max_boxes and box_count >= max_boxes):
                pallet_status = "FULL"
            elif box_count > 0:
                pallet_status = "IN_USE"
            else:
                pallet_status = "EMPTY"
            cursor.execute(
                """
                INSERT INTO pallets (
                    pallet_code, current_zone_id, pallet_status,
                    max_weight_kg, max_volume_cm3, max_height_cm, created_by
                )
                VALUES (%s, %s, %s, %s, %s, %s, %s)
                ON DUPLICATE KEY UPDATE
                    current_zone_id = VALUES(current_zone_id),
                    pallet_status = VALUES(pallet_status),
                    max_weight_kg = VALUES(max_weight_kg),
                    max_volume_cm3 = VALUES(max_volume_cm3),
                    max_height_cm = VALUES(max_height_cm)
                """,
                (
                    pallet_code,
                    self.default_zone_id,
                    pallet_status,
                    self._float_or_none(row.get("max_weight_kg")),
                    self._float_or_none(row.get("max_volume_cm3")),
                    self._float_or_none(row.get("max_height_cm")),
                    self.default_user_id,
                ),
            )

    def _sync_guide_boxes(self, cursor, final_boxes: list[dict[str, Any]]) -> None:
        for row in final_boxes:
            box_code = row.get("box_id") or row.get("box_code")
            if not box_code:
                continue
            pallet_code = row.get("pallet_id") or row.get("pallet_code") or None
            current_pallet_id = self._get_pallet_pk(cursor, pallet_code) if pallet_code else None
            cursor.execute(
                """
                INSERT INTO boxes (
                    box_code, current_pallet_id, length_cm, breadth_cm,
                    height_cm, weight_kg, box_status, created_by
                )
                VALUES (%s, %s, %s, %s, %s, %s, %s, %s)
                ON DUPLICATE KEY UPDATE
                    current_pallet_id = VALUES(current_pallet_id),
                    length_cm = VALUES(length_cm),
                    breadth_cm = VALUES(breadth_cm),
                    height_cm = VALUES(height_cm),
                    weight_kg = VALUES(weight_kg),
                    box_status = VALUES(box_status)
                """,
                (
                    box_code,
                    current_pallet_id,
                    self._float_or_default(row.get("length_cm"), 10.0),
                    self._float_or_default(row.get("width_cm"), 8.0),
                    self._float_or_default(row.get("height_cm"), 5.0),
                    self._float_or_none(row.get("weight_kg")),
                    self._box_status_for_core(row.get("status")),
                    self.default_user_id,
                ),
            )

    def _sync_guide_assignments(self, cursor, final_boxes: list[dict[str, Any]]) -> None:
        for row in final_boxes:
            box_code = row.get("box_id") or row.get("box_code")
            pallet_code = row.get("pallet_id") or row.get("pallet_code") or None
            if not box_code:
                continue
            box_pk = self._get_box_pk(cursor, box_code)
            if box_pk is None:
                continue
            cursor.execute(
                """
                UPDATE pallet_box_assignments
                SET removed_at = NOW()
                WHERE box_id = %s AND removed_at IS NULL
                """,
                (box_pk,),
            )
            if not pallet_code:
                continue
            pallet_pk = self._get_pallet_pk(cursor, pallet_code)
            if pallet_pk is None:
                continue
            cursor.execute(
                """
                INSERT INTO pallet_box_assignments (
                    pallet_id, box_id, slot_no, row_no, column_no, layer_no
                )
                VALUES (%s, %s, %s, %s, %s, %s)
                """,
                (
                    pallet_pk,
                    box_pk,
                    self._int_or_none(row.get("slot_no")),
                    self._slot_row(row.get("slot_no")),
                    self._slot_col(row.get("slot_no")),
                    self._int_or_none(row.get("layer_no")),
                ),
            )

    def _sync_guide_events(self, cursor, events: list[dict[str, Any]]) -> None:
        for event in events:
            event_type = self._core_event_type(event.get("event_type"))
            box_code = event.get("box_id")
            if event_type is None or not box_code:
                continue
            box_pk = self._get_box_pk(cursor, box_code)
            if box_pk is None:
                continue
            cursor.execute(
                """
                INSERT INTO events (
                    event_type, box_id, from_pallet_id, to_pallet_id, remarks
                )
                VALUES (%s, %s, %s, %s, %s)
                """,
                (
                    event_type,
                    box_pk,
                    self._get_pallet_pk(cursor, event.get("from_pallet")),
                    self._get_pallet_pk(cursor, event.get("to_pallet") or event.get("pallet_id")),
                    event.get("description") or "",
                ),
            )

    @staticmethod
    def _int_or_none(value: Any) -> Optional[int]:
        try:
            if value is None or value == "":
                return None
            return int(float(value))
        except Exception:
            return None

    @staticmethod
    def _float_or_none(value: Any) -> Optional[float]:
        try:
            if value is None or value == "":
                return None
            return float(value)
        except Exception:
            return None

    @staticmethod
    def _float_or_default(value: Any, default: float) -> float:
        parsed = DatabaseManager._float_or_none(value)
        return default if parsed is None else parsed

    @staticmethod
    def _datetime_or_none(value: Any) -> Optional[datetime]:
        if not value:
            return None
        if isinstance(value, datetime):
            return value
        try:
            return datetime.fromisoformat(str(value))
        except Exception:
            return None

    @staticmethod
    def _truthy(value: Any) -> bool:
        return str(value).strip().lower() in {"1", "true", "yes", "y"}

    @staticmethod
    def _slot_row(slot_no: Any, grid_cols: int = 3) -> Optional[int]:
        slot = DatabaseManager._int_or_none(slot_no)
        if not slot or slot < 1:
            return None
        return ((slot - 1) // grid_cols) + 1

    @staticmethod
    def _slot_col(slot_no: Any, grid_cols: int = 3) -> Optional[int]:
        slot = DatabaseManager._int_or_none(slot_no)
        if not slot or slot < 1:
            return None
        return ((slot - 1) % grid_cols) + 1

    @staticmethod
    def _box_status_for_core(status: Any) -> str:
        value = str(status or "").upper()
        if value in {"ON_PALLET", "PLACING", "STACKED"}:
            return "ON_PALLET"
        if value in {"MOVED", "MOVING"}:
            return "MOVED"
        if value in {"REMOVED", "BOX_REMOVED"}:
            return "REMOVED"
        if value in {"COMPLETE_REMOVED", "TEMP_MISSING", "PALLET_REMOVED", "MISSING"}:
            return "MISSING"
        return "RECEIVED"

    @staticmethod
    def _core_event_type(event_type: Any) -> Optional[str]:
        mapping = {
            "BOX_ADDED": "ADD",
            "BOX_RE_ADDED": "RE_ADD",
            "BOX_MOVED": "MOVE",
            "BOX_SWAP": "SWAP",
            "BOX_STACKED": "STACK",
            "BOX_REMOVED": "REMOVE",
            "BOX_COMPLETE_REMOVED": "REMOVE",
        }
        return mapping.get(str(event_type or "").upper())

    @staticmethod
    def _get_pallet_pk(cursor, pallet_code: Optional[str]) -> Optional[int]:
        if not pallet_code:
            return None
        cursor.execute("SELECT pallet_id FROM pallets WHERE pallet_code = %s", (pallet_code,))
        row = cursor.fetchone()
        return int(row["pallet_id"]) if row else None

    @staticmethod
    def _get_box_pk(cursor, box_code: Optional[str]) -> Optional[int]:
        if not box_code:
            return None
        cursor.execute("SELECT box_id FROM boxes WHERE box_code = %s", (box_code,))
        row = cursor.fetchone()
        return int(row["box_id"]) if row else None


EXTENSION_DDL = [
    """
    CREATE TABLE IF NOT EXISTS video_processing_runs (
        run_id BIGINT AUTO_INCREMENT PRIMARY KEY,
        video_name VARCHAR(255) NOT NULL,
        input_video_path TEXT,
        model_path TEXT,
        device VARCHAR(100),
        gpu_name VARCHAR(255),
        source_width INT,
        source_height INT,
        source_fps DECIMAL(10,3),
        video_duration_sec DECIMAL(12,3),
        frame_skip INT,
        process_width INT,
        process_height INT,
        yolo_imgsz INT,
        processed_frames INT,
        source_frames_covered INT,
        total_processing_time_sec DECIMAL(12,3),
        end_to_end_time_sec DECIMAL(12,3),
        speed_vs_realtime DECIMAL(12,3),
        realtime_factor DECIMAL(12,3),
        status ENUM('RUNNING','COMPLETED','FAILED') DEFAULT 'COMPLETED',
        started_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
        completed_at TIMESTAMP NULL,
        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
    )
    """,
    """
    CREATE TABLE IF NOT EXISTS video_events (
        video_event_id BIGINT AUTO_INCREMENT PRIMARY KEY,
        run_id BIGINT NOT NULL,
        processed_index INT,
        video_time VARCHAR(20),
        timestamp_sec DECIMAL(12,3),
        event_type VARCHAR(80),
        severity VARCHAR(30),
        box_code VARCHAR(50),
        pallet_code VARCHAR(50),
        from_pallet_code VARCHAR(50),
        to_pallet_code VARCHAR(50),
        description TEXT,
        event_created_at DATETIME NULL,
        inserted_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
        KEY idx_video_events_run_type (run_id, event_type),
        KEY idx_video_events_box (box_code),
        KEY idx_video_events_pallet (pallet_code, from_pallet_code, to_pallet_code),
        CONSTRAINT fk_video_events_run
            FOREIGN KEY (run_id) REFERENCES video_processing_runs(run_id)
            ON DELETE CASCADE
    )
    """,
    """
    CREATE TABLE IF NOT EXISTS video_final_boxes (
        final_box_id BIGINT AUTO_INCREMENT PRIMARY KEY,
        run_id BIGINT NOT NULL,
        box_code VARCHAR(50) NOT NULL,
        status VARCHAR(80),
        pallet_code VARCHAR(50),
        slot_no INT,
        row_no INT,
        column_no INT,
        layer_no INT,
        length_cm DECIMAL(10,2),
        width_cm DECIMAL(10,2),
        height_cm DECIMAL(10,2),
        weight_kg DECIMAL(10,2),
        volume_cm3 DECIMAL(12,2),
        inserted_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
        KEY idx_video_final_boxes_run (run_id, pallet_code, status),
        CONSTRAINT fk_video_final_boxes_run
            FOREIGN KEY (run_id) REFERENCES video_processing_runs(run_id)
            ON DELETE CASCADE
    )
    """,
    """
    CREATE TABLE IF NOT EXISTS video_final_pallets (
        final_pallet_id BIGINT AUTO_INCREMENT PRIMARY KEY,
        run_id BIGINT NOT NULL,
        pallet_code VARCHAR(50) NOT NULL,
        active TINYINT(1) DEFAULT 0,
        status VARCHAR(80),
        box_count INT,
        max_boxes INT,
        total_weight_kg DECIMAL(12,3),
        max_weight_kg DECIMAL(12,3),
        weight_utilization_percent DECIMAL(8,2),
        total_volume_cm3 DECIMAL(14,3),
        max_volume_cm3 DECIMAL(14,3),
        volume_efficiency_percent DECIMAL(8,2),
        max_stack_height_cm DECIMAL(10,3),
        max_height_cm DECIMAL(10,3),
        height_utilization_percent DECIMAL(8,2),
        bottleneck_utilization_percent DECIMAL(8,2),
        reasons TEXT,
        inserted_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
        KEY idx_video_final_pallets_run (run_id, pallet_code, status),
        CONSTRAINT fk_video_final_pallets_run
            FOREIGN KEY (run_id) REFERENCES video_processing_runs(run_id)
            ON DELETE CASCADE
    )
    """,
    """
    CREATE TABLE IF NOT EXISTS video_processing_metrics (
        metric_id BIGINT AUTO_INCREMENT PRIMARY KEY,
        run_id BIGINT NOT NULL,
        metric_name VARCHAR(128) NOT NULL,
        metric_value TEXT,
        inserted_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
        KEY idx_video_metrics_run (run_id, metric_name),
        CONSTRAINT fk_video_metrics_run
            FOREIGN KEY (run_id) REFERENCES video_processing_runs(run_id)
            ON DELETE CASCADE
    )
    """,
    """
    CREATE OR REPLACE VIEW v_latest_video_run AS
    SELECT *
    FROM video_processing_runs
    WHERE run_id = (SELECT MAX(run_id) FROM video_processing_runs)
    """,
    """
    CREATE OR REPLACE VIEW v_latest_video_pallets AS
    SELECT p.*
    FROM video_final_pallets p
    JOIN v_latest_video_run r ON p.run_id = r.run_id
    """,
    """
    CREATE OR REPLACE VIEW v_latest_video_boxes AS
    SELECT b.*
    FROM video_final_boxes b
    JOIN v_latest_video_run r ON b.run_id = r.run_id
    """,
    """
    CREATE OR REPLACE VIEW v_latest_video_events AS
    SELECT e.*
    FROM video_events e
    JOIN v_latest_video_run r ON e.run_id = r.run_id
    """,
]
