from __future__ import annotations

from collections.abc import Iterator, Mapping, Sequence
from contextlib import contextmanager
from datetime import datetime
from typing import Any, Protocol

from visiontrack.lifecycle.errors import (
    InvalidEventError,
    LifecycleSchemaError,
)
from visiontrack.lifecycle.models import (
    LifecycleState,
    LifecycleTarget,
    LocationContext,
    OrderContext,
    SourceVideoEvent,
)


class ConnectionFactory(Protocol):
    def connect(self): ...


class LifecycleTransactionProtocol(Protocol):
    def lock_next_new_event(self) -> Mapping[str, Any] | None: ...

    def create_event_savepoint(self) -> None: ...

    def rollback_event_savepoint(self) -> None: ...

    def resolve_box(self, box_code: str) -> int: ...

    def resolve_order_contexts(self, box_id: int) -> list[OrderContext]: ...

    def latest_box_state(self, box_id: int) -> LifecycleState | None: ...

    def resolve_location_id(self, kind: str, code: str) -> int: ...

    def resolve_carrier_context(
        self,
        dock_id: int,
        event_time: datetime,
        *,
        require_attached: bool,
    ) -> LocationContext: ...

    def insert_box_lifecycle(
        self,
        *,
        event: SourceVideoEvent,
        target: LifecycleTarget,
        box_id: int,
        order_contexts: Sequence[OrderContext],
        location: LocationContext,
        recorded_by: int | None,
        remarks: str,
    ) -> int: ...

    def order_dispatch_complete(self, order_id: int) -> bool: ...

    def order_already_dispatched(self, order_id: int) -> bool: ...

    def insert_order_dispatch(
        self,
        *,
        event: SourceVideoEvent,
        order_id: int,
        recorded_by: int | None,
    ) -> int: ...

    def mark_processed(self, video_event_id: int) -> None: ...

    def mark_error(self, video_event_id: int, message: str) -> None: ...


class LifecycleRepositoryProtocol(Protocol):
    @contextmanager
    def transaction(self) -> Iterator[LifecycleTransactionProtocol]: ...


class MySQLLifecycleRepository:
    """Transactional MySQL adapter for lifecycle promotion.

    One repository owns one persistent connection. Every source event still
    uses its own transaction, and ``FOR UPDATE SKIP LOCKED`` allows multiple
    workers to cooperate without processing the same row.
    """

    REQUIRED_SCHEMA: Mapping[str, frozenset[str]] = {
        "video_events": frozenset(
            {
                "video_event_id",
                "run_id",
                "processed_index",
                "video_time",
                "timestamp_sec",
                "event_type",
                "severity",
                "box_code",
                "pallet_code",
                "from_pallet_code",
                "to_pallet_code",
                "conveyor_code",
                "from_zone_code",
                "to_zone_code",
                "dock_code",
                "confidence_score",
                "qr_read_success",
                "processing_status",
                "processed_at",
                "processing_error",
                "description",
                "event_created_at",
                "inserted_at",
            }
        ),
        "boxes": frozenset({"box_id", "box_code"}),
        "pallets": frozenset({"pallet_id", "pallet_code"}),
        "conveyors": frozenset({"conveyor_id", "conveyor_code"}),
        "zones": frozenset({"zone_id", "zone_code"}),
        "docks": frozenset({"dock_id", "dock_code"}),
        "orders": frozenset({"order_id"}),
        "order_items": frozenset({"order_item_id", "order_id"}),
        "order_item_products": frozenset(
            {"order_item_product_id", "order_item_id"}
        ),
        "order_boxes": frozenset(
            {"box_id", "order_item_product_id"}
        ),
        "order_lifecycle_events": frozenset(
            {
                "order_lifecycle_event_id",
                "entity_type",
                "lifecycle_status",
                "event_type",
                "order_id",
                "order_item_product_id",
                "box_id",
                "conveyor_id",
                "pallet_id",
                "zone_id",
                "dock_id",
                "trailer_id",
                "truck_id",
                "trailer_dock_assignment_id",
                "trailer_truck_assignment_id",
                "event_source",
                "source_reference",
                "idempotency_key",
                "confidence_score",
                "qr_read_success",
                "severity",
                "event_time",
                "recorded_by",
                "remarks",
            }
        ),
        "trailer_dock_assignments": frozenset(
            {
                "trailer_dock_assignment_id",
                "trailer_id",
                "dock_id",
                "scheduled_from",
                "scheduled_to",
                "actual_start",
                "actual_end",
                "assignment_status",
            }
        ),
        "trailer_truck_assignments": frozenset(
            {
                "trailer_truck_assignment_id",
                "trailer_id",
                "truck_id",
                "assignment_status",
                "assigned_at",
                "attached_at",
                "detached_at",
            }
        ),
        "trailers": frozenset({"trailer_id"}),
        "trucks": frozenset({"truck_id"}),
    }

    def __init__(self, connection) -> None:
        self.connection = connection

    @contextmanager
    def transaction(self) -> Iterator["MySQLLifecycleTransaction"]:
        cursor = self.connection.cursor()
        try:
            self.connection.start_transaction()
            yield MySQLLifecycleTransaction(cursor)
            self.connection.commit()
        except Exception:
            self.connection.rollback()
            raise
        finally:
            cursor.close()

    def validate_schema(self) -> None:
        tables = tuple(sorted(self.REQUIRED_SCHEMA))
        placeholders = ", ".join(["%s"] * len(tables))
        cursor = self.connection.cursor()
        try:
            cursor.execute(
                f"""
                SELECT table_name, column_name
                FROM information_schema.COLUMNS
                WHERE table_schema = DATABASE()
                  AND table_name IN ({placeholders})
                """,
                tables,
            )
            available: dict[str, set[str]] = {}
            for table, column in cursor.fetchall():
                available.setdefault(str(table), set()).add(str(column))
            self.connection.commit()
        except Exception:
            self.connection.rollback()
            raise
        finally:
            cursor.close()

        missing_tables = [
            table for table in tables if table not in available
        ]
        missing_columns = [
            (table, sorted(columns - available.get(table, set())))
            for table, columns in self.REQUIRED_SCHEMA.items()
            if table in available and columns - available[table]
        ]
        if missing_tables or missing_columns:
            problems: list[str] = []
            if missing_tables:
                problems.append("missing tables: " + ", ".join(missing_tables))
            problems.extend(
                f"{table} missing columns: {', '.join(columns)}"
                for table, columns in missing_columns
            )
            raise LifecycleSchemaError("; ".join(problems))

    def retry_errors(self, limit: int) -> int:
        if limit < 1:
            raise ValueError("retry error limit must be at least 1")
        cursor = self.connection.cursor()
        try:
            self.connection.start_transaction()
            cursor.execute(
                """
                UPDATE video_events ve
                JOIN (
                    SELECT video_event_id
                    FROM video_events
                    WHERE processing_status = 'ERROR'
                    ORDER BY video_event_id
                    LIMIT %s
                ) selected
                  ON selected.video_event_id = ve.video_event_id
                SET ve.processing_status = 'NEW',
                    ve.processed_at = NULL,
                    ve.processing_error = NULL
                """,
                (limit,),
            )
            count = int(cursor.rowcount)
            self.connection.commit()
            return count
        except Exception:
            self.connection.rollback()
            raise
        finally:
            cursor.close()


class MySQLLifecycleTransaction:
    _LOCATION_QUERIES = {
        "conveyor": (
            "SELECT conveyor_id FROM conveyors "
            "WHERE UPPER(conveyor_code) = UPPER(%s) LIMIT 1"
        ),
        "pallet": (
            "SELECT pallet_id FROM pallets "
            "WHERE UPPER(pallet_code) = UPPER(%s) LIMIT 1"
        ),
        "zone": (
            "SELECT zone_id FROM zones "
            "WHERE UPPER(zone_code) = UPPER(%s) LIMIT 1"
        ),
        "dock": (
            "SELECT dock_id FROM docks "
            "WHERE UPPER(dock_code) = UPPER(%s) LIMIT 1"
        ),
    }

    def __init__(self, cursor) -> None:
        self.cursor = cursor

    def lock_next_new_event(self) -> Mapping[str, Any] | None:
        self.cursor.execute(
            """
            SELECT
                video_event_id, run_id, processed_index, video_time,
                timestamp_sec, event_type, severity, box_code, pallet_code,
                from_pallet_code, to_pallet_code, conveyor_code,
                from_zone_code, to_zone_code, dock_code, confidence_score,
                qr_read_success, processing_status, description,
                event_created_at, inserted_at
            FROM video_events
            WHERE processing_status = 'NEW'
            ORDER BY video_event_id
            LIMIT 1
            FOR UPDATE SKIP LOCKED
            """
        )
        row = self.cursor.fetchone()
        if row is None:
            return None
        columns = [str(item[0]) for item in self.cursor.description]
        return dict(zip(columns, row))

    def create_event_savepoint(self) -> None:
        self.cursor.execute("SAVEPOINT lifecycle_event_work")

    def rollback_event_savepoint(self) -> None:
        # The video_events row was locked before this savepoint, so rolling back
        # event work retains that lock until mark_error() commits.
        self.cursor.execute("ROLLBACK TO SAVEPOINT lifecycle_event_work")

    def resolve_box(self, box_code: str) -> int:
        self.cursor.execute(
            """
            SELECT box_id
            FROM boxes
            WHERE UPPER(box_code) = UPPER(%s)
            LIMIT 1
            """,
            (box_code,),
        )
        row = self.cursor.fetchone()
        if row is None:
            raise InvalidEventError(f"Unknown box_code: {box_code}")
        return int(row[0])

    def resolve_order_contexts(self, box_id: int) -> list[OrderContext]:
        self.cursor.execute(
            """
            SELECT DISTINCT oi.order_id, oip.order_item_product_id
            FROM order_boxes ob
            JOIN order_item_products oip
              ON oip.order_item_product_id = ob.order_item_product_id
            JOIN order_items oi
              ON oi.order_item_id = oip.order_item_id
            WHERE ob.box_id = %s
            ORDER BY oi.order_id, oip.order_item_product_id
            """,
            (box_id,),
        )
        contexts = [
            OrderContext(int(order_id), int(product_id))
            for order_id, product_id in self.cursor.fetchall()
        ]
        if not contexts:
            raise InvalidEventError(
                f"Box ID {box_id} is not linked to an order through order_boxes"
            )
        return contexts

    def latest_box_state(self, box_id: int) -> LifecycleState | None:
        self.cursor.execute(
            """
            SELECT
                lifecycle_status, conveyor_id, pallet_id, zone_id, dock_id,
                trailer_id, truck_id, trailer_dock_assignment_id,
                trailer_truck_assignment_id, event_time
            FROM order_lifecycle_events
            WHERE entity_type = 'BOX'
              AND box_id = %s
              AND lifecycle_status IN (
                  'RECEIVED', 'ON_FLOOR', 'ON_CONVEYOR', 'IN_ZONE',
                  'ON_PALLET', 'AT_DOCK', 'LOADED_TO_TRAILER',
                  'DISPATCHED', 'BOX_EXCEPTION', 'REMOVED', 'MISSING'
              )
              AND COALESCE(source_reference, '') NOT LIKE 'CV:RUN:%%'
              AND COALESCE(source_reference, '') NOT LIKE 'box-status-%%'
              AND COALESCE(source_reference, '') NOT LIKE 'video-final-box-%%'
              AND COALESCE(event_type, '') NOT IN (
                  'CV_FINAL_STATE', 'BOX_STATUS_CHANGED',
                  'VIDEO_FINAL_BOX_SYNC'
              )
            ORDER BY event_time DESC, order_lifecycle_event_id DESC
            LIMIT 1
            """,
            (box_id,),
        )
        row = self.cursor.fetchone()
        if row is None:
            return None
        return LifecycleState(
            lifecycle_status=str(row[0]),
            conveyor_id=self._optional_int(row[1]),
            pallet_id=self._optional_int(row[2]),
            zone_id=self._optional_int(row[3]),
            dock_id=self._optional_int(row[4]),
            trailer_id=self._optional_int(row[5]),
            truck_id=self._optional_int(row[6]),
            trailer_dock_assignment_id=self._optional_int(row[7]),
            trailer_truck_assignment_id=self._optional_int(row[8]),
            event_time=row[9] if isinstance(row[9], datetime) else None,
        )

    def resolve_location_id(self, kind: str, code: str) -> int:
        try:
            query = self._LOCATION_QUERIES[kind]
        except KeyError as exc:
            raise ValueError(f"Unsupported location kind: {kind}") from exc
        self.cursor.execute(query, (code,))
        row = self.cursor.fetchone()
        if row is None:
            raise InvalidEventError(f"Unknown {kind}_code: {code}")
        return int(row[0])

    def resolve_carrier_context(
        self,
        dock_id: int,
        event_time: datetime,
        *,
        require_attached: bool,
    ) -> LocationContext:
        status_filter = (
            "AND assignment_status = 'ACTIVE'"
            if require_attached
            else "AND assignment_status IN ('ACTIVE', 'SCHEDULED')"
        )
        self.cursor.execute(
            f"""
            SELECT trailer_dock_assignment_id, trailer_id
            FROM trailer_dock_assignments
            WHERE dock_id = %s
              {status_filter}
              AND scheduled_from <= %s
              AND (actual_end IS NULL OR actual_end >= %s)
              AND (
                  assignment_status = 'ACTIVE'
                  OR scheduled_to >= %s
              )
            ORDER BY
                CASE assignment_status WHEN 'ACTIVE' THEN 0 ELSE 1 END,
                COALESCE(actual_start, scheduled_from) DESC,
                trailer_dock_assignment_id DESC
            LIMIT 1
            """,
            (dock_id, event_time, event_time, event_time),
        )
        assignment = self.cursor.fetchone()
        if assignment is None:
            return LocationContext(dock_id=dock_id)

        dock_assignment_id = int(assignment[0])
        trailer_id = int(assignment[1])
        attached_filter = (
            "AND UPPER(assignment_status) = 'ATTACHED'"
            if require_attached
            else (
                "AND UPPER(COALESCE(assignment_status, '')) "
                "NOT IN ('DETACHED', 'CANCELLED')"
            )
        )
        self.cursor.execute(
            f"""
            SELECT trailer_truck_assignment_id, truck_id
            FROM trailer_truck_assignments
            WHERE trailer_id = %s
              {attached_filter}
              AND assigned_at <= %s
              AND (attached_at IS NULL OR attached_at <= %s)
              AND (detached_at IS NULL OR detached_at > %s)
            ORDER BY
                CASE UPPER(assignment_status)
                    WHEN 'ATTACHED' THEN 0 ELSE 1
                END,
                COALESCE(attached_at, assigned_at) DESC,
                trailer_truck_assignment_id DESC
            LIMIT 1
            """,
            (trailer_id, event_time, event_time, event_time),
        )
        truck_assignment = self.cursor.fetchone()
        return LocationContext(
            dock_id=dock_id,
            trailer_id=trailer_id,
            truck_id=(
                int(truck_assignment[1]) if truck_assignment is not None else None
            ),
            trailer_dock_assignment_id=dock_assignment_id,
            trailer_truck_assignment_id=(
                int(truck_assignment[0])
                if truck_assignment is not None
                else None
            ),
        )

    def insert_box_lifecycle(
        self,
        *,
        event: SourceVideoEvent,
        target: LifecycleTarget,
        box_id: int,
        order_contexts: Sequence[OrderContext],
        location: LocationContext,
        recorded_by: int | None,
        remarks: str,
    ) -> int:
        inserted = 0
        source_reference = self.source_reference(event.video_event_id)
        for order_context in order_contexts:
            idempotency_key = self.box_idempotency_key(
                event.video_event_id,
                box_id,
                order_context,
            )
            self.cursor.execute(
                """
                INSERT INTO order_lifecycle_events (
                    entity_type, lifecycle_status, event_type,
                    order_id, order_item_product_id, box_id,
                    conveyor_id, pallet_id, zone_id, dock_id,
                    trailer_id, truck_id,
                    trailer_dock_assignment_id,
                    trailer_truck_assignment_id,
                    event_source, source_reference, idempotency_key,
                    confidence_score, qr_read_success, severity,
                    event_time, recorded_by, remarks
                ) VALUES (
                    'BOX', %s, %s,
                    %s, %s, %s,
                    %s, %s, %s, %s,
                    %s, %s,
                    %s, %s,
                    'COMPUTER_VISION', %s, %s,
                    %s, %s, %s,
                    %s, %s, %s
                )
                ON DUPLICATE KEY UPDATE
                    idempotency_key = VALUES(idempotency_key)
                """,
                (
                    target.lifecycle_status,
                    target.business_event_type,
                    order_context.order_id,
                    order_context.order_item_product_id,
                    box_id,
                    location.conveyor_id,
                    location.pallet_id,
                    location.zone_id,
                    location.dock_id,
                    location.trailer_id,
                    location.truck_id,
                    location.trailer_dock_assignment_id,
                    location.trailer_truck_assignment_id,
                    source_reference,
                    idempotency_key,
                    event.confidence_score,
                    event.qr_read_success,
                    event.severity,
                    event.event_time,
                    recorded_by,
                    remarks[:500],
                ),
            )
            if int(self.cursor.rowcount or 0) == 1:
                inserted += 1
        return inserted

    def order_dispatch_complete(self, order_id: int) -> bool:
        self.cursor.execute(
            """
            WITH ranked_box_states AS (
                SELECT
                    box_id,
                    lifecycle_status,
                    ROW_NUMBER() OVER (
                        PARTITION BY box_id
                        ORDER BY event_time DESC,
                                 order_lifecycle_event_id DESC
                    ) AS state_rank
                FROM order_lifecycle_events
                WHERE entity_type = 'BOX'
                  AND box_id IS NOT NULL
                  AND lifecycle_status IN (
                      'RECEIVED', 'ON_FLOOR', 'ON_CONVEYOR', 'IN_ZONE',
                      'ON_PALLET', 'AT_DOCK', 'LOADED_TO_TRAILER',
                      'DISPATCHED', 'BOX_EXCEPTION', 'REMOVED', 'MISSING'
                  )
                  AND COALESCE(source_reference, '') NOT LIKE 'CV:RUN:%%'
                  AND COALESCE(source_reference, '') NOT LIKE 'box-status-%%'
                  AND COALESCE(source_reference, '')
                      NOT LIKE 'video-final-box-%%'
                  AND COALESCE(event_type, '') NOT IN (
                      'CV_FINAL_STATE', 'BOX_STATUS_CHANGED',
                      'VIDEO_FINAL_BOX_SYNC'
                  )
            ),
            order_box_set AS (
                SELECT DISTINCT ob.box_id
                FROM order_items oi
                JOIN order_item_products oip
                  ON oip.order_item_id = oi.order_item_id
                JOIN order_boxes ob
                  ON ob.order_item_product_id = oip.order_item_product_id
                WHERE oi.order_id = %s
            )
            SELECT
                COUNT(*) AS total_boxes,
                SUM(
                    CASE WHEN latest.lifecycle_status = 'DISPATCHED'
                         THEN 1 ELSE 0 END
                ) AS dispatched_boxes
            FROM order_box_set
            LEFT JOIN ranked_box_states latest
              ON latest.box_id = order_box_set.box_id
             AND latest.state_rank = 1
            """,
            (order_id,),
        )
        total, dispatched = self.cursor.fetchone()
        total_count = int(total or 0)
        return total_count > 0 and total_count == int(dispatched or 0)

    def order_already_dispatched(self, order_id: int) -> bool:
        self.cursor.execute(
            """
            SELECT 1
            FROM order_lifecycle_events
            WHERE entity_type = 'ORDER'
              AND order_id = %s
              AND lifecycle_status = 'DISPATCHED'
            LIMIT 1
            """,
            (order_id,),
        )
        return self.cursor.fetchone() is not None

    def insert_order_dispatch(
        self,
        *,
        event: SourceVideoEvent,
        order_id: int,
        recorded_by: int | None,
    ) -> int:
        self.cursor.execute(
            """
            INSERT INTO order_lifecycle_events (
                entity_type, lifecycle_status, event_type, order_id,
                event_source, source_reference, idempotency_key,
                confidence_score, qr_read_success, severity,
                event_time, recorded_by, remarks
            ) VALUES (
                'ORDER', 'DISPATCHED', 'ORDER_DISPATCHED', %s,
                'COMPUTER_VISION', %s, %s,
                %s, %s, %s,
                %s, %s,
                'All physical boxes linked to this order are DISPATCHED.'
            )
            ON DUPLICATE KEY UPDATE
                idempotency_key = VALUES(idempotency_key)
            """,
            (
                order_id,
                self.source_reference(event.video_event_id),
                self.order_dispatch_idempotency_key(order_id),
                event.confidence_score,
                event.qr_read_success,
                event.severity,
                event.event_time,
                recorded_by,
            ),
        )
        return 1 if int(self.cursor.rowcount or 0) == 1 else 0

    def mark_processed(self, video_event_id: int) -> None:
        self.cursor.execute(
            """
            UPDATE video_events
            SET processing_status = 'PROCESSED',
                processed_at = CURRENT_TIMESTAMP,
                processing_error = NULL
            WHERE video_event_id = %s
              AND processing_status = 'NEW'
            """,
            (video_event_id,),
        )
        if int(self.cursor.rowcount or 0) != 1:
            raise InvalidEventError(
                f"Source event {video_event_id} is no longer NEW"
            )

    def mark_error(self, video_event_id: int, message: str) -> None:
        self.cursor.execute(
            """
            UPDATE video_events
            SET processing_status = 'ERROR',
                processed_at = NULL,
                processing_error = %s
            WHERE video_event_id = %s
              AND processing_status = 'NEW'
            """,
            (message[:500], video_event_id),
        )

    @staticmethod
    def source_reference(video_event_id: int) -> str:
        return f"VIDEO_EVENT:{int(video_event_id)}"

    @staticmethod
    def box_idempotency_key(
        video_event_id: int,
        box_id: int,
        order_context: OrderContext,
    ) -> str:
        return (
            f"video-event:{int(video_event_id)}:box:{int(box_id)}:"
            f"order:{order_context.order_id}:"
            f"item:{order_context.order_item_product_id}"
        )

    @staticmethod
    def order_dispatch_idempotency_key(order_id: int) -> str:
        return f"order:{int(order_id)}:dispatched"

    @staticmethod
    def _optional_int(value: Any) -> int | None:
        return None if value is None else int(value)
