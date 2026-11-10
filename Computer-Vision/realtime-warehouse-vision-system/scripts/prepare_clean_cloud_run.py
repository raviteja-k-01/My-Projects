from __future__ import annotations

import argparse
from contextlib import closing
from dataclasses import dataclass
import sys

from visiontrack.config.settings import load_settings
from visiontrack.database.cloud_repository import (
    CloudDatabaseConfig,
    CloudDatabaseConnectionFactory,
)
from visiontrack.database.connection import DatabaseConfig, DatabaseConnectionFactory


BOX_COLUMNS = (
    "box_id",
    "box_code",
    "length_cm",
    "breadth_cm",
    "height_cm",
    "weight_kg",
    "box_status",
    "created_by",
    "created_at",
)
RUN_TABLES = (
    "video_events",
    "video_final_boxes",
    "video_final_pallets",
    "video_processing_metrics",
    "video_processing_runs",
)


@dataclass(frozen=True)
class PreparationReport:
    local_box_count: int
    cloud_box_count_before: int
    missing_codes: tuple[str, ...]
    differing_codes: tuple[str, ...]
    run_rows_before: tuple[tuple[str, int], ...]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Synchronize the cloud box master from visiontrack_local and clear "
            "only VisionTrack inference-run records."
        )
    )
    parser.add_argument(
        "--apply",
        action="store_true",
        help="Perform the inserts and cleanup; without this flag the command is read-only.",
    )
    return parser.parse_args()


def expected_box_codes() -> set[str]:
    return {
        f"BOX_{number:03d}" if number < 1000 else "BOX_1000"
        for number in range(1, 1001)
    }


def box_code_number(code: str) -> int:
    return int(code.removeprefix("BOX_"))


def load_factories():
    local = load_settings(database_target="local").database
    cloud = load_settings(database_target="cloud").database
    if local.host.lower() not in {"localhost", "127.0.0.1", "::1"}:
        raise RuntimeError(f"Refusing non-local source database host: {local.host}")
    if cloud.host.lower() in {"localhost", "127.0.0.1", "::1"}:
        raise RuntimeError("Cloud target unexpectedly resolves to a local host.")

    local_factory = DatabaseConnectionFactory(
        DatabaseConfig(
            host=local.host,
            port=local.port,
            user=local.user,
            password=local.password,
            database=local.name,
        )
    )
    cloud_factory = CloudDatabaseConnectionFactory(
        CloudDatabaseConfig(
            host=cloud.host,
            port=cloud.port,
            user=cloud.user,
            password=cloud.password,
            database=cloud.name,
        )
    )
    return local, cloud, local_factory, cloud_factory


def fetch_boxes(connection) -> dict[str, tuple]:
    columns = ", ".join(BOX_COLUMNS)
    with closing(connection.cursor()) as cursor:
        cursor.execute(
            f"""
            SELECT {columns}
            FROM boxes
            ORDER BY box_id
            """
        )
        return {str(row[1]): tuple(row) for row in cursor.fetchall()}


def inspect(local_connection, cloud_connection) -> PreparationReport:
    local_boxes = fetch_boxes(local_connection)
    cloud_boxes = fetch_boxes(cloud_connection)
    expected = expected_box_codes()
    missing_local = sorted(expected - set(local_boxes), key=box_code_number)
    if missing_local:
        raise RuntimeError(
            "Local master is incomplete; refusing cloud sync. Missing: "
            + ", ".join(missing_local[:10])
        )

    cloud_ids = {int(row[0]): code for code, row in cloud_boxes.items()}
    missing_codes = tuple(
        sorted(expected - set(cloud_boxes), key=box_code_number)
    )
    collisions = [
        f"{code} needs id {local_boxes[code][0]}, occupied by {cloud_ids[int(local_boxes[code][0])]}"
        for code in missing_codes
        if int(local_boxes[code][0]) in cloud_ids
    ]
    if collisions:
        raise RuntimeError(
            "Cloud box ID collisions detected; refusing sync: "
            + "; ".join(collisions[:10])
        )

    comparable_slice = slice(0, 8)
    differing_codes = tuple(
        sorted(
            [
                code
                for code in expected & set(cloud_boxes)
                if local_boxes[code][comparable_slice]
                != cloud_boxes[code][comparable_slice]
            ],
            key=box_code_number,
        )
    )

    run_rows = []
    with closing(cloud_connection.cursor()) as cursor:
        for table in RUN_TABLES:
            cursor.execute(f"SELECT COUNT(*) FROM {table}")
            run_rows.append((table, int(cursor.fetchone()[0])))

    return PreparationReport(
        local_box_count=len(expected & set(local_boxes)),
        cloud_box_count_before=len(expected & set(cloud_boxes)),
        missing_codes=missing_codes,
        differing_codes=differing_codes,
        run_rows_before=tuple(run_rows),
    )


def apply_changes(local_connection, cloud_connection, report: PreparationReport) -> None:
    local_boxes = fetch_boxes(local_connection)
    placeholders = ", ".join(["%s"] * len(BOX_COLUMNS))
    columns = ", ".join(BOX_COLUMNS)
    insert_sql = f"INSERT INTO boxes ({columns}) VALUES ({placeholders})"

    try:
        with closing(cloud_connection.cursor()) as cursor:
            if report.missing_codes:
                cursor.executemany(
                    insert_sql,
                    [local_boxes[code] for code in report.missing_codes],
                )
            for table in RUN_TABLES:
                cursor.execute(f"DELETE FROM {table}")
        cloud_connection.commit()
    except Exception:
        cloud_connection.rollback()
        raise

    with closing(cloud_connection.cursor()) as cursor:
        for table in RUN_TABLES:
            cursor.execute(f"ALTER TABLE {table} AUTO_INCREMENT = 1")
    cloud_connection.commit()


def verify(
    cloud_connection,
) -> tuple[int, tuple[tuple[str, int], ...], tuple[tuple[str, int], ...]]:
    cloud_boxes = fetch_boxes(cloud_connection)
    matched_count = len(expected_box_codes() & set(cloud_boxes))
    table_counts = []
    auto_increments = []
    with closing(cloud_connection.cursor()) as cursor:
        for table in RUN_TABLES:
            cursor.execute(f"SELECT COUNT(*) FROM {table}")
            table_counts.append((table, int(cursor.fetchone()[0])))
        cursor.execute(
            """
            SELECT TABLE_NAME, AUTO_INCREMENT
            FROM information_schema.TABLES
            WHERE TABLE_SCHEMA = DATABASE()
              AND TABLE_NAME IN (
                  'video_events',
                  'video_final_boxes',
                  'video_final_pallets',
                  'video_processing_metrics',
                  'video_processing_runs'
              )
            ORDER BY TABLE_NAME
            """
        )
        auto_increments.extend(
            (str(table), int(value or 1)) for table, value in cursor.fetchall()
        )
    return matched_count, tuple(table_counts), tuple(auto_increments)


def main() -> int:
    args = parse_args()
    try:
        local, cloud, local_factory, cloud_factory = load_factories()
        with closing(local_factory.connect()) as local_connection:
            with closing(cloud_factory.connect()) as cloud_connection:
                report = inspect(local_connection, cloud_connection)
                print(f"Local source: {local.name} @ {local.host}")
                print(f"Cloud target: {cloud.name} @ {cloud.host}")
                print(f"Validated local boxes: {report.local_box_count}")
                print(f"Cloud boxes before: {report.cloud_box_count_before}")
                print(f"Boxes to insert: {len(report.missing_codes)}")
                if report.missing_codes:
                    print(
                        "Missing range/sample: "
                        f"{report.missing_codes[0]} ... {report.missing_codes[-1]}"
                    )
                print(f"Existing box differences: {len(report.differing_codes)}")
                print("Run rows before:")
                for table, count in report.run_rows_before:
                    print(f"  {table}: {count}")

                if not args.apply:
                    box_count, _run_counts, auto_increments = verify(
                        cloud_connection
                    )
                    print(f"Cloud validated boxes now: {box_count}")
                    print("Next auto-increment values:")
                    for table, value in auto_increments:
                        print(f"  {table}: {value}")
                    print("Read-only inspection complete. Use --apply to make changes.")
                    return 0

                apply_changes(local_connection, cloud_connection, report)
                box_count, run_counts, auto_increments = verify(cloud_connection)
    except Exception as exc:
        print(f"Cloud preparation failed: {exc}", file=sys.stderr)
        return 1

    print(f"Cloud validated boxes after: {box_count}")
    print("Run rows after:")
    for table, count in run_counts:
        print(f"  {table}: {count}")
    print("Next auto-increment values:")
    for table, value in auto_increments:
        print(f"  {table}: {value}")
    if (
        box_count != 1000
        or any(count != 0 for _, count in run_counts)
        or any(value != 1 for _, value in auto_increments)
    ):
        print("Post-change verification failed.", file=sys.stderr)
        return 1
    print("Cloud master synchronization and run cleanup completed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
