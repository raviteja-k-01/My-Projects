from __future__ import annotations

import argparse
from contextlib import closing
import sys

from visiontrack.cli import build_repository
from visiontrack.config.settings import load_settings
from visiontrack.database.cloud_repository import (
    CloudDatabaseError,
    CloudVisionTrackRepository,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Read back one VisionTrack cloud run without changing data."
    )
    parser.add_argument("run_id", type=int, help="Cloud video_processing_runs.run_id")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    try:
        repository = build_repository(load_settings(database_target="cloud"))
        if not isinstance(repository, CloudVisionTrackRepository):
            raise CloudDatabaseError("Cloud repository was not selected.")

        with closing(repository.cloud_connection_factory.connect()) as connection:
            with closing(connection.cursor()) as cursor:
                cursor.execute(
                    """
                    SELECT video_name, status, processed_frames,
                           source_frames_covered, completed_at
                    FROM video_processing_runs
                    WHERE run_id = %s
                    """,
                    (args.run_id,),
                )
                run = cursor.fetchone()
                if run is None:
                    print(f"Cloud run {args.run_id} was not found.", file=sys.stderr)
                    return 1

                cursor.execute(
                    """
                    SELECT event_type, COUNT(*)
                    FROM video_events
                    WHERE run_id = %s
                    GROUP BY event_type
                    ORDER BY event_type
                    """,
                    (args.run_id,),
                )
                event_counts = cursor.fetchall()

                cursor.execute(
                    """
                    SELECT video_event_id, event_type, box_code,
                           from_zone_code, to_zone_code, dock_code,
                           confidence_score, qr_read_success,
                           processing_status, processed_at, processing_error
                    FROM video_events
                    WHERE run_id = %s
                    ORDER BY video_event_id
                    """,
                    (args.run_id,),
                )
                event_rows = cursor.fetchall()

                cursor.execute(
                    """
                    SELECT ve.video_event_id, COUNT(ole.order_lifecycle_event_id)
                    FROM video_events ve
                    LEFT JOIN order_lifecycle_events ole
                      ON ole.source_reference = CONCAT(
                          'VIDEO_EVENT:', ve.video_event_id
                      )
                    WHERE ve.run_id = %s
                    GROUP BY ve.video_event_id
                    ORDER BY ve.video_event_id
                    """,
                    (args.run_id,),
                )
                lifecycle_counts = dict(cursor.fetchall())

                cursor.execute(
                    """
                    SELECT pallet_code, box_count, status
                    FROM video_final_pallets
                    WHERE run_id = %s
                    ORDER BY pallet_code
                    """,
                    (args.run_id,),
                )
                pallets = cursor.fetchall()

                cursor.execute(
                    """
                    SELECT metric_name, metric_value
                    FROM video_processing_metrics
                    WHERE run_id = %s
                      AND metric_name IN (
                          'database_matched_box_count',
                          'database_unmatched_box_codes',
                          'database_unmatched_pallet_codes'
                      )
                    ORDER BY metric_name
                    """,
                    (args.run_id,),
                )
                master_metrics = cursor.fetchall()

                counts = {}
                for table in (
                    "video_events",
                    "video_final_boxes",
                    "video_final_pallets",
                    "video_processing_metrics",
                ):
                    cursor.execute(
                        f"SELECT COUNT(*) FROM {table} WHERE run_id = %s",
                        (args.run_id,),
                    )
                    counts[table] = int(cursor.fetchone()[0])
    except CloudDatabaseError as exc:
        print(f"Cloud run report failed: {exc}", file=sys.stderr)
        return 1

    video_name, status, processed_frames, covered_frames, completed_at = run
    print(f"Cloud run: {args.run_id}")
    print(f"Video: {video_name}")
    print(f"Status: {status}")
    print(f"Frames: processed={processed_frames}, covered={covered_frames}")
    print(f"Completed at: {completed_at}")
    print("Rows:")
    for table, count in counts.items():
        print(f"  {table}: {count}")
    print("Events:")
    for event_type, count in event_counts:
        print(f"  {event_type}: {count}")
    print("Event-processor handoff:")
    for event_row in event_rows:
        (
            event_id,
            event_type,
            box_code,
            from_zone,
            to_zone,
            dock_code,
            confidence_score,
            qr_read_success,
            processing_status,
            processed_at,
            processing_error,
        ) = event_row
        print(
            f"  #{event_id} {event_type} {box_code}: "
            f"{from_zone or '-'} -> {to_zone or '-'} "
            f"dock={dock_code or '-'} confidence={confidence_score} "
            f"qr_read={qr_read_success} status={processing_status} "
            f"processed_at={processed_at or '-'} "
            f"lifecycle_rows={int(lifecycle_counts.get(event_id, 0))} "
            f"error={processing_error or '-'}"
        )
    print("Master-data matching:")
    for metric_name, metric_value in master_metrics:
        print(f"  {metric_name}: {metric_value}")
    print("Final pallets:")
    for pallet_code, box_count, pallet_status in pallets:
        print(f"  {pallet_code}: boxes={box_count}, status={pallet_status}")
    print("No data was changed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
