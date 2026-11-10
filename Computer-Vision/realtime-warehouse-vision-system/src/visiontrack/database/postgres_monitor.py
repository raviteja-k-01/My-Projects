"""PostgreSQL storage for video evidence, separate from operational MySQL state."""

from contextlib import contextmanager
import os

from psycopg import connect
from psycopg.rows import dict_row

from visiontrack.config.settings import project_path


@contextmanager
def connection():
    # Libpq also supports a password file and the standard PG* variables.
    with connect(
        host=os.getenv("VISIONTRACK_PG_HOST", os.getenv("PGHOST", "localhost")),
        port=os.getenv("VISIONTRACK_PG_PORT", os.getenv("PGPORT", "5432")),
        dbname=os.getenv("VISIONTRACK_PG_DATABASE", os.getenv("PGDATABASE", "visiontrack")),
        user=os.getenv("VISIONTRACK_PG_USER", os.getenv("PGUSER", "postgres")),
        password=os.getenv("VISIONTRACK_PG_PASSWORD") or None,
        connect_timeout=5,
        autocommit=True,
        row_factory=dict_row,
    ) as database:
        yield database


def migrate():
    script = project_path("database/migrations/003_postgresql_monitoring.sql")
    with connection() as database, database.transaction():
        database.execute(script.read_text(encoding="utf-8"))


def create_run(database, run_id, video, model_path):
    database.execute(
        """INSERT INTO visiontrack_monitor.runs
           (id, source_name, source_bytes, source_width, source_height, source_fps,
            source_frames, model_path, status)
           VALUES (%s, %s, %s, %s, %s, %s, %s, %s, 'PROCESSING')""",
        (run_id, video.name, video.size, video.width, video.height, video.fps,
         video.frames, str(model_path)),
    )


def persist(database, run_id, events, inventory, conflicts):
    with database.transaction(), database.cursor() as cursor:
        if events:
            cursor.executemany(
                """INSERT INTO visiontrack_monitor.events
                   (run_id, frame, source_seconds, object_code, event_type, from_pallet,
                    to_pallet, from_zone, to_zone, confidence, remarks)
                   VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
                   ON CONFLICT DO NOTHING""",
                [
                    (run_id, frame, seconds, event.box_code, event.event_type.value,
                     event.from_pallet_code, event.to_pallet_code, event.from_zone_code,
                     event.to_zone_code, event.confidence_score, event.remarks)
                    for frame, seconds, event in events
                ],
            )
        if inventory:
            cursor.executemany(
                """INSERT INTO visiontrack_monitor.inventory
                   (run_id, kind, object_code, pallet, location, last_seen_seconds)
                   VALUES (%s, %s, %s, %s, %s, %s)
                   ON CONFLICT (run_id, kind, object_code) DO UPDATE SET
                   pallet = excluded.pallet, location = excluded.location,
                   last_seen_seconds = excluded.last_seen_seconds
                   WHERE inventory.last_seen_seconds <= excluded.last_seen_seconds""",
                [(run_id, *row) for row in inventory],
            )
        if conflicts:
            cursor.executemany(
                """INSERT INTO visiontrack_monitor.exceptions
                   (run_id, object_code, first_frame, last_frame, frames)
                   VALUES (%s, %s, %s, %s, %s)
                   ON CONFLICT (run_id, object_code) DO UPDATE SET
                   last_frame = excluded.last_frame, frames = excluded.frames
                   WHERE exceptions.last_frame <= excluded.last_frame""",
                [(run_id, code, *counts) for code, counts in conflicts.items()],
            )


def finish_run(database, run_id, status, frames, elapsed, device, output,
               unresolved=0, unresolved_movements=0, error=None):
    database.execute(
        """UPDATE visiontrack_monitor.runs SET status = %s, processed_frames = %s,
           elapsed_seconds = %s, device = %s, output_path = %s, unresolved = %s,
           unresolved_movements = %s, error = %s, finished_at = now()
           WHERE id = %s AND status = 'PROCESSING'""",
        (status, frames, elapsed, device, str(output) if output else None,
         unresolved, unresolved_movements, error, run_id),
    )


def list_runs():
    with connection() as database:
        return database.execute(
            """SELECT * FROM visiontrack_monitor.runs ORDER BY started_at DESC LIMIT 100"""
        ).fetchall()


def load_run(run_id):
    with connection() as database, database.transaction():
        database.execute("SET TRANSACTION ISOLATION LEVEL REPEATABLE READ READ ONLY")
        run = database.execute(
            "SELECT * FROM visiontrack_monitor.runs WHERE id = %s", (run_id,),
        ).fetchone()
        events = database.execute(
            """SELECT * FROM visiontrack_monitor.events WHERE run_id = %s
               ORDER BY frame DESC, object_code, event_type""", (run_id,),
        ).fetchall()
        inventory = database.execute(
            """SELECT * FROM visiontrack_monitor.inventory WHERE run_id = %s
               ORDER BY kind, object_code""", (run_id,),
        ).fetchall()
        exceptions = database.execute(
            """SELECT object_code, first_frame, last_frame, frames
               FROM visiontrack_monitor.exceptions WHERE run_id = %s
               ORDER BY last_frame DESC""", (run_id,),
        ).fetchall()
    return run, events, inventory, exceptions


def recent_events(run_id):
    with connection() as database:
        return database.execute(
            """SELECT source_seconds, object_code, event_type, to_pallet, to_zone
               FROM visiontrack_monitor.events WHERE run_id = %s
               ORDER BY frame DESC LIMIT 8""", (run_id,),
        ).fetchall()


def object_history(object_code):
    with connection() as database:
        return database.execute(
            """SELECT e.*, r.source_name, r.status AS run_status
               FROM visiontrack_monitor.events e JOIN visiontrack_monitor.runs r
               ON e.run_id = r.id WHERE e.object_code = %s
               ORDER BY e.recorded_at DESC, e.frame DESC""", (object_code,),
        ).fetchall()


if __name__ == "__main__":
    migrate()
    print("VisionTrack PostgreSQL migration applied.")
