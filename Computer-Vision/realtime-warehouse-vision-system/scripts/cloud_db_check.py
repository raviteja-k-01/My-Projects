from __future__ import annotations

import sys

from visiontrack.cli import build_repository
from visiontrack.config.settings import load_settings
from visiontrack.database.cloud_repository import (
    CloudDatabaseError,
    CloudVisionTrackRepository,
)


def main() -> int:
    """Check cloud connectivity and schema compatibility without writing data."""
    try:
        settings = load_settings(database_target="cloud")
        repository = build_repository(settings)
        if not isinstance(repository, CloudVisionTrackRepository):
            raise CloudDatabaseError("Cloud repository was not selected.")

        health = repository.check_connection()
        schema = repository.validate_schema()
    except CloudDatabaseError as exc:
        print(f"Cloud database check failed: {exc}", file=sys.stderr)
        return 1

    print("Cloud database connection: OK")
    print(f"Selected database: {health.database}")
    print(f"MySQL server: {health.server_version}")
    print(f"Schema: {schema.describe()}")
    print("No data was changed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
