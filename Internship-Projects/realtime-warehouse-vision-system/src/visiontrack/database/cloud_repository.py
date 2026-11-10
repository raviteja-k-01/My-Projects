from __future__ import annotations

from collections.abc import Callable, Mapping
from contextlib import closing
from dataclasses import dataclass
import time
from typing import Any

from visiontrack.database.repository import VisionTrackRepository


class CloudDatabaseError(RuntimeError):
    """Base error for cloud database configuration and connectivity failures."""


class CloudDatabaseConfigurationError(CloudDatabaseError):
    """Raised when required cloud settings are missing or invalid."""


class CloudDatabaseDependencyError(CloudDatabaseError):
    """Raised when the MySQL client dependency is unavailable."""


class CloudDatabaseConnectionError(CloudDatabaseError):
    """Raised when a cloud connection cannot be established safely."""


class CloudDatabaseSchemaError(CloudDatabaseError):
    """Raised when the cloud schema cannot support VisionTrack persistence."""


@dataclass(frozen=True)
class CloudDatabaseConfig:
    host: str
    port: int
    user: str
    password: str
    database: str
    connect_timeout_seconds: int = 10
    max_connect_attempts: int = 3
    retry_backoff_seconds: float = 1.0
    ssl_disabled: bool = False
    ssl_ca: str | None = None
    ssl_verify_cert: bool = True
    ssl_verify_identity: bool = True

    def validate(self) -> None:
        missing = [
            name
            for name, value in (
                ("CLOUD_DB_HOST", self.host),
                ("CLOUD_DB_NAME", self.database),
                ("CLOUD_DB_USER", self.user),
                ("CLOUD_DB_PASSWORD", self.password),
            )
            if not str(value).strip()
        ]
        if missing:
            raise CloudDatabaseConfigurationError(
                "Missing cloud database settings: " + ", ".join(missing)
            )
        if not 1 <= int(self.port) <= 65535:
            raise CloudDatabaseConfigurationError(
                "CLOUD_DB_PORT must be between 1 and 65535."
            )
        if int(self.connect_timeout_seconds) < 1:
            raise CloudDatabaseConfigurationError(
                "Cloud connection timeout must be at least one second."
            )
        if int(self.max_connect_attempts) < 1:
            raise CloudDatabaseConfigurationError(
                "Cloud connection attempts must be at least one."
            )
        if float(self.retry_backoff_seconds) < 0:
            raise CloudDatabaseConfigurationError(
                "Cloud retry backoff cannot be negative."
            )
        if self.ssl_verify_identity and not self.ssl_verify_cert:
            raise CloudDatabaseConfigurationError(
                "SSL identity verification requires certificate verification."
            )


@dataclass(frozen=True)
class CloudHealthReport:
    database: str
    server_version: str


@dataclass(frozen=True)
class CloudSchemaReport:
    missing_tables: tuple[str, ...]
    missing_columns: tuple[tuple[str, tuple[str, ...]], ...]

    @property
    def is_compatible(self) -> bool:
        return not self.missing_tables and not self.missing_columns

    def describe(self) -> str:
        problems: list[str] = []
        if self.missing_tables:
            problems.append("missing tables: " + ", ".join(self.missing_tables))
        for table, columns in self.missing_columns:
            problems.append(f"{table} missing columns: {', '.join(columns)}")
        return "; ".join(problems) or "schema compatible"


def _load_mysql_connector():
    try:
        import mysql.connector as mysql_connector
    except Exception as exc:  # pragma: no cover - environment dependent
        raise CloudDatabaseDependencyError(
            "mysql-connector-python is required for cloud persistence."
        ) from exc
    return mysql_connector


class CloudDatabaseConnectionFactory:
    """Create cloud MySQL connections with bounded, connection-only retries."""

    TRANSIENT_ERROR_CODES = frozenset({1040, 1205, 2002, 2003, 2006, 2013, 2055})

    def __init__(
        self,
        config: CloudDatabaseConfig,
        *,
        connector_loader: Callable[[], Any] = _load_mysql_connector,
        sleeper: Callable[[float], None] = time.sleep,
    ) -> None:
        config.validate()
        self.config = config
        self._connector_loader = connector_loader
        self._sleeper = sleeper

    def connect(self):
        connector = self._connector_loader()
        attempts = int(self.config.max_connect_attempts)

        for attempt in range(1, attempts + 1):
            try:
                return connector.connect(**self._connection_arguments())
            except Exception as exc:
                if attempt >= attempts or not self._is_transient(exc):
                    raise CloudDatabaseConnectionError(
                        "Cloud MySQL connection failed "
                        f"after {attempt} attempt(s)."
                    ) from exc
                delay = float(self.config.retry_backoff_seconds) * (2 ** (attempt - 1))
                if delay > 0:
                    self._sleeper(delay)

        raise CloudDatabaseConnectionError("Cloud MySQL connection failed.")

    def _connection_arguments(self) -> dict[str, Any]:
        arguments: dict[str, Any] = {
            "host": self.config.host,
            "port": int(self.config.port),
            "user": self.config.user,
            "password": self.config.password,
            "database": self.config.database,
            "autocommit": False,
            "connection_timeout": int(self.config.connect_timeout_seconds),
            "charset": "utf8mb4",
            "use_unicode": True,
            "ssl_disabled": bool(self.config.ssl_disabled),
        }
        if self.config.ssl_ca:
            arguments.update(
                {
                    "ssl_ca": self.config.ssl_ca,
                    "ssl_verify_cert": bool(self.config.ssl_verify_cert),
                    "ssl_verify_identity": bool(self.config.ssl_verify_identity),
                }
            )
        return arguments

    @classmethod
    def _is_transient(cls, error: Exception) -> bool:
        error_code = getattr(error, "errno", None)
        if error_code in cls.TRANSIENT_ERROR_CODES:
            return True
        return isinstance(error, (ConnectionError, TimeoutError, OSError))


class CloudVisionTrackRepository(VisionTrackRepository):
    """Cloud-specific repository with read-only readiness validation.

    Persistence operations are inherited from ``VisionTrackRepository`` so the
    local and cloud targets share one SQL contract. Only connection creation is
    retried; write transactions are never replayed automatically.
    """

    REQUIRED_SCHEMA: Mapping[str, frozenset[str]] = {
        "boxes": frozenset(
            {
                "box_code",
                "length_cm",
                "breadth_cm",
                "height_cm",
                "weight_kg",
                "volume_cm3",
            }
        ),
        "pallets": frozenset(
            {
                "pallet_code",
                "max_weight_kg",
                "max_volume_cm3",
                "max_height_cm",
            }
        ),
        "video_processing_runs": frozenset(
            {
                "run_id",
                "video_name",
                "input_video_path",
                "model_path",
                "device",
                "gpu_name",
                "source_width",
                "source_height",
                "source_fps",
                "video_duration_sec",
                "frame_skip",
                "process_width",
                "process_height",
                "yolo_imgsz",
                "processed_frames",
                "source_frames_covered",
                "total_processing_time_sec",
                "end_to_end_time_sec",
                "speed_vs_realtime",
                "realtime_factor",
                "status",
                "completed_at",
            }
        ),
        "video_events": frozenset(
            {
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
            }
        ),
        "video_final_boxes": frozenset(
            {
                "run_id",
                "box_code",
                "status",
                "pallet_code",
                "slot_no",
                "row_no",
                "column_no",
                "layer_no",
                "length_cm",
                "width_cm",
                "height_cm",
                "weight_kg",
                "volume_cm3",
            }
        ),
        "video_final_pallets": frozenset(
            {
                "run_id",
                "pallet_code",
                "active",
                "status",
                "box_count",
                "max_boxes",
                "total_weight_kg",
                "max_weight_kg",
                "weight_utilization_percent",
                "total_volume_cm3",
                "max_volume_cm3",
                "volume_efficiency_percent",
                "max_stack_height_cm",
                "max_height_cm",
                "height_utilization_percent",
                "bottleneck_utilization_percent",
                "reasons",
            }
        ),
        "video_processing_metrics": frozenset(
            {"run_id", "metric_name", "metric_value"}
        ),
    }

    def __init__(self, connection_factory: CloudDatabaseConnectionFactory) -> None:
        super().__init__(connection_factory)
        self.cloud_connection_factory = connection_factory

    @classmethod
    def from_config(
        cls,
        config: CloudDatabaseConfig,
        *,
        connector_loader: Callable[[], Any] = _load_mysql_connector,
        sleeper: Callable[[float], None] = time.sleep,
    ) -> CloudVisionTrackRepository:
        return cls(
            CloudDatabaseConnectionFactory(
                config,
                connector_loader=connector_loader,
                sleeper=sleeper,
            )
        )

    def check_connection(self) -> CloudHealthReport:
        """Perform a read-only connection check without changing cloud state."""
        with closing(self.cloud_connection_factory.connect()) as connection:
            with closing(connection.cursor()) as cursor:
                cursor.execute("SELECT DATABASE(), VERSION()")
                row = cursor.fetchone()
        if not row or not row[0]:
            raise CloudDatabaseConnectionError(
                "Cloud connection succeeded but no database is selected."
            )
        return CloudHealthReport(
            database=str(row[0]),
            server_version=str(row[1] or "unknown"),
        )

    def inspect_schema(self) -> CloudSchemaReport:
        """Inspect required tables and columns using information_schema only."""
        table_names = tuple(sorted(self.REQUIRED_SCHEMA))
        placeholders = ", ".join(["%s"] * len(table_names))
        query = f"""
            SELECT TABLE_NAME, COLUMN_NAME
            FROM information_schema.COLUMNS
            WHERE TABLE_SCHEMA = %s
              AND TABLE_NAME IN ({placeholders})
        """
        parameters = (
            self.cloud_connection_factory.config.database,
            *table_names,
        )

        with closing(self.cloud_connection_factory.connect()) as connection:
            with closing(connection.cursor()) as cursor:
                cursor.execute(query, parameters)
                rows = cursor.fetchall()

        actual: dict[str, set[str]] = {}
        for table_name, column_name in rows:
            actual.setdefault(str(table_name), set()).add(str(column_name))

        missing_tables = tuple(
            table for table in table_names if table not in actual
        )
        missing_columns = tuple(
            (
                table,
                tuple(sorted(required_columns - actual.get(table, set()))),
            )
            for table, required_columns in sorted(self.REQUIRED_SCHEMA.items())
            if table in actual and required_columns - actual.get(table, set())
        )
        return CloudSchemaReport(
            missing_tables=missing_tables,
            missing_columns=missing_columns,
        )

    def validate_schema(self) -> CloudSchemaReport:
        """Fail clearly when the cloud schema is unsafe for persistence."""
        report = self.inspect_schema()
        if not report.is_compatible:
            raise CloudDatabaseSchemaError(
                "Cloud database schema is incompatible: " + report.describe()
            )
        return report
