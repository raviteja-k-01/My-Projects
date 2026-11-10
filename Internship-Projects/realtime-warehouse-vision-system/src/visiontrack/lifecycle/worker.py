from __future__ import annotations

from collections.abc import Callable
import logging
import time

from visiontrack.lifecycle.models import BatchStats
from visiontrack.lifecycle.repository import (
    ConnectionFactory,
    MySQLLifecycleRepository,
)
from visiontrack.lifecycle.service import LifecycleEventProcessor


class LifecycleWorker:
    """Run the processor with one reusable database connection."""

    def __init__(
        self,
        connection_factory: ConnectionFactory,
        processor: LifecycleEventProcessor,
        *,
        logger: logging.Logger | None = None,
        sleeper: Callable[[float], None] = time.sleep,
    ) -> None:
        self.connection_factory = connection_factory
        self.processor = processor
        self.logger = logger or logging.getLogger(__name__)
        self.sleeper = sleeper

    def run_once(self, *, validate_schema: bool = True) -> BatchStats:
        connection = self.connection_factory.connect()
        try:
            repository = MySQLLifecycleRepository(connection)
            if validate_schema:
                repository.validate_schema()
            return self.process_batch(repository)
        finally:
            self._close(connection)

    def run_forever(
        self,
        stop_requested: Callable[[], bool],
        *,
        validate_schema: bool = True,
    ) -> BatchStats:
        total = BatchStats()
        connection = None
        repository = None
        reconnect_delay = 1.0
        schema_checked = False

        while not stop_requested():
            try:
                if connection is None:
                    connection = self.connection_factory.connect()
                    repository = MySQLLifecycleRepository(connection)
                    schema_checked = False
                self._ping(connection)
                if validate_schema and not schema_checked:
                    repository.validate_schema()
                    schema_checked = True

                batch = self.process_batch(repository)
                total.merge(batch)
                reconnect_delay = 1.0
                if batch.fetched == 0 and not stop_requested():
                    self.sleeper(
                        self.processor.settings.poll_ms / 1000.0
                    )
            except Exception:
                self.logger.exception(
                    "Lifecycle worker database cycle failed; reconnecting"
                )
                self._close(connection)
                connection = None
                repository = None
                schema_checked = False
                if not stop_requested():
                    self.sleeper(reconnect_delay)
                    reconnect_delay = min(reconnect_delay * 2.0, 30.0)

        self._close(connection)
        return total

    def process_batch(
        self,
        repository: MySQLLifecycleRepository,
    ) -> BatchStats:
        stats = BatchStats()
        for _ in range(self.processor.settings.batch_size):
            result = self.processor.process_next(repository)
            if result is None:
                break
            stats.add(result)
            log = (
                self.logger.error
                if result.outcome.value == "ERROR"
                else self.logger.info
            )
            log(
                "video_event_id=%s outcome=%s reason=%s box_rows=%s "
                "order_rows=%s",
                result.video_event_id,
                result.outcome.value,
                result.reason,
                result.inserted_box_rows,
                result.inserted_order_rows,
            )
        return stats

    def retry_errors(self, limit: int, *, validate_schema: bool = True) -> int:
        connection = self.connection_factory.connect()
        try:
            repository = MySQLLifecycleRepository(connection)
            if validate_schema:
                repository.validate_schema()
            return repository.retry_errors(limit)
        finally:
            self._close(connection)

    @staticmethod
    def _ping(connection) -> None:
        ping = getattr(connection, "ping", None)
        if callable(ping):
            ping(reconnect=True, attempts=2, delay=1)

    @staticmethod
    def _close(connection) -> None:
        if connection is None:
            return
        try:
            connection.close()
        except Exception:
            pass
