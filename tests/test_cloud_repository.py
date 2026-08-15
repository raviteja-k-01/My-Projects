from __future__ import annotations

import unittest
from types import SimpleNamespace

from visiontrack.cli import build_repository
from visiontrack.database.cloud_repository import (
    CloudDatabaseConfig,
    CloudDatabaseConfigurationError,
    CloudDatabaseConnectionError,
    CloudDatabaseConnectionFactory,
    CloudDatabaseSchemaError,
    CloudVisionTrackRepository,
)
from visiontrack.database.repository import VisionTrackRepository


def cloud_config(**overrides) -> CloudDatabaseConfig:
    values = {
        "host": "mysql.example.test",
        "port": 3306,
        "user": "visiontrack",
        "password": "test-password",
        "database": "visiontrack_cloud",
        "connect_timeout_seconds": 5,
        "max_connect_attempts": 3,
        "retry_backoff_seconds": 0.25,
    }
    values.update(overrides)
    return CloudDatabaseConfig(**values)


class FakeMySQLError(Exception):
    def __init__(self, message: str, errno: int) -> None:
        super().__init__(message)
        self.errno = errno


class FakeCursor:
    def __init__(self, *, one=None, rows=()) -> None:
        self.one = one
        self.rows = list(rows)
        self.query = None
        self.parameters = None
        self.closed = False

    def execute(self, query, parameters=None) -> None:
        self.query = query
        self.parameters = parameters

    def fetchone(self):
        return self.one

    def fetchall(self):
        return list(self.rows)

    def close(self) -> None:
        self.closed = True


class FakeConnection:
    def __init__(self, cursor: FakeCursor) -> None:
        self._cursor = cursor
        self.closed = False

    def cursor(self) -> FakeCursor:
        return self._cursor

    def close(self) -> None:
        self.closed = True


class FakeConnector:
    def __init__(self, *outcomes) -> None:
        self.outcomes = list(outcomes)
        self.calls = []

    def connect(self, **arguments):
        self.calls.append(arguments)
        outcome = self.outcomes.pop(0)
        if isinstance(outcome, Exception):
            raise outcome
        return outcome


class CloudConfigurationTests(unittest.TestCase):
    def test_required_cloud_values_are_validated(self) -> None:
        with self.assertRaisesRegex(
            CloudDatabaseConfigurationError,
            "CLOUD_DB_PASSWORD",
        ):
            CloudDatabaseConnectionFactory(cloud_config(password=""))

    def test_ssl_identity_verification_requires_certificate_verification(self) -> None:
        with self.assertRaisesRegex(
            CloudDatabaseConfigurationError,
            "identity verification",
        ):
            CloudDatabaseConnectionFactory(
                cloud_config(ssl_verify_cert=False, ssl_verify_identity=True)
            )


class CloudConnectionFactoryTests(unittest.TestCase):
    def test_connection_is_lazy_and_uses_safe_arguments(self) -> None:
        connection = FakeConnection(FakeCursor())
        connector = FakeConnector(connection)
        factory = CloudDatabaseConnectionFactory(
            cloud_config(),
            connector_loader=lambda: connector,
        )

        self.assertEqual(connector.calls, [])
        self.assertIs(factory.connect(), connection)
        self.assertEqual(len(connector.calls), 1)
        arguments = connector.calls[0]
        self.assertEqual(arguments["database"], "visiontrack_cloud")
        self.assertEqual(arguments["connection_timeout"], 5)
        self.assertFalse(arguments["autocommit"])
        self.assertFalse(arguments["ssl_disabled"])

    def test_transient_connection_failure_is_retried_with_backoff(self) -> None:
        connection = FakeConnection(FakeCursor())
        connector = FakeConnector(
            FakeMySQLError("temporary outage", errno=2003),
            connection,
        )
        delays = []
        factory = CloudDatabaseConnectionFactory(
            cloud_config(),
            connector_loader=lambda: connector,
            sleeper=delays.append,
        )

        self.assertIs(factory.connect(), connection)
        self.assertEqual(len(connector.calls), 2)
        self.assertEqual(delays, [0.25])

    def test_permanent_connection_failure_is_not_retried(self) -> None:
        connector = FakeConnector(
            FakeMySQLError("access denied for test-password", errno=1045)
        )
        factory = CloudDatabaseConnectionFactory(
            cloud_config(),
            connector_loader=lambda: connector,
            sleeper=lambda _delay: None,
        )

        with self.assertRaises(CloudDatabaseConnectionError) as context:
            factory.connect()

        self.assertEqual(len(connector.calls), 1)
        self.assertNotIn("test-password", str(context.exception))


class CloudRepositoryReadinessTests(unittest.TestCase):
    def test_connection_check_is_read_only(self) -> None:
        cursor = FakeCursor(one=("visiontrack_cloud", "8.0.40"))
        connection = FakeConnection(cursor)
        connector = FakeConnector(connection)
        repository = CloudVisionTrackRepository.from_config(
            cloud_config(),
            connector_loader=lambda: connector,
        )

        report = repository.check_connection()

        self.assertEqual(report.database, "visiontrack_cloud")
        self.assertEqual(report.server_version, "8.0.40")
        self.assertEqual(cursor.query, "SELECT DATABASE(), VERSION()")
        self.assertTrue(cursor.closed)
        self.assertTrue(connection.closed)

    def test_complete_schema_is_reported_as_compatible(self) -> None:
        rows = [
            (table, column)
            for table, columns in CloudVisionTrackRepository.REQUIRED_SCHEMA.items()
            for column in columns
        ]
        cursor = FakeCursor(rows=rows)
        connector = FakeConnector(FakeConnection(cursor))
        repository = CloudVisionTrackRepository.from_config(
            cloud_config(),
            connector_loader=lambda: connector,
        )

        report = repository.inspect_schema()

        self.assertTrue(report.is_compatible)
        self.assertEqual(report.describe(), "schema compatible")
        self.assertIn("information_schema.COLUMNS", cursor.query)
        self.assertEqual(cursor.parameters[0], "visiontrack_cloud")

    def test_incompatible_schema_fails_before_persistence(self) -> None:
        cursor = FakeCursor(rows=[("boxes", "box_code")])
        connector = FakeConnector(FakeConnection(cursor))
        repository = CloudVisionTrackRepository.from_config(
            cloud_config(),
            connector_loader=lambda: connector,
        )

        with self.assertRaises(CloudDatabaseSchemaError) as context:
            repository.validate_schema()

        message = str(context.exception)
        self.assertIn("missing tables", message)
        self.assertIn("boxes missing columns", message)


class CloudCliRoutingTests(unittest.TestCase):
    def test_cli_routes_each_database_target(self) -> None:
        off = build_repository(
            SimpleNamespace(database=SimpleNamespace(enabled=False))
        )
        local = build_repository(
            SimpleNamespace(
                database=SimpleNamespace(
                    enabled=True,
                    target="local",
                    host="localhost",
                    port=3306,
                    name="visiontrack_local",
                    user="root",
                    password="",
                )
            )
        )
        cloud = build_repository(
            SimpleNamespace(
                database=SimpleNamespace(
                    enabled=True,
                    target="cloud",
                    host="mysql.example.test",
                    port=3306,
                    name="visiontrack_cloud",
                    user="visiontrack",
                    password="test-password",
                )
            )
        )

        self.assertIsNone(off)
        self.assertIs(type(local), VisionTrackRepository)
        self.assertIsInstance(cloud, CloudVisionTrackRepository)


if __name__ == "__main__":
    unittest.main()
