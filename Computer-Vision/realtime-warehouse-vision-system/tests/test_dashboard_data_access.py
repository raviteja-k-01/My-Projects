from __future__ import annotations

import unittest

from visiontrack.dashboard.data_access import (
    DashboardDatabaseConfig,
    MySQLReadClient,
)


class _Cursor:
    description = (("value",),)

    def __init__(self) -> None:
        self.closed = False

    def execute(self, sql: str, params: tuple) -> None:
        self.sql = sql
        self.params = params

    def fetchall(self) -> list[dict[str, int]]:
        return [{"value": 42}]

    def close(self) -> None:
        self.closed = True


class _Connection:
    def __init__(self) -> None:
        self.cursor_count = 0
        self.cursors: list[_Cursor] = []

    def is_connected(self) -> bool:
        return True

    def cursor(self, *, dictionary: bool) -> _Cursor:
        self.assert_dictionary = dictionary
        self.cursor_count += 1
        cursor = _Cursor()
        self.cursors.append(cursor)
        return cursor


class DashboardDataAccessTests(unittest.TestCase):
    def setUp(self) -> None:
        self.client = MySQLReadClient(
            DashboardDatabaseConfig(
                host="localhost",
                port=3306,
                user="reader",
                password="private",
                database="visiontrack",
            )
        )

    def test_reuses_one_connection_for_multiple_queries(self) -> None:
        connection = _Connection()
        self.client._connection = connection

        first = self.client.query("SELECT 42 AS value")
        second = self.client.query("SELECT 42 AS value")

        self.assertEqual(connection.cursor_count, 2)
        self.assertEqual(first.iloc[0]["value"], 42)
        self.assertEqual(second.iloc[0]["value"], 42)
        self.assertTrue(all(cursor.closed for cursor in connection.cursors))

    def test_rejects_mutating_sql(self) -> None:
        with self.assertRaisesRegex(ValueError, "read-only"):
            self.client.query("DELETE FROM boxes")


if __name__ == "__main__":
    unittest.main()
