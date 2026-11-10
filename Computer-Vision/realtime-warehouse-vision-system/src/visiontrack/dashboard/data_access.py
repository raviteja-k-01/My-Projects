from __future__ import annotations

from dataclasses import dataclass
import re
from threading import RLock
from typing import Any

import pandas as pd


_READ_ONLY_STATEMENT = re.compile(
    r"^\s*(?:SELECT|SHOW|DESCRIBE|DESC|EXPLAIN|WITH)\b",
    flags=re.IGNORECASE,
)
_TRANSIENT_MYSQL_ERRORS = frozenset(
    {1040, 1205, 2002, 2003, 2006, 2013, 2055}
)


@dataclass(frozen=True)
class DashboardDatabaseConfig:
    host: str
    port: int
    user: str
    password: str
    database: str
    connection_timeout_seconds: int = 10


class MySQLReadClient:
    """Thread-safe, reconnecting MySQL reader for the Streamlit dashboard.

    A Streamlit rerun executes the complete dashboard file. Keeping this object
    in ``st.cache_resource`` preserves one connection across those reruns and
    avoids a new cloud TLS/authentication handshake for every SELECT.
    """

    def __init__(self, config: DashboardDatabaseConfig) -> None:
        self.config = config
        self._connection: Any | None = None
        self._lock = RLock()

    def query(self, sql: str, params: tuple[Any, ...] = ()) -> pd.DataFrame:
        if not _READ_ONLY_STATEMENT.match(sql):
            raise ValueError("Dashboard database access is read-only.")

        with self._lock:
            try:
                return self._execute(sql, params)
            except Exception as exc:
                if not self._is_transient(exc):
                    raise
                self._close_unlocked()
                return self._execute(sql, params)

    def close(self) -> None:
        with self._lock:
            self._close_unlocked()

    def _execute(self, sql: str, params: tuple[Any, ...]) -> pd.DataFrame:
        connection = self._connected_session()
        cursor = connection.cursor(dictionary=True)
        try:
            cursor.execute(sql, params)
            rows = cursor.fetchall()
            columns = (
                [description[0] for description in cursor.description]
                if cursor.description
                else []
            )
            return pd.DataFrame(rows, columns=columns)
        finally:
            cursor.close()

    def _connected_session(self):
        if self._connection is not None:
            try:
                if self._connection.is_connected():
                    return self._connection
            except Exception:
                self._close_unlocked()

        try:
            import mysql.connector as mysql_connector
        except Exception as exc:  # pragma: no cover - dependency failure
            raise RuntimeError(
                "mysql-connector-python is required for the dashboard."
            ) from exc

        self._connection = mysql_connector.connect(
            host=self.config.host,
            port=int(self.config.port),
            user=self.config.user,
            password=self.config.password,
            database=self.config.database,
            connection_timeout=int(self.config.connection_timeout_seconds),
            autocommit=True,
            charset="utf8mb4",
            use_unicode=True,
        )
        return self._connection

    def _close_unlocked(self) -> None:
        connection, self._connection = self._connection, None
        if connection is None:
            return
        try:
            connection.close()
        except Exception:
            pass

    @staticmethod
    def _is_transient(exc: Exception) -> bool:
        errno = getattr(exc, "errno", None)
        return errno in _TRANSIENT_MYSQL_ERRORS
