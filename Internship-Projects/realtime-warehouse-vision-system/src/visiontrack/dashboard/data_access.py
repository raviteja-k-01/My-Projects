from __future__ import annotations

from dataclasses import dataclass
import re
from threading import RLock
from typing import Any

import pandas as pd


_READ_ONLY_STATEMENT = re.compile(
    r"^\s*(?:SELECT|SHOW|DESCRIBE|DESC|EXPLAIN)\b",
    flags=re.IGNORECASE,
)
_READ_SIDE_EFFECT = re.compile(
    r"\b(?:FOR\s+UPDATE|INTO\s+(?:OUTFILE|DUMPFILE))\b",
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

    def is_ready(self, database_target: str = "local") -> bool:
        """Return whether the selected profile has enough values to connect.

        A local MySQL installation may intentionally use an empty password.
        Cloud profiles must always provide one.
        """
        target = database_target.strip().lower()
        required = (self.host.strip(), self.user.strip(), self.database.strip())
        return bool(all(required) and (target != "cloud" or self.password))


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
        self._validate_read_only(sql)

        with self._lock:
            try:
                return self._execute(sql, params)
            except Exception as exc:
                if not self._is_transient(exc):
                    raise
                self._close_unlocked()
                return self._execute(sql, params)

    @staticmethod
    def _validate_read_only(sql: str) -> None:
        statement = str(sql).strip()
        without_trailing_terminator = statement[:-1].rstrip() if statement.endswith(";") else statement
        if (
            not _READ_ONLY_STATEMENT.match(statement)
            or ";" in without_trailing_terminator
            or _READ_SIDE_EFFECT.search(statement)
        ):
            raise ValueError("Dashboard database access is read-only.")

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
