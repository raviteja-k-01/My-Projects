from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class DatabaseConfig:
    host: str
    port: int
    user: str
    password: str
    database: str


class DatabaseConnectionFactory:
    def __init__(self, config: DatabaseConfig):
        self.config = config

    def connect(self):
        try:
            import mysql.connector as mysql_connector
        except Exception as exc:  # pragma: no cover
            raise RuntimeError(
                "mysql-connector-python is required for database persistence."
            ) from exc

        return mysql_connector.connect(
            host=self.config.host,
            port=self.config.port,
            user=self.config.user,
            password=self.config.password,
            database=self.config.database,
            autocommit=False,
        )
