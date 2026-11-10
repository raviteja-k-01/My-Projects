from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml

try:
    from dotenv import load_dotenv
except Exception:  # pragma: no cover
    load_dotenv = None

PROJECT_ROOT = Path(__file__).resolve().parents[3]

if load_dotenv is not None:
    load_dotenv(PROJECT_ROOT / ".env")


def load_yaml(path: str | Path) -> dict[str, Any]:
    with Path(path).open("r", encoding="utf-8") as handle:
        return yaml.safe_load(handle) or {}


def project_path(value: str | Path) -> Path:
    path = Path(value)
    return path if path.is_absolute() else PROJECT_ROOT / path


def env_bool(key: str, default: bool) -> bool:
    value = os.getenv(key)
    if value is None:
        return default
    return value.strip().lower() in {"1", "true", "yes", "on"}


def env_int(key: str, default: int) -> int:
    try:
        return int(os.getenv(key, str(default)))
    except ValueError:
        return default


def env_str(key: str, default: str = "") -> str:
    return os.getenv(key, default).strip()


@dataclass(frozen=True)
class DatabaseSettings:
    enabled: bool
    target: str
    host: str
    port: int
    name: str
    user: str
    password: str


@dataclass(frozen=True)
class AppSettings:
    app: dict[str, Any]
    models: dict[str, Any]
    zones: dict[str, Any]
    database: DatabaseSettings


def _load_database(database_config: dict[str, Any], target: str | None) -> DatabaseSettings:
    selected = (target or env_str("DB_TARGET", "local")).lower()
    if selected not in {"off", "local", "cloud"}:
        raise ValueError("DB target must be off, local, or cloud")

    prefix = "CLOUD_DB" if selected == "cloud" else "LOCAL_DB"
    enabled = selected != "off" and env_bool(
        database_config.get("enabled_env", "DB_ENABLED"), True
    )

    def value(suffix: str, default: str = "") -> str:
        profile_value = env_str(f"{prefix}_{suffix}", "")
        if profile_value or selected != "local":
            return profile_value or default
        return env_str(f"DB_{suffix}", default)

    port_key = f"{prefix}_PORT"
    port = env_int(port_key, 3306)
    if selected == "local" and os.getenv(port_key) is None:
        port = env_int("DB_PORT", 3306)

    return DatabaseSettings(
        enabled=enabled,
        target=selected,
        host=value("HOST"),
        port=port,
        name=value("NAME"),
        user=value("USER"),
        password=value("PASSWORD"),
    )


def load_settings(
    app_config: str | Path = "configs/app.yaml",
    models_config: str | Path = "configs/models.yaml",
    zones_config: str | Path = "configs/zones.yaml",
    database_target: str | None = None,
) -> AppSettings:
    app = load_yaml(project_path(app_config))
    return AppSettings(
        app=app,
        models=load_yaml(project_path(models_config)),
        zones=load_yaml(project_path(zones_config)),
        database=_load_database(app.get("database", {}), database_target),
    )
