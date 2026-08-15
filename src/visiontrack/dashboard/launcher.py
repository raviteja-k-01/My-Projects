from __future__ import annotations

import os
from pathlib import Path
import subprocess
import sys
from typing import Mapping


PROJECT_ROOT = Path(__file__).resolve().parents[3]
DASHBOARD_APP = Path(__file__).with_name("app.py")


def dashboard_app_path() -> Path:
    """Return the packaged Streamlit app without importing its UI module."""
    if not DASHBOARD_APP.is_file():
        raise FileNotFoundError(f"Dashboard app was not found: {DASHBOARD_APP}")
    return DASHBOARD_APP


def dashboard_environment(database_target: str) -> dict[str, str]:
    """Map a VisionTrack database profile to the dashboard's DB_* contract."""
    from visiontrack.config.settings import load_settings

    target = database_target.strip().lower()
    if target not in {"local", "cloud"}:
        raise ValueError("Dashboard database target must be local or cloud.")

    database = load_settings(database_target=target).database
    environment = os.environ.copy()
    environment.update(
        {
            "DB_HOST": database.host,
            "DB_PORT": str(database.port),
            "DB_USER": database.user,
            "DB_PASSWORD": database.password,
            "DB_NAME": database.name,
            "VISIONTRACK_DASHBOARD_DATABASE": target,
        }
    )
    return environment


def dashboard_command(
    *,
    bind_address: str = "127.0.0.1",
    port: int = 8501,
    headless: bool = False,
) -> list[str]:
    """Build the isolated Streamlit command used by the main application."""
    return [
        sys.executable,
        "-m",
        "streamlit",
        "run",
        str(dashboard_app_path()),
        "--server.address",
        bind_address,
        "--server.port",
        str(port),
        "--server.headless",
        str(headless).lower(),
        "--browser.gatherUsageStats",
        "false",
    ]


def launch_dashboard(
    *,
    database_target: str = "cloud",
    bind_address: str = "127.0.0.1",
    port: int = 8501,
    headless: bool = False,
    extra_environment: Mapping[str, str] | None = None,
) -> int:
    """Start Sarfaraz's unchanged dashboard UI in a separate process."""
    environment = dashboard_environment(database_target)
    if extra_environment:
        environment.update(extra_environment)

    return subprocess.call(
        dashboard_command(
            bind_address=bind_address,
            port=port,
            headless=headless,
        ),
        cwd=PROJECT_ROOT,
        env=environment,
    )
