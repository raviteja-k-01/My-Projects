from __future__ import annotations

import argparse
from collections.abc import Sequence
import sys


def create_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="visiontrack",
        description="VisionTrack video inference and warehouse dashboard.",
    )
    commands = parser.add_subparsers(dest="command", required=True)

    commands.add_parser(
        "video",
        add_help=False,
        help="Run the existing video inference CLI.",
    )

    monitor = commands.add_parser("monitor", help="Open the VisionTrack 2.0 PostgreSQL monitor.")
    monitor.add_argument("--port", type=int, default=8501)

    dashboard = commands.add_parser(
        "dashboard",
        help="Open Sarfaraz's read-only Streamlit dashboard.",
    )
    dashboard.add_argument(
        "--database",
        choices=("local", "cloud"),
        default="cloud",
        help="Database profile exposed to the unchanged dashboard UI.",
    )
    dashboard.add_argument(
        "--bind-address",
        default="127.0.0.1",
        help="Address on which Streamlit listens.",
    )
    dashboard.add_argument(
        "--port",
        type=int,
        default=8501,
        help="Streamlit server port.",
    )
    dashboard.add_argument(
        "--headless",
        action="store_true",
        help="Start without opening a browser automatically.",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = create_parser()
    arguments, passthrough = parser.parse_known_args(argv)

    if arguments.command == "monitor":
        if passthrough:
            parser.error("unrecognized monitor arguments: " + " ".join(passthrough))
        from pathlib import Path
        import subprocess

        app = Path(__file__).with_name("dashboard") / "monitor_app.py"
        return subprocess.call([
            sys.executable, "-m", "streamlit", "run", str(app),
            "--server.address", "127.0.0.1", "--server.port", str(arguments.port),
            "--browser.gatherUsageStats", "false", "--client.toolbarMode", "minimal",
            "--theme.base", "dark", "--theme.primaryColor", "#35d5c0",
            "--theme.backgroundColor", "#0b1320", "--theme.secondaryBackgroundColor", "#162438",
            "--theme.textColor", "#edf4fc", "--theme.baseFontSize", "17",
            "--theme.sidebar.backgroundColor", "#111d30",
            "--theme.sidebar.secondaryBackgroundColor", "#1c3048",
        ])

    if arguments.command == "dashboard":
        if passthrough:
            parser.error(
                "unrecognized dashboard arguments: " + " ".join(passthrough)
            )
        from visiontrack.dashboard import launch_dashboard

        return launch_dashboard(
            database_target=arguments.database,
            bind_address=arguments.bind_address,
            port=arguments.port,
            headless=arguments.headless,
        )

    from visiontrack.cli import main as video_main

    previous_argv = sys.argv
    try:
        sys.argv = ["visiontrack video", *passthrough]
        video_main()
    finally:
        sys.argv = previous_argv
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
