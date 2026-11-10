from __future__ import annotations

import sys
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from visiontrack.dashboard.launcher import (
    dashboard_app_path,
    dashboard_command,
    dashboard_environment,
)


class DashboardIntegrationTests(unittest.TestCase):
    def test_packaged_dashboard_preserves_the_ui_contract(self) -> None:
        source = dashboard_app_path().read_text(encoding="utf-8")
        expected_ui_text = (
            "VisionTrack Warehouse Dashboard",
            "Live Operations",
            "Order Control Center",
            "Exceptions & Events",
            "Refresh dashboard",
            "VISIONTRACK LIVE OPERATIONS",
        )
        for text in expected_ui_text:
            self.assertIn(text, source)

    def test_database_profile_is_mapped_to_dashboard_contract(self) -> None:
        settings = SimpleNamespace(
            database=SimpleNamespace(
                host="cloud.example",
                port=3307,
                user="dashboard_user",
                password="private-value",
                name="visiontrack_cloud",
            )
        )
        with patch(
            "visiontrack.config.settings.load_settings",
            return_value=settings,
        ) as load_settings:
            environment = dashboard_environment("cloud")

        load_settings.assert_called_once_with(database_target="cloud")
        self.assertEqual(environment["DB_HOST"], "cloud.example")
        self.assertEqual(environment["DB_PORT"], "3307")
        self.assertEqual(environment["DB_NAME"], "visiontrack_cloud")
        self.assertEqual(environment["VISIONTRACK_DASHBOARD_DATABASE"], "cloud")

    def test_streamlit_command_uses_packaged_app_and_current_python(self) -> None:
        command = dashboard_command(
            bind_address="127.0.0.1",
            port=8601,
            headless=True,
        )

        self.assertEqual(command[:4], [sys.executable, "-m", "streamlit", "run"])
        self.assertEqual(command[4], str(dashboard_app_path()))
        self.assertIn("8601", command)
        self.assertIn("true", command)


if __name__ == "__main__":
    unittest.main()
