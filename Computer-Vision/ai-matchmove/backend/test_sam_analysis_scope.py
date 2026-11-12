"""Verify SAM frame selection is independent of sparse-solve frame count."""
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import patch
from uuid import uuid4

from . import main


def test_foreground_budget_is_independent_of_solve_limit():
    with TemporaryDirectory() as temp:
        folder = Path(temp)
        sequence = {"frame_details": [{"frame_number": i} for i in range(12)]}
        captured = []

        def fake_create_foreground(sequence_id, request, progress):
            captured.append(request.discover_frame_numbers)
            return {"masked_frames": 0}

        def fake_submit(folder, operation, work, **options):
            work(Path(temp) / "dummy", lambda *args: None)
            return {"job_id": uuid4().hex, "status": "done"}

        with patch.object(main, "sequence_folder", return_value=folder), \
             patch.object(main, "read_json", return_value=sequence), \
             patch.object(main, "_create_foreground", side_effect=fake_create_foreground), \
             patch.object(main.jobs, "submit", side_effect=fake_submit):
            main.create_job(uuid4(), main.JobRequest(
                operation="foreground", max_frames=3, sam_analysis_frames=9,
                foreground=main.ForegroundRequest(discover=True),
            ))
            main.create_job(uuid4(), main.JobRequest(
                operation="foreground", max_frames=3,
                foreground=main.ForegroundRequest(discover=True),
            ))
            main.create_job(uuid4(), main.JobRequest(
                operation="foreground", max_frames=3, sam_analysis_frames=9,
                foreground=main.ForegroundRequest(discover=True, discover_frame_numbers=[4, 6]),
            ))

        assert len(captured[0]) == 9
        assert len(captured[1]) == 3
        assert captured[0][0] == 0 and captured[0][-1] == 11
        assert captured[1][0] == 0 and captured[1][-1] == 11
        assert captured[2] == [4, 6]


def test_budget_validation():
    from pydantic import ValidationError
    for value in (0, -1, 121):
        try:
            main.JobRequest(sam_analysis_frames=value)
        except ValidationError:
            pass
        else:
            raise AssertionError(f"Budget {value} should have been rejected")
