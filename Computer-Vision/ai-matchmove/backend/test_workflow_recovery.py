"""Regression tests for inexpensive mask review and persistent processing state."""
from io import BytesIO
from pathlib import Path
from tempfile import TemporaryDirectory
from threading import Event
from time import monotonic, sleep
from unittest.mock import patch

import numpy as np
from fastapi.testclient import TestClient
from PIL import Image

from . import main
from . import jobs
from .reconstruction import ReconstructionError
from .sequence import read_json, write_json


def png(width=64, height=40):
    stream = BytesIO()
    Image.new("RGB", (width, height), (100, 120, 140)).save(stream, format="PNG")
    return stream.getvalue()


class FakeSam:
    device = "cpu"
    input_size = 512

    def segment(self, image, prompts):
        masks = []
        for prompt in prompts:
            mask = np.zeros(image.shape[:2], dtype=bool)
            if prompt.box[0] < 25:
                mask[7:19, 7:21] = True
            else:
                mask[20:34, 39:56] = True
            masks.append((mask, 0.91))
        return masks


def _setup(client):
    response = client.post("/upload/", files=[
        ("files", ("shot_0001.png", png(), "image/png")),
        ("files", ("shot_0002.png", png(), "image/png")),
        ("files", ("shot_0003.png", png(), "image/png")),
    ])
    assert response.status_code == 200, response.text
    return response.json()


def test_review_individual_sam_regions_without_reinference_and_restore_after_reload():
    with TemporaryDirectory() as temp, patch.object(main, "DATA_DIR", Path(temp)):
        client = TestClient(main.app)
        sequence = _setup(client)
        root = f"/sequences/{sequence['sequence_id']}"
        with patch("backend.foreground.get_sam", return_value=FakeSam()) as load_sam:
            created = client.post(root + "/foreground", json={"prompts": [
                {"frame_number": 2, "box": [5, 5, 22, 22], "decision": "exclude"},
                {"frame_number": 2, "box": [38, 18, 59, 36], "decision": "exclude"},
            ]})
        assert created.status_code == 200, created.text
        old = created.json()
        assert old["masked_frames"] == 1
        load_sam.assert_called_once()

        # Reject one bad proposal without repeating SAM inference.
        reviewed = client.post(old["manifest_url"] + "/review", json={
            "decisions": [{"frame_number": 2, "region_index": 0, "decision": "keep"}]
        })
        assert reviewed.status_code == 200, reviewed.text
        new = reviewed.json()
        assert new["mask_set_id"] != old["mask_set_id"]
        assert new["reviewed_regions"] == 1
        assert new["frames"][1]["regions"][0]["review_decision"] == "keep"
        assert new["frames"][1]["regions"][1]["excluded"] is True

        # Bytes of original mask and manifest remain untouched.
        assert client.get(old["manifest_url"]).json()["frames"][1]["regions"][0]["excluded"]
        data = client.get(new["frames"][1]["mask_url"])
        assert data.status_code == 200
        mask = np.asarray(Image.open(BytesIO(data.content)).convert("L"))
        assert mask[13, 13] == 0  # kept region
        assert mask[25, 45] == 255  # other region still excluded

        # Second review can reverse a decision with stored masks; no rerun.
        again = client.post(new["manifest_url"] + "/review", json={
            "decisions": [{"frame_number": 2, "region_index": 1, "decision": "keep"}]
        })
        assert again.status_code == 200, again.text
        assert again.json()["masked_frames"] == 0

        masks = client.get(root + "/foreground").json()["mask_sets"]
        assert masks[0]["mask_set_id"] == again.json()["mask_set_id"]
        assert len(masks) == 3


def test_legacy_mask_set_supports_single_region_review_and_rejects_multiregion_ambiguity():
    with TemporaryDirectory() as temp, patch.object(main, "DATA_DIR", Path(temp)):
        client = TestClient(main.app)
        sequence = _setup(client)
        root = f"/sequences/{sequence['sequence_id']}"
        with patch("backend.foreground.get_sam", return_value=FakeSam()):
            created = client.post(root + "/foreground", json={"prompts": [
                {"frame_number": 1, "box": [5, 5, 22, 22], "decision": "exclude"},
                {"frame_number": 2, "box": [5, 5, 22, 22], "decision": "exclude"},
                {"frame_number": 2, "box": [38, 18, 59, 36], "decision": "exclude"},
            ]})
        assert created.status_code == 200
        existing = created.json()
        source = Path(temp) / sequence["sequence_id"] / "foreground" / existing["mask_set_id"]
        # Simulate old manifests that predate individual mask storage.
        for item in (source / "region_masks").glob("*.png"):
            item.unlink()
        okay = client.post(existing["manifest_url"] + "/review", json={
            "decisions": [{"frame_number": 1, "region_index": 0, "decision": "keep"}]
        })
        assert okay.status_code == 200, okay.text
        fail = client.post(existing["manifest_url"] + "/review", json={
            "decisions": [{"frame_number": 2, "region_index": 0, "decision": "keep"}]
        })
        assert fail.status_code == 409


def test_persisted_job_listing_handles_multiple_jobs_and_missing_results():
    with TemporaryDirectory() as temp, patch.object(main, "DATA_DIR", Path(temp)):
        client = TestClient(main.app)
        sequence = _setup(client)
        root = Path(temp) / sequence["sequence_id"] / "jobs"
        root.mkdir(exist_ok=True)
        identifier = "12345678123412341234123412341234"
        (root / identifier).mkdir()
        write_json(root / identifier / "status.json", {
            "job_id": identifier, "operation": "solve", "status": "done",
            "result": {"statistics": {"registered_frames": 9}},
        })
        restored = client.get(f"/sequences/{sequence['sequence_id']}/jobs")
        assert restored.status_code == 200, restored.text
        jobs = restored.json()["jobs"]
        assert len(jobs) == 1
        assert jobs[0]["result"]["statistics"]["registered_frames"] == 9
        assert jobs[0]["status_url"].endswith(identifier)


def test_interrupted_job_can_be_retried_with_its_original_settings():
    with TemporaryDirectory() as temp, patch.object(main, "DATA_DIR", Path(temp)):
        client = TestClient(main.app)
        sequence = _setup(client)
        identifier = "12345678123412341234123412341234"
        interrupted = Path(temp) / sequence["sequence_id"] / "jobs" / identifier
        interrupted.mkdir(parents=True)
        parameters = main.JobRequest(max_frames=7).model_dump(mode="json")
        write_json(interrupted / "status.json", {
            "job_id": identifier, "operation": "solve", "status": "running", "parameters": parameters,
        })
        url = f"/sequences/{sequence['sequence_id']}/jobs/{identifier}"
        assert client.get(url).json()["stage"] == "Interrupted"
        with patch.object(main, "reconstruct", side_effect=ReconstructionError("No parallax.")) as solve:
            response = client.post(url + "/retry")
            assert response.status_code == 202, response.text
            retry = response.json()
            deadline = monotonic() + 5
            while monotonic() < deadline:
                status = client.get(retry["status_url"]).json()
                if status["status"] == "failed":
                    break
                sleep(0.05)
        assert status["error"] == "No parallax."
        assert status["parameters"] == parameters and status["retry_of"] == identifier
        assert solve.call_args.kwargs["max_frames"] == 7
        assert read_json(interrupted / "status.json")["status"] == "running"


def test_queued_job_cancels_immediately_and_running_job_signals_native_work():
    with TemporaryDirectory() as temp:
        folder = Path(temp)
        started, stop = Event(), Event()

        def native_work(target, progress):
            progress.bind_cancel(stop.set)
            started.set()
            assert stop.wait(5)
            return {"would_be_result": True}

        running = jobs.submit(folder, "solve", native_work)
        assert started.wait(5)
        try:
            queued = jobs.submit(folder, "solve", lambda *args: {"unexpected": True})
            target = folder / "jobs" / queued["job_id"]
            assert jobs.cancel(target)["status"] == "cancelled"
            assert "result" not in jobs.status(target)
        finally:
            target = folder / "jobs" / running["job_id"]
            jobs.cancel(target)
            assert stop.wait(5)
            deadline = monotonic() + 5
            while monotonic() < deadline:
                current = jobs.status(target)
                if current["status"] == "cancelled":
                    break
                sleep(0.05)
        assert current["status"] == "cancelled" and "result" not in current


def test_invalid_review_rejected_without_modifying_existing_artifacts():
    with TemporaryDirectory() as temp, patch.object(main, "DATA_DIR", Path(temp)):
        client = TestClient(main.app)
        sequence = _setup(client)
        root = f"/sequences/{sequence['sequence_id']}"
        with patch("backend.foreground.get_sam", return_value=FakeSam()):
            created = client.post(root + "/foreground", json={"prompts": [
                {"frame_number": 2, "box": [5, 5, 22, 22], "decision": "exclude"},
            ]})
        assert created.status_code == 200
        existing = created.json()
        for item in [{"frame_number": 999, "region_index": 0, "decision": "keep"},
                     {"frame_number": 2, "region_index": 4, "decision": "exclude"}]:
            response = client.post(existing["manifest_url"] + "/review", json={"decisions": [item]})
            assert response.status_code == 422
        assert len(client.get(root + "/foreground").json()["mask_sets"]) == 1


def test_automatic_exclusions_cannot_bypass_review_on_any_solve_route():
    with TemporaryDirectory() as temp, patch.object(main, "DATA_DIR", Path(temp)):
        client = TestClient(main.app)
        sequence = _setup(client)
        root = f"/sequences/{sequence['sequence_id']}"
        with patch("backend.foreground.get_sam", return_value=FakeSam()), patch(
            "backend.foreground.motion_evidence", return_value=[
                {"neighbor": number, "tracked_points": 20, "decision": "moving"} for number in (1, 3)
            ]
        ):
            generated = client.post(root + "/foreground", json={"prompts": [
                {"frame_number": 2, "box": [5, 5, 22, 22], "decision": "auto"},
            ]}).json()
        mask_id = generated["mask_set_id"]
        with patch.object(main, "reconstruct") as solve:
            assert client.post(root + "/reconstruct", params={"mask_set_id": mask_id}).status_code == 409
            assert client.post(generated["manifest_url"] + "/compare").status_code == 409
            assert client.post(root + "/jobs", json={"sam": True, "mask_set_id": mask_id}).status_code == 409
            assert client.post(root + "/jobs", json={"operation": "compare", "mask_set_id": mask_id}).status_code == 409
            assert client.post(root + "/jobs", json={"sam": True}).status_code == 422
            assert client.post(root + "/jobs", json={"mask_set_id": mask_id}).status_code == 422
            solve.assert_not_called()
        reviewed = client.post(generated["manifest_url"] + "/review", json={
            "decisions": [{"frame_number": 2, "region_index": 0, "decision": "exclude"}],
        }).json()
        path = Path(temp) / sequence["sequence_id"] / "foreground" / reviewed["mask_set_id"]
        assert main.require_exclusion(path)["masked_frames"] == 1
        mask_file = path / "colmap" / (sequence["frame_details"][1]["filename"] + ".png")
        Image.new("L", (2, 2), 255).save(mask_file)
        assert client.post(root + "/reconstruct", params={"mask_set_id": reviewed["mask_set_id"]}).status_code == 409
        assert client.get("/sequences").json()["sequences"][0]["sequence_id"] == sequence["sequence_id"]
