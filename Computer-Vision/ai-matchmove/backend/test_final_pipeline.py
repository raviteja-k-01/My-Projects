import io
import json
import shutil
import subprocess
import tempfile
import time
import unittest
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from threading import Event
from unittest.mock import patch
from zipfile import ZipFile

import numpy as np
from fastapi.testclient import TestClient
from PIL import Image

from . import main
from .export_camera import blender_data
from .test_pipeline import png


def example_result():
    rotation = [[0, 0, 1], [0, 1, 0], [-1, 0, 0]]
    return {
        "status": "partial", "fps": 24000 / 1001, "start_frame": 10,
        "end_frame": 15, "resolution": {"width": 80, "height": 60},
        "cameras": [{"camera_id": 1, "model": "SIMPLE_RADIAL", "width": 80, "height": 60,
                     "parameters": [70, 37, 28, 0.06], "calibration_matrix": [[70, 0, 37], [0, 70, 28], [0, 0, 1]]}],
        "registered_frames": [
            {"frame_number": n, "source_name": f"shot_{n}.png", "filename": f"frame_{n:015d}.png",
             "camera_id": 1, "position_world": [1, 2, 3], "rotation_camera_to_world": rotation,
             "observations": [[40, 31, 1]]}
            for n in (10, 12, 15)
        ],
        "source_frame_count": 3,
        "sparse_points": [{"xyz_world": [4, 5, 6], "rgb": [100, 200, 240]}],
        "statistics": {"registered_frames": 3},
    }


class ExportTests(unittest.TestCase):
    def test_pose_axis_conversion_and_off_center_projection(self):
        result = example_result()
        data = blender_data(result)
        pose = data["frames"][0]
        self.assertEqual([frame["frame"] for frame in data["frames"]], [1, 3, 6])
        self.assertEqual([frame["source_frame"] for frame in data["frames"]], [10, 12, 15])
        colmap_rotation = np.array(result["registered_frames"][0]["rotation_camera_to_world"])
        world_ray = colmap_rotation @ np.array([0.1, -0.2, 1])
        blender_ray = np.array(pose["rotation"]).T @ world_ray
        np.testing.assert_allclose(blender_ray, [0.1, 0.2, -1])
        lens = pose["lens"]
        scale = lens["focal_mm"] / 36 * 80
        x = 40 - lens["shift_x"] * 80 + scale * blender_ray[0] / -blender_ray[2]
        y = 30 + lens["shift_y"] * 80 - scale * blender_ray[1] / -blender_ray[2]
        np.testing.assert_allclose([x, y], [37 + 70 * 0.1, 28 - 70 * 0.2])
        self.assertEqual(lens["radial_k"], 0.06)

    def test_rejects_unknown_fps_and_unsupported_lens(self):
        result = example_result()
        result["fps"] = None
        with self.assertRaisesRegex(ValueError, "source FPS"):
            blender_data(result)
        result["fps"] = 24
        result["cameras"][0]["model"] = "FISHEYE"
        with self.assertRaisesRegex(ValueError, "does not support"):
            blender_data(result)


class JobTests(unittest.TestCase):
    def setUp(self):
        self.root = Path(self.enterContext(tempfile.TemporaryDirectory()))
        self.enterContext(patch.object(main, "DATA_DIR", self.root / "uploads"))
        self.client = self.enterContext(TestClient(main.app))

    def wait_job(self, url):
        deadline = time.monotonic() + 15
        while time.monotonic() < deadline:
            response = self.client.get(url)
            response.raise_for_status()
            result = response.json()
            if result["status"] not in {"running", "queued"}:
                return result
            time.sleep(0.05)
        self.fail("Timed out waiting for job")

    def test_solve_job_export_and_isolation(self):
        upload = self.client.post("/upload/", files=[
            ("files", (f"shot_{n}.png", png(80, 60), "image/png")) for n in (10, 12, 15)
        ]).json()
        root = "/sequences/" + upload["sequence_id"]
        result = {**example_result(), "fps": None, "fps_rational": None, "fps_source": None}

        def fake_solve(sequence, folder, **options):
            options["progress"]("Saving test model")
            main.write_json(options["output_path"], result)
            return result

        with patch.object(main, "reconstruct", side_effect=fake_solve):
            job = self.client.post(root + "/jobs", json={"operation": "solve"})
            self.assertEqual(job.status_code, 202)
            solved = self.wait_job(job.json()["status_url"])
        self.assertEqual(solved["status"], "done", solved)
        self.assertEqual(self.client.get(solved["result"]["tracks_url"]).json()["frames"]["10"], [[40, 31, 1]])
        self.assertIn("element vertex 1", self.client.get(solved["result"]["points_url"]).text)
        options = {"operation": "export", "source_job_id": solved["job_id"]}
        self.assertEqual(self.client.post(root + "/jobs", json=options).status_code, 422)
        export = self.client.post(root + "/jobs", json={**options, "export_fps": "24000/1001"})
        state = self.wait_job(export.json()["status_url"])
        self.assertEqual(state["status"], "done", state)
        response = self.client.get(state["result"]["export_url"])
        with ZipFile(io.BytesIO(response.content)) as package:
            self.assertIn("import_matchmove.py", package.namelist())
            self.assertIn("element vertex 1", package.read("points.ply").decode())
            camera = json.loads(package.read("camera.json"))
            self.assertEqual(len(camera["frames"]), 3)
            self.assertAlmostEqual(camera["fps"], 24000 / 1001)
            self.assertEqual(json.loads(package.read("reconstruction.json"))["fps_rational"], [24000, 1001])
            self.assertEqual(Image.open(io.BytesIO(package.read("plates/0000000001.png"))).size, (80, 60))
        self.assertIsNone(self.client.get(solved["result"]["result_url"]).json()["fps"])
        self.assertEqual(self.client.get(root + "/jobs/" + state["job_id"] + "/files/tracks.json").status_code, 404)

    def test_comparison_can_be_opened_as_a_solve_without_recomputing(self):
        project = self.client.post("/upload/", files=[
            ("files", (f"shot_{n}.png", png(80, 60), "image/png")) for n in (10, 12, 15)
        ]).json()
        project_id = project["sequence_id"]
        base = f"/sequences/{project_id}/jobs"
        mask_id, report_id, compare_job = ("a" * 32, "b" * 32, "c" * 32)
        directory = main.DATA_DIR / project_id
        mask = directory / "foreground" / mask_id
        (mask / report_id).mkdir(parents=True)
        main.write_json(mask / "manifest.json", {"mask_set_id": mask_id})
        main.write_json(mask / report_id / "masked.json", example_result())
        original = directory / "jobs" / compare_job
        original.mkdir(parents=True)
        main.write_json(original / "status.json", {"job_id": compare_job, "operation": "compare", "status": "done",
                        "result": {"mask_set_id": mask_id, "comparison_id": report_id,
                                   "baseline": {"status": "failed"}, "masked": {"status": "partial"}}})
        with patch.object(main, "reconstruct") as reconstruct:
            options = {"comparison_job_id": compare_job}
            self.assertEqual(self.client.post(base, json={**options, "comparison_mode": "baseline"}).status_code, 422)
            opened = self.client.post(base, json={**options, "comparison_mode": "masked"})
            self.assertEqual(opened.status_code, 202, opened.text)
            state = self.wait_job(opened.json()["status_url"])
            reconstruct.assert_not_called()
        self.assertEqual(state["status"], "done")
        self.assertTrue(state["result"]["masking_applied"])
        self.assertEqual(state["result"]["comparison_job_id"], compare_job)
        self.assertEqual(self.client.get(state["result"]["result_url"]).json(), example_result())

    def test_cancellation_stops_at_next_stage_and_blocks_download(self):
        upload = self.client.post("/upload/", files=[
            ("files", (f"shot_{n}.png", png(), "image/png")) for n in (1, 2, 3)
        ]).json()
        started, release = Event(), Event()

        def staged_solve(sequence, folder, **options):
            started.set()
            release.wait(timeout=10)
            options["progress"]("Next stage")
            self.fail("Cancelled work continued")

        with patch.object(main, "reconstruct", side_effect=staged_solve):
            response = self.client.post(f"/sequences/{upload['sequence_id']}/jobs", json={})
            url = response.json()["status_url"]
            try:
                self.assertTrue(started.wait(timeout=5))
                self.assertTrue(self.client.post(url + "/cancel").json()["cancellation_requested"])
            finally:
                release.set()
            self.assertEqual(self.wait_job(url)["status"], "cancelled")
            self.assertEqual(self.client.get(url + "/files/reconstruction.json").status_code, 404)

    def test_preview_is_bounded_and_original_frame_remains_full_resolution(self):
        upload = self.client.post("/upload/", files={
            "files": ("shot_1.png", png(1920, 1080), "image/png"),
        }).json()
        original_url = upload["frame_details"][0]["url"]
        preview_url = original_url.replace("/frames/", "/previews/", 1)
        preview = self.client.get(preview_url)
        self.assertEqual(preview.status_code, 200)
        self.assertEqual(preview.headers["content-type"], "image/jpeg")
        self.assertEqual(Image.open(io.BytesIO(preview.content)).size, (960, 540))
        self.assertEqual(self.client.get(preview_url).content, preview.content)
        original = self.client.get(original_url)
        self.assertEqual(Image.open(io.BytesIO(original.content)).size, (1920, 1080))
        self.assertEqual(self.client.get("/sequences/" + upload["sequence_id"]).json(), upload)

    def test_job_status_is_readable_during_progress_updates(self):
        def work(target, progress):
            for index in range(40):
                progress("Processing", index, 40)
            return {}

        job = main.jobs.submit(self.root, "test", work)
        target = self.root / "jobs" / job["job_id"]
        with ThreadPoolExecutor(max_workers=4) as readers:
            states = list(readers.map(lambda _: main.jobs.status(target)["status"], range(100)))
        self.assertLessEqual(set(states), {"queued", "running", "done"})
        deadline = time.monotonic() + 5
        while main.jobs.status(target)["status"] != "done" and time.monotonic() < deadline:
            time.sleep(.05)
        self.assertEqual(main.jobs.status(target)["status"], "done")

    @unittest.skipUnless(shutil.which("ffmpeg") and shutil.which("ffprobe"), "FFmpeg tools required")
    def test_video_decode_sampling_and_corrupt_upload(self):
        video = self.root / "video.mp4"
        subprocess.run([
            "ffmpeg", "-v", "error", "-f", "lavfi", "-i", "testsrc2=size=80x60:rate=25",
            "-frames:v", "7", "-threads", "1", "-c:v", "mpeg4", str(video),
        ], check=True, timeout=20)
        response = self.client.post("/upload/", data={"video_stride": "2"},
                                    files={"files": ("video.mp4", video.read_bytes(), "video/mp4")})
        self.assertEqual(response.status_code, 200, response.text)
        data = response.json()
        self.assertEqual(data["source_frame_count"], 7)
        self.assertEqual(data["fps_rational"], [25, 1])
        self.assertEqual(data["resolution"], {"width": 80, "height": 60})
        self.assertEqual([frame["frame_number"] for frame in data["frame_details"]], [0, 2, 4, 6])
        np.testing.assert_allclose([frame["timestamp_seconds"] for frame in data["frame_details"]], [0, .08, .16, .24])
        failed = self.client.post("/upload/", files={"files": ("bad.mp4", b"broken", "video/mp4")})
        self.assertEqual(failed.status_code, 422)


if __name__ == "__main__":
    unittest.main()
