import io
import tempfile
import time
import unittest
from pathlib import Path
from unittest.mock import Mock, patch

import numpy as np
from PIL import Image, ImageDraw
from fastapi.testclient import TestClient

from . import main
from .geometry_diagnostics import failure_diagnosis
from .motion import load_cv, motion_evidence, project_candidate_box, sample_corners
from .reconstruction import ReconstructionError, SolverUnavailable, load_solver, match_statistics, wider_pairs
from .sequence import check_frame_size, FrameLimitError


class GeometryTests(unittest.TestCase):
    def test_failure_diagnosis_does_not_equate_homographies_with_rotation(self):
        text = failure_diagnosis({"homography_fraction": 0.9}, [{"initializations": 0}])
        self.assertIn("Homographies dominate", text)
        self.assertIn("planar structure", text)
        self.assertIn("seed pair initialized", failure_diagnosis({}, [{"initializations": 1}]))
        low_angle = {"sampled_pairs": [{"provisional_geometry": "CALIBRATED", "provisional_triangulation_angle_deg": 1.42}]}
        self.assertIn("1.42°", failure_diagnosis(low_angle, [{"initializations": 0}]))

    def test_frame_limit_accepts_4k_and_rejects_oversized_inputs(self):
        check_frame_size(3840, 2160)
        with self.assertRaises(FrameLimitError):
            check_frame_size(10000, 10000)

    def test_wider_pair_budget_excludes_existing_matches(self):
        names = [f"frame_{i:02}.png" for i in range(30)]
        existing = {tuple(sorted((names[0], names[15])))}
        pairs = wider_pairs(names, existing, limit=5)
        self.assertEqual(len(pairs), 5)
        self.assertTrue(existing.isdisjoint(pairs))
        self.assertEqual(len(set(pairs)), len(pairs))
        self.assertTrue(all(abs(names.index(a) - names.index(b)) >= 15 for a, b in pairs))

    def test_pair_counts_ignore_empty_colmap_rows(self):
        database = Mock()
        database.num_matches.return_value = 12
        database.num_inlier_matches.return_value = 5
        database.read_num_matches.return_value = ([1, 2, 3], [0, 7, 5])
        database.read_two_view_geometry_num_inliers.return_value = ([1, 2, 3], [0, 5, 0])
        stats = match_statistics(database)
        self.assertEqual((stats["matched_pairs"], stats["verified_pairs"]), (2, 1))
        self.assertEqual(stats["geometric_inlier_ratio"], 5 / 12)

    def test_motion_seed_requires_spatially_distributed_tracks(self):
        cv = load_cv()
        points = np.array([(x, y) for x in range(30, 80, 8) for y in range(30, 80, 8)], dtype=np.float32)
        shifted = points + np.array([11, 7], dtype=np.float32)
        self.assertEqual(project_candidate_box(cv, (25, 25, 85, 85), (points, shifted), (120, 140)), (36, 32, 96, 92))
        cluster = points / 30 + 30
        self.assertIsNone(project_candidate_box(cv, (25, 25, 85, 85), (cluster, cluster + 4), (120, 140)))

    def test_dominant_background_warp_detects_coherent_motion_along_a_scanline(self):
        cv = load_cv()
        points = np.array([(x, y) for x in range(12, 180, 10) for y in range(12, 160, 10)], dtype=np.float32)
        mask = np.zeros((160, 180), dtype=bool)
        mask[50:91, 60:111] = True
        x, y = points.astype(int).T
        moving = mask[y, x]
        shifted = points + np.array([2, 1], dtype=np.float32)
        shifted[moving, 0] += 7
        result = motion_evidence(cv, [(7, (points, shifted))], mask)[0]
        self.assertEqual((result["background_model"], result["decision"]), ("dominant_warp", "moving"))
        self.assertAlmostEqual(result["median_residual_px"], 7, places=3)
        shifted[np.flatnonzero(moving)[::2], 0] -= 14
        result = motion_evidence(cv, [(7, (points, shifted))], mask)[0]
        self.assertEqual(result["decision"], "inconclusive")

    def test_motion_corners_cover_low_contrast_tiles_in_source_coordinates(self):
        cv = load_cv()
        image = np.zeros((240, 360), dtype=np.uint8)
        image[10:24, 10:24] = 250
        image[214:228, 334:348] = 12
        points = sample_corners(cv, image)[:, 0]
        self.assertTrue(any(x < 25 and y < 25 for x, y in points))
        self.assertTrue(any(x >= 330 and y >= 210 for x, y in points))
        self.assertLessEqual(len(points), 1800)


class DiagnosticJobTests(unittest.TestCase):
    def setUp(self):
        root = Path(self.enterContext(tempfile.TemporaryDirectory()))
        self.enterContext(patch.object(main, "DATA_DIR", root))
        self.client = self.enterContext(TestClient(main.app))

    def wait_job(self, url):
        deadline = time.monotonic() + 45
        while time.monotonic() < deadline:
            state = self.client.get(url).json()
            if state["status"] not in {"queued", "running"}:
                return state
            time.sleep(0.1)
        self.fail("Job did not reach a terminal state.")

    def test_failure_retains_measurements_and_never_exposes_reconstruction(self):
        content = io.BytesIO()
        Image.new("RGB", (32, 32)).save(content, "PNG")
        response = self.client.post("/upload/", files=[
            ("files", (f"shot_{i}.png", content.getvalue(), "image/png")) for i in range(3)
        ])
        root = "/sequences/" + response.json()["sequence_id"]
        with patch.object(main, "reconstruct", side_effect=ReconstructionError("No usable seed.", {"verified_pairs": 8})):
            job = self.client.post(root + "/jobs", json={}).json()
            state = self.wait_job(job["status_url"])
        self.assertEqual(state["status"], "failed")
        self.assertEqual(state["measurements"]["verified_pairs"], 8)
        self.assertEqual(self.client.get(job["status_url"] + "/files/diagnostics.json").json()["error"], "No usable seed.")
        self.assertEqual(self.client.get(job["status_url"] + "/files/reconstruction.json").status_code, 404)

    def test_real_pure_rotation_returns_geometry_diagnostics_without_camera(self):
        try:
            load_solver()
        except SolverUnavailable as exc:
            self.skipTest(str(exc))
        cv = load_cv()
        rng = np.random.default_rng(641)
        width, height = 480, 360
        image = Image.new("RGB", (width, height), (50, 60, 70))
        draw = ImageDraw.Draw(image)
        for _ in range(280):
            x, y = (int(v) for v in rng.integers([0, 0], [width, height]))
            color = tuple(int(v) for v in rng.integers(30, 250, size=3))
            draw.ellipse((x, y, x+9, y+7), fill=color)
        camera = np.array([[575, 0, width/2], [0, 575, height/2], [0, 0, 1]])
        files = []
        for index, angle in enumerate(np.linspace(-0.13, 0.13, 5)):
            rotation, _ = cv.Rodrigues(np.array([0., angle, 0.]))
            projection = camera @ rotation @ np.linalg.inv(camera)
            frame = cv.warpPerspective(np.asarray(image), projection, (width, height))
            content = io.BytesIO()
            Image.fromarray(frame).save(content, "PNG")
            files.append(("files", (f"rotation_{index}.png", content.getvalue(), "image/png")))
        upload = self.client.post("/upload/", files=files).json()
        job = self.client.post("/sequences/" + upload["sequence_id"] + "/jobs", json={"initial_focal_px": 575}).json()
        state = self.wait_job(job["status_url"])
        self.assertEqual(state["status"], "failed", state)
        measurements = state["measurements"]
        self.assertGreater(measurements["verified_pairs"], 0)
        self.assertGreaterEqual(measurements["diagnostics"]["geometry"]["homography_fraction"], 0.5)
        self.assertEqual(measurements["diagnostics"]["geometry"]["calibration"]["database_focal_px"], [575.0])
        self.assertEqual(self.client.get(job["status_url"] + "/files/reconstruction.json").status_code, 404)
