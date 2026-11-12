import tempfile
import unittest
from concurrent.futures import ThreadPoolExecutor
from io import BytesIO
from pathlib import Path
from unittest.mock import Mock, patch

import Imath
import numpy as np
import OpenEXR
from fastapi.testclient import TestClient
from PIL import Image

from . import main
from .reconstruction import SolverUnavailable, pose_record


def png(width=8, height=6, color=(50, 100, 150)):
    stream = BytesIO()
    Image.new("RGB", (width, height), color).save(stream, format="PNG")
    return stream.getvalue()


def exr(path, numerator=24000, denominator=1001, cropped=False):
    width, height = 8, 6
    header = OpenEXR.Header(width, height)
    header["framesPerSecond"] = Imath.Rational(numerator, denominator)
    if cropped:
        header["dataWindow"] = Imath.Box2i(Imath.V2i(2, 1), Imath.V2i(5, 3))
        width, height = 4, 3
    file = OpenEXR.OutputFile(str(path), header)
    try:
        pixels = np.full((height, width), 0.5, dtype=np.float32).tobytes()
        file.writePixels(dict.fromkeys("RGB", pixels))
    finally:
        file.close()
    return path.read_bytes()


class SequenceTests(unittest.TestCase):
    def setUp(self):
        self.temp = self.enterContext(tempfile.TemporaryDirectory())
        self.enterContext(patch.object(main, "DATA_DIR", Path(self.temp) / "sequences"))
        self.client = self.enterContext(TestClient(main.app))

    def upload(self, files, fps=None):
        data = {"fps": fps} if fps is not None else {}
        return self.client.post("/upload/", files=[("files", file) for file in files], data=data)

    def solve(self, sequence):
        return self.client.post(f"/sequences/{sequence['sequence_id']}/reconstruct")

    def test_order_resolution_fps_and_corrupt_frames(self):
        response = self.upload([
            ("shot_0012.png", png(), "image/png"),
            ("shot_0009.png", png(), "image/png"),
            ("shot_0011.png", b"bad image", "image/png"),
        ], fps="24000/1001")
        self.assertEqual(response.status_code, 200)
        sequence = response.json()
        self.assertEqual(sequence["status"], "partial")
        self.assertEqual(sequence["fps_rational"], [24000, 1001])
        self.assertEqual(sequence["resolution"], {"width": 8, "height": 6})
        self.assertEqual(sequence["start_frame"], 9)
        self.assertEqual(sequence["end_frame"], 12)
        self.assertEqual(sequence["missing_frame_ranges"], [{"start": 10, "end": 11}])
        self.assertEqual([frame["frame_number"] for frame in sequence["frame_details"]], [9, 12])
        self.assertEqual([frame["source_name"] for frame in sequence["frame_details"]], ["shot_0009.png", "shot_0012.png"])
        self.assertEqual(sequence["failed_uploads"][0]["frame_number"], 11)
        for name in sequence["frames"]:
            response = self.client.get(f"/frames/{name}")
            self.assertEqual(response.status_code, 200)
            self.assertEqual(Image.open(BytesIO(response.content)).size, (8, 6))

    def test_exr_embedded_fps_is_exact_and_conflicts_are_rejected(self):
        content = exr(Path(self.temp) / "test.exr")
        files = [("shot.001.exr", content, "application/octet-stream")]
        response = self.upload(files)
        self.assertEqual(response.status_code, 200, response.text)
        sequence = response.json()
        self.assertEqual(sequence["status"], "done", sequence)
        self.assertEqual(sequence["fps_source"], "exr")
        self.assertEqual(sequence["fps_rational"], [24000, 1001])
        self.assertEqual(sequence["resolution"], {"width": 8, "height": 6})
        self.assertEqual(self.upload(files, fps="30").status_code, 422)
        self.assertEqual(len(list(main.DATA_DIR.iterdir())), 1)

    def test_unnumbered_frames_keep_upload_order_and_fps_unknown(self):
        sequence = self.upload([
            ("zebra.png", png(), "image/png"),
            ("alpha.png", png(), "image/png"),
        ]).json()
        self.assertEqual(sequence["numbering"], "upload_order")
        self.assertEqual(sequence["fps"], None)
        self.assertEqual([frame["source_name"] for frame in sequence["frame_details"]], ["zebra.png", "alpha.png"])

    def test_exr_cropped_data_keeps_display_resolution_and_position(self):
        content = exr(Path(self.temp) / "cropped.exr", cropped=True)
        sequence = self.upload([("shot_1.exr", content, "application/octet-stream")]).json()
        self.assertEqual(sequence["resolution"], {"width": 8, "height": 6})
        response = self.client.get(sequence["frame_details"][0]["url"])
        with Image.open(BytesIO(response.content)) as image:
            self.assertEqual(image.getpixel((0, 0)), (0, 0, 0))
            self.assertGreater(image.getpixel((2, 1))[0], 0)
            self.assertEqual(image.getpixel((6, 3)), (0, 0, 0))

    def test_signed_numbers_and_hyphen_separator_sort_correctly(self):
        names = ["shot-001.png", "shot_-002.png", "shot_-010.png", "shot_000.png"]
        sequence = self.upload([(name, png(), "image/png") for name in names]).json()
        self.assertEqual([frame["frame_number"] for frame in sequence["frame_details"]], [-10, -2, 0, 1])
        self.assertEqual(sequence["frames"], sorted(sequence["frames"]))

    def test_duplicate_frame_numbers_are_reported_and_kept_as_gaps(self):
        sequence = self.upload([
            ("shot_1.png", png(), "image/png"),
            ("shot_2.png", png(), "image/png"),
            ("shot_02.png", png(), "image/png"),
            ("shot_3.png", png(), "image/png"),
        ]).json()
        self.assertEqual(len(sequence["failed_uploads"]), 2)
        self.assertEqual(sequence["missing_frame_ranges"], [{"start": 2, "end": 2}])

    def test_concurrent_uploads_isolate_identical_filenames(self):
        def upload_color(color):
            return self.upload([("shot_1.png", png(color=color), "image/png")]).json()

        colors = [(255, 0, 0), (0, 255, 0)]
        with ThreadPoolExecutor(max_workers=2) as pool:
            sequences = list(pool.map(upload_color, colors))
        self.assertNotEqual(sequences[0]["sequence_id"], sequences[1]["sequence_id"])
        for sequence, color in zip(sequences, colors):
            response = self.client.get(sequence["frame_details"][0]["url"])
            self.assertEqual(response.status_code, 200)
            self.assertEqual(Image.open(BytesIO(response.content)).getpixel((0, 0)), color)

    def test_all_failed_upload_and_no_saved_result(self):
        sequence = self.upload([("shot_3.png", b"bad", "image/png")]).json()
        self.assertEqual(sequence["status"], "failed")
        self.assertEqual(sequence["frames"], [])
        self.assertEqual(self.solve(sequence).status_code, 422)
        result = self.client.get(f"/sequences/{sequence['sequence_id']}/reconstruction.json")
        self.assertEqual(result.status_code, 404)

    def test_mixed_resolutions_cannot_enter_solver(self):
        files = [(f"shot_{i}.png", png(width=10 + i), "image/png") for i in range(3)]
        sequence = self.upload(files).json()
        response = self.solve(sequence)
        self.assertEqual(response.status_code, 422)
        self.assertIn("same resolution", response.json()["detail"])

    def test_unavailable_native_solver_returns_503(self):
        files = [(f"shot_{i}.png", png(), "image/png") for i in range(3)]
        sequence = self.upload(files).json()
        with patch("backend.reconstruction.load_solver", side_effect=SolverUnavailable("Native module unavailable")):
            response = self.solve(sequence)
        self.assertEqual(response.status_code, 503)
        self.assertIn("Native module unavailable", response.json()["detail"])

    def test_verified_matches_without_model_returns_422(self):
        files = [(f"shot_{i}.png", png(), "image/png") for i in range(3)]
        sequence = self.upload(files).json()
        solver = Mock()
        solver.Database.open.return_value.num_verified_image_pairs.return_value = 3
        solver.Database.open.return_value.num_inlier_matches.return_value = 50
        solver.incremental_mapping.return_value = {}
        with patch("backend.reconstruction.load_solver", return_value=solver):
            response = self.solve(sequence)
        self.assertEqual(response.status_code, 422)
        self.assertIn("no model registered", response.json()["detail"])
        self.assertFalse((main.DATA_DIR / sequence["sequence_id"] / "reconstruction.json").exists())


class PoseTests(unittest.TestCase):
    def test_colmap_translation_is_not_camera_center(self):
        rotation = np.array([[0, -1, 0], [1, 0, 0], [0, 0, 1]])
        center = np.array([2, 3, 4])
        translation = -rotation @ center
        pose = pose_record(rotation, translation)
        np.testing.assert_array_equal(pose["position_world"], center)
        np.testing.assert_array_equal(pose["rotation_camera_to_world"], rotation.T)
        np.testing.assert_array_equal(pose["translation_world_to_camera"], translation)

    def test_pose_preserves_right_down_forward_and_round_trips(self):
        rotation = np.array([[0, 0, 1], [0, 1, 0], [-1, 0, 0]])
        center = np.array([1, 2, 3])
        pose = pose_record(rotation, -rotation @ center)
        camera_to_world = np.array(pose["rotation_camera_to_world"])
        np.testing.assert_array_equal(camera_to_world[:, 0], [0, 0, 1])
        np.testing.assert_array_equal(camera_to_world[:, 1], [0, 1, 0])
        np.testing.assert_array_equal(camera_to_world[:, 2], [-1, 0, 0])
        camera_point = np.array([5, 7, 11])
        world_point = camera_to_world @ camera_point + pose["position_world"]
        np.testing.assert_array_equal(
            np.array(pose["rotation_world_to_camera"]) @ world_point + pose["translation_world_to_camera"],
            camera_point,
        )


if __name__ == "__main__":
    unittest.main()
