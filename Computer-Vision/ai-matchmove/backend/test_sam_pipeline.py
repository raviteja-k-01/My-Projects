import os
import tempfile
import unittest
from contextlib import nullcontext
from io import BytesIO
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, patch

import numpy as np
from fastapi.testclient import TestClient
from PIL import Image, ImageDraw

from . import main
from .motion import load_cv, motion_evidence
from .reconstruction import ReconstructionError, SolverUnavailable, count_masked_keypoints, load_solver
from .sam_service import Prompt, SamService, SamUnavailable, export_mask, get_sam, validate_prompt
from .test_pipeline import png


class PromptTests(unittest.TestCase):
    def test_prompt_uses_source_xy_and_rejects_invalid_coordinates(self):
        validate_prompt(Prompt(frame_number=1, box=(0, 0, 80, 40), points=[(79, 39, 1)]), 80, 40)
        for prompt in [
            Prompt(frame_number=1),
            Prompt(frame_number=1, points=[(80, 39, 1)]),
            Prompt(frame_number=1, box=(5, 5, 5, 10)),
            Prompt(frame_number=1, points=[(30, 20, 0)]),
        ]:
            with self.subTest(prompt=prompt), self.assertRaises(ValueError):
                validate_prompt(prompt, 80, 40)

    def test_model_reports_missing_checkpoint(self):
        with patch.dict(os.environ, {"MATCHMOVE_SAM_CHECKPOINT": ""}):
            with self.assertRaisesRegex(SamUnavailable, "MATCHMOVE_SAM_CHECKPOINT"):
                get_sam()

    def test_inference_memory_error_clears_predictor_and_explains_recovery(self):
        class OutOfMemoryError(RuntimeError):
            pass

        predictor = Mock()
        predictor.set_image.side_effect = OutOfMemoryError()
        torch = SimpleNamespace(OutOfMemoryError=OutOfMemoryError, inference_mode=nullcontext)
        service = SamService(predictor, torch, "cuda", 1024)
        with self.assertRaisesRegex(SamUnavailable, "MATCHMOVE_SAM_DEVICE=cpu"):
            service.segment(Image.new("RGB", (30, 30)), [Prompt(frame_number=1, points=[(5, 5, 1)])])
        predictor.reset_image.assert_called_once()

    @unittest.skipUnless(os.environ.get("MATCHMOVE_SAM_CHECKPOINT"), "set the existing ViT-B checkpoint for real inference")
    def test_real_vit_b_box_and_point_masks_and_reuse(self):
        image = Image.new("RGB", (280, 168), (38, 46, 58))
        ImageDraw.Draw(image).rounded_rectangle((60, 40, 218, 135), radius=12, fill=(235, 125, 44))
        service = get_sam()
        self.assertIs(service, get_sam())
        output = service.segment(image, [
            Prompt(frame_number=1, box=(56, 36, 222, 139), points=[(130, 70, 1), (30, 26, 0)]),
            Prompt(frame_number=1, points=[(130, 70, 1)]),
        ])
        for mask, score in output:
            self.assertEqual(mask.shape, (168, 280))
            self.assertTrue(mask[70, 130])
            self.assertFalse(mask[8, 8])
            self.assertGreater(score, 0.4)


class GeometricTests(unittest.TestCase):
    def test_static_parallax_is_kept_and_independent_motion_is_detected(self):
        cv = load_cv()
        rng = np.random.default_rng(702)
        xyz = rng.uniform((-2, -1, 4), (2, 1, 8), (400, 3))
        start = (520 * xyz[:, :2] / xyz[:, 2:] + (320, 200)).astype(np.float32)
        shifted = xyz - (0.4, 0.1, 0.2)
        static = (520 * shifted[:, :2] / shifted[:, 2:] + (320, 200)).astype(np.float32)
        mask = np.zeros((400, 640), dtype=bool)
        x, y = start[:30].astype(int).T
        mask[y, x] = True
        moving = static.copy()
        moving[:30, 1] += 23
        self.assertEqual(motion_evidence(cv, [(2, (start, static))], mask)[0]["decision"], "consistent")
        self.assertEqual(motion_evidence(cv, [(2, (start, moving))], mask)[0]["decision"], "moving")

    def test_native_colmap_mask_removes_keypoints_in_exact_pixels(self):
        try:
            solver = load_solver()
        except SolverUnavailable as exc:
            self.skipTest(str(exc))
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            images, masks = root / "images", root / "masks"
            images.mkdir()
            masks.mkdir()
            image = np.random.default_rng(73).integers(0, 256, (320, 440), dtype=np.uint8)
            Image.fromarray(image).save(images / "shot.png")
            keep = np.ones(image.shape, dtype=bool)
            keep[:, :220] = False
            export_mask(masks / "shot.png.png", keep)
            frame = {"frame_number": 17, "filename": "shot.png", "width": 440, "height": 320}
            counts = []
            for mode, settings in [("baseline", {}), ("masked", {"mask_path": masks})]:
                path = root / (mode + ".db")
                solver.extract_features(
                    path, images, reader_options=solver.ImageReaderOptions(**settings),
                    extraction_options=solver.FeatureExtractionOptions(num_threads=2, use_gpu=False),
                    device=solver.Device.cpu,
                )
                database = solver.Database.open(path)
                try:
                    counts.append((database.num_keypoints(), count_masked_keypoints(database, masks, [frame])["17"]))
                finally:
                    database.close()
            self.assertGreater(counts[0][1], 0)
            self.assertEqual(counts[1], (counts[0][0] - counts[0][1], 0))


class ForegroundApiTests(unittest.TestCase):
    def setUp(self):
        self.directory = Path(self.enterContext(tempfile.TemporaryDirectory()))
        self.enterContext(patch.object(main, "DATA_DIR", self.directory))
        self.client = self.enterContext(TestClient(main.app))
        result = self.client.post("/upload/", files=[
            ("files", (f"shot_{number}.png", png(64, 32), "image/png")) for number in (11, 13, 14)
        ])
        result.raise_for_status()
        self.sequence_url = "/sequences/" + result.json()["sequence_id"]

    def request_mask(self, decision="exclude"):
        mask = np.zeros((32, 64), dtype=bool)
        mask[8:20, 20:40] = True
        service = Mock(device="test", input_size=512)
        service.segment.return_value = [(mask, 0.9)]
        with patch("backend.foreground.get_sam", return_value=service):
            return self.client.post(self.sequence_url + "/foreground", json={"prompts": [
                {"frame_number": 13, "box": [20, 8, 40, 20], "decision": decision},
            ]})

    def test_preview_export_and_source_number_align(self):
        response = self.request_mask()
        self.assertEqual(response.status_code, 200, response.text)
        manifest = response.json()
        self.assertEqual(manifest["masked_frames"], 1)
        frame = manifest["frames"][1]
        self.assertEqual(frame["frame_number"], 13)
        self.assertEqual(self.client.get(frame["preview_url"]).headers["content-type"], "image/png")
        for kind in ("mask", "candidate"):
            image = Image.open(BytesIO(self.client.get(frame[kind + "_url"]).content))
            self.assertEqual(image.size, (64, 32))
            self.assertEqual(image.getpixel((25, 10)), 255)
            self.assertEqual(image.getpixel((2, 2)), 0)

    def test_keep_override_does_not_enable_masked_solve(self):
        result = self.request_mask("keep").json()
        self.assertEqual(result["masked_frames"], 0)
        response = self.client.post(self.sequence_url + "/reconstruct", params={"mask_set_id": result["mask_set_id"]})
        self.assertEqual(response.status_code, 422)

    def test_missing_frame_invalid_prompt_and_unavailable_sam(self):
        endpoint = self.sequence_url + "/foreground"
        for number, box in [(12, [1, 1, 20, 20]), (13, [1, 1, 70, 20])]:
            self.assertEqual(self.client.post(endpoint, json={
                "prompts": [{"frame_number": number, "box": box, "decision": "exclude"}],
            }).status_code, 422)
        with patch("backend.foreground.get_sam", side_effect=SamUnavailable("checkpoint unavailable")):
            response = self.client.post(endpoint, json={
                "prompts": [{"frame_number": 13, "box": [1, 1, 20, 20], "decision": "exclude"}],
            })
        self.assertEqual(response.status_code, 503)

    def test_comparison_retains_failed_mode_and_applies_masks_only_in_masked_mode(self):
        manifest = self.request_mask().json()
        with patch.object(main, "reconstruct", side_effect=[
            ReconstructionError("no model", {"keypoints_in_exclusion_regions": 12}),
            {"status": "complete", "statistics": {"keypoints_in_exclusion_regions": 0}},
        ]) as solve:
            response = self.client.post(manifest["manifest_url"] + "/compare")
        self.assertEqual(response.status_code, 200, response.text)
        baseline, masked = [call.kwargs for call in solve.call_args_list]
        self.assertIsNone(baseline["mask_dir"])
        self.assertEqual(masked["mask_dir"], baseline["audit_mask_dir"])
        result = response.json()
        self.assertEqual(result["baseline"]["status"], "failed")
        self.assertEqual(result["excluded_baseline_keypoints"], 12)
        self.assertEqual(result["masked"]["status"], "complete")
        self.assertEqual(self.client.get(result["comparison_url"]).json(), result)


if __name__ == "__main__":
    unittest.main()
