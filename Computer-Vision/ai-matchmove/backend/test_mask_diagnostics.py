"""Fast, deterministic checks for foreground-report semantics (no model download)."""

from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import patch

import numpy as np
from PIL import Image

from . import foreground


def _sequence(directory):
    frames = []
    for index in range(3):
        name = f"frame_{index:03}.png"
        Image.fromarray(np.full((40, 60, 3), 50, dtype=np.uint8)).save(directory / name)
        frames.append({"filename": name, "frame_number": index, "width": 60, "height": 40})
    return {"resolution": {"width": 60, "height": 40}, "frame_details": frames}


def test_discovery_without_candidates_reports_no_proposal():
    with TemporaryDirectory() as tmp:
        root = Path(tmp)
        inputs = root / "frames"
        inputs.mkdir()
        sequence = _sequence(inputs)
        destination = root / "masks"
        destination.mkdir()
        # Tracking returns no motion evidence, so SAM must not be called.
        empty = np.empty((0, 2), dtype=np.float32)
        with patch.object(foreground, "track_pair", return_value=(empty, empty)), \
             patch.object(foreground, "get_sam") as get_sam:
            output = foreground.build_masks(
                sequence, inputs, destination,
                foreground.ForegroundRequest(discover=True, discover_frame_numbers=[1]),
            )
        get_sam.assert_not_called()
        assert output["analyzed_frames"] == 1
        assert output["masked_frames"] == 0
        assert output["frame_status_counts"]["no_proposal"] == 1
        assert output["frame_status_counts"]["not_analyzed"] == 2
        assert output["frames"][1]["mask_status"] == "no_proposal"
        assert output["frames"][0]["mask_status"] == "not_analyzed"
        for frame in sequence["frame_details"]:
            keep = np.asarray(Image.open(destination / "colmap" / (frame["filename"] + ".png")))
            assert np.all(keep == 255)

class FakeSam:
    device = "cpu"
    input_size = 512

    def segment(self, image, prompts):
        mask = np.zeros(image.shape[:2], dtype=bool)
        mask[8:22, 12:26] = True
        return [(mask, 0.92) for _ in prompts]


def test_manual_exclusion_is_reported_and_written_to_colmap():
    with TemporaryDirectory() as tmp:
        root = Path(tmp)
        inputs = root / "frames"
        inputs.mkdir()
        sequence = _sequence(inputs)
        destination = root / "masks"
        destination.mkdir()
        prompt = foreground.Prompt(frame_number=1, box=(10, 7, 27, 23), decision="exclude")
        with patch.object(foreground, "get_sam", return_value=FakeSam()):
            output = foreground.build_masks(sequence, inputs, destination,
                                            foreground.ForegroundRequest(prompts=[prompt]))
        assert output["masked_frames"] == 1
        assert output["frames"][1]["mask_status"] == "approved_exclusion"
        assert output["frame_status_counts"]["approved_exclusion"] == 1
        keep = np.asarray(Image.open(destination / "colmap" / "frame_001.png.png"))
        assert keep[15, 19] == 0
        assert keep[1, 1] == 255


def test_manual_keep_is_not_counted_as_exclusion():
    with TemporaryDirectory() as tmp:
        root = Path(tmp)
        inputs = root / "frames"
        inputs.mkdir()
        sequence = _sequence(inputs)
        destination = root / "masks"
        destination.mkdir()
        prompt = foreground.Prompt(frame_number=1, box=(10, 7, 27, 23), decision="keep")
        with patch.object(foreground, "get_sam", return_value=FakeSam()):
            output = foreground.build_masks(sequence, inputs, destination,
                                            foreground.ForegroundRequest(prompts=[prompt]))
        assert output["masked_frames"] == 0
        assert output["frames"][1]["mask_status"] == "manual_keep"
        assert output["frame_status_counts"]["manual_keep"] == 1
        keep = np.asarray(Image.open(destination / "colmap" / "frame_001.png.png"))
        assert np.all(keep == 255)
