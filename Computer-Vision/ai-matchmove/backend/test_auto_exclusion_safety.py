"""Conservative automatic-exclusion guard: no SAM model or solver required."""
import unittest
from unittest.mock import Mock, patch
from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np
from PIL import Image

from . import foreground


class AutoExclusionEvidenceTests(unittest.TestCase):
    def test_frame280_one_neighbor_cannot_auto_exclude_static_railing(self):
        # Reproduces the *shape* of evidence in the supplied frame-280 manifest:
        # 23 tracks, 91.3% outliers to source frame 275; no forward neighbor.
        evidence = [{"neighbor": 275, "tracked_points": 23,
                     "outlier_fraction": 0.913, "median_residual_px": 4.05,
                     "decision": "moving"}]
        self.assertFalse(foreground.automatic_exclusion_supported(evidence))

    def test_two_independent_moving_neighbors_allow_auto_exclusion(self):
        evidence = [
            {"neighbor": 165, "tracked_points": 12, "decision": "moving"},
            {"neighbor": 175, "tracked_points": 11, "decision": "moving"},
        ]
        self.assertTrue(foreground.automatic_exclusion_supported(evidence))

    def test_inconclusive_or_consistent_neighbor_prevents_automatic_exclusion(self):
        moving = {"neighbor": 165, "tracked_points": 12, "decision": "moving"}
        other = {"neighbor": 175, "tracked_points": 2, "decision": "inconclusive"}
        self.assertFalse(foreground.automatic_exclusion_supported([moving, other]))
        self.assertFalse(foreground.automatic_exclusion_supported(
            [moving, {**other, "tracked_points": 12, "decision": "consistent"}]))

    def test_same_neighbor_twice_is_not_independent_support(self):
        evidence = [{"neighbor": 165, "tracked_points": 12, "decision": "moving"}] * 2
        self.assertFalse(foreground.automatic_exclusion_supported(evidence))

    def test_low_track_count_not_auto_approved(self):
        evidence = [
            {"neighbor": 165, "tracked_points": 3, "decision": "moving"},
            {"neighbor": 175, "tracked_points": 5, "decision": "moving"},
        ]
        self.assertFalse(foreground.automatic_exclusion_supported(evidence))

    def test_manual_exclude_still_allowed_with_no_temporal_support(self):
        # Integrates with real build_masks() logic using mocked segmentation.
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            imgs = root / 'images'
            imgs.mkdir()
            frame_specs = []
            for i in [0, 1, 2]:
                file = f'frame_{i}.png'
                Image.fromarray(np.full((40, 70, 3), 100, dtype=np.uint8)).save(imgs / file)
                frame_specs.append({'frame_number': i, 'filename': file, 'width': 70, 'height': 40})
            sequence = {'frame_details': frame_specs, 'resolution': {'width': 70, 'height': 40}}
            mask = np.zeros((40, 70), dtype=bool)
            mask[5:20, 10:30] = True
            service = Mock(device='test', input_size=512)
            service.segment.return_value = [(mask, 0.9)]
            (root/'out').mkdir()
            with patch.object(foreground, 'get_sam', return_value=service):
                result = foreground.build_masks(
                    sequence, imgs, root/'out',
                    foreground.ForegroundRequest(prompts=[
                        foreground.Prompt(frame_number=1, box=(10, 5, 30, 20), decision='exclude')]))
            self.assertEqual(result['masked_frames'], 1)
            self.assertTrue(result['frames'][1]['regions'][0]['excluded'])


if __name__ == '__main__':
    unittest.main()
