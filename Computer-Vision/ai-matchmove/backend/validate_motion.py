"""Compare real SAM masks using global and tiled corner sampling on labeled footage."""

import argparse
import json
from pathlib import Path
from unittest.mock import patch

import numpy as np
from PIL import Image

from . import motion
from .foreground import automatic_exclusion_supported
from .sam_service import Prompt, get_sam


def rgb(path):
    with Image.open(path) as image:
        return np.asarray(image.convert("RGB"))


def global_corners(cv, gray):
    return cv.goodFeaturesToTrack(gray, maxCorners=1800, qualityLevel=0.008, minDistance=6)


def validate(folder, offset, threshold, numbers=None):
    images = {int(path.stem): path for path in (folder / "images").glob("*.jpg")}
    annotations = sorted((folder / "masks").glob("*.png"))
    selected = [path for path in annotations if numbers is None or int(path.stem) in numbers]
    if not selected:
        raise ValueError("No annotated frames were selected.")
    cv = motion.load_cv()
    sam = get_sam()
    stats = {name: dict(frames_with_masks=0, target_frames=0, suggestions_hit=0, suggestions_off_target=0)
             for name in ("global", "tiled")}
    for label in selected:
        number = int(label.stem)
        neighbors = (number - offset, number + offset)
        image = rgb(images[number])
        adjacent = [(n, rgb(images[n])) for n in neighbors]
        with Image.open(label) as annotated:
            target = np.asarray(annotated) > 0
        if target.shape != image.shape[:2]:
            raise ValueError(f"Annotation dimensions differ from frame {number}.")

        tracks = {"tiled": [(n, motion.track_pair(cv, image, neighbor)) for n, neighbor in adjacent]}
        with patch.object(motion, "sample_corners", global_corners):
            tracks["global"] = [(n, motion.track_pair(cv, image, neighbor)) for n, neighbor in adjacent]
        proposals = {}
        for name, pair_tracks in tracks.items():
            cv.setRNGSeed(513)
            proposals[name] = motion.suggest_boxes(cv, pair_tracks, target.shape)
        boxes = dict.fromkeys(box for candidates in proposals.values() for box in candidates)
        masks = dict(zip(boxes, sam.segment(image, [Prompt(frame_number=number, box=box) for box in boxes]), strict=True))

        for name, candidates in proposals.items():
            counts = stats[name]
            counts["frames_with_masks"] += bool(candidates)
            hit_frame = False
            for box in candidates:
                predicted, _ = masks[box]
                iou = np.count_nonzero(predicted & target) / max(1, np.count_nonzero(predicted | target))
                hit = bool(iou >= threshold)
                hit_frame |= hit
                evidence = motion.motion_evidence(cv, tracks[name], predicted)
                if automatic_exclusion_supported(evidence):
                    counts["suggestions_hit" if hit else "suggestions_off_target"] += 1
            counts["target_frames"] += hit_frame
    return {"frames": len(selected), "iou_threshold": threshold, "results": stats}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("clip", type=Path, help="Folder containing images/ and masks/ with numeric filenames")
    parser.add_argument("--neighbor-offset", type=int, default=3)
    parser.add_argument("--iou", type=float, default=0.2)
    parser.add_argument("--frames", type=int, nargs="+")
    args = parser.parse_args()
    if args.neighbor_offset < 1 or not 0 < args.iou <= 1:
        parser.error("Use a positive neighbor offset and an IoU in (0, 1].")
    print(json.dumps(validate(args.clip, args.neighbor_offset, args.iou, args.frames), indent=2))
