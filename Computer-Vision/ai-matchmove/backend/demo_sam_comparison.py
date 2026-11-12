"""Run a real SAM/PyCOLMAP comparison on a scene with a known moving occluder."""

import argparse
from io import BytesIO
from pathlib import Path

import numpy as np
from fastapi.testclient import TestClient
from PIL import Image, ImageDraw

from . import main
from .demo_reconstruction import aligned_pose_errors, render_sequence, texture
from .sequence import write_json


def moving_sequence():
    files, centers, rotations = render_sequence()
    rng = np.random.default_rng(301)
    size = (216, 168)
    pattern = texture(size, rng)
    draw = ImageDraw.Draw(pattern)
    draw.rounded_rectangle((0, 0, size[0] - 1, size[1] - 1), radius=22, outline=(244, 208, 54), width=11)
    alpha = Image.new("L", size)
    ImageDraw.Draw(alpha).rounded_rectangle((0, 0, size[0] - 1, size[1] - 1), radius=22, fill=255)
    frames, prompts, truth = [], [], {}
    for i, (_, (name, content, media_type)) in enumerate(files):
        number = 101 + i
        frame = Image.open(BytesIO(content)).convert("RGB")
        x, y = 300 + 20 * i, 80 + 34 * i
        frame.paste(pattern, (x, y), alpha)
        expected = Image.new("L", frame.size)
        expected.paste(alpha, (x, y))
        truth[number] = np.asarray(expected) > 0
        prompts.append({
            "frame_number": number,
            "box": [x - 4, y - 4, x + size[0] + 4, y + size[1] + 4],
            "points": [[x + size[0] / 2, y + size[1] / 2, 1]],
        })
        buffer = BytesIO()
        frame.save(buffer, format="PNG")
        frames.append(("files", (name, buffer.getvalue(), media_type)))
    return frames, prompts, truth, centers, rotations


def run(output):
    main.DATA_DIR = output.resolve()
    files, prompts, truth, centers, rotations = moving_sequence()
    with TestClient(main.app) as client:
        upload = client.post("/upload/", files=files[::-1], data={"fps": "24000/1001"})
        upload.raise_for_status()
        sequence = upload.json()
        response = client.post(f"/sequences/{sequence['sequence_id']}/foreground", json={"prompts": prompts})
        if not response.is_success:
            raise RuntimeError(f"Foreground HTTP {response.status_code}: {response.text}")
        manifest = response.json()
        assert manifest["model"] == "SAM ViT-B"
        overlap = {}
        for frame in manifest["frames"]:
            number = frame["frame_number"]
            mask_response = client.get(frame["candidate_url"])
            mask_response.raise_for_status()
            mask = np.asarray(Image.open(BytesIO(mask_response.content)).convert("L")) > 0
            expected = truth[number]
            assert mask.shape == expected.shape
            overlap[str(number)] = float(np.count_nonzero(mask & expected) / np.count_nonzero(mask | expected))
        print("Masked frames:", manifest["masked_frames"], "SAM IoU:", overlap, flush=True)
        assert manifest["masked_frames"] >= 5
        assert np.mean(list(overlap.values())) >= 0.75

        # This fixture's known moving object provides the human-review equivalent.
        reviews = [{"frame_number": frame["frame_number"], "region_index": index, "decision": "exclude"}
                   for frame in manifest["frames"] for index, region in enumerate(frame["regions"])
                   if region["excluded"]]
        reviewed = client.post(manifest["manifest_url"] + "/review", json={"decisions": reviews})
        reviewed.raise_for_status()
        manifest = reviewed.json()
        compared = client.post(manifest["manifest_url"] + "/compare")
        if not compared.is_success:
            raise RuntimeError(f"Comparison HTTP {compared.status_code}: {compared.text}")
        report = compared.json()
        fetched = client.get(report["comparison_url"])
        fetched.raise_for_status()
        assert fetched.json() == report

        pose_errors = {}
        for mode in ("baseline", "masked"):
            run_result = report[mode]
            if "result_url" not in run_result:
                continue
            response = client.get(run_result["result_url"])
            response.raise_for_status()
            result = response.json()
            registered = result["registered_frames"]
            indices = [frame["frame_number"] - 101 for frame in registered]
            assert result["solver"]["foreground_masking"] == (mode == "masked")
            if len(indices) >= 3:
                position_rmse, max_rotation_error = aligned_pose_errors(registered, centers[indices], rotations[indices])
                pose_errors[mode] = {
                    "position_rmse_aligned": position_rmse,
                    "maximum_rotation_error_deg_aligned": max_rotation_error,
                }

    target = main.DATA_DIR / sequence["sequence_id"] / "foreground" / manifest["mask_set_id"] / report["comparison_id"]
    validation = {
        **report, "masked_frames": manifest["masked_frames"], "sam_iou_by_frame": overlap,
        "sam_mean_iou": float(np.mean(list(overlap.values()))), "pose_errors": pose_errors,
    }
    write_json(target / "validation.json", validation)
    print("Validation:", target / "validation.json")
    for mode in ("baseline", "masked"):
        print(mode, report[mode]["status"], report[mode]["statistics"])
    print("Pose errors:", pose_errors)
    assert report["excluded_baseline_keypoints"] > 0
    assert report["masked"]["status"] in {"complete", "partial"}
    assert report["masked"]["statistics"]["keypoints_in_exclusion_regions"] == 0
    assert report["masked"]["statistics"]["registered_frames"] >= 5


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    run(parser.parse_args().output)
