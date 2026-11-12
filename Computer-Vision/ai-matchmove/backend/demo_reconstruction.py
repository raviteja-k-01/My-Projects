"""Generate a textured multi-depth scene, upload it, and validate the actual solve."""

import argparse
from io import BytesIO
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

WIDTH, HEIGHT, FOCAL = 960, 640, 1050
MARGIN = 360
CANVAS = (WIDTH + 2 * MARGIN, HEIGHT + 2 * MARGIN)


def texture(size, rng):
    pixels = rng.integers(75, 180, (size[1], size[0], 3), dtype=np.uint8)
    image = Image.fromarray(pixels)
    draw = ImageDraw.Draw(image)
    for _ in range(1800):
        x, y = (int(value) for value in rng.integers((0, 0), size))
        width, height = (int(value) for value in rng.integers(4, 42, size=2))
        color = tuple(int(value) for value in rng.integers(15, 245, size=3))
        shape = draw.rectangle if width < height else draw.ellipse
        shape((x, y, x + width, y + height), fill=color, outline=(20, 20, 20), width=2)
    return image


def render_sequence():
    rng = np.random.default_rng(821)
    layers = [(7.5, texture(CANVAS, rng).convert("RGBA"))]
    for depth, position, size in [
        (5.0, (520, 720), (650, 260)),
        (3.7, (335, 420), (350, 265)),
        (3.1, (1000, 440), (330, 310)),
    ]:
        layer = Image.new("RGBA", CANVAS)
        layer.paste(texture(size, rng).convert("RGBA"), position)
        layers.append((depth, layer))

    intrinsics = np.array([[FOCAL, 0, WIDTH / 2], [0, FOCAL, HEIGHT / 2], [0, 0, 1]])
    inverse_intrinsics = np.linalg.inv(intrinsics)
    texture_intrinsics = intrinsics.copy()
    texture_intrinsics[:2, 2] += MARGIN
    centers, rotations = [], []
    files = []
    for number, phase in enumerate(np.linspace(-1, 1, 9), start=101):
        center = np.array([0.85 * phase, 0.26 * (1 - phase**2), 0.32 * np.sin(np.pi * phase / 2)])
        yaw, pitch = -0.20 * phase, 0.08 * np.cos(np.pi * phase)
        cy, sy = np.cos(yaw), np.sin(yaw)
        cp, sp = np.cos(pitch), np.sin(pitch)
        rotation = np.array([[cy, 0, sy], [0, 1, 0], [-sy, 0, cy]]) @ np.array(
            [[1, 0, 0], [0, cp, -sp], [0, sp, cp]]
        )
        centers.append(center)
        rotations.append(rotation)
        frame = Image.new("RGBA", (WIDTH, HEIGHT))
        for depth, layer in layers:
            # Map output pixel rays to their intersections with the plane in the reference view.
            plane = (depth - center[2]) * np.eye(3) + np.outer(center, [0, 0, 1])
            homography = texture_intrinsics @ plane @ rotation @ inverse_intrinsics
            mapping = (homography / homography[2, 2]).ravel()[:8]
            projected = layer.transform(
                frame.size, Image.Transform.PERSPECTIVE, mapping, Image.Resampling.BILINEAR
            )
            frame = Image.alpha_composite(frame, projected)
        buffer = BytesIO()
        frame.convert("RGB").save(buffer, format="PNG")
        files.append(("files", (f"synthetic_{number:04d}.png", buffer.getvalue(), "image/png")))
    return files, np.array(centers), np.array(rotations)


def aligned_pose_errors(registered, expected_centers, expected_rotations):
    centers = np.array([frame["position_world"] for frame in registered])
    estimated = centers - centers.mean(axis=0)
    expected = expected_centers - expected_centers.mean(axis=0)
    left, singular_values, right = np.linalg.svd(estimated.T @ expected)
    signs = np.array([1, 1, np.linalg.det(left @ right)])
    alignment = (left * signs) @ right
    scale = (singular_values @ signs) / np.sum(estimated**2)
    rmse = np.sqrt(np.mean(np.sum((scale * estimated @ alignment - expected) ** 2, axis=1)))
    rotations = np.array([frame["rotation_camera_to_world"] for frame in registered])
    aligned_rotations = alignment.T @ rotations
    rotation_errors = expected_rotations.transpose(0, 2, 1) @ aligned_rotations
    cos_angles = (np.trace(rotation_errors, axis1=1, axis2=2) - 1) / 2
    max_angle = np.degrees(np.arccos(np.clip(cos_angles, -1, 1))).max()
    return float(rmse), float(max_angle)


def run(output):
    from fastapi.testclient import TestClient

    from . import main

    main.DATA_DIR = output.resolve()
    files, expected_centers, expected_rotations = render_sequence()
    with TestClient(main.app) as client:
        upload = client.post("/upload/", files=files[::-1], data={"fps": "24000/1001"})
        upload.raise_for_status()
        sequence = upload.json()
        assert sequence["status"] == "done", sequence["failed_uploads"]
        solve = client.post(f"/sequences/{sequence['sequence_id']}/reconstruct")
        if not solve.is_success:
            raise RuntimeError(f"Solve HTTP {solve.status_code}: {solve.text}")
        response = client.get(solve.json()["result_url"])
        response.raise_for_status()
        result = response.json()

    frames = result["registered_frames"]
    print("Saved:", main.DATA_DIR / sequence["sequence_id"] / "reconstruction.json")
    print("Statistics:", result["statistics"])
    assert [frame["frame_number"] for frame in frames] == list(range(101, 110))
    assert result["fps_rational"] == [24000, 1001]
    assert result["resolution"] == {"width": WIDTH, "height": HEIGHT}
    assert result["statistics"]["sparse_points"] >= 100
    assert result["statistics"]["mean_reprojection_error_px"] < 1.0
    rmse, max_angle = aligned_pose_errors(frames, expected_centers, expected_rotations)
    print(f"Center RMSE after similarity alignment: {rmse:.5f} scene units")
    print(f"Maximum rotation error after alignment: {max_angle:.3f} degrees")
    print("Camera intrinsics:", result["cameras"])
    for frame in frames:
        print(f"Frame {frame['frame_number']}: center {np.round(frame['position_world'], 5).tolist()}")
    assert rmse < 0.06
    assert max_angle < 5.0
    assert abs(result["cameras"][0]["calibration_matrix"][0][0] / FOCAL - 1) < 0.03


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--frames-only", action="store_true", help="Write the nine demo PNGs for upload in the browser.")
    args = parser.parse_args()
    if args.frames_only:
        try:
            args.output.mkdir(parents=True)
        except FileExistsError:
            parser.error("Choose a new output directory to generate demo frames.")
        files, _, _ = render_sequence()
        for _, (name, pixels, _) in files:
            (args.output / name).write_bytes(pixels)
        print(f"Wrote {len(files)} frames to {args.output}. Set Sequence FPS to 24000/1001 on upload.")
    else:
        run(args.output)
