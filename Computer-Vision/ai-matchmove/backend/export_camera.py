"""Package the sparse reconstruction and a Blender importer."""

import json
from pathlib import Path
from zipfile import ZIP_DEFLATED, ZipFile

import numpy as np


def blender_data(result):
    fps = result.get("fps")
    if not fps or not np.isfinite(fps) or not 1 <= fps <= 32767:
        raise ValueError("Set a source FPS between 1 and 32767 before exporting to Blender.")
    cameras = {}
    for camera in result["cameras"]:
        if camera["model"] != "SIMPLE_RADIAL":
            raise ValueError(f"Blender export does not support {camera['model']}.")
        f, cx, cy, k = camera["parameters"]
        width, height = camera["width"], camera["height"]
        cameras[camera["camera_id"]] = {
            "focal_mm": 36 * f / width, "shift_x": (width / 2 - cx) / width,
            "shift_y": (cy - height / 2) / width, "radial_k": k,
        }
    frames = []
    for frame in result["registered_frames"]:
        frames.append({
            "frame": frame["frame_number"] - result["start_frame"] + 1,
            "source_frame": frame["frame_number"],
            "timestamp_seconds": frame.get("timestamp_seconds"),
            "source_name": frame["source_name"],
            "position": frame["position_world"],
            "rotation": (np.asarray(frame["rotation_camera_to_world"]) @ np.diag([1, -1, -1])).tolist(),
            "lens": cameras[frame["camera_id"]],
        })
    return {
        "fps": fps, "resolution": result["resolution"], "frames": frames,
        "scale": "Arbitrary monocular reconstruction units",
    }


def point_cloud_ply(points):
    header = (
        f"ply\nformat ascii 1.0\nelement vertex {len(points)}\n"
        "property float x\nproperty float y\nproperty float z\n"
        "property uchar red\nproperty uchar green\nproperty uchar blue\nend_header\n"
    )
    return header + "".join(" ".join(map(str, point["xyz_world"] + point["rgb"])) + "\n" for point in points)


def export_package(result, image_dir, target, progress=None):
    data = blender_data(result)
    with ZipFile(target, "w", compression=ZIP_DEFLATED) as package:
        package.writestr("reconstruction.json", json.dumps(result, allow_nan=False))
        package.writestr("camera.json", json.dumps(data, indent=2, allow_nan=False))
        quality = {**result["statistics"], "solver": result.get("solver"), "diagnostics": result.get("diagnostics")}
        quality["frames"] = {frame["frame_number"]: frame["quality"] for frame in result["registered_frames"] if "quality" in frame}
        package.writestr("quality.json", json.dumps(quality, indent=2, allow_nan=False))
        package.writestr("points.ply", point_cloud_ply(result["sparse_points"]))
        package.write(Path(__file__).with_name("import_matchmove.py"), "import_matchmove.py")
        package.writestr(
            "README.txt",
            "Extract the entire ZIP. In Blender's Scripting workspace open import_matchmove.py and run it.\n"
            "The importer creates the animated camera and sparse point mesh. Source frame numbers map to\n"
            "Blender frames in camera.json. Keyframes exist only for registered frames. Treat intervening\n"
            "poses as unavailable. Scale is arbitrary. Corrected plates in plates/ are undistorted to the\n"
            "exported camera and are present only for registered frames. Source timestamps are in camera.json.\n"
            "For variable-rate sources the Blender timeline uses average FPS. Use timestamps for retiming.\n",
        )
        for index, (frame, pose) in enumerate(zip(result["registered_frames"], data["frames"], strict=True)):
            if progress:
                progress("Writing corrected plates", index, len(data["frames"]))
            original = image_dir / frame["filename"]
            name = f"plates/{pose['frame']:010d}.png"
            k = pose["lens"]["radial_k"]
            if abs(k) < 1e-12:
                package.write(original, name)
                continue
            try:
                import cv2
            except ImportError as exc:
                raise ValueError("OpenCV is required to export corrected plates for this lens.") from exc
            camera = next(item for item in result["cameras"] if item["camera_id"] == frame["camera_id"])
            pixels = cv2.imdecode(np.frombuffer(original.read_bytes(), np.uint8), cv2.IMREAD_UNCHANGED)
            matrix = np.array(camera["calibration_matrix"], dtype=np.float64)
            # COLMAP pixel centers are at .5; OpenCV pixel centers are integers.
            matrix[:2, 2] -= 0.5
            corrected = cv2.undistort(pixels, matrix, np.array([k, 0, 0, 0, 0]))
            ok, png = cv2.imencode(".png", corrected)
            if not ok:
                raise ValueError("A corrected frame could not be encoded.")
            package.writestr(name, png.tobytes())
