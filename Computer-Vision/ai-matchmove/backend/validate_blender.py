"""Check an imported .blend against its extracted package inside Blender."""

import json
from pathlib import Path
import sys

import bpy
from bpy_extras.object_utils import world_to_camera_view
from mathutils import Matrix, Vector


def validate(root):
    package = json.loads((root / "camera.json").read_text(encoding="utf-8"))
    solve = json.loads((root / "reconstruction.json").read_text(encoding="utf-8"))
    scene = bpy.context.scene
    camera = scene.camera
    width, height = package["resolution"]["width"], package["resolution"]["height"]
    assert (scene.render.resolution_x, scene.render.resolution_y) == (width, height)
    assert abs(scene.render.fps / scene.render.fps_base - solve["fps"]) < 1e-4
    assert scene.frame_start == 1 and scene.frame_end == solve["end_frame"] - solve["start_frame"] + 1
    points = solve["sparse_points"]
    mesh = bpy.data.objects["MatchMove Sparse Points"].data
    assert len(mesh.vertices) == len(points)
    xyz = {}
    for vertex, point in zip(mesh.vertices, points, strict=True):
        expected = Vector(point["xyz_world"])
        assert (vertex.co - expected).length <= max(1.0, expected.length) * 1e-6
        xyz[point["point_id"]] = expected

    intrinsics = {item["camera_id"]: item["parameters"] for item in solve["cameras"]}
    available = {frame["frame_number"] - solve["start_frame"] + 1 for frame in solve["registered_frames"]}
    assert {pose["frame"]: pose["source_frame"] for pose in package["frames"]} == {
        frame: solve["start_frame"] + frame - 1 for frame in available
    }
    plates = root / "plates"
    assert {path.name for path in plates.glob("*.png")} == {f"{frame:010d}.png" for frame in available}
    for frame in {min(available), max(available)}:
        plate = bpy.data.images.load(str(plates / f"{frame:010d}.png"))
        assert tuple(plate.size) == (width, height)
        bpy.data.images.remove(plate)
    channels = camera.animation_data.action.fcurves
    location = next(channel for channel in channels if channel.data_path == "location")
    assert {int(key.co.x) for key in location.keyframe_points} == available
    gaps = [frame for frame in range(scene.frame_start, scene.frame_end + 1) if frame not in available]
    for frame in gaps:
        scene.frame_set(frame)
        assert not camera["matchmove_pose_available"], frame

    max_pixel_difference = 0.0
    checked = 0
    for pose in solve["registered_frames"]:
        frame = pose["frame_number"] - solve["start_frame"] + 1
        scene.frame_set(frame)
        assert camera["matchmove_pose_available"], frame
        world_to_camera = Matrix(pose["rotation_world_to_camera"])
        expected_rotation = Matrix(pose["rotation_camera_to_world"]) @ Matrix.Diagonal((1, -1, -1))
        center = Vector(pose["position_world"])
        assert (camera.location - center).length <= max(1.0, center.length) * 1e-6
        assert all((a - b).length < 1e-5 for a, b in zip(camera.rotation_quaternion.to_matrix(), expected_rotation))
        f, cx, cy, _ = intrinsics[pose["camera_id"]]
        assert abs(camera.data.lens - 36 * f / width) < 1e-4
        assert abs(camera.data.shift_x - (width / 2 - cx) / width) < 1e-6
        assert abs(camera.data.shift_y - (cy - height / 2) / width) < 1e-6
        for _, _, point_id in pose["observations"][::max(1, len(pose["observations"]) // 12)]:
            point = xyz[point_id]
            local = world_to_camera @ (point - center)
            assert local.z > 0
            x, y, _ = world_to_camera_view(scene, camera, point)
            expected_x = f * local.x / local.z + cx
            expected_y = f * local.y / local.z + cy
            difference = max(abs(x * width - expected_x), abs((1 - y) * height - expected_y))
            max_pixel_difference = max(max_pixel_difference, difference)
            checked += 1
    assert checked > 0 and max_pixel_difference < 0.02, max_pixel_difference
    print(json.dumps({"poses": len(available), "missing_pose_frames": len(gaps), "points": len(points),
                      "projection_samples": checked, "max_pixel_difference": max_pixel_difference}))


if __name__ == "__main__":
    validate(Path(sys.argv[sys.argv.index("--") + 1]).resolve())
