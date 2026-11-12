"""Run in Blender from the extracted camera package."""

import json
from pathlib import Path
import sys

import bpy
from mathutils import Matrix


def key_camera(scene, obj, poses, frame_end):
    camera = obj.data
    edit = bpy.context.preferences.edit
    interpolation = edit.keyframe_new_interpolation_type
    edit.keyframe_new_interpolation_type = "CONSTANT"
    try:
        previous_quaternion = None
        previous_frame = 0
        for pose in poses:
            frame = pose["frame"]
            if frame > previous_frame + 1:
                obj["matchmove_pose_available"] = False
                obj.keyframe_insert(data_path='["matchmove_pose_available"]', frame=previous_frame + 1)
                scene.timeline_markers.new("No solved pose", frame=previous_frame + 1)
            obj["matchmove_pose_available"] = True
            obj.keyframe_insert(data_path='["matchmove_pose_available"]', frame=frame)
            obj.location = pose["position"]
            quaternion = Matrix(pose["rotation"]).to_quaternion()
            if previous_quaternion is not None:
                quaternion.make_compatible(previous_quaternion)
            previous_quaternion = quaternion.copy()
            previous_frame = frame
            obj.rotation_quaternion = quaternion
            obj.keyframe_insert(data_path="location", frame=frame)
            obj.keyframe_insert(data_path="rotation_quaternion", frame=frame)
            for prop, field in (("lens", "focal_mm"), ("shift_x", "shift_x"), ("shift_y", "shift_y")):
                setattr(camera, prop, pose["lens"][field])
                camera.keyframe_insert(data_path=prop, frame=frame)
            scene.timeline_markers.new(f"Source {pose['source_frame']}", frame=frame)
        if previous_frame < frame_end:
            obj["matchmove_pose_available"] = False
            obj.keyframe_insert(data_path='["matchmove_pose_available"]', frame=previous_frame + 1)
            scene.timeline_markers.new("No solved pose", frame=previous_frame + 1)
    finally:
        edit.keyframe_new_interpolation_type = interpolation


def import_camera(root):
    data = json.loads((root / "camera.json").read_text(encoding="utf-8"))
    result = json.loads((root / "reconstruction.json").read_text(encoding="utf-8"))
    scene = bpy.context.scene
    camera = bpy.data.cameras.new("MatchMove Lens")
    camera.sensor_fit = "HORIZONTAL"
    camera.sensor_width = 36
    obj = bpy.data.objects.new("MatchMove Camera", camera)
    scene.collection.objects.link(obj)
    scene.camera = obj
    obj.rotation_mode = "QUATERNION"
    scene.render.resolution_x = data["resolution"]["width"]
    scene.render.resolution_y = data["resolution"]["height"]
    scene.render.resolution_percentage = 100
    scene.render.pixel_aspect_x = scene.render.pixel_aspect_y = 1
    scene.render.fps = max(1, round(data["fps"]))
    scene.render.fps_base = scene.render.fps / data["fps"]
    scene.frame_start = 1
    scene.frame_end = result["end_frame"] - result["start_frame"] + 1
    key_camera(scene, obj, data["frames"], scene.frame_end)
    scene.frame_set(data["frames"][0]["frame"])
    mesh = bpy.data.meshes.new("MatchMove Sparse Points")
    mesh.from_pydata([point["xyz_world"] for point in result["sparse_points"]], [], [])
    mesh.update()
    scene.collection.objects.link(bpy.data.objects.new(mesh.name, mesh))
    return scene


if __name__ == "__main__":
    args = sys.argv[sys.argv.index("--") + 1:] if "--" in sys.argv else []
    space = bpy.context.space_data
    text = space.text if space and space.type == "TEXT_EDITOR" else None
    root = Path(args[0]) if args else Path(bpy.path.abspath(text.filepath) if text else __file__).resolve().parent
    import_camera(root)
    if len(args) > 1:
        bpy.ops.wm.save_as_mainfile(filepath=str(Path(args[1]).resolve()))
