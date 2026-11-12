import os
from pathlib import Path
from tempfile import TemporaryDirectory
from threading import Lock
from time import perf_counter

import numpy as np
from PIL import Image

_solve_lock = Lock()

if __package__:
    from .geometry_diagnostics import assess_model, failure_diagnosis, inspect_geometry
    from .sequence import write_json
else:
    from geometry_diagnostics import assess_model, failure_diagnosis, inspect_geometry
    from sequence import write_json


class ReconstructionError(Exception):
    """The input did not yield a usable sparse reconstruction."""

    def __init__(self, message, statistics=None):
        super().__init__(message)
        self.statistics = statistics or {}


class SolverUnavailable(Exception):
    """PyCOLMAP could not be loaded in the running Python environment."""


def load_solver():
    try:
        import pycolmap
    except (ImportError, RuntimeError, OSError) as exc:
        cause = exc.__cause__ or exc
        raise SolverUnavailable(f"PyCOLMAP could not load: {cause}") from exc
    return pycolmap


def pose_record(rotation, translation):
    """COLMAP stores world-to-camera R and t, with x right, y down, z forward."""
    rotation = np.asarray(rotation)
    translation = np.asarray(translation)
    camera_to_world = rotation.T
    return {
        "position_world": (-camera_to_world @ translation).tolist(),
        "rotation_camera_to_world": camera_to_world.tolist(),
        "rotation_world_to_camera": rotation.tolist(),
        "translation_world_to_camera": translation.tolist(),
    }


def serialize_model(model, sequence, metrics, version):
    frame_by_name = {frame["filename"]: frame for frame in sequence["frame_details"]}
    point_by_id = dict(model.points3D.items())
    cameras = []
    registered = []
    for image_id in model.reg_image_ids():
        image = model.images[image_id]
        frame = frame_by_name[image.name]
        transform = image.cam_from_world()
        observations = [
            [float(point.xy[0]), float(point.xy[1]), int(point.point3D_id)]
            for point in image.points2D if point.has_point3D()
        ]
        residuals = []
        if observations:
            world = np.array([point_by_id[point_id].xyz for _, _, point_id in observations])
            local = world @ transform.rotation.matrix().T + transform.translation
            projected = model.cameras[image.camera_id].img_from_cam(local)
            residuals = np.linalg.norm(projected - np.array([item[:2] for item in observations]), axis=1)
            residuals = residuals[np.isfinite(residuals)]
        registered.append({
            **frame,
            "image_id": int(image_id),
            "camera_id": int(image.camera_id),
            "num_points3d": image.num_points3D,
            "observations": observations,
            "quality": {
                "observations": len(observations), "projectable_observations": len(residuals),
                "median_error_px": float(np.median(residuals)) if len(residuals) else None,
                "p90_error_px": float(np.percentile(residuals, 90)) if len(residuals) else None,
            },
            **pose_record(transform.rotation.matrix(), transform.translation),
        })
    registered.sort(key=lambda frame: frame["frame_number"])

    for camera_id in sorted({frame["camera_id"] for frame in registered}):
        camera = model.cameras[camera_id]
        cameras.append({
            "camera_id": camera_id,
            "model": camera.model_name,
            "width": camera.width,
            "height": camera.height,
            "parameter_names": [name.strip() for name in camera.params_info.split(",")],
            "parameters": camera.params.tolist(),
            "calibration_matrix": camera.calibration_matrix().tolist(),
        })

    points = [
        {
            "point_id": int(point_id),
            "xyz_world": point.xyz.tolist(),
            "rgb": point.color.tolist(),
            "mean_reprojection_error_px": float(point.error) if point.error >= 0 else None,
            "track_length": point.track.length(),
        }
        for point_id, point in sorted(point_by_id.items())
    ]
    solved_names = {frame["filename"] for frame in registered}
    unregistered = [frame for name, frame in frame_by_name.items() if name not in solved_names]
    return {
        "schema_version": 1,
        "status": "partial" if unregistered or sequence["missing_frame_ranges"] or sequence["failed_uploads"] else "complete",
        "sequence_id": sequence["sequence_id"],
        "solver": {"name": "PyCOLMAP", "version": version, "method": "incremental", "device": "cpu"},
        "coordinates": {
            "world": "COLMAP reconstruction world, arbitrary origin, orientation and scale",
            "camera_axes": {"x": "right", "y": "down", "z": "forward"},
            "matrix_layout": "row-major, column vectors",
            "world_to_camera": "x_camera = R_world_to_camera @ x_world + t_world_to_camera",
            "camera_to_world": "x_world = R_camera_to_world @ x_camera + position_world",
        },
        "fps": sequence["fps"],
        "fps_rational": sequence["fps_rational"],
        "fps_source": sequence["fps_source"],
        "resolution": sequence["resolution"],
        "numbering": sequence["numbering"],
        "start_frame": sequence["start_frame"],
        "end_frame": sequence["end_frame"],
        "missing_frame_ranges": sequence["missing_frame_ranges"],
        "failed_uploads": sequence["failed_uploads"],
        "cameras": cameras,
        "registered_frames": registered,
        "unregistered_frames": unregistered,
        "sparse_points": points,
        "statistics": {
            **metrics,
            "input_frames": len(frame_by_name),
            "registered_frames": len(registered),
            "registration_fraction": len(registered) / len(frame_by_name),
            "unregistered_frames": len(unregistered),
            "sparse_points": len(points),
            "observations": model.compute_num_observations(),
            "mean_track_length": model.compute_mean_track_length(),
            "mean_reprojection_error_px": model.compute_mean_reprojection_error(),
        },
    }


def count_masked_keypoints(database, mask_dir, frames):
    counts = {}
    for frame in frames:
        filename = frame["filename"]
        with Image.open(mask_dir / (filename + ".png")) as image:
            keep = np.asarray(image.convert("L")) > 0
        if keep.shape != (frame["height"], frame["width"]):
            raise ReconstructionError(f"Mask size differs from frame {frame['frame_number']}.")
        image = database.read_image_with_name(filename)
        if image is None:
            raise ReconstructionError(f"Feature extraction skipped frame {frame['frame_number']}.")
        x, y = np.floor(database.read_keypoints(image.image_id)[:, :2]).astype(int).T
        within_image = (x >= 0) & (y >= 0) & (x < frame["width"]) & (y < frame["height"])
        counts[str(frame["frame_number"])] = int(np.count_nonzero(~keep[y[within_image], x[within_image]]))
    return counts


def reconstruct(sequence, folder, *, mask_dir=None, audit_mask_dir=None, output_path=None, max_frames=None,
                progress=None, preflight=False, initial_focal_px=None):
    with _solve_lock:
        return _reconstruct(sequence, folder, mask_dir, audit_mask_dir, output_path, max_frames, progress,
                            preflight, initial_focal_px)


def sampled_indices(count, maximum):
    if not maximum or count <= maximum:
        return range(count)
    return np.linspace(0, count - 1, maximum, dtype=int).tolist()


def wider_pairs(names, existing, limit=16):
    # Spread the pair budget across the sequence instead of matching all pairs.
    anchors = sampled_indices(len(names), 8)
    pairs = []
    for first in anchors:
        for offset in (len(names) // 2, len(names) * 3 // 4):
            second = first + offset if first + offset < len(names) else first - offset
            if second < 0 or second == first:
                continue
            pair = tuple(sorted((names[first], names[second])))
            if pair not in existing and pair not in pairs:
                pairs.append(pair)
    return pairs[:limit]


def match_statistics(database):
    raw = database.num_matches()
    inliers = database.num_inlier_matches()
    if not isinstance(raw, int):
        return {"matched_pairs": database.num_matched_image_pairs(),
                "verified_pairs": database.num_verified_image_pairs(), "inlier_matches": inliers}
    # COLMAP's table row counts also include attempted pairs with empty matches.
    _, raw_counts = database.read_num_matches()
    _, inlier_counts = database.read_two_view_geometry_num_inliers()
    return {
        "matched_pairs": sum(count > 0 for count in raw_counts),
        "verified_pairs": sum(count > 0 for count in inlier_counts),
        "inlier_matches": inliers, "raw_matches": raw,
        "geometric_inlier_ratio": inliers / raw if raw else None,
    }


def map_with_diagnostics(solver, database_path, image_dir, model_dir, frames, threads, progress, *, inspect=True, cancellation=None):
    attempts = []
    geometry = {}

    def mapping(label):
        if progress:
            progress("Estimating camera poses and sparse points" if not attempts else "Mapping wider-baseline pairs")
        target = model_dir / label
        target.mkdir()
        counters = {"initializations": 0, "additional_registrations": 0}

        def count(key):
            counters[key] += 1

        started = perf_counter()
        models = solver.incremental_mapping(
            database_path, image_dir, target,
            options=solver.IncrementalPipelineOptions(num_threads=threads, min_model_size=3, ba_use_gpu=False),
            initial_image_pair_callback=lambda: count("initializations"),
            next_image_callback=lambda: count("additional_registrations"),
            **(cancellation or {}),
        )
        candidates = [(model, assess_model(model)) for model in models.values()]
        attempts.append({"configuration": label, "elapsed_seconds": perf_counter() - started, **counters,
                         "candidate_models": [report for _, report in candidates]})
        accepted = [model for model, report in candidates if report["accepted"]]
        return max(accepted, key=lambda model: (model.num_reg_images(), model.num_points3D()), default=None)

    model = mapping("sequential")
    if not inspect:
        return model, geometry, attempts
    if progress:
        progress("Checking correspondence geometry")
    database = solver.Database.open(database_path)
    try:
        geometry = inspect_geometry(solver, database, frames, probe=model is None)
        names = {image.image_id: image.name for image in database.read_all_images()}
        raw_ids, _ = database.read_num_matches()
        matched = {tuple(sorted(names[i] for i in solver.pair_id_to_image_pair(pair))) for pair in raw_ids}
    finally:
        database.close()
    # Preserve mapper thresholds. A retry only adds a small number of fresh
    # correspondences when the graph contains non-homography 3D evidence.
    if model is None and (geometry["homography_fraction"] or 0) < 0.5:
        ray_evidence = any(
            pair.get("provisional_geometry") in {"CALIBRATED", "UNCALIBRATED"}
            and pair.get("provisional_triangulation_angle_deg", 0) >= 4
            for pair in geometry["sampled_pairs"]
        )
        pairs = wider_pairs([frame["filename"] for frame in frames], matched) if ray_evidence else []
        if pairs:
            if progress:
                progress("Verifying additional wider-baseline pairs")
            pair_path = model_dir / "recovery_pairs.txt"
            pair_path.write_text("".join(f"{first} {second}\n" for first, second in pairs), encoding="utf-8")
            solver.match_image_pairs(
                database_path,
                pairing_options=solver.ImportedPairingOptions(match_list_path=pair_path),
                matching_options=solver.FeatureMatchingOptions(num_threads=threads, use_gpu=False),
                device=solver.Device.cpu,
                **(cancellation or {}),
            )
            model = mapping("wider_baseline")
            attempts[-1]["additional_pairs_requested"] = len(pairs)
            database = solver.Database.open(database_path)
            try:
                geometry = inspect_geometry(solver, database, frames, probe=model is None)
            finally:
                database.close()
    return model, geometry, attempts


def _reconstruct(sequence, folder, mask_dir, audit_mask_dir, output_path, max_frames, progress, preflight, initial_focal_px):
    start = perf_counter()
    frames = sequence["frame_details"]
    if len(frames) < 3:
        raise ReconstructionError("At least three valid overlapping images are required.")
    if sequence["resolution"] is None:
        raise ReconstructionError("All solve images must have the same resolution.")
    if initial_focal_px is not None and (not np.isfinite(initial_focal_px) or initial_focal_px <= 0):
        raise ReconstructionError("The initial focal length must be positive and finite.")
    if sequence.get("video_metadata", {}).get("sample_aspect_ratio") not in (None, "1:1", "N/A", "0:1"):
        raise ReconstructionError("This solve requires square pixels. Convert the video to square pixels first.")
    omitted = []
    if max_frames and len(frames) > max_frames:
        selected = set(sampled_indices(len(frames), max_frames))
        omitted = [frame["frame_number"] for index, frame in enumerate(frames) if index not in selected]
        frames = [frame for index, frame in enumerate(frames) if index in selected]
        sequence = {**sequence, "frame_details": frames}

    if progress:
        progress("Extracting SIFT features")
    solver = load_solver()
    cancellation = {}
    if callable(getattr(progress, "bind_cancel", None)):
        token = solver.CancellationToken()
        progress.bind_cancel(token.cancel)
        cancellation["cancellation_token"] = token
    threads = min(os.cpu_count() or 1, 2)
    image_dir = folder / "frames"
    with TemporaryDirectory(prefix="matchmove-colmap-") as workspace:
        workspace = Path(workspace)
        database_path = workspace / "database.db"
        model_dir = workspace / "models"
        model_dir.mkdir()

        reader_settings = {"camera_model": "SIMPLE_RADIAL"}
        if initial_focal_px is not None:
            width, height = sequence["resolution"]["width"], sequence["resolution"]["height"]
            reader_settings["camera_params"] = f"{initial_focal_px},{width / 2},{height / 2},0"
        if mask_dir is not None:
            reader_settings["mask_path"] = mask_dir
        solver.extract_features(
            database_path, image_dir,
            image_names=[frame["filename"] for frame in frames],
            camera_mode=solver.CameraMode.SINGLE,
            reader_options=solver.ImageReaderOptions(**reader_settings),
            extraction_options=solver.FeatureExtractionOptions(num_threads=threads, max_image_size=1920, use_gpu=False),
            device=solver.Device.cpu,
            **cancellation,
        )
        if progress:
            progress("Matching features and verifying geometry")
        # This API stores geometrically verified inliers as well as descriptor matches.
        solver.match_sequential(
            database_path,
            matching_options=solver.FeatureMatchingOptions(num_threads=threads, use_gpu=False),
            pairing_options=solver.SequentialPairingOptions(overlap=8, loop_detection=False),
            device=solver.Device.cpu,
            **cancellation,
        )
        database = solver.Database.open(database_path)
        try:
            metrics = {
                "selected_frame_numbers": [frame["frame_number"] for frame in frames],
                "sift_max_image_size": 1920,
                "initial_focal_px": initial_focal_px,
                "detected_keypoints": database.num_keypoints(),
                **match_statistics(database),
            }
            if audit_mask_dir is not None or mask_dir is not None:
                inside = count_masked_keypoints(database, audit_mask_dir or mask_dir, frames)
                metrics["keypoints_in_exclusion_regions_by_frame"] = inside
                metrics["keypoints_in_exclusion_regions"] = sum(inside.values())
                if mask_dir is not None and metrics["keypoints_in_exclusion_regions"]:
                    raise RuntimeError("Masked extraction retained keypoints inside exclusion regions.")
            if preflight:
                if progress:
                    progress("Evaluating provisional geometry")
                geometry = inspect_geometry(solver, database, frames, probe=True)
                return {"statistics": {**metrics, "elapsed_seconds": perf_counter() - start},
                        "geometry": geometry, "assessment": "Preflight only. No camera poses were estimated."}
        finally:
            database.close()
        if not metrics["inlier_matches"]:
            raise ReconstructionError("No geometrically verified feature matches were found.", {
                **metrics, "stage": "feature_matching", "elapsed_seconds": perf_counter() - start,
            })

        model, geometry, attempts = map_with_diagnostics(
            solver, database_path, image_dir, model_dir, frames, threads, progress,
            inspect="raw_matches" in metrics, cancellation=cancellation,
        )
        diagnostics = {"geometry": geometry, "attempts": attempts}
        if len(attempts) > 1:
            database = solver.Database.open(database_path)
            try:
                metrics.update(match_statistics(database))
            finally:
                database.close()
        if model is None:
            raise ReconstructionError(
                f"Found {metrics['verified_pairs']} verified pairs, but no model registered "
                "at least three images and triangulated 3D points with sufficient support. "
                + failure_diagnosis(geometry, attempts),
                {**metrics, "stage": "mapping", "diagnostics": diagnostics,
                 "elapsed_seconds": perf_counter() - start},
            )
        if progress:
            progress("Saving camera poses and feature tracks")
        result = serialize_model(model, sequence, {
            **metrics, "model_count": len(attempts[-1]["candidate_models"]),
            "elapsed_seconds": perf_counter() - start,
        }, solver.__version__)
        result["diagnostics"] = diagnostics
        result["solver"]["pairing"] = attempts[-1]["configuration"]
        result["solver"]["sift_max_image_size"] = 1920
        result["unsampled_frames"] = omitted
        result["source_frame_count"] = sequence.get("source_frame_count", len(frames) + len(omitted))
        if sequence.get("input_type") == "video":
            result.update(input_type="video", video_metadata=sequence["video_metadata"],
                          video_stride=sequence["video_stride"])
        if omitted or sequence.get("video_stride", 1) > 1:
            result["status"] = "partial"
        result["solver"]["foreground_masking"] = mask_dir is not None
        destination = output_path or folder / "reconstruction.json"
        with TemporaryDirectory(prefix=".solve-", dir=destination.parent) as output_dir:
            output = Path(output_dir) / "reconstruction.json"
            write_json(output, result)
            output.replace(destination)
    return result
