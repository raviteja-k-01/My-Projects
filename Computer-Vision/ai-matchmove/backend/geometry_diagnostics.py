"""Bounded diagnostics derived from COLMAP's actual correspondence graph."""

from collections import Counter, defaultdict
from functools import lru_cache

import numpy as np


HOMOGRAPHY = {"PLANAR", "PANORAMIC", "PLANAR_OR_PANORAMIC"}
EPIPOLAR = {"CALIBRATED", "UNCALIBRATED"}
# These are the default COLMAP absolute-pose support and final residual limits.
MIN_MULTIVIEW_TRACKS = 30
MAX_MEAN_RESIDUAL_PX = 4.0


def inspect_geometry(solver, database, frames, *, probe=False):
    number_by_name = {frame["filename"]: frame["frame_number"] for frame in frames}
    images = {image.image_id: image for image in database.read_all_images()}
    numbers = {image_id: number_by_name[image.name] for image_id, image in images.items()}
    counts = Counter()
    edges = defaultdict(set)
    pairs = []
    pair_ids, geometries = database.read_two_view_geometries()
    for pair_id, geometry in zip(pair_ids, geometries, strict=True):
        first, second = solver.pair_id_to_image_pair(pair_id)
        support = len(geometry.inlier_matches)
        if support < 15:
            continue
        name = solver.TwoViewGeometryConfiguration(geometry.config).name
        counts[name] += 1
        edges[first].add(second)
        edges[second].add(first)
        pairs.append((first, second, name, support, geometry))
    remaining = set(images)
    components = []
    while remaining:
        stack = [remaining.pop()]
        size = 0
        while stack:
            size += 1
            linked = edges[stack.pop()] & remaining
            remaining.difference_update(linked)
            stack.extend(linked)
        components.append(size)
    ordered = sorted(pairs, key=lambda item: (-item[3], item[0], item[1]))
    widest = sorted(pairs, key=lambda item: (-abs(numbers[item[0]] - numbers[item[1]]), -item[3]))
    selected = {(pair[0], pair[1]): pair for pair in ordered[:8] + widest[:8]}
    sampled = []

    @lru_cache(maxsize=16)
    def keypoints(image_id):
        return np.ascontiguousarray(database.read_keypoints(image_id)[:, :2], dtype=np.float64)

    cameras = {camera.camera_id: camera for camera in database.read_all_cameras()}
    for first, second, name, support, geometry in selected.values():
        matches = geometry.inlier_matches
        points1, points2 = keypoints(first), keypoints(second)
        camera1, camera2 = cameras[images[first].camera_id], cameras[images[second].camera_id]
        a, b = points1[matches[:, 0]], points2[matches[:, 1]]
        occupied = []
        for points, camera in ((a, camera1), (b, camera2)):
            cells = np.clip((points / (camera.width, camera.height) * 4).astype(int), 0, 3)
            occupied.append(len(np.unique(cells[:, 0] + 4 * cells[:, 1])) / 16)
        sample = {
            "frames": [numbers[first], numbers[second]], "geometry": name, "inliers": support,
            "grid_coverage_fraction": occupied,
            "median_motion_px": float(np.median(np.linalg.norm(a - b, axis=1))),
        }
        if probe and name in EPIPOLAR | HOMOGRAPHY:
            # Re-estimate a calibrated model on the stored inliers. Its angle is
            # conditional on the provisional database intrinsics.
            options = solver.TwoViewGeometryOptions(compute_relative_pose=True)
            options.ransac.max_num_trials = 512
            relative = solver.estimate_calibrated_two_view_geometry(
                camera1, points1, camera2, points2, matches, options
            )
            angle = np.degrees(relative.tri_angle)
            if relative.cam2_from_cam1 is not None and np.isfinite(angle) and angle >= 0:
                sample["provisional_triangulation_angle_deg"] = float(angle)
                sample["provisional_geometry"] = solver.TwoViewGeometryConfiguration(relative.config).name
        sampled.append(sample)
    eligible = sum(counts.values())
    return {
        "pair_configurations": dict(counts), "pairs_with_at_least_15_inliers": eligible,
        "homography_fraction": sum(counts[name] for name in HOMOGRAPHY) / eligible if eligible else None,
        "component_sizes": sorted(components, reverse=True),
        "frames_with_verified_neighbors": len(edges),
        "sampled_pairs": sampled,
        "calibration": {
            "model": "SIMPLE_RADIAL", "shared_lens": True, "principal_point": "fixed at image center",
            "focal_length": "refined by bundle adjustment", "radial_distortion": "refined",
            "database_focal_px": [float(camera.focal_length) for camera in cameras.values()],
        },
    }


def failure_diagnosis(geometry, attempts):
    if any(attempt["initializations"] for attempt in attempts):
        return "A seed pair initialized, but registration or multi-view model quality was insufficient."
    if geometry.get("homography_fraction", 0) is not None and geometry.get("homography_fraction", 0) >= 0.5:
        return ("No initialization succeeded. Homographies dominate verified pairs; rotation, planar structure, "
                "or weak parallax can explain strong matches without stable 3D.")
    if len(geometry.get("component_sizes", [])) > 1:
        return "No initialization succeeded. The verified correspondence graph has disconnected frame groups."
    angles = [pair["provisional_triangulation_angle_deg"] for pair in geometry.get("sampled_pairs", [])
              if pair.get("provisional_geometry") in EPIPOLAR and "provisional_triangulation_angle_deg" in pair]
    if angles and max(angles) < 4:
        return (f"No initialization succeeded. The strongest sampled triangulation angle was {max(angles):.2f}° "
                "under the provisional lens calibration, which indicates weak parallax for the tested seed pairs.")
    return ("No initialization succeeded under the tested calibration and pair selection. "
            "The available matches did not establish a usable 3D seed.")


def assess_model(model):
    center_by_image = {image_id: model.images[image_id].projection_center() for image_id in model.reg_image_ids()}
    centers = np.array(list(center_by_image.values()))
    points = list(model.points3D.values())
    track_lengths = [point.track.length() for point in points]
    supported = sum(length >= 3 for length in track_lengths)
    error = float(model.compute_mean_reprojection_error())
    finite = bool(len(centers) and np.isfinite(centers).all() and np.isfinite(error))
    translation = bool(finite and np.any(centers != centers[0]))
    angles = []
    for point in points[::max(1, (len(points) + 255) // 256)]:
        rays = np.array([point.xyz - center_by_image[item.image_id]
                         for item in point.track.elements if item.image_id in center_by_image])
        if len(rays) < 2:
            continue
        norms = np.linalg.norm(rays, axis=1)
        if not np.isfinite(norms).all() or np.any(norms == 0):
            continue
        rays /= norms[:, None]
        # Largest angular separation from one observation, a bounded lower bound
        # on the largest pairwise angle in this track.
        angles.append(float(np.degrees(np.arccos(np.clip(np.min(rays[1:] @ rays[0]), -1, 1)))))
    return {
        "registered_frames": len(centers), "sparse_points": len(track_lengths),
        "tracks_in_at_least_3_frames": supported, "mean_reprojection_error_px": error if finite else None,
        "has_camera_translation": translation,
        "sampled_track_median_angle_deg": float(np.median(angles)) if angles else None,
        "accepted": bool(len(centers) >= 3 and supported >= MIN_MULTIVIEW_TRACKS and translation
                         and 0 <= error <= MAX_MEAN_RESIDUAL_PX),
    }
