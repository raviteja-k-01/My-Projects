"""Use forward/backward LK and two-view geometry to test independent motion."""

import numpy as np

if __package__:
    from .sam_service import SamUnavailable
else:
    from sam_service import SamUnavailable


def load_cv():
    try:
        import cv2
    except (ImportError, OSError) as exc:
        raise SamUnavailable(f"OpenCV motion analysis could not load: {exc}") from exc
    cv2.setNumThreads(min(cv2.getNumThreads(), 2))
    return cv2


def sample_corners(cv, gray):
    height, width = gray.shape
    if height < 28 or width < 42:
        return cv.goodFeaturesToTrack(gray, maxCorners=1800, qualityLevel=0.008, minDistance=6)
    # Cap the contribution of highly textured areas and retain corners in
    # nearby low-contrast regions. The total tracking budget stays at 1,800.
    rows = np.linspace(0, height, 5, dtype=int)
    columns = np.linspace(0, width, 7, dtype=int)
    tiles = []
    for top, bottom in zip(rows[:-1], rows[1:]):
        for left, right in zip(columns[:-1], columns[1:]):
            points = cv.goodFeaturesToTrack(gray[top:bottom, left:right], maxCorners=75,
                                            qualityLevel=0.008, minDistance=6)
            if points is not None:
                points += np.array([left, top], dtype=np.float32)
                tiles.append(points)
    return np.concatenate(tiles) if tiles else None


def track_pair(cv, current, adjacent):
    gray = cv.cvtColor(current, cv.COLOR_RGB2GRAY)
    next_gray = cv.cvtColor(adjacent, cv.COLOR_RGB2GRAY)
    start = sample_corners(cv, gray)
    empty = np.empty((0, 2), np.float32)
    if start is None:
        return empty, empty
    options = {"winSize": (25, 25), "maxLevel": 4}
    end, forward, _ = cv.calcOpticalFlowPyrLK(gray, next_gray, start, None, **options)
    if end is None:
        return empty, empty
    back, reverse, _ = cv.calcOpticalFlowPyrLK(next_gray, gray, end, None, **options)
    if back is None:
        return empty, empty
    height, width = gray.shape
    target = end[:, 0]
    valid = (
        (forward.ravel() == 1) & (reverse.ravel() == 1)
        & (np.linalg.norm((back - start)[:, 0], axis=1) <= 1.5)
        & (target[:, 0] >= 0) & (target[:, 0] < width)
        & (target[:, 1] >= 0) & (target[:, 1] < height)
    )
    return start[valid, 0], target[valid]


def epipolar_errors(cv, start, end, background):
    if np.count_nonzero(background) < 45:
        return None
    fundamental, inliers = cv.findFundamentalMat(
        start[background], end[background], cv.FM_RANSAC, 1.5, 0.995, 4000
    )
    if fundamental is None or fundamental.shape != (3, 3) or inliers is None:
        return None
    if np.count_nonzero(inliers) < 40 or inliers.mean() < 0.55:
        return None
    p = np.column_stack([start, np.ones(len(start))])
    q = np.column_stack([end, np.ones(len(end))])
    f_p, f_q = p @ fundamental.T, q @ fundamental
    denominator = np.sqrt((f_p[:, :2] ** 2 + f_q[:, :2] ** 2).sum(axis=1))
    return np.abs((q * f_p).sum(axis=1)) / np.maximum(denominator, 1e-12)


def background_motion(cv, start, end, background, shape):
    """Use a dominant image warp only when it explains spatially distributed tracks."""
    source = start[background]
    if len(source) >= 45:
        transform, support = cv.findHomography(source, end[background], cv.RANSAC, 1.5, maxIters=500)
        if transform is not None and support is not None and support.mean() >= 0.85:
            height, width = shape
            supported = source[support.ravel().astype(bool)]
            cells = np.clip((supported / (width, height) * 4).astype(int), 0, 3)
            if len(np.unique(cells[:, 0] + 4 * cells[:, 1])) >= 8:
                projected = cv.perspectiveTransform(start[:, None, :], transform)[:, 0]
                vectors = end - projected
                if np.isfinite(vectors).all():
                    return np.linalg.norm(vectors, axis=1), "dominant_warp", vectors
    return epipolar_errors(cv, start, end, background), "epipolar", None


def motion_evidence(cv, tracks, mask):
    decisions = []
    for neighbor, (start, end) in tracks:
        x, y = np.floor(start).astype(int).T
        inside = mask[y, x]
        count = int(np.count_nonzero(inside))
        errors, model, vectors = background_motion(cv, start, end, ~inside, mask.shape) if count >= 8 else (None, None, None)
        if errors is None:
            decisions.append({"neighbor": neighbor, "tracked_points": count, "decision": "inconclusive"})
            continue
        residuals = errors[inside]
        fraction = float(np.mean(residuals > 2.5))
        decision = "moving" if fraction >= 0.65 else "consistent"
        if decision == "moving" and vectors is not None:
            moving = inside & (errors > 2.5)
            unit = vectors[moving] / errors[moving, None]
            if np.linalg.norm(unit.mean(axis=0)) < 0.65:
                decision = "inconclusive"
        decisions.append({
            "neighbor": neighbor,
            "tracked_points": count,
            "outlier_fraction": fraction,
            "median_residual_px": float(np.median(residuals)),
            "decision": decision, "background_model": model,
        })
    return decisions


def suggest_boxes(cv, tracks, shape):
    height, width = shape
    candidates = []
    for _, (start, end) in tracks:
        errors, _, _ = background_motion(cv, start, end, np.ones(len(start), dtype=bool), shape)
        if errors is not None:
            candidates.extend(start[errors > 3.5])
    if len(candidates) < 8:
        return []
    points = np.asarray(candidates, dtype=int)
    raster = np.zeros(shape, dtype=np.uint8)
    raster[points[:, 1], points[:, 0]] = 1
    size = max(11, min(height, width) // 25) | 1
    connected = cv.dilate(raster, np.ones((size, size), np.uint8))
    _, labels, stats, _ = cv.connectedComponentsWithStats(connected)
    counts = np.bincount(labels[points[:, 1], points[:, 0]], minlength=len(stats))
    boxes = []
    for index in np.argsort(counts[1:])[::-1][:2] + 1:
        x, y, w, h, _ = stats[index]
        if counts[index] >= 8 and 0.002 * width * height <= w * h <= 0.35 * width * height:
            boxes.append((int(x), int(y), int(x + w), int(y + h)))
    return boxes


def project_candidate_box(cv, box, tracks, shape):
    """Propose a one-hop box. A proposal still needs fresh SAM and motion validation."""
    start, end = tracks
    x0, y0, x1, y1 = box
    inside = ((start[:, 0] >= x0) & (start[:, 0] < x1)
              & (start[:, 1] >= y0) & (start[:, 1] < y1))
    if np.count_nonzero(inside) < 12:
        return None
    src, dst = start[inside], end[inside]
    transform, inliers = cv.estimateAffinePartial2D(src, dst, method=cv.RANSAC, ransacReprojThreshold=2.5)
    if transform is None or inliers is None or inliers.mean() < 0.7:
        return None
    support = src[inliers.ravel().astype(bool)]
    if np.count_nonzero(inliers) < 10 or np.any(np.ptp(support, axis=0) < 0.25 * np.array([x1-x0, y1-y0])):
        return None
    scale = np.linalg.norm(transform[:, 0])
    if not 0.7 <= scale <= 1.4:
        return None
    corners = np.array([[x0, y0, 1], [x1, y0, 1], [x0, y1, 1], [x1, y1, 1]])
    projected = corners @ transform.T
    height, width = shape
    low = np.maximum(np.floor(projected.min(axis=0)), 0)
    high = np.minimum(np.ceil(projected.max(axis=0)), [width, height])
    if np.any(high - low < 8):
        return None
    return (*map(int, low), *map(int, high))
