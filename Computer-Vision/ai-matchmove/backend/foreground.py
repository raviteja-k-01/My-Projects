"""Build immutable, source-aligned foreground mask sets for a sequence."""

from functools import lru_cache
from io import BytesIO
from time import perf_counter

import numpy as np
from PIL import Image
from pydantic import BaseModel, Field

if __package__:
    from .motion import load_cv, motion_evidence, project_candidate_box, suggest_boxes, track_pair
    from .sam_service import Prompt, export_mask, get_sam, save_preview, validate_prompt
else:
    from motion import load_cv, motion_evidence, project_candidate_box, suggest_boxes, track_pair
    from sam_service import Prompt, export_mask, get_sam, save_preview, validate_prompt


class ForegroundRequest(BaseModel):
    prompts: list[Prompt] = Field(default_factory=list)
    discover: bool = False
    discover_frame_numbers: list[int] | None = None


def automatic_exclusion_supported(evidence):
    """Require independent support on both sides of an automatically proposed frame.

    One epipolar outlier result is not sufficient to identify a physically moving
    object: model error or bad correspondences can mislabel static structures.
    Boundary frames with only one neighbor remain candidates for manual review.
    """
    moving_neighbors = {
        item["neighbor"] for item in evidence
        if item.get("decision") == "moving" and item.get("tracked_points", 0) >= 8
    }
    return len(moving_neighbors) >= 2 and not any(
        item.get("decision") == "consistent" for item in evidence
    )


def build_masks(sequence, image_dir, destination, request, progress=None):
    start = perf_counter()
    frames = sequence["frame_details"]
    if not frames:
        raise ValueError("The sequence has no valid frames.")
    if not request.prompts and not request.discover:
        raise ValueError("Supply prompts or enable automatic motion proposals.")
    if request.discover and sequence["resolution"] is None:
        raise ValueError("Automatic motion proposals need a consistent frame resolution.")

    by_number = {frame["frame_number"]: frame for frame in frames}
    by_frame = {}
    for prompt in request.prompts:
        frame = by_number.get(prompt.frame_number)
        if frame is None:
            raise ValueError(f"Frame {prompt.frame_number} is missing from the sequence.")
        validate_prompt(prompt, frame["width"], frame["height"])
        by_frame.setdefault(prompt.frame_number, []).append(prompt)

    cv = load_cv()
    if request.discover_frame_numbers is not None and any(number not in by_number for number in request.discover_frame_numbers):
        raise ValueError("An automatic analysis frame is missing from the sequence.")
    for directory in ("foreground", "candidate", "preview", "colmap", "region_masks"):
        (destination / directory).mkdir()

    @lru_cache(maxsize=4)
    def read_frame(index):
        with Image.open(image_dir / frames[index]["filename"]) as image:
            return np.asarray(image.convert("RGB"))

    @lru_cache(maxsize=10)
    def pair_tracks(first, second):
        return track_pair(cv, read_frame(first), read_frame(second))

    @lru_cache(maxsize=4)
    def keep_every_pixel(width, height):
        buffer = BytesIO()
        Image.new("L", (width, height), 255).save(buffer, format="PNG")
        return buffer.getvalue()

    def tracks_for(index):
        return [
            (frames[neighbor]["frame_number"], pair_tracks(index, neighbor))
            for neighbor in (index - 1, index + 1)
            if 0 <= neighbor < len(frames) and read_frame(index).shape == read_frame(neighbor).shape
        ]

    service = None
    inference_seconds = 0.0
    details = []
    discovery = set(request.discover_frame_numbers) if request.discover_frame_numbers is not None else set(by_number)
    direct = {}
    nearby = {}
    if request.discover:
        analyzed = [index for index, frame in enumerate(frames)
                    if frame["frame_number"] in discovery and frame["frame_number"] not in by_frame]
        nearby = {index: [neighbor for neighbor in analyzed[max(0, order - 1):order + 2]
                          if 0 < abs(neighbor - index) <= 3] for order, index in enumerate(analyzed)}
        for count, index in enumerate(analyzed, 1):
            frame = frames[index]
            direct[index] = suggest_boxes(cv, tracks_for(index), (frame["height"], frame["width"]))
            if progress:
                progress("Finding motion proposals", count, len(analyzed))
    processed_count = sum(frame["frame_number"] in by_frame or (request.discover and frame["frame_number"] in discovery)
                          for frame in frames)
    proposal_seconds = perf_counter() - start
    processed = 0
    for index, frame in enumerate(frames):
        number, filename = frame["frame_number"], frame["filename"]
        shape = frame["height"], frame["width"]
        prompts = by_frame.get(number, [])
        discover = request.discover and number in discovery
        excluded_pixels = 0
        regions = []
        if prompts or discover:
            rgb = read_frame(index)
            tracks = []
            if discover or any(prompt.decision == "auto" for prompt in prompts):
                tracks = tracks_for(index)

            origin = "user"
            if not prompts:
                boxes = direct.get(index, [])
                origin = "motion_proposal"
                if not boxes:
                    # Only direct neighbor proposals can seed a hop. Never chain masks.
                    for neighbor in nearby.get(index, []):
                        if direct.get(neighbor):
                            projected = project_candidate_box(cv, direct[neighbor][0], pair_tracks(neighbor, index), shape)
                            if projected is not None:
                                boxes = [projected]
                                origin = "temporal_proposal"
                                break
                prompts = [Prompt(frame_number=number, box=box) for box in boxes]
            if prompts:
                service = service or get_sam()
                candidates = np.zeros(shape, dtype=bool)
                exclusion = np.zeros(shape, dtype=bool)
                keep = np.zeros(shape, dtype=bool)
                inference_started = perf_counter()
                masks = service.segment(rgb, prompts)
                inference_seconds += perf_counter() - inference_started
                for region_index, (prompt, (mask, score)) in enumerate(zip(prompts, masks, strict=True)):
                    # Preserve each SAM mask so keep/exclude reviews do not require new inference.
                    export_mask(destination / "region_masks" / f"{filename}.{region_index}.png", mask)
                    evidence = motion_evidence(cv, tracks, mask) if prompt.decision == "auto" else []
                    moving = automatic_exclusion_supported(evidence)
                    auto_validation = (
                        "confirmed_both_neighbors" if moving else
                        "insufficient_independent_motion_evidence"
                        if any(item["decision"] == "moving" for item in evidence) else
                        "no_moving_motion_evidence"
                    ) if prompt.decision == "auto" else "manual_decision"
                    approved = bool(mask.any() and (prompt.decision == "exclude" or moving))
                    if approved:
                        exclusion |= mask
                    if prompt.decision == "keep":
                        keep |= mask
                    candidates |= mask
                    regions.append({
                        "prompt": prompt.model_dump(),
                        "origin": origin,
                        "sam_score": score,
                        "candidate_pixels": int(np.count_nonzero(mask)),
                        "excluded": approved,
                        "auto_validation": auto_validation,
                        "decision": ("manual_exclude" if approved and prompt.decision == "exclude"
                                     else "auto_validated" if approved else "manual_keep" if prompt.decision == "keep"
                                     else "consistent" if any(item["decision"] == "consistent" for item in evidence)
                                     else "uncertain"),
                        "motion": evidence,
                    })
                exclusion = cv.dilate(exclusion.astype(np.uint8), np.ones((5, 5), np.uint8)).astype(bool) & ~keep
                excluded_pixels = int(np.count_nonzero(exclusion))
                save_preview(destination / "preview" / filename, rgb, candidates, exclusion)
                export_mask(destination / "candidate" / filename, candidates)
                export_mask(destination / "foreground" / filename, exclusion)
        colmap_mask = destination / "colmap" / (filename + ".png")
        if excluded_pixels:
            export_mask(colmap_mask, ~exclusion)
        else:
            colmap_mask.write_bytes(keep_every_pixel(frame["width"], frame["height"]))
        analyzed = bool(discover or number in by_frame)
        # Distinguish "SAM did not run" from "SAM ran but the region was
        # geometrically consistent / uncertain". No change to mask decisions.
        mask_status = (
            "not_analyzed" if not analyzed else
            "no_proposal" if not regions else
            "approved_exclusion" if excluded_pixels else
            "manual_keep" if all(region["decision"] == "manual_keep" for region in regions) else
            "no_approved_exclusion"
        )
        details.append({
            "frame_number": number,
            "width": frame["width"],
            "height": frame["height"],
            "excluded_pixels": excluded_pixels,
            "analyzed": analyzed,
            "mask_status": mask_status,
            "regions": regions,
        })
        if discover or number in by_frame:
            processed += 1
            if progress:
                progress("Checking foreground motion and SAM masks", processed, processed_count)
    return {
        "model": "SAM ViT-B" if service is not None else None,
        "device": service.device if service is not None else None,
        "encoder_input_size": service.input_size if service is not None else None,
        "motion_corner_sampling": "spatial_tiles",
        "elapsed_seconds": perf_counter() - start,
        "proposal_seconds": proposal_seconds,
        "sam_inference_seconds": inference_seconds,
        "frames": details,
        "masked_frames": sum(frame["excluded_pixels"] > 0 for frame in details),
        "analyzed_frames": processed_count,
        "frames_with_proposals": sum(bool(frame["regions"]) for frame in details),
        "frame_status_counts": {
            status: sum(frame["mask_status"] == status for frame in details)
            for status in ("not_analyzed", "no_proposal", "approved_exclusion", "manual_keep", "no_approved_exclusion")
        },
        "temporal_proposals": sum(region["origin"] == "temporal_proposal" for frame in details for region in frame["regions"]),
    }
