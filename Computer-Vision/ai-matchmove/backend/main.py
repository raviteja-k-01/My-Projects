import logging
import os
from copy import deepcopy
from shutil import copy2
from fractions import Fraction
from math import isfinite
from pathlib import Path
from tempfile import NamedTemporaryFile, TemporaryDirectory
from typing import Annotated, Literal
from uuid import UUID, uuid4

from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from pydantic import BaseModel, Field
from PIL import Image

if __package__:
    from . import jobs
    from .export_camera import export_package, point_cloud_ply
    from .video import VIDEO_EXTENSIONS, decode_video
    from .foreground import ForegroundRequest, build_masks
    from .sam_service import SamUnavailable, export_mask, save_preview
    from .sequence import build_sequence, read_json, write_json
    from .reconstruction import ReconstructionError, SolverUnavailable, reconstruct, sampled_indices
else:
    import jobs
    from export_camera import export_package, point_cloud_ply
    from video import VIDEO_EXTENSIONS, decode_video
    from foreground import ForegroundRequest, build_masks
    from sam_service import SamUnavailable, export_mask, save_preview
    from sequence import build_sequence, read_json, write_json
    from reconstruction import ReconstructionError, SolverUnavailable, reconstruct, sampled_indices

DATA_DIR = Path(os.environ.get("MATCHMOVE_DATA_DIR", Path(__file__).resolve().parent / "data"))
logger = logging.getLogger(__name__)
app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/")
def read_root():
    return {"status": "backend running"}


def parse_fps(value):
    try:
        rate = Fraction(value.strip()) if value and value.strip() else None
        if rate is not None and (not isfinite(float(rate)) or rate <= 0):
            raise ValueError
        return rate
    except (ValueError, ZeroDivisionError, OverflowError) as exc:
        raise HTTPException(422, "Enter a positive FPS, such as 24 or 24000/1001.") from exc


@app.post("/upload/")
def upload_files(
    files: Annotated[list[UploadFile], File()],
    fps: Annotated[str | None, Form()] = None,
    video_stride: Annotated[int, Form(ge=1, le=1200)] = 1,
):
    rate = parse_fps(fps)
    sequence_id = uuid4().hex
    if sum(item.size or 0 for item in files) > 2 * 1024**3:
        raise HTTPException(413, "This upload exceeds the 2 GiB input limit. Split the sequence or trim the clip.")
    videos = [item for item in files if Path(item.filename or "").suffix.lower() in VIDEO_EXTENSIONS]
    if videos and len(files) != 1:
        raise HTTPException(422, "Upload one video or an image sequence in each upload.")
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    with TemporaryDirectory(prefix=".upload-", dir=DATA_DIR) as staging:
        folder = Path(staging)
        try:
            sequence = (
                decode_video(videos[0], folder, sequence_id, video_stride, rate)
                if videos else build_sequence(files, folder, sequence_id, rate)
            )
        except ValueError as exc:
            raise HTTPException(422, str(exc)) from exc
        write_json(folder / "sequence.json", sequence)
        folder.rename(DATA_DIR / sequence_id)
    return sequence


def sequence_folder(sequence_id: UUID) -> Path:
    folder = DATA_DIR / sequence_id.hex
    if not (folder / "sequence.json").is_file():
        raise HTTPException(404, "Sequence not found.")
    return folder


@app.get("/sequences")
def list_sequences():
    if not DATA_DIR.is_dir():
        return {"sequences": []}
    entries = []
    for folder in DATA_DIR.iterdir():
        try:
            identifier = UUID(folder.name).hex
            if folder.name != identifier:
                continue
            path = folder / "sequence.json"
            saved = read_json(path)
            if saved.get("sequence_id") != identifier or not isinstance(saved.get("frame_details"), list):
                continue
            entries.append({
                "sequence_id": identifier, "source_name": saved.get("source_name")
                or (saved["frame_details"][0]["source_name"] if saved["frame_details"] else "Image sequence"),
                "frame_count": len(saved["frame_details"]), "created_order": path.stat().st_mtime_ns,
            })
        except (OSError, ValueError, KeyError, TypeError, AttributeError):
            continue
    return {"sequences": sorted(entries, key=lambda item: item["created_order"], reverse=True)}


@app.get("/sequences/{sequence_id}")
def get_sequence(sequence_id: UUID):
    return read_json(sequence_folder(sequence_id) / "sequence.json")


@app.get("/frames/{sequence_id}/{filename}")
def serve_frame(sequence_id: UUID, filename: str):
    folder = sequence_folder(sequence_id)
    path = folder / "frames" / filename
    if path.suffix != ".png" or Path(filename).name != filename or not path.is_file():
        raise HTTPException(404, "Frame not found.")
    return FileResponse(path, media_type="image/png")


@app.get("/previews/{sequence_id}/{filename}")
def serve_preview(sequence_id: UUID, filename: str):
    folder = sequence_folder(sequence_id)
    source = folder / "frames" / filename
    if source.suffix != ".png" or Path(filename).name != filename or not source.is_file():
        raise HTTPException(404, "Frame not found.")
    directory = folder / "previews"
    directory.mkdir(exist_ok=True)
    target = directory / (source.stem + ".jpg")
    if not target.is_file():
        with Image.open(source) as image:
            image.thumbnail((960, 960), Image.Resampling.BILINEAR)
            with NamedTemporaryFile(dir=directory, suffix=".jpg", delete=False) as stream:
                temporary = Path(stream.name)
                try:
                    image.convert("RGB").save(stream, format="JPEG", quality=78)
                    stream.close()
                    temporary.replace(target)
                finally:
                    temporary.unlink(missing_ok=True)
    return FileResponse(target, media_type="image/jpeg", headers={"Cache-Control": "private, max-age=86400, immutable"})


def foreground_folder(folder, mask_set_id):
    target = folder / "foreground" / mask_set_id.hex
    if not (target / "manifest.json").is_file():
        raise HTTPException(404, "Foreground mask set not found.")
    return target


def require_exclusion(target):
    try:
        manifest = read_json(target / "manifest.json")
        sequence = read_json(target.parent.parent / "sequence.json")
        saved_frames = {frame["frame_number"]: frame for frame in manifest["frames"]}
        expected = {frame["frame_number"]: frame for frame in sequence["frame_details"]}
        if (manifest["mask_set_id"] != target.name or saved_frames.keys() != expected.keys()
                or len(saved_frames) != len(manifest["frames"])
                or manifest.get("manifest_url") != f"/sequences/{sequence['sequence_id']}/foreground/{target.name}"
                or manifest.get("sequence_id", sequence["sequence_id"]) != sequence["sequence_id"]):
            raise ValueError("Mask revision does not match this sequence.")
        pending = 0
        masked = 0
        for number, frame in saved_frames.items():
            original = expected[number]
            if (frame["width"], frame["height"]) != (original["width"], original["height"]):
                raise ValueError("Manifest frame dimensions do not match this sequence.")
            regions = [region for region in frame["regions"] if region.get("excluded")]
            pending += sum(
                region.get("review_decision") != "exclude"
                and not (region.get("origin") == "user" and region.get("prompt", {}).get("decision") == "exclude"
                         and region.get("review_decision") is None)
                for region in regions
            )
            path = target / "colmap" / (original["filename"] + ".png")
            with Image.open(path) as image:
                if image.mode != "L" or image.size != (original["width"], original["height"]):
                    raise ValueError("Mask dimensions or format do not match this sequence.")
                counts = image.histogram()
                if sum(counts[1:255]) or counts[0] != frame["excluded_pixels"] or (counts[0] and not regions):
                    raise ValueError("Mask contents do not match the saved region decisions.")
                masked += counts[0] > 0
        if masked != manifest["masked_frames"]:
            raise ValueError("Mask revision coverage is inconsistent.")
    except (KeyError, OSError, TypeError, ValueError) as exc:
        raise HTTPException(409, "This mask revision is incomplete or incompatible. Generate a new mask set.") from exc
    if pending:
        raise HTTPException(409, f"Review {pending} proposed exclusion(s) before using this mask revision.")
    if not masked:
        raise HTTPException(422, "No foreground region was approved for exclusion in this mask set.")
    return manifest


@app.post("/sequences/{sequence_id}/foreground")
def create_foreground(sequence_id: UUID, request: ForegroundRequest):
    return _create_foreground(sequence_id, request)


def _create_foreground(sequence_id, request, progress=None):
    folder = sequence_folder(sequence_id)
    sequence = read_json(folder / "sequence.json")
    output = folder / "foreground"
    output.mkdir(exist_ok=True)
    mask_set_id = uuid4().hex
    base_url = f"/sequences/{sequence_id.hex}/foreground/{mask_set_id}"
    with TemporaryDirectory(prefix=".masks-", dir=output) as temporary:
        staging = Path(temporary)
        try:
            manifest = build_masks(sequence, folder / "frames", staging, request, progress=progress)
        except SamUnavailable as exc:
            raise HTTPException(503, str(exc)) from exc
        except ValueError as exc:
            raise HTTPException(422, str(exc)) from exc
        for frame in manifest["frames"]:
            if frame["regions"]:
                base = f"{base_url}/frames/{frame['frame_number']}"
                frame.update({f"{kind}_url": f"{base}/{kind}" for kind in ("preview", "mask", "candidate")})
        manifest.update({"mask_set_id": mask_set_id, "manifest_url": base_url, "sequence_id": sequence_id.hex})
        write_json(staging / "manifest.json", manifest)
        if progress:
            progress("Saving foreground revision")
        commit = getattr(progress, "commit", None)
        if callable(commit):
            commit(lambda: staging.rename(output / mask_set_id))
        else:
            staging.rename(output / mask_set_id)
    return manifest


@app.get("/sequences/{sequence_id}/foreground")
def list_foreground_sets(sequence_id: UUID):
    """Recover the newest reviewed mask set after a browser refresh."""
    folder = sequence_folder(sequence_id) / "foreground"
    if not folder.exists():
        return {"mask_sets": []}
    entries = []
    for path in folder.iterdir():
        if not path.is_dir() or not (path / "manifest.json").is_file():
            continue
        try:
            identifier = UUID(path.name)
            report = read_json(path / "manifest.json")
            if identifier.hex != path.name or report.get("mask_set_id") != identifier.hex:
                continue
            entries.append({
                "mask_set_id": identifier.hex,
                "manifest_url": report["manifest_url"],
                "masked_frames": report.get("masked_frames", 0),
                "reviewed_regions": report.get("reviewed_regions", 0),
                "created_order": path.stat().st_mtime_ns,
            })
        except (OSError, ValueError, KeyError):
            continue
    entries.sort(key=lambda item: item["created_order"], reverse=True)
    return {"mask_sets": entries[:50]}


@app.get("/sequences/{sequence_id}/foreground/{mask_set_id}")
def get_foreground(sequence_id: UUID, mask_set_id: UUID):
    target = foreground_folder(sequence_folder(sequence_id), mask_set_id)
    return read_json(target / "manifest.json")


class RegionDecision(BaseModel):
    frame_number: int
    region_index: int = Field(ge=0)
    decision: Literal["keep", "exclude"]


class MaskReviewRequest(BaseModel):
    # Changes are applied to a derived immutable mask set. The source is not edited.
    decisions: list[RegionDecision] = Field(min_length=1, max_length=200)


@app.post("/sequences/{sequence_id}/foreground/{mask_set_id}/review")
def review_foreground(sequence_id: UUID, mask_set_id: UUID, request: MaskReviewRequest):
    """Approve/reject saved SAM regions without reloading the model or running SfM."""
    folder = sequence_folder(sequence_id)
    previous = foreground_folder(folder, mask_set_id)
    original = read_json(previous / "manifest.json")
    sequence = read_json(folder / "sequence.json")
    frames = {frame["frame_number"]: frame for frame in sequence["frame_details"]}
    source_regions = {frame["frame_number"]: frame["regions"] for frame in original["frames"]}
    overrides = {}
    for item in request.decisions:
        key = (item.frame_number, item.region_index)
        if item.frame_number not in frames or item.region_index >= len(source_regions.get(item.frame_number, [])):
            raise HTTPException(422, "The requested SAM region does not exist.")
        if key in overrides:
            raise HTTPException(422, "Each region can be reviewed only once per request.")
        overrides[key] = item.decision

    # Reject legacy multi-region outputs when individual masks are not saved.
    # A union mask cannot be separated safely into its original components.
    for (number, index), decision in overrides.items():
        filename = frames[number]["filename"]
        specific = previous / "region_masks" / f"{filename}.{index}.png"
        fallback = previous / "candidate" / filename
        if not specific.is_file() and not (len(source_regions[number]) == 1 and fallback.is_file()):
            raise HTTPException(409, "This older mask set has no individual region masks. Regenerate the masks once to enable independent review.")

    from numpy import asarray, zeros, uint8, count_nonzero, ones
    new_id = uuid4().hex
    base = f"/sequences/{sequence_id.hex}/foreground/{new_id}"
    output = folder / "foreground"
    with TemporaryDirectory(prefix=".review-", dir=output) as temporary:
        staging = Path(temporary)
        for name in ("region_masks", "foreground", "candidate", "preview", "colmap"):
            (staging / name).mkdir()
        report = deepcopy(original)
        changed_frames = {number for number, _ in overrides}
        for saved in report["frames"]:
            number = saved["frame_number"]
            filename = frames[number]["filename"]
            if number not in changed_frames:
                # Fast path: preserve verified output bytes; no SAM/model/CV work.
                for directory, name in (("colmap", filename + ".png"),
                                         ("candidate", filename), ("foreground", filename),
                                         ("preview", filename)):
                    source_file = previous / directory / name
                    if source_file.is_file():
                        copy2(source_file, staging / directory / name)
                for index in range(len(saved["regions"])):
                    name = f"{filename}.{index}.png"
                    if (previous / "region_masks" / name).is_file():
                        copy2(previous / "region_masks" / name, staging / "region_masks" / name)
                if saved["regions"]:
                    frame_url = f"{base}/frames/{number}"
                    saved.update({f"{kind}_url": f"{frame_url}/{kind}" for kind in ("preview", "mask", "candidate")})
                continue
            shape = frames[number]["height"], frames[number]["width"]
            excluded, kept, candidates = (zeros(shape, dtype=bool) for _ in range(3))
            for index, region in enumerate(saved["regions"]):
                mask_path = previous / "region_masks" / f"{filename}.{index}.png"
                if not mask_path.is_file():
                    if len(saved["regions"]) != 1:
                        raise HTTPException(409, "Legacy multi-region masks need regeneration for individual review.")
                    mask_path = previous / "candidate" / filename
                if not mask_path.is_file():
                    raise HTTPException(409, "This mask set is missing its saved SAM segmentation data.")
                with Image.open(mask_path) as image:
                    mask = asarray(image.convert("L")) > 0
                if mask.shape != shape:
                    raise HTTPException(409, "Saved SAM region dimensions differ from the original frame.")
                copy2(mask_path, staging / "region_masks" / f"{filename}.{index}.png")
                candidates |= mask
                change = overrides.get((number, index))
                if change is not None:
                    region["review_decision"] = change
                    region["decision"] = "manual_exclude" if change == "exclude" else "manual_keep"
                    region["excluded"] = change == "exclude"
                if region.get("excluded"):
                    excluded |= mask
                if region.get("decision") == "manual_keep":
                    kept |= mask
            # Match the original output's 5×5 exclusion-margin logic.
            if saved["regions"]:
                import cv2
                excluded = cv2.dilate(excluded.astype(uint8), ones((5, 5), dtype=uint8)).astype(bool) & ~kept
                with Image.open(folder / "frames" / filename) as image:
                    rgb = asarray(image.convert("RGB"))
                save_preview(staging / "preview" / filename, rgb, candidates, excluded)
                export_mask(staging / "candidate" / filename, candidates)
                export_mask(staging / "foreground" / filename, excluded)
                frame_url = f"{base}/frames/{number}"
                saved.update({f"{kind}_url": f"{frame_url}/{kind}" for kind in ("preview", "mask", "candidate")})
            export_mask(staging / "colmap" / (filename + ".png"), ~excluded)
            saved["excluded_pixels"] = int(count_nonzero(excluded))
            saved["mask_status"] = (
                "not_analyzed" if not saved.get("analyzed") else
                "no_proposal" if not saved["regions"] else
                "approved_exclusion" if excluded.any() else
                "manual_keep" if all(r["decision"] == "manual_keep" for r in saved["regions"]) else
                "no_approved_exclusion"
            )
        report["mask_set_id"] = new_id
        report["manifest_url"] = base
        report["reviewed_from_mask_set_id"] = mask_set_id.hex
        report["sequence_id"] = sequence_id.hex
        report["review_changes"] = [item.model_dump() for item in request.decisions]
        report["reviewed_regions"] = original.get("reviewed_regions", 0) + len(overrides)
        report["masked_frames"] = sum(frame["excluded_pixels"] > 0 for frame in report["frames"])
        report["frame_status_counts"] = {
            status: sum(frame["mask_status"] == status for frame in report["frames"])
            for status in ("not_analyzed", "no_proposal", "approved_exclusion", "manual_keep", "no_approved_exclusion")
        }
        write_json(staging / "manifest.json", report)
        staging.rename(output / new_id)
    return report


@app.get("/sequences/{sequence_id}/foreground/{mask_set_id}/frames/{frame_number}/{kind}")
def get_foreground_image(
    sequence_id: UUID, mask_set_id: UUID, frame_number: int,
    kind: Literal["preview", "mask", "candidate"],
):
    folder = sequence_folder(sequence_id)
    target = foreground_folder(folder, mask_set_id)
    sequence = read_json(folder / "sequence.json")
    frame = next((item for item in sequence["frame_details"] if item["frame_number"] == frame_number), None)
    if frame is None:
        raise HTTPException(404, "Frame not found.")
    path = target / ("foreground" if kind == "mask" else kind) / frame["filename"]
    if not path.is_file():
        raise HTTPException(404, "There is no segmentation for this frame.")
    return FileResponse(path, media_type="image/png")


@app.post("/sequences/{sequence_id}/reconstruct")
def solve_sequence(sequence_id: UUID, mask_set_id: UUID | None = None):
    folder = sequence_folder(sequence_id)
    options = {}
    result_url = f"/sequences/{sequence_id.hex}/reconstruction.json"
    if mask_set_id is not None:
        target = foreground_folder(folder, mask_set_id)
        require_exclusion(target)
        options = {"mask_dir": target / "colmap", "output_path": target / "reconstruction.json"}
        result_url += f"?mask_set_id={mask_set_id.hex}"
    try:
        result = reconstruct(read_json(folder / "sequence.json"), folder, **options)
    except SolverUnavailable as exc:
        raise HTTPException(503, str(exc)) from exc
    except ReconstructionError as exc:
        raise HTTPException(422, str(exc)) from exc
    except (OSError, RuntimeError) as exc:
        logger.exception("Reconstruction failed for sequence %s", sequence_id)
        raise HTTPException(500, "Reconstruction failed while running PyCOLMAP or saving results.") from exc
    return {
        "status": result["status"],
        "statistics": result["statistics"],
        "result_url": result_url,
    }


@app.get("/sequences/{sequence_id}/reconstruction.json")
def get_reconstruction(sequence_id: UUID, mask_set_id: UUID | None = None):
    folder = sequence_folder(sequence_id)
    if mask_set_id is not None:
        folder = foreground_folder(folder, mask_set_id)
    path = folder / "reconstruction.json"
    if not path.is_file():
        raise HTTPException(404, "No reconstruction has been saved for this sequence.")
    return FileResponse(path, media_type="application/json", filename="reconstruction.json")


@app.post("/sequences/{sequence_id}/foreground/{mask_set_id}/compare")
def compare_reconstructions(sequence_id: UUID, mask_set_id: UUID):
    return _compare_reconstructions(sequence_id, mask_set_id)


def _compare_reconstructions(sequence_id, mask_set_id, max_frames=None, progress=None, initial_focal_px=None):
    folder = sequence_folder(sequence_id)
    target = foreground_folder(folder, mask_set_id)
    manifest = require_exclusion(target)
    sequence = read_json(folder / "sequence.json")
    frames = sequence["frame_details"]
    selected = {frames[index]["frame_number"] for index in sampled_indices(len(frames), max_frames)}
    if not any(frame["excluded_pixels"] and frame["frame_number"] in selected for frame in manifest["frames"]):
        raise HTTPException(422, "The selected comparison frames contain no approved exclusions.")
    comparison_id = uuid4().hex
    base = f"/sequences/{sequence_id.hex}/foreground/{mask_set_id.hex}/comparisons/{comparison_id}"
    mask_dir = target / "colmap"
    with TemporaryDirectory(prefix=".comparison-", dir=target) as temporary:
        staging = Path(temporary)
        runs = {}
        for mode in ("baseline", "masked"):
            mode_progress = None
            if progress:
                def mode_progress(stage, completed=None, total=None):
                    progress(f"{mode.capitalize()} · {stage}", completed, total)

                if callable(getattr(progress, "bind_cancel", None)):
                    mode_progress.bind_cancel = progress.bind_cancel
            try:
                result = reconstruct(
                    sequence, folder, audit_mask_dir=mask_dir,
                    mask_dir=mask_dir if mode == "masked" else None,
                    output_path=staging / f"{mode}.json",
                    max_frames=max_frames,
                    progress=mode_progress,
                    initial_focal_px=initial_focal_px,
                )
            except ReconstructionError as exc:
                runs[mode] = {"status": "failed", "error": str(exc), "statistics": exc.statistics}
            except SolverUnavailable as exc:
                raise HTTPException(503, str(exc)) from exc
            except (OSError, RuntimeError) as exc:
                logger.exception("Comparison solve failed in mode %s", mode)
                raise HTTPException(500, f"The {mode} solve could not run or save results.") from exc
            else:
                runs[mode] = {
                    "status": result["status"], "statistics": result["statistics"],
                    "result_url": f"{base}/{mode}",
                }
        report = {
            "comparison_id": comparison_id,
            "comparison_url": base,
            "sequence_id": sequence_id.hex,
            "mask_set_id": mask_set_id.hex,
            "mask_generation_seconds": manifest["elapsed_seconds"],
            "excluded_baseline_keypoints": runs["baseline"]["statistics"].get("keypoints_in_exclusion_regions"),
            **runs,
        }
        write_json(staging / "comparison.json", report)
        if progress:
            progress("Saving comparison")
        commit = getattr(progress, "commit", None)
        if callable(commit):
            commit(lambda: staging.rename(target / comparison_id))
        else:
            staging.rename(target / comparison_id)
    return report


@app.get("/sequences/{sequence_id}/foreground/{mask_set_id}/comparisons/{comparison_id}")
def get_comparison(sequence_id: UUID, mask_set_id: UUID, comparison_id: UUID):
    folder = foreground_folder(sequence_folder(sequence_id), mask_set_id)
    path = folder / comparison_id.hex / "comparison.json"
    if not path.is_file():
        raise HTTPException(404, "Comparison not found.")
    return read_json(path)


@app.get("/sequences/{sequence_id}/foreground/{mask_set_id}/comparisons/{comparison_id}/{mode}")
def get_comparison_solve(
    sequence_id: UUID, mask_set_id: UUID, comparison_id: UUID, mode: Literal["baseline", "masked"],
):
    folder = foreground_folder(sequence_folder(sequence_id), mask_set_id)
    path = folder / comparison_id.hex / f"{mode}.json"
    if not path.is_file():
        raise HTTPException(404, "No successful solve was saved for this mode.")
    return FileResponse(path, media_type="application/json", filename=f"{mode}.json")


class JobRequest(BaseModel):
    operation: Literal["solve", "foreground", "compare", "export", "preflight"] = "solve"
    max_frames: int = Field(24, ge=3, le=120)
    # Independent SAM analysis budget; omitted requests retain the old behavior.
    sam_analysis_frames: int | None = Field(default=None, ge=1, le=120)
    sam: bool = False
    mask_set_id: UUID | None = None
    source_job_id: UUID | None = None
    comparison_job_id: UUID | None = None
    comparison_mode: Literal["baseline", "masked"] | None = None
    initial_focal_px: float | None = Field(default=None, gt=0, allow_inf_nan=False)
    export_fps: str | None = Field(default=None, max_length=64)
    foreground: ForegroundRequest = Field(default_factory=ForegroundRequest)


def job_folder(sequence_id, job_id):
    target = sequence_folder(sequence_id) / "jobs" / job_id.hex
    if not jobs.exists(target):
        raise HTTPException(404, "Processing job not found.")
    return target


def save_solve_outputs(target, root, result, manifest):
    frame_quality = {frame["frame_number"]: frame["quality"] for frame in result["registered_frames"] if "quality" in frame}
    write_json(target / "quality.json", {
        **result["statistics"], "solver": result.get("solver"), "diagnostics": result.get("diagnostics"),
        "frames": frame_quality,
    })
    (target / "points.ply").write_text(point_cloud_ply(result["sparse_points"]), encoding="ascii")
    write_json(target / "tracks.json", {
        "coordinates": "source pixels, origin at top left",
        "fields": ["x", "y", "point_id"],
        "frames": {frame["frame_number"]: frame["observations"] for frame in result["registered_frames"]},
    })
    return {
        "statistics": result["statistics"], "status": result["status"],
        "solver": result.get("solver"), "diagnostics": result.get("diagnostics"),
        "source_frame_count": result["source_frame_count"], "frame_quality": frame_quality,
        "masking_applied": manifest is not None, "foreground": manifest,
        "camera_path": [frame["position_world"] for frame in result["registered_frames"]],
        "result_url": f"{root}/files/reconstruction.json",
        "quality_url": f"{root}/files/quality.json",
        "tracks_url": f"{root}/files/tracks.json",
        "points_url": f"{root}/files/points.ply",
    }


@app.post("/sequences/{sequence_id}/jobs", status_code=202)
def create_job(sequence_id: UUID, request: JobRequest):
    return _create_job(sequence_id, request)


def _create_job(sequence_id, request, retry_of=None):
    folder = sequence_folder(sequence_id)
    sequence = read_json(folder / "sequence.json")
    mask_folder = foreground_folder(folder, request.mask_set_id) if request.mask_set_id else None
    source = None
    saved_comparison = None
    if request.comparison_job_id or request.comparison_mode:
        if request.operation != "solve" or not request.comparison_job_id or not request.comparison_mode or request.sam or mask_folder:
            raise HTTPException(422, "Select one completed comparison and one result to open as a solve.")
        compared = jobs.status(job_folder(sequence_id, request.comparison_job_id))
        if compared["operation"] != "compare" or compared["status"] != "done":
            raise HTTPException(422, "The comparison must finish before you can open its results.")
        report = compared["result"]
        if report[request.comparison_mode]["status"] == "failed":
            raise HTTPException(422, "This comparison mode did not produce a camera solve.")
        compared_mask = foreground_folder(folder, UUID(report["mask_set_id"]))
        saved_comparison = compared_mask / UUID(report["comparison_id"]).hex / f"{request.comparison_mode}.json"
        if not saved_comparison.is_file():
            raise HTTPException(404, "The comparison camera data is missing.")
    if request.operation == "compare" and mask_folder is None:
        raise HTTPException(422, "Create or select a reviewed mask set first.")
    if request.operation == "solve" and request.sam != (mask_folder is not None):
        raise HTTPException(422, "Select baseline without a mask or enable SAM with a reviewed mask revision.")
    if request.operation == "preflight" and (request.sam or mask_folder is not None):
        raise HTTPException(422, "Geometry preflight currently uses baseline features.")
    if mask_folder is not None and request.operation in {"solve", "compare"}:
        require_exclusion(mask_folder)
    if request.operation == "export":
        if request.source_job_id is None:
            raise HTTPException(422, "Select a completed camera solve to export.")
        source = job_folder(sequence_id, request.source_job_id)
        if jobs.status(source)["status"] != "done" or not (source / "reconstruction.json").is_file():
            raise HTTPException(422, "Select a completed camera solve to export.")
        export_rate = parse_fps(request.export_fps)
        if export_rate is not None and sequence.get("fps"):
            raise HTTPException(422, "The source frame rate is already recorded and will be used for export.")
        if not sequence.get("fps") and (export_rate is None or not 1 <= export_rate <= 32767):
            raise HTTPException(422, "Supply the source FPS between 1 and 32767 to prepare a Blender package.")
    prefix = f"/sequences/{sequence_id.hex}/jobs"

    def work(target, progress):
        root = f"{prefix}/{target.name}"
        if saved_comparison is not None:
            progress("Opening saved comparison result")
            result = read_json(saved_comparison)
            manifest = read_json(compared_mask / "manifest.json") if request.comparison_mode == "masked" else None
            copy2(saved_comparison, target / "reconstruction.json")
            return {
                **save_solve_outputs(target, root, result, manifest),
                "comparison_job_id": request.comparison_job_id.hex, "comparison_mode": request.comparison_mode,
            }
        frames = sequence["frame_details"]
        selected = [frames[i]["frame_number"] for i in sampled_indices(len(frames), request.max_frames)]
        # SAM may inspect more frames than the sparse solver uses. Do not bind
        # mask coverage to the camera solve's frame limit.
        mask_budget = request.sam_analysis_frames if request.sam_analysis_frames is not None else request.max_frames
        sam_selected = [frames[i]["frame_number"] for i in sampled_indices(len(frames), mask_budget)]
        foreground_request = request.foreground
        if foreground_request.discover and foreground_request.discover_frame_numbers is None:
            foreground_request = foreground_request.model_copy(update={"discover_frame_numbers": sam_selected})
        if request.operation == "foreground":
            return _create_foreground(sequence_id, foreground_request, progress)
        if request.operation == "compare":
            return _compare_reconstructions(sequence_id, request.mask_set_id, request.max_frames, progress,
                                            request.initial_focal_px)
        if request.operation == "preflight":
            try:
                report = reconstruct(sequence, folder, max_frames=request.max_frames, progress=progress, preflight=True,
                                     initial_focal_px=request.initial_focal_px)
            except SolverUnavailable as exc:
                raise HTTPException(503, str(exc)) from exc
            write_json(target / "preflight.json", report)
            return {**report, "report_url": f"{root}/files/preflight.json"}
        if request.operation == "export":
            progress("Packaging camera, points and corrected plates")
            try:
                export_result = read_json(source / "reconstruction.json")
                if export_rate is not None:
                    export_result.update(fps=float(export_rate), fps_rational=[export_rate.numerator, export_rate.denominator],
                                         fps_source="provided_at_export")
                export_package(export_result, folder / "frames", target / "camera-package.zip", progress=progress)
            except ValueError as exc:
                raise HTTPException(422, str(exc)) from exc
            return {"export_url": f"{root}/files/camera-package.zip", "source_job_id": request.source_job_id.hex}

        manifest, mask_dir = None, None
        if mask_folder is not None:
            manifest = require_exclusion(mask_folder)
            mask_dir = mask_folder / "colmap"
        if mask_dir is not None:
            excluded = {frame["frame_number"] for frame in manifest["frames"] if frame["excluded_pixels"]}
            if not excluded.intersection(selected):
                raise HTTPException(422, "The selected solve frames contain no approved masks. Increase the frame limit or revise the mask set.")
        try:
            result = reconstruct(
                sequence, folder, mask_dir=mask_dir, max_frames=request.max_frames,
                output_path=target / "reconstruction.json", progress=progress, initial_focal_px=request.initial_focal_px,
            )
        except SolverUnavailable as exc:
            raise HTTPException(503, str(exc)) from exc
        return save_solve_outputs(target, root, result, manifest)

    try:
        state = jobs.submit(folder, request.operation, work, parameters=request.model_dump(mode="json"), retry_of=retry_of)
    except ValueError as exc:
        raise HTTPException(429, str(exc)) from exc
    return {**state, "status_url": f"{prefix}/{state['job_id']}"}


@app.get("/sequences/{sequence_id}/jobs")
def list_sequence_jobs(sequence_id: UUID):
    """Recover processing results after reload without repeating costly inference."""
    base = sequence_folder(sequence_id) / "jobs"
    if not base.exists():
        return {"jobs": []}
    rows = []
    for directory in base.iterdir():
        if not directory.is_dir() or not (directory / "status.json").is_file():
            continue
        try:
            row = jobs.status(directory)
            # UUID guard: no arbitrary filesystem directory names become URLs.
            identifier = UUID(row["job_id"])
            if identifier.hex != directory.name:
                continue
        except (ValueError, KeyError, OSError):
            continue
        row["status_url"] = f"/sequences/{sequence_id.hex}/jobs/{directory.name}"
        if "created_order" not in row:
            row["created_order"] = directory.stat().st_mtime_ns
        rows.append(row)
    rows.sort(key=lambda row: row["created_order"], reverse=True)
    return {"jobs": rows[:50]}


@app.get("/sequences/{sequence_id}/jobs/{job_id}")
def get_job(sequence_id: UUID, job_id: UUID):
    return jobs.status(job_folder(sequence_id, job_id))


@app.post("/sequences/{sequence_id}/jobs/{job_id}/cancel")
def cancel_job(sequence_id: UUID, job_id: UUID):
    return jobs.cancel(job_folder(sequence_id, job_id))


@app.post("/sequences/{sequence_id}/jobs/{job_id}/retry", status_code=202)
def retry_job(sequence_id: UUID, job_id: UUID):
    original = jobs.status(job_folder(sequence_id, job_id))
    if original["status"] not in {"failed", "cancelled"} or not original.get("parameters"):
        raise HTTPException(409, "Only failed or cancelled jobs with saved settings can be retried.")
    return _create_job(sequence_id, JobRequest.model_validate(original["parameters"]), retry_of=job_id.hex)


@app.get("/sequences/{sequence_id}/jobs/{job_id}/files/{filename}")
def job_file(
    sequence_id: UUID, job_id: UUID,
    filename: Literal["reconstruction.json", "tracks.json", "quality.json", "camera-package.zip", "diagnostics.json", "points.ply", "preflight.json"],
):
    folder = job_folder(sequence_id, job_id)
    path = folder / filename
    required_status = "failed" if filename == "diagnostics.json" else "done"
    if jobs.status(folder)["status"] != required_status or not path.is_file():
        raise HTTPException(404, "The requested result is not available.")
    return FileResponse(path, filename=filename)
