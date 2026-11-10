"""Background processing for the Streamlit monitor."""

from dataclasses import dataclass, replace
import logging
import math
from pathlib import Path
from threading import Event, Lock, Thread
from time import monotonic
from uuid import UUID, uuid4

import cv2

from visiontrack.cli import draw, fit_display
from visiontrack.config.model_registry import choose_model_profile
from visiontrack.config.settings import load_settings, project_path
from visiontrack.database import postgres_monitor as repository
from visiontrack.output_panel import HighQualityVideoWriter
from visiontrack.perception.class_mapper import canonical_class_name
from visiontrack.perception.detector import Detector
from visiontrack.pipeline.pallet_zone_pipeline import PalletZonePipeline

LOGGER = logging.getLogger(__name__)
MODEL_KEY = "six_class_final"
CLASSES = {"box", "pallet", "qr_code", "conveyor_belt", "dock", "zone"}


@dataclass(frozen=True)
class Video:
    path: Path
    name: str
    size: int
    width: int
    height: int
    fps: float
    frames: int


def inspect_video(path, name):
    capture = cv2.VideoCapture(str(path))
    try:
        if not capture.isOpened():
            raise ValueError("OpenCV could not read this MP4.")
        width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = capture.get(cv2.CAP_PROP_FPS)
        frames = max(0, int(capture.get(cv2.CAP_PROP_FRAME_COUNT)))
        if min(width, height) < 2 or not math.isfinite(fps) or fps <= 0:
            raise ValueError("The video must have a valid resolution and frame rate.")
        return Video(path, name, path.stat().st_size, width, height, fps, frames)
    finally:
        capture.release()


@dataclass(frozen=True)
class Progress:
    run_id: UUID
    status: str = "PROCESSING"
    frames: int = 0
    total: int = 0
    elapsed: float = 0
    device: str | None = None
    preview: bytes | None = None
    error: str | None = None
    output: Path | None = None


class Monitor:
    """One active worker per Streamlit server, shared across browser refreshes."""

    def __init__(self):
        self.lock = Lock()
        self.cancel = Event()
        self.progress: Progress | None = None
        self.detector: Detector | None = None

    def start(self, video, *, source_owner=None, max_frames=None):
        with self.lock:
            if self.progress and self.progress.status == "PROCESSING":
                return self.progress.run_id
            self.cancel = Event()
            run_id = uuid4()
            total = min(video.frames or max_frames, max_frames) if max_frames else video.frames
            self.progress = Progress(run_id, total=total)
            # Holding source_owner keeps the temporary upload alive after a browser refresh.
            Thread(
                target=self._run, args=(run_id, video, max_frames, source_owner),
                name=f"visiontrack-{run_id}", daemon=True,
            ).start()
            return run_id

    def stop(self):
        with self.lock:
            if self.progress and self.progress.status == "PROCESSING":
                self.cancel.set()

    def snapshot(self):
        with self.lock:
            return self.progress

    def _publish(self, **values):
        with self.lock:
            self.progress = replace(self.progress, **values)

    def _run(self, run_id, video, max_frames, source_owner):
        started = monotonic()
        frame_number = 0
        device = None
        created = False
        folder = project_path("outputs/monitor") / str(run_id)
        partial = folder / "encoding.mp4"
        output = folder / "annotated.mp4"
        try:
            with repository.connection() as database:
                model_path = project_path(f"models/{MODEL_KEY}.pt")
                repository.create_run(database, run_id, video, model_path)
                created = True
                settings = load_settings(database_target="off")
                profile = choose_model_profile(settings.models, MODEL_KEY)
                if profile.path.resolve() != model_path.resolve():
                    raise ValueError(f"The monitor requires {model_path.name} in models/.")
                frame_number, device, unresolved, pending = self._process(
                    database, run_id, video, settings.app, profile, partial, max_frames,
                )
                stopped = self.cancel.is_set()
                if not stopped:
                    partial.replace(output)
                elapsed = monotonic() - started
                status = "STOPPED" if stopped else "COMPLETED"
                repository.finish_run(
                    database, run_id, status, frame_number, elapsed, device,
                    None if stopped else output, unresolved, pending,
                )
        except Exception as error:
            LOGGER.exception("VisionTrack run %s failed", run_id)
            self.detector = None
            progress = self.snapshot()
            error_message = f"{type(error).__name__}: {error}"
            if created:
                try:
                    with repository.connection() as database:
                        repository.finish_run(
                            database, run_id, "FAILED", progress.frames, monotonic() - started,
                            progress.device, None, error=error_message,
                        )
                except Exception:
                    LOGGER.exception("Could not persist failure for run %s", run_id)
                    error_message += " (PostgreSQL could not save the failure status.)"
            self._discard(partial, output, folder)
            self._publish(status="FAILED", elapsed=monotonic() - started, error=error_message)
            return

        if stopped:
            self._discard(partial, output, folder)
        self._publish(status=status, frames=frame_number, elapsed=elapsed, device=device,
                      output=None if stopped else output)

    @staticmethod
    def _discard(partial, output, folder):
        try:
            partial.unlink(missing_ok=True)
            output.unlink(missing_ok=True)
            if folder.exists():
                folder.rmdir()
        except OSError:
            LOGGER.exception("Could not clean up output in %s", folder)

    def _process(self, database, run_id, video, app, profile, partial, max_frames):
        if self.cancel.is_set():
            return 0, None, 0, 0
        # The integrated monitor accepts arbitrary cameras, so legacy floor ROIs are disabled.
        dynamic = app.setdefault("dynamic_location", {})
        dynamic["record_region_arrivals"] = True
        dynamic["floor_inference"] = {"enabled": False}
        app.setdefault("perception", {})["guard_qr_identity"] = True
        detector = self.detector
        if detector is None or detector.profile != profile:
            detector = Detector(profile)
            names = {canonical_class_name(name, profile.class_aliases) for name in detector.names.values()}
            if names != CLASSES or len(detector.names) != len(CLASSES):
                raise ValueError(f"Unexpected model classes: {', '.join(sorted(names))}")
            self.detector = detector
        else:
            detector.reset_tracking()
        pipeline = PalletZonePipeline(app, profile, detector)
        device = detector.device_label
        self._publish(device=device)
        scale = min(1.0, 1280 / video.width, 720 / video.height)
        width = max(2, int(video.width * scale) // 2 * 2)
        height = max(2, int(video.height * scale) // 2 * 2)
        capture = cv2.VideoCapture(str(video.path))
        writer = None
        events = []
        dirty_objects = set()
        pending_conflicts = set()
        conflicts = {}
        last_seen = {}
        last_confirmed = {}
        unresolved = set()
        recent = None
        frame_number = 0
        started = monotonic()
        publish_at = started
        flush_at = started + 1

        def flush():
            if not (events or dirty_objects or pending_conflicts):
                return
            inventory = []
            for kind, code in sorted(dirty_objects):
                states = pipeline.engine.boxes if kind == "box" else pipeline.engine.pallets
                state = states.get(code)
                if state is not None:
                    pallet = getattr(state, "current_pallet_code", None)
                    zone = state.current_zone_code
                    if not any(pipeline.engine._is_temporary_identity(value) for value in (pallet, zone)):
                        last_confirmed[kind, code] = pallet, zone
                    inventory.append((kind, code, *last_confirmed.get((kind, code), (None, None)),
                                      last_seen[kind, code]))
            repository.persist(database, run_id, events, inventory,
                               {code: conflicts[code] for code in pending_conflicts})
            events.clear()
            dirty_objects.clear()
            pending_conflicts.clear()

        try:
            if not capture.isOpened():
                raise ValueError("The uploaded MP4 could not be reopened.")
            while not self.cancel.is_set() and (max_frames is None or frame_number < max_frames):
                ok, frame = capture.read()
                if not ok:
                    break
                if writer is None:
                    writer = HighQualityVideoWriter(partial, width, height, video.fps, crf=20)
                frame_number += 1
                result = pipeline.process_frame(frame, frame_number)
                seconds = (frame_number - 1) / video.fps
                for code in pipeline.perception.identity_conflicts:
                    first, _, count = conflicts.get(code, (frame_number, frame_number, 0))
                    conflicts[code] = (first, frame_number, count + 1)
                    pending_conflicts.add(code)
                for observation in result.observations:
                    kind, code = observation.class_name, observation.qr_value
                    if kind not in {"box", "pallet"}:
                        continue
                    if code and observation.qr_read_success:
                        key = (kind, code)
                        last_seen[key] = seconds
                        dirty_objects.add(key)
                        unresolved.discard((kind, observation.track_id))
                    elif observation.track_id is not None:
                        key = (kind, observation.track_id)
                        identity = pipeline.engine.track_identities.get(
                            (kind.upper(), observation.track_id)
                        )
                        if not identity or pipeline.engine._is_temporary_identity(identity):
                            unresolved.add(key)
                        else:
                            last_seen[kind, identity] = seconds
                            dirty_objects.add((kind, identity))

                for event in result.events:
                    if event.qr_read_success is True:
                        events.append((frame_number, seconds, event))
                        key = ("box", event.box_code)
                        if key in last_seen:
                            dirty_objects.add(key)
                        recent = (frame_number, event)
                draw(frame, result.display_observations, result.qrs, result.events)
                annotate(frame, result.display_observations, pipeline.engine,
                         pipeline.perception.identity_conflicts, recent, frame_number, video.fps)
                if frame.shape[:2] != (height, width):
                    frame = cv2.resize(frame, (width, height), interpolation=cv2.INTER_AREA)
                writer.write(frame)
                now = monotonic()
                if now >= flush_at or len(events) >= 128:
                    flush()
                    flush_at = now + 1
                if now >= publish_at:
                    ok, jpeg = cv2.imencode(
                        ".jpg", fit_display(frame, max_width=900, max_height=506),
                        [cv2.IMWRITE_JPEG_QUALITY, 78],
                    )
                    self._publish(frames=frame_number, elapsed=now - started,
                                  preview=jpeg.tobytes() if ok else None)
                    publish_at = now + 0.35
            flush()
            if not self.cancel.is_set():
                if frame_number == 0:
                    raise ValueError("The video contained no readable frames.")
                expected = min(video.frames, max_frames) if max_frames else video.frames
                if expected and frame_number < expected:
                    raise RuntimeError(f"Video decoding stopped at frame {frame_number} of {expected}.")
        finally:
            capture.release()
            self._publish(frames=frame_number)
            if writer:
                writer.release()
        pending = len(pipeline.engine.pending_events) + pipeline.engine.dropped_unresolved_event_count
        return frame_number, device, len(unresolved), pending


def annotate(frame, observations, engine, conflicts, recent, frame_number, fps):
    for observation in observations:
        kind = observation.class_name
        if kind not in {"box", "pallet"}:
            continue
        identity = engine.track_identities.get((kind.upper(), observation.track_id))
        states = engine.boxes if kind == "box" else engine.pallets
        state = states.get(identity or observation.qr_value)
        label = kind
        if observation.track_id is not None:
            label += f" #{observation.track_id}"
        code = identity or observation.qr_value
        if code in conflicts or observation.qr_value in conflicts:
            label += " / QR conflict"
        elif state and code and "_TRACK_" not in code:
            label += f" {code}"
            location = getattr(state, "current_pallet_code", None) or state.current_zone_code
            if location and "_TRACK_" not in location:
                label += f" / {location}"
        x, y = observation.bbox[:2]
        caption(frame, label, (x, max(24, y - 4)))
    if recent and frame_number - recent[0] <= 2 * fps:
        event = recent[1]
        caption(frame, f"{event.event_type.value}  {event.box_code}", (15, 32))


def caption(frame, label, point):
    scale = max(0.55, min(1.25, frame.shape[1] / 1250))
    cv2.putText(frame, label, point, cv2.FONT_HERSHEY_SIMPLEX, scale, (5, 15, 25), 5, cv2.LINE_AA)
    cv2.putText(frame, label, point, cv2.FONT_HERSHEY_SIMPLEX, scale, (230, 255, 245), 2, cv2.LINE_AA)
