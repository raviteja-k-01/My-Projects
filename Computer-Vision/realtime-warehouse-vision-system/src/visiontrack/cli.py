from __future__ import annotations

import argparse
from collections import Counter
from copy import deepcopy
from dataclasses import replace
from datetime import datetime
from pathlib import Path
import re
import time


# =====================================================================
# CLI ARGUMENTS
# =====================================================================


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="VisionTrack pallet-zone processor"
    )

    parser.add_argument(
        "--video",
        required=True,
        help="Video path or camera stream URL",
    )

    parser.add_argument(
        "--model-key",
        default="six_class_final",
        help="Model key, .pt filename, or model path",
    )

    parser.add_argument(
        "--device",
        default="auto",
        help="Inference device: auto, cpu, cuda, cuda:0, or GPU index like 0",
    )

    parser.add_argument(
        "--show-conveyor-belt",
        action="store_true",
        help="Deprecated; conveyor_belt detections are drawn by default",
    )

    parser.add_argument(
        "--hide-conveyor-belt",
        action="store_true",
        help="Hide conveyor_belt detections in the preview window",
    )

    parser.add_argument(
        "--status-every",
        type=int,
        default=None,
        help="Print one status line every N processed frames",
    )

    parser.add_argument(
        "--estimate-carton-boxes",
        action="store_true",
        help="Deprecated alias for --estimate-missing-boxes",
    )

    parser.add_argument(
        "--estimate-missing-boxes",
        action="store_true",
        help=(
            "Create approximate carton boxes from package QR codes "
            "when the model misses a box"
        ),
    )

    parser.add_argument(
        "--no-display",
        action="store_true",
        help="Disable OpenCV display",
    )

    parser.add_argument(
        "--output-video",
        default=None,
        help="Optional path for the annotated video with a separate logs panel",
    )

    parser.add_argument(
        "--output-crf",
        type=int,
        default=17,
        help="H.264 output quality: 0 is lossless, 17 is visually near-lossless",
    )

    parser.add_argument(
        "--max-frames",
        type=int,
        default=None,
        help="Optional frame limit for a short preview or test",
    )

    parser.add_argument(
        "--db-mode",
        choices=("off", "local", "cloud"),
        default=None,
        help="Database destination; defaults to DB_TARGET in .env",
    )

    parser.add_argument(
        "--db-disabled",
        action="store_true",
        help="Deprecated alias for --db-mode off",
    )

    return parser.parse_args()


# =====================================================================
# VIDEO SOURCE
# =====================================================================


def resolve_video_source(
    video: str,
    input_video_dir: str | None,
) -> str:

    direct = Path(video)

    if direct.exists():
        return str(direct)

    if "://" in video:
        return video

    search_dir = Path(
        input_video_dir or "data/input_videos"
    )

    if not search_dir.is_absolute():
        search_dir = Path.cwd() / search_dir

    named = search_dir / video

    if named.exists():
        return str(named)

    if not video.lower().endswith(".mp4"):

        if video.isdigit():

            exact_numbered = sorted(
                search_dir.glob(f"{video}.*.mp4")
            )

            exact_input = (
                search_dir / f"input_video{video}.mp4"
            )

            if len(exact_numbered) == 1:
                return str(exact_numbered[0])

            if exact_input.exists():
                return str(exact_input)

        matches = sorted(
            search_dir.glob(f"{video}*.mp4")
        )

        if len(matches) == 1:
            return str(matches[0])

        if len(matches) > 1:
            options = "\n".join(
                str(item.name)
                for item in matches
            )

            raise ValueError(
                f"Multiple videos match '{video}':\n"
                f"{options}"
            )

    raise FileNotFoundError(
        f"Video '{video}' was not found directly "
        f"or under {search_dir}."
    )


def automatic_output_video_path(
    video_source: str,
) -> Path:

    source_name = (
        "stream"
        if "://" in video_source
        else Path(video_source).stem
    )

    safe_name = re.sub(
        r"[^A-Za-z0-9_-]+",
        "_",
        source_name,
    ).strip("_")

    if not safe_name:
        safe_name = "video"

    timestamp = datetime.now().strftime(
        "%Y%m%d_%H%M%S"
    )

    return (
        Path("outputs/videos")
        / f"{safe_name}_{timestamp}_annotated.mp4"
    )


# =====================================================================
# DISPLAY HELPERS
# =====================================================================


def fit_display(
    frame,
    max_width: int = 1280,
    max_height: int = 820,
):
    import cv2

    height, width = frame.shape[:2]

    scale = min(
        max_width / max(width, 1),
        max_height / max(height, 1),
        1.0,
    )

    if scale >= 1.0:
        return frame

    display_width = max(
        1,
        int(width * scale),
    )

    display_height = max(
        1,
        int(height * scale),
    )

    return cv2.resize(
        frame,
        (display_width, display_height),
        interpolation=cv2.INTER_AREA,
    )


def draw(
    frame,
    observations,
    qrs,
    events,
    hidden_classes: set[str] | None = None,
) -> None:

    import cv2

    height, width = frame.shape[:2]

    thickness = max(
        3,
        int(round(width / 700)),
    )

    font_scale = max(
        0.55,
        min(1.1, width / 1400),
    )

    hidden_classes = (
        hidden_classes or set()
    )

    used_label_boxes: list[
        tuple[int, int, int, int]
    ] = []

    drawable_observations = (
        suppress_duplicate_display_boxes(
            observations
        )
    )

    for observation in drawable_observations:

        if observation.class_name in hidden_classes:
            continue

        if observation.class_name == "qr_code":
            continue

        x1, y1, x2, y2 = observation.bbox

        color = {
            "box": (0, 125, 0),
            "pallet": (180, 70, 0),
            "conveyor_belt": (135, 0, 135),
            "dock": (0, 0, 200),
            "zone": (190, 95, 0),
        }.get(
            observation.class_name,
            (80, 80, 80),
        )

        cv2.rectangle(
            frame,
            (x1, y1),
            (x2, y2),
            color,
            thickness,
        )

        if observation.class_name in {
            "conveyor_belt",
            "dock",
            "zone",
        }:

            draw_label(
                frame,
                observation.qr_value
                or observation.class_name,
                (x1, y1),
                color,
                font_scale,
                used_label_boxes,
            )

    for qr in qrs:

        color = (0, 220, 220)

        x1, y1, x2, y2 = qr["bbox"]

        points = qr.get("points")

        if points and len(points) >= 4:

            for index, start in enumerate(points):

                end = points[
                    (index + 1) % len(points)
                ]

                cv2.line(
                    frame,
                    start,
                    end,
                    color,
                    thickness,
                )

        else:

            cv2.rectangle(
                frame,
                (x1, y1),
                (x2, y2),
                color,
                thickness,
            )

        draw_label(
            frame,
            qr["value"],
            (x1, y1),
            color,
            font_scale,
            used_label_boxes,
        )


def draw_label(
    frame,
    text: str,
    anchor: tuple[int, int],
    color: tuple[int, int, int],
    font_scale: float,
    used_boxes: list[
        tuple[int, int, int, int]
    ],
) -> None:

    import cv2

    height, width = frame.shape[:2]

    font = cv2.FONT_HERSHEY_SIMPLEX

    text_thickness = max(
        1,
        int(round(width / 1400)),
    )

    (
        text_width,
        text_height,
    ), baseline = cv2.getTextSize(
        text,
        font,
        font_scale,
        text_thickness,
    )

    pad = 4

    x = max(
        0,
        min(
            anchor[0],
            width
            - text_width
            - pad * 2,
        ),
    )

    preferred_y = (
        anchor[1]
        - text_height
        - baseline
        - pad * 2
    )

    y = (
        preferred_y
        if preferred_y >= 0
        else anchor[1] + pad
    )

    label_box = (
        x,
        y,
        x + text_width + pad * 2,
        y
        + text_height
        + baseline
        + pad * 2,
    )

    while (
        any(
            overlaps(label_box, used)
            for used in used_boxes
        )
        and label_box[3] < height - 1
    ):

        y += (
            text_height
            + baseline
            + pad * 2
        )

        label_box = (
            x,
            y,
            x + text_width + pad * 2,
            y
            + text_height
            + baseline
            + pad * 2,
        )

    y = min(
        y,
        max(
            0,
            height
            - text_height
            - baseline
            - pad * 2,
        ),
    )

    label_box = (
        x,
        y,
        x + text_width + pad * 2,
        y
        + text_height
        + baseline
        + pad * 2,
    )

    used_boxes.append(label_box)

    cv2.rectangle(
        frame,
        (label_box[0], label_box[1]),
        (label_box[2], label_box[3]),
        (20, 20, 20),
        -1,
    )

    cv2.rectangle(
        frame,
        (label_box[0], label_box[1]),
        (label_box[2], label_box[3]),
        color,
        1,
    )

    cv2.putText(
        frame,
        text,
        (
            x + pad,
            y + pad + text_height,
        ),
        font,
        font_scale,
        color,
        text_thickness,
    )


# =====================================================================
# BOUNDING BOX HELPERS
# =====================================================================


def overlaps(
    a: tuple[int, int, int, int],
    b: tuple[int, int, int, int],
) -> bool:

    return not (
        a[2] <= b[0]
        or b[2] <= a[0]
        or a[3] <= b[1]
        or b[3] <= a[1]
    )


def suppress_duplicate_display_boxes(
    observations,
):

    kept = []

    for observation in observations:

        duplicate_index = (
            duplicate_display_index(
                observation,
                kept,
            )
        )

        if duplicate_index is None:

            kept.append(
                observation
            )

            continue

        current = kept[
            duplicate_index
        ]

        if (
            display_observation_score(
                observation
            )
            >
            display_observation_score(
                current
            )
        ):

            kept[
                duplicate_index
            ] = observation

    return kept


def duplicate_display_index(
    observation,
    kept,
) -> int | None:

    if observation.class_name not in {
        "box",
        "pallet",
    }:
        return None

    for index, current in enumerate(
        kept
    ):

        if (
            current.class_name
            != observation.class_name
        ):
            continue

        if (
            bbox_iou(
                observation.bbox,
                current.bbox,
            )
            >= 0.78
        ):
            return index

        if (
            bbox_overlap_min_area(
                observation.bbox,
                current.bbox,
            )
            >= 0.88
        ):
            return index

    return None


def display_observation_score(
    observation,
) -> tuple[int, float]:

    has_qr = (
        1
        if getattr(
            observation,
            "qr_value",
            None,
        )
        else 0
    )

    return (
        has_qr,
        float(
            getattr(
                observation,
                "confidence",
                0.0,
            )
        ),
    )


def bbox_iou(
    a: tuple[int, int, int, int],
    b: tuple[int, int, int, int],
) -> float:

    intersection = (
        bbox_intersection_area(
            a,
            b,
        )
    )

    if intersection <= 0:
        return 0.0

    union = (
        bbox_area(a)
        + bbox_area(b)
        - intersection
    )

    return (
        intersection
        / max(union, 1)
    )


def bbox_overlap_min_area(
    a: tuple[int, int, int, int],
    b: tuple[int, int, int, int],
) -> float:

    intersection = (
        bbox_intersection_area(
            a,
            b,
        )
    )

    return (
        intersection
        / max(
            min(
                bbox_area(a),
                bbox_area(b),
            ),
            1,
        )
    )


def bbox_intersection_area(
    a: tuple[int, int, int, int],
    b: tuple[int, int, int, int],
) -> int:

    left = max(
        a[0],
        b[0],
    )

    top = max(
        a[1],
        b[1],
    )

    right = min(
        a[2],
        b[2],
    )

    bottom = min(
        a[3],
        b[3],
    )

    return (
        max(0, right - left)
        * max(0, bottom - top)
    )


def bbox_area(
    bbox: tuple[int, int, int, int],
) -> int:

    return (
        max(
            0,
            bbox[2] - bbox[0],
        )
        *
        max(
            0,
            bbox[3] - bbox[1],
        )
    )


# =====================================================================
# TERMINAL REPORTER
# =====================================================================


class TerminalReporter:

    def __init__(
        self,
        total_frames: int,
        database_enabled: bool,
        update_every: int = 30,
    ):

        self.total_frames = total_frames

        self.database_enabled = (
            database_enabled
        )

        self.update_every = max(
            1,
            update_every,
        )

        self.seen_boxes: set[str] = set()

        self.seen_pallets: set[str] = set()

        self.seen_qrs: set[str] = set()

        self.event_count = 0

    def update(
        self,
        frame_number: int,
        observations,
        qrs,
        events,
        elapsed: float,
    ) -> None:

        counts = Counter(
            item.class_name
            for item in observations
        )

        for qr in qrs:
            self.seen_qrs.add(
                qr["value"]
            )

        for observation in observations:

            if observation.qr_value:

                self.seen_qrs.add(
                    observation.qr_value
                )

            if (
                observation.class_name
                == "box"
                and observation.qr_value
            ):

                self.seen_boxes.add(
                    observation.qr_value
                )

            if (
                observation.class_name
                == "pallet"
                and observation.qr_value
            ):

                self.seen_pallets.add(
                    observation.qr_value
                )

        for event in events:

            self.event_count += 1

            print(
                self._format_event(
                    self.event_count,
                    frame_number,
                    event,
                ),
                flush=True,
            )

        if (
            frame_number == 1
            or frame_number
            % self.update_every
            == 0
            or frame_number
            == self.total_frames
        ):

            fps = (
                frame_number
                / max(elapsed, 0.001)
            )

            total = (
                self.total_frames
                if self.total_frames > 0
                else "?"
            )

            percent = self._percent(
                frame_number
            )

            decoded = (
                self._format_values(
                    [
                        qr["value"]
                        for qr in qrs
                    ],
                    max_items=6,
                )
            )

            print(
                f"[STATUS] frame "
                f"{frame_number:>4}/{total} "
                f"{percent:>6} | "
                f"{fps:>5.2f} fps | "
                f"box={counts.get('box', 0)} "
                f"pallet={counts.get('pallet', 0)} "
                f"conveyor="
                f"{counts.get('conveyor_belt', 0)} "
                f"qr={counts.get('qr_code', 0)} | "
                f"events={self.event_count} "
                f"(+{len(events)})",
                flush=True,
            )

            print(
                f"         decoded: "
                f"{decoded} | "
                f"seen boxes="
                f"{len(self.seen_boxes)} "
                f"pallets="
                f"{len(self.seen_pallets)} "
                f"qrs="
                f"{len(self.seen_qrs)}",
                flush=True,
            )

    def finish(
        self,
        frame_number: int,
        elapsed: float,
    ) -> None:

        print()

        print("Summary")

        print(
            f"  Frames : "
            f"{frame_number}"
        )

        print(
            f"  FPS    : "
            f"{frame_number / max(elapsed, 0.001):.2f} "
            f"average"
        )

        print(
            f"  Boxes  : "
            f"{self._format_values(sorted(self.seen_boxes), max_items=12)}"
        )

        print(
            f"  Pallets: "
            f"{self._format_values(sorted(self.seen_pallets), max_items=12)}"
        )

        print(
            f"  QR     : "
            f"{len(self.seen_qrs)} unique"
        )

        print(
            f"  Events : "
            f"{self.event_count}"
        )

    @staticmethod
    def _format_values(
        values,
        max_items: int = 8,
    ) -> str:

        items = [
            str(value)
            for value in values
            if value
        ]

        if not items:
            return "-"

        visible = items[:max_items]

        suffix = (
            f", +{len(items) - max_items} more"
            if len(items) > max_items
            else ""
        )

        return (
            ", ".join(visible)
            + suffix
        )

    def _db_label(self) -> str:

        return (
            "db"
            if self.database_enabled
            else "off"
        )

    def _percent(
        self,
        frame_number: int,
    ) -> str:

        if self.total_frames <= 0:
            return ""

        return (
            f"{frame_number / self.total_frames * 100:5.1f}%"
        )

    def _format_event(
        self,
        event_number: int,
        frame_number: int,
        event,
    ) -> str:

        slot = ""

        if event.slot_no is not None:

            slot = (
                f" | slot={event.slot_no} "
                f"row={event.row_no} "
                f"col={event.column_no} "
                f"layer={event.layer_no}"
            )

        return (
            f"[EVENT {event_number:03}] "
            f"frame {frame_number} | "
            f"{event.event_type.value} | "
            f"box={event.box_code} | "
            f"from="
            f"{event.from_pallet_code or '-'} | "
            f"to="
            f"{event.to_pallet_code or '-'} | "
            f"zone="
            f"{event.to_zone_code or '-'}"
            f"{slot} | "
            f"target={self._db_label()} | "
            f"{event.remarks}"
        )


# =====================================================================
# DATABASE REPOSITORY
# =====================================================================


def build_repository(settings):

    from visiontrack.database.repository import (
        VisionTrackRepository,
    )

    if not settings.database.enabled:
        return None

    db = settings.database

    # --------------------------------------------------------------
    # CLOUD DATABASE
    # --------------------------------------------------------------

    if db.target == "cloud":

        from visiontrack.database.cloud_repository import (
            CloudDatabaseConfig,
            CloudVisionTrackRepository,
        )

        return CloudVisionTrackRepository.from_config(
            CloudDatabaseConfig(
                host=db.host,
                port=db.port,
                user=db.user,
                password=db.password,
                database=db.name,
            )
        )

    # --------------------------------------------------------------
    # LOCAL DATABASE
    # --------------------------------------------------------------

    if db.target != "local":

        raise RuntimeError(
            f"Unsupported database target: "
            f"{db.target!r}"
        )

    if not all(
        [
            db.host,
            db.name,
            db.user,
        ]
    ):

        raise RuntimeError(
            "Database is enabled but "
            "DB_HOST, DB_NAME, or DB_USER "
            "is missing."
        )

    from visiontrack.database.connection import (
        DatabaseConfig,
        DatabaseConnectionFactory,
    )

    factory = DatabaseConnectionFactory(
        DatabaseConfig(
            host=db.host,
            port=db.port,
            user=db.user,
            password=db.password,
            database=db.name,
        )
    )

    return VisionTrackRepository(
        factory
    )


# =====================================================================
# MODEL / GPU HELPERS
# =====================================================================


def _normalise_imgsz(
    value,
) -> int | None:

    if value is None:
        return None

    if isinstance(
        value,
        (list, tuple),
    ):

        if not value:
            return None

        value = max(value)

    try:
        return int(value)

    except (
        TypeError,
        ValueError,
    ):
        return None


def _gpu_name(
    device_label: str,
) -> str | None:

    if "cpu" in device_label.lower():
        return None

    try:

        import torch

        if not torch.cuda.is_available():
            return None

        device_index = (
            torch.cuda.current_device()
        )

        if ":" in device_label:

            possible_index = (
                device_label.rsplit(
                    ":",
                    1,
                )[-1]
            )

            if possible_index.isdigit():
                device_index = int(
                    possible_index
                )

        return torch.cuda.get_device_name(
            device_index
        )

    except Exception:
        return None


# =====================================================================
# MAIN
# =====================================================================


def main() -> None:

    application_started = (
        time.perf_counter()
    )

    args = parse_args()

    print(
        "VisionTrack: loading runtime...",
        flush=True,
    )

    import cv2

    from visiontrack.config.model_registry import (
        choose_model_profile,
    )

    from visiontrack.config.settings import (
        load_settings,
    )

    from visiontrack.output_panel import (
        HighQualityVideoWriter,
        OutputPanel,
    )

    from visiontrack.pipeline.pallet_zone_pipeline import (
        PalletZonePipeline,
    )

    from visiontrack.run_logging import (
        RunLogger,
    )

    # --------------------------------------------------------------
    # SETTINGS
    # --------------------------------------------------------------

    database_target = (
        "off"
        if args.db_disabled
        else args.db_mode
    )

    settings = load_settings(
        database_target=database_target
    )

    model_profile = replace(
        choose_model_profile(
            settings.models,
            args.model_key,
        ),
        device=args.device,
    )

    repository = build_repository(
        settings
    )

    # --------------------------------------------------------------
    # APPLICATION CONFIGURATION
    # --------------------------------------------------------------

    app_config = deepcopy(
        settings.app
    )

    if (
        args.estimate_missing_boxes
        or args.estimate_carton_boxes
    ):

        app_config.setdefault(
            "perception",
            {},
        )[
            "synthesize_missing_boxes_from_qr"
        ] = True

    # --------------------------------------------------------------
    # MODEL
    # --------------------------------------------------------------

    print(
        f"VisionTrack: loading model "
        f"{model_profile.path.name} "
        f"on {args.device}...",
        flush=True,
    )

    pipeline = PalletZonePipeline(
        app_config=app_config,
        model_profile=model_profile,
    )

    print(
        "VisionTrack: model ready.",
        flush=True,
    )

    # --------------------------------------------------------------
    # VIDEO
    # --------------------------------------------------------------

    video_source = resolve_video_source(
        args.video,
        settings.app.get(
            "runtime",
            {},
        ).get(
            "input_video_dir",
            "data/input_videos",
        ),
    )

    cap = cv2.VideoCapture(
        video_source
    )

    if not cap.isOpened():

        raise RuntimeError(
            f"Unable to open video source: "
            f"{video_source}"
        )

    source_total_frames = int(
        cap.get(
            cv2.CAP_PROP_FRAME_COUNT
        )
    )

    if (
        args.max_frames is not None
        and args.max_frames <= 0
    ):

        cap.release()

        raise ValueError(
            "--max-frames must be "
            "greater than zero"
        )

    total_frames = (
        source_total_frames
    )

    if args.max_frames is not None:

        total_frames = (
            min(
                source_total_frames,
                args.max_frames,
            )
            if source_total_frames > 0
            else args.max_frames
        )

    source_width = int(
        cap.get(
            cv2.CAP_PROP_FRAME_WIDTH
        )
    )

    source_height = int(
        cap.get(
            cv2.CAP_PROP_FRAME_HEIGHT
        )
    )

    source_fps = float(
        cap.get(
            cv2.CAP_PROP_FPS
        )
    )

    if (
        source_fps != source_fps
        or source_fps <= 0
    ):
        source_fps = 0.0

    estimated_duration_sec = (
        total_frames / source_fps
        if (
            total_frames > 0
            and source_fps > 0
        )
        else None
    )

    # --------------------------------------------------------------
    # RUNTIME SETTINGS
    # --------------------------------------------------------------

    runtime = settings.app.get(
        "runtime",
        {},
    )

    window_name = runtime.get(
        "display_window_name",
        "VisionTrack Warehouse Monitor",
    )

    max_display_width = int(
        runtime.get(
            "display_max_width",
            1280,
        )
    )

    max_display_height = int(
        runtime.get(
            "display_max_height",
            820,
        )
    )

    max_output_width = int(
        runtime.get(
            "output_max_width",
            1920,
        )
    )

    max_output_height = int(
        runtime.get(
            "output_max_height",
            1080,
        )
    )

    # --------------------------------------------------------------
    # REPORTER
    # --------------------------------------------------------------

    reporter = TerminalReporter(
        total_frames=total_frames,
        database_enabled=(
            repository is not None
        ),
        update_every=int(
            args.status_every
            or runtime.get(
                "terminal_update_frames",
                90,
            )
        ),
    )

    # --------------------------------------------------------------
    # OUTPUT PANEL
    # --------------------------------------------------------------

    output_panel = OutputPanel()

    source_panel_width = (
        output_panel.panel_width_for(
            source_width
        )
    )

    source_canvas_width = (
        source_width
        + source_panel_width
    )

    output_scale = min(
        max_output_width
        / max(
            source_canvas_width,
            1,
        ),
        max_output_height
        / max(
            source_height,
            1,
        ),
        1.0,
    )

    output_video_width = max(
        2,
        int(
            source_width
            * output_scale
        )
        // 2
        * 2,
    )

    output_height = max(
        2,
        int(
            source_height
            * output_scale
        )
        // 2
        * 2,
    )

    output_panel_width = (
        output_panel.panel_width_for(
            output_video_width
        )
    )

    output_width = (
        output_video_width
        + output_panel_width
    )

    # --------------------------------------------------------------
    # MODEL METADATA
    # --------------------------------------------------------------

    model_imgsz = _normalise_imgsz(
        getattr(
            model_profile,
            "imgsz",
            None,
        )
    )

    if model_imgsz is None:

        model_imgsz = _normalise_imgsz(
            getattr(
                pipeline.perception.detector,
                "imgsz",
                None,
            )
        )

    device_label = str(
        pipeline.perception.detector.device_label
    )

    # --------------------------------------------------------------
    # PATHS
    # --------------------------------------------------------------

    stored_video_path = (
        video_source
    )

    if "://" not in video_source:

        stored_video_path = str(
            Path(
                video_source
            ).resolve()
        )

    video_name = (
        Path(video_source).name
        or args.video
    )

    output_video_path = (
        Path(args.output_video)
        if args.output_video
        else automatic_output_video_path(
            video_source
        )
    )

    log_directory = (
        settings.app.get(
            "runtime",
            {},
        ).get(
            "log_directory",
            "outputs/logs",
        )
    )

    run_logger = RunLogger(
        log_directory,
        video_name,
    )

    model_path = (
        str(
            model_profile.path
        )
        if getattr(
            model_profile,
            "path",
            None,
        )
        else None
    )

    # --------------------------------------------------------------
    # RUN STATE
    # --------------------------------------------------------------

    video_writer = None

    run_id = None

    frame_number = 0

    processing_started = None

    database_run_completed = False

    # ==============================================================
    # PROCESSING
    # ==============================================================

    try:

        if not args.no_display:

            cv2.namedWindow(
                window_name,
                cv2.WINDOW_NORMAL,
            )

        output_fps = (
            source_fps
            if source_fps > 0
            else 30.0
        )

        video_writer = (
            HighQualityVideoWriter(
                path=output_video_path,
                width=output_width,
                height=output_height,
                fps=output_fps,
                crf=args.output_crf,
            )
        )

        # ----------------------------------------------------------
        # CREATE DATABASE RUN
        # ----------------------------------------------------------

        if repository is not None:

            run_id = repository.create_run(
                video_name=video_name,
                input_video_path=(
                    stored_video_path
                ),
                model_path=model_path,
                device=device_label,
                gpu_name=_gpu_name(
                    device_label
                ),
                source_width=source_width,
                source_height=source_height,
                source_fps=source_fps,
                video_duration_sec=(
                    estimated_duration_sec
                ),
                frame_skip=1,
                process_width=source_width,
                process_height=source_height,
                yolo_imgsz=model_imgsz,
            )

        # ----------------------------------------------------------
        # START INFORMATION
        # ----------------------------------------------------------

        print("VisionTrack")
        print("-----------")

        print(
            f"Video    : "
            f"{video_source}"
        )

        print(
            f"Source   : "
            f"{source_width}x"
            f"{source_height} "
            f"@ {source_fps:.2f} FPS"
        )

        print(
            f"Model    : "
            f"{model_profile.key} "
            f"-> {model_profile.path}"
        )

        print(
            f"Device   : "
            f"{device_label}"
        )

        print(
            "Database : "
            + (
                f"ON - run {run_id}"
                if repository is not None
                else "OFF - preview only"
            )
        )

        print(
            f"Status   : every "
            f"{reporter.update_every} "
            f"frame(s)"
        )

        print(
            f"Output   : "
            f"{output_video_path}"
        )

        print(
            f"Log      : "
            f"{run_logger.path}"
        )

        run_logger.info(
            f"RUN STARTED | "
            f"video={video_source}"
        )

        run_logger.info(
            f"source="
            f"{source_width}x"
            f"{source_height}@"
            f"{source_fps:.2f}"
        )

        run_logger.info(
            f"model="
            f"{model_profile.key} "
            f"path={model_profile.path}"
        )

        run_logger.info(
            f"device="
            f"{device_label}"
        )

        run_logger.info(
            f"database="
            f"{settings.database.target}"
        )

        run_logger.info(
            f"output="
            f"{output_video_path}"
        )

        print(
            f"Canvas   : "
            f"{output_width}x"
            f"{output_height} "
            f"(video "
            f"{output_video_width}x"
            f"{output_height} "
            f"+ panel "
            f"{output_panel_width}px)"
        )

        print(
            f"Quality  : "
            f"H.264 CRF "
            f"{args.output_crf}"
        )

        print()

        hidden_classes = (
            {"conveyor_belt"}
            if args.hide_conveyor_belt
            else set()
        )

        processing_started = (
            time.perf_counter()
        )

        # ==========================================================
        # FRAME LOOP
        # ==========================================================

        while True:

            ok, frame = cap.read()

            if not ok:
                break

            frame_number += 1

            result = (
                pipeline.process_frame(
                    frame,
                    frame_number,
                )
            )

            # ------------------------------------------------------
            # LOG EVENTS
            # ------------------------------------------------------

            for event in result.events:

                run_logger.info(
                    f"EVENT | "
                    f"frame={frame_number} | "
                    f"type="
                    f"{event.event_type.value} | "
                    f"box="
                    f"{event.box_code} | "
                    f"from="
                    f"{event.from_pallet_code or '-'} | "
                    f"to="
                    f"{event.to_pallet_code or '-'} | "
                    f"zone="
                    f"{event.to_zone_code or '-'} | "
                    f"{event.remarks}"
                )

            timestamp_sec = (
                (frame_number - 1)
                / source_fps
                if source_fps > 0
                else 0.0
            )

            # ------------------------------------------------------
            # STORE RAW / STABILIZED VIDEO EVENTS
            # ------------------------------------------------------

            if (
                repository is not None
                and run_id is not None
            ):

                repository.persist_events(
                    run_id=run_id,
                    processed_index=(
                        frame_number
                    ),
                    timestamp_sec=(
                        timestamp_sec
                    ),
                    events=(
                        result.events
                    ),
                )

            elapsed = (
                time.perf_counter()
                - processing_started
            )

            reporter.update(
                frame_number,
                result.observations,
                result.qrs,
                result.events,
                elapsed,
            )

            output_panel.update(
                frame_number,
                result.display_observations,
                result.qrs,
                result.events,
                pipeline.engine,
            )

            annotated_frame = (
                frame.copy()
            )

            draw(
                annotated_frame,
                result.display_observations,
                result.qrs,
                result.events,
                hidden_classes,
            )

            if (
                annotated_frame.shape[1]
                != output_video_width
                or
                annotated_frame.shape[0]
                != output_height
            ):

                annotated_frame = (
                    cv2.resize(
                        annotated_frame,
                        (
                            output_video_width,
                            output_height,
                        ),
                        interpolation=(
                            cv2.INTER_AREA
                        ),
                    )
                )

            output_frame = (
                output_panel.compose(
                    annotated_frame,
                    total_frames,
                )
            )

            video_writer.write(
                output_frame
            )

            # ------------------------------------------------------
            # DISPLAY
            # ------------------------------------------------------

            if not args.no_display:

                display_frame = (
                    fit_display(
                        output_frame.copy(),
                        max_width=(
                            max_display_width
                        ),
                        max_height=(
                            max_display_height
                        ),
                    )
                )

                if frame_number == 1:

                    cv2.resizeWindow(
                        window_name,
                        display_frame.shape[1],
                        display_frame.shape[0],
                    )

                cv2.imshow(
                    window_name,
                    display_frame,
                )

                if cv2.waitKey(1) == 27:
                    break

            # ------------------------------------------------------
            # OPTIONAL FRAME LIMIT
            # ------------------------------------------------------

            if (
                args.max_frames is not None
                and frame_number
                >= args.max_frames
            ):
                break

        # ==========================================================
        # END OF VIDEO
        # ==========================================================

        if frame_number == 0:

            raise RuntimeError(
                f"No frames could be "
                f"read from: "
                f"{video_source}"
            )

        video_writer.release()

        video_writer = None

        output_size = (
            output_video_path
            .stat()
            .st_size
        )

        run_logger.info(
            f"OUTPUT SAVED | "
            f"path="
            f"{output_video_path.resolve()} | "
            f"bytes="
            f"{output_size}"
        )

        processing_elapsed = max(
            time.perf_counter()
            - processing_started,
            0.001,
        )

        end_to_end_elapsed = max(
            time.perf_counter()
            - application_started,
            0.001,
        )

        reporter.finish(
            frame_number,
            processing_elapsed,
        )

        run_logger.info(
            f"RUN COMPLETED | "
            f"frames={frame_number} | "
            f"average_fps="
            f"{frame_number / processing_elapsed:.2f} | "
            f"boxes="
            f"{len(reporter.seen_boxes)} | "
            f"pallets="
            f"{len(reporter.seen_pallets)} | "
            f"qrs="
            f"{len(reporter.seen_qrs)} | "
            f"events="
            f"{reporter.event_count}"
        )

        # ==========================================================
        # COMPLETE DATABASE RUN
        # ==========================================================

        if (
            repository is not None
            and run_id is not None
        ):

            processed_duration_sec = (
                frame_number / source_fps
                if source_fps > 0
                else None
            )

            repository.complete_run(
                run_id=run_id,
                boxes=(
                    pipeline.engine.boxes
                ),
                pallets=(
                    pipeline.engine.pallets
                ),
                processed_frames=(
                    frame_number
                ),
                source_frames_covered=(
                    frame_number
                ),
                video_duration_sec=(
                    processed_duration_sec
                ),
                total_processing_time_sec=(
                    processing_elapsed
                ),
                end_to_end_time_sec=(
                    end_to_end_elapsed
                ),
                metrics={
                    "average_processing_fps": (
                        frame_number
                        / processing_elapsed
                    ),
                    "unique_boxes_seen": (
                        len(
                            reporter.seen_boxes
                        )
                    ),
                    "unique_pallets_seen": (
                        len(
                            reporter.seen_pallets
                        )
                    ),
                    "unique_qrs_seen": (
                        len(
                            reporter.seen_qrs
                        )
                    ),
                    "pending_identity_event_count": (
                        len(
                            pipeline.engine.pending_events
                        )
                    ),
                    "dropped_unresolved_event_count": (
                        pipeline.engine
                        .dropped_unresolved_event_count
                    ),
                    "source_total_frames": (
                        source_total_frames
                    ),
                    "expected_processed_frames": (
                        total_frames
                    ),
                    "requested_max_frames": (
                        args.max_frames
                    ),
                    "ended_before_expected_frames": (
                        total_frames > 0
                        and frame_number
                        < total_frames
                    ),
                },
            )

            database_run_completed = True

            print(
                f"Database run "
                f"{run_id}: COMPLETED"
            )

            # ======================================================
            # EVENT-PROCESSOR HANDOFF
            # ======================================================
            # complete_run() intentionally remains responsible for final CV
            # snapshots, metrics, and run completion. Operational lifecycle
            # promotion is owned exclusively by the separate event processor,
            # preventing cv_sync from overwriting a later DISPATCHED state with
            # a final-snapshot ON_PALLET state.

            print(
                "CV run complete. NEW video_events are ready "
                "for event_processor.py."
            )

            run_logger.info(
                "EVENT PROCESSOR HANDOFF | "
                f"run_id={run_id} | lifecycle promotion deferred"
            )

        # ==========================================================
        # FINAL OUTPUT
        # ==========================================================

        print(
            f"Annotated video: "
            f"{output_video_path.resolve()}"
        )

        print(
            f"Run log        : "
            f"{run_logger.path.resolve()}"
        )

    # =================================================================
    # FAILURE HANDLING
    # =================================================================

    except Exception as exc:

        run_logger.exception(
            f"RUN FAILED | {exc}"
        )

        failure_elapsed = max(
            time.perf_counter()
            - (
                processing_started
                if processing_started
                is not None
                else application_started
            ),
            0.001,
        )

        # Only mark the database run FAILED if
        # complete_run() has NOT already succeeded.

        if (
            repository is not None
            and run_id is not None
            and not database_run_completed
        ):

            try:

                repository.fail_run(
                    run_id=run_id,
                    error_message=str(exc),
                    processed_frames=(
                        frame_number
                    ),
                    source_frames_covered=(
                        frame_number
                    ),
                    elapsed_sec=(
                        failure_elapsed
                    ),
                )

                print(
                    f"Database run "
                    f"{run_id}: FAILED"
                )

            except Exception as database_error:

                print(
                    f"Could not mark database "
                    f"run {run_id} as FAILED: "
                    f"{database_error}"
                )

        raise

    # =================================================================
    # CLEANUP
    # =================================================================

    finally:

        cap.release()

        if video_writer is not None:

            video_writer.release()

        cv2.destroyAllWindows()

        run_logger.close()


# =====================================================================
# ENTRY POINT
# =====================================================================


if __name__ == "__main__":
    main()
