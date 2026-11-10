from __future__ import annotations

import argparse
import os
import math
import re
import time
from collections import defaultdict, deque
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Optional

import cv2
import numpy as np
import pandas as pd
import torch
from ultralytics import YOLO

from db_manager import DatabaseConfig, DatabaseManager


# ============================================================
# PATHS
# ============================================================

PROJECT_DIR = Path(__file__).resolve().parent

try:
    from dotenv import load_dotenv
    load_dotenv(PROJECT_DIR / ".env")
except Exception:
    # The project can still run if python-dotenv is not installed,
    # but DB credentials then must be passed by command-line arguments.
    pass


def project_path_from_env(key: str, default: str) -> Path:
    value = os.getenv(key, default)
    path = Path(value)
    return path if path.is_absolute() else PROJECT_DIR / path


MODEL_PATH = project_path_from_env("MODEL_PATH", "models/dynamic_pallet_best.pt")
INPUT_VIDEO_DIR = project_path_from_env("INPUT_VIDEO_DIR", "data/input_videos")

DATA_DIR = project_path_from_env("MASTER_DATA_DIR", "data/master")
OUTPUT_DATA_DIR = project_path_from_env("OUTPUT_DATA_DIR", "outputs/csv")
OUTPUT_VIDEO_DIR = project_path_from_env("OUTPUT_VIDEO_DIR", "outputs/videos")

BOX_MASTER_CSV = DATA_DIR / "box_master.csv"
PALLET_MASTER_CSV = DATA_DIR / "pallet_master.csv"



# ============================================================
# PERFORMANCE PROFILE
# ============================================================

# High-speed profile for an RTX 3050 Laptop GPU with 4 GB VRAM.
# Accuracy fallback: change PROCESS_WIDTH to 832, FRAME_SKIP to 2,
# and YOLO_IMGSZ to 768 only when small QR detections are weak.
PROCESS_WIDTH = 768
YOLO_IMGSZ = 640
FRAME_SKIP = 3
YOLO_CONF = 0.18
YOLO_IOU = 0.50
MAX_DETECTIONS = 40
TRACKER = "bytetrack.yaml"

CLASS_CONFIDENCE = {
    "pallet": 0.25,
    "box": 0.25,
    "qr_code": 0.15,
}

# Decode only unidentified QR tracks, then cache their decoded identity.
QR_SCAN_EVERY = 6
QR_CACHE_MAX_AGE = 300

# Reuse dashboard graphics to reduce repeated drawing work.
DASHBOARD_EVERY = 5
PRINT_PROGRESS_EVERY = 25

DISPLAY_HEIGHT = 640
DASHBOARD_WIDTH = 420

SHOW_WINDOW = True
SAVE_OUTPUT_VIDEO = False
DRAW_TRAILS = False
WARMUP_MODEL = True

DEVICE = 0 if torch.cuda.is_available() else "cpu"
USE_HALF = torch.cuda.is_available()

# Confirmation values are in PROCESSED frames, not source-video frames.
PALLET_ADD_CONFIRM = 3
PALLET_REMOVE_CONFIRM = 15

BOX_PLACE_CONFIRM = 3
BOX_MOVE_CONFIRM = 4
BOX_REMOVE_CONFIRM = 6
BOX_COMPLETE_REMOVE_CONFIRM = 30

SWAP_WINDOW_PROCESSED_FRAMES = 40

GRID_ROWS = 2
GRID_COLS = 3
MAX_LAYERS_PER_SLOT = 3
PALLET_MARGIN = 14

WARNING_PERCENT = 80.0
CRITICAL_PERCENT = 95.0


# ============================================================
# DEFAULT MASTER DATA
# ============================================================

DEFAULT_BOX_LENGTH_CM = 10.0
DEFAULT_BOX_WIDTH_CM = 8.0
DEFAULT_BOX_HEIGHT_CM = 5.0
DEFAULT_BOX_WEIGHT_KG = 1.0

DEFAULT_MAX_WEIGHT_KG = 50.0
DEFAULT_MAX_HEIGHT_CM = 60.0
DEFAULT_MAX_VOLUME_CM3 = 150000.0
DEFAULT_MAX_BOXES = GRID_ROWS * GRID_COLS * MAX_LAYERS_PER_SLOT


# ============================================================
# DISPLAY COLOURS
# ============================================================

SAFE = (45, 190, 90)
WARNING = (0, 190, 255)
DANGER = (30, 40, 235)
TEXT = (235, 238, 245)
PANEL = (23, 26, 33)
CARD = (38, 42, 51)
BOX_COLOUR = (255, 170, 40)
QR_COLOUR = (255, 255, 0)
UNKNOWN = (160, 160, 160)


# ============================================================
# STATE
# ============================================================

@dataclass
class PalletState:
    pallet_id: str
    bbox: Optional[tuple[int, int, int, int]] = None
    track_id: Optional[int] = None
    active: bool = False
    present_count: int = 0
    missing_count: int = 0
    last_seen_processed: int = 0


@dataclass
class BoxState:
    box_id: str
    bbox: Optional[tuple[int, int, int, int]] = None
    track_id: Optional[int] = None

    current_pallet: Optional[str] = None
    candidate_pallet: Optional[str] = None

    candidate_count: int = 0
    outside_count: int = 0
    missing_count: int = 0

    slot_no: Optional[int] = None
    layer_no: Optional[int] = None

    status: str = "UNASSIGNED"
    assigned_processed: int = 0
    ever_assigned: bool = False
    last_seen_processed: int = 0
    recommended_pallet: Optional[str] = None


# ============================================================
# HELPERS
# ============================================================

def canonical_class_name(name: str) -> str:
    value = name.strip().lower().replace("-", "_").replace(" ", "_")
    aliases = {
        "boxes": "box",
        "pallets": "pallet",
        "qr": "qr_code",
        "qrcode": "qr_code",
        "qr_codes": "qr_code",
    }
    return aliases.get(value, value)


def normalize_qr_text(text: str) -> Optional[str]:
    if not text:
        return None

    value = text.strip().upper().replace("-", "_").replace(" ", "_")

    pallet_match = re.search(r"PALLET_?0*(\d+)", value)
    if pallet_match:
        return f"PALLET_{int(pallet_match.group(1)):02d}"

    box_match = re.search(r"BOX_?0*(\d+)", value)
    if box_match:
        return f"BOX_{int(box_match.group(1)):03d}"

    return None


def bbox_center(bbox: tuple[int, int, int, int]) -> tuple[int, int]:
    x1, y1, x2, y2 = bbox
    return (x1 + x2) // 2, (y1 + y2) // 2


def bbox_area(bbox: tuple[int, int, int, int]) -> int:
    x1, y1, x2, y2 = bbox
    return max(0, x2 - x1) * max(0, y2 - y1)


def intersection_area(
    first: tuple[int, int, int, int],
    second: tuple[int, int, int, int],
) -> int:
    ax1, ay1, ax2, ay2 = first
    bx1, by1, bx2, by2 = second

    x1 = max(ax1, bx1)
    y1 = max(ay1, by1)
    x2 = min(ax2, bx2)
    y2 = min(ay2, by2)

    return max(0, x2 - x1) * max(0, y2 - y1)


def point_inside(
    point: tuple[int, int],
    bbox: tuple[int, int, int, int],
    margin: int = 0,
) -> bool:
    px, py = point
    x1, y1, x2, y2 = bbox
    return (
        x1 - margin <= px <= x2 + margin
        and y1 - margin <= py <= y2 + margin
    )


def smooth_bbox(
    old_bbox: Optional[tuple[int, int, int, int]],
    new_bbox: tuple[int, int, int, int],
    old_weight: float = 0.55,
) -> tuple[int, int, int, int]:
    if old_bbox is None:
        return new_bbox

    return tuple(
        int(old_weight * old + (1.0 - old_weight) * new)
        for old, new in zip(old_bbox, new_bbox)
    )


def format_video_time(seconds: float) -> str:
    seconds = max(0, int(seconds))
    return f"{seconds // 60:02d}:{seconds % 60:02d}"


def fit_display(frame: np.ndarray, max_width: int = 1500, max_height: int = 850) -> np.ndarray:
    height, width = frame.shape[:2]
    scale = min(max_width / width, max_height / height, 1.0)

    if scale == 1.0:
        return frame

    return cv2.resize(
        frame,
        (int(width * scale), int(height * scale)),
        interpolation=cv2.INTER_AREA,
    )


def risk_colour(status: str) -> tuple[int, int, int]:
    if status == "SAFE":
        return SAFE
    if status in {"WARNING", "CRITICAL"}:
        return WARNING
    return DANGER


# ============================================================
# MASTER FILES
# ============================================================

def ensure_project_files() -> None:
    for directory in (
        DATA_DIR,
        OUTPUT_DATA_DIR,
        OUTPUT_VIDEO_DIR,
        MODEL_PATH.parent,
        INPUT_VIDEO_DIR,
    ):
        directory.mkdir(parents=True, exist_ok=True)

    if not BOX_MASTER_CSV.exists():
        rows = [
            {
                "box_id": f"BOX_{number:03d}",
                "length_cm": DEFAULT_BOX_LENGTH_CM,
                "width_cm": DEFAULT_BOX_WIDTH_CM,
                "height_cm": DEFAULT_BOX_HEIGHT_CM,
                "weight_kg": DEFAULT_BOX_WEIGHT_KG,
            }
            for number in range(1, 31)
        ]
        pd.DataFrame(rows).to_csv(BOX_MASTER_CSV, index=False)

    if not PALLET_MASTER_CSV.exists():
        rows = [
            {
                "pallet_id": f"PALLET_{number:02d}",
                "max_weight_kg": DEFAULT_MAX_WEIGHT_KG,
                "max_height_cm": DEFAULT_MAX_HEIGHT_CM,
                "max_volume_cm3": DEFAULT_MAX_VOLUME_CM3,
                "max_boxes": DEFAULT_MAX_BOXES,
            }
            for number in range(1, 11)
        ]
        pd.DataFrame(rows).to_csv(PALLET_MASTER_CSV, index=False)


def load_box_master() -> dict[str, dict[str, float]]:
    data: dict[str, dict[str, float]] = {}

    for _, row in pd.read_csv(BOX_MASTER_CSV).iterrows():
        box_id = normalize_qr_text(str(row["box_id"]))
        if not box_id or not box_id.startswith("BOX_"):
            continue

        length = float(row["length_cm"])
        width = float(row["width_cm"])
        height = float(row["height_cm"])

        data[box_id] = {
            "length_cm": length,
            "width_cm": width,
            "height_cm": height,
            "weight_kg": float(row["weight_kg"]),
            "volume_cm3": length * width * height,
        }

    return data


def load_pallet_master() -> dict[str, dict[str, float]]:
    data: dict[str, dict[str, float]] = {}

    for _, row in pd.read_csv(PALLET_MASTER_CSV).iterrows():
        pallet_id = normalize_qr_text(str(row["pallet_id"]))
        if not pallet_id or not pallet_id.startswith("PALLET_"):
            continue

        data[pallet_id] = {
            "max_weight_kg": float(row["max_weight_kg"]),
            "max_height_cm": float(row["max_height_cm"]),
            "max_volume_cm3": float(row["max_volume_cm3"]),
            "max_boxes": int(row["max_boxes"]),
        }

    return data


# ============================================================
# FAST DYNAMIC PALLET ENGINE
# ============================================================

class FastDynamicPalletSystem:
    def __init__(
        self,
        model: YOLO,
        box_master: dict[str, dict[str, float]],
        pallet_master: dict[str, dict[str, float]],
        source_fps: float,
    ) -> None:
        self.model = model
        self.box_master = box_master
        self.pallet_master = pallet_master
        self.source_fps = source_fps

        self.qr_detector = cv2.QRCodeDetector()

        self.pallets: dict[str, PalletState] = {}
        self.boxes: dict[str, BoxState] = {}

        self.pallet_track_to_id: dict[int, str] = {}
        self.box_track_to_id: dict[int, str] = {}

        # qr_track_id -> {"text": str, "last_seen": int}
        self.qr_cache: dict[int, dict[str, Any]] = {}

        self.events: list[dict[str, Any]] = []
        self.recent_events: deque[dict[str, Any]] = deque(maxlen=5)
        self.recent_moves: deque[dict[str, Any]] = deque(maxlen=20)
        self.logged_swaps: set[tuple[str, str, str, str]] = set()

        self.latest_banner: Optional[dict[str, Any]] = None

        self.trails: dict[str, deque[tuple[int, int]]] = defaultdict(
            lambda: deque(maxlen=18)
        )

    # --------------------------------------------------------
    # Metadata
    # --------------------------------------------------------

    def box_metadata(self, box_id: str) -> dict[str, float]:
        return self.box_master.get(
            box_id,
            {
                "length_cm": DEFAULT_BOX_LENGTH_CM,
                "width_cm": DEFAULT_BOX_WIDTH_CM,
                "height_cm": DEFAULT_BOX_HEIGHT_CM,
                "weight_kg": DEFAULT_BOX_WEIGHT_KG,
                "volume_cm3": (
                    DEFAULT_BOX_LENGTH_CM
                    * DEFAULT_BOX_WIDTH_CM
                    * DEFAULT_BOX_HEIGHT_CM
                ),
            },
        )

    def pallet_capacity(self, pallet_id: str) -> dict[str, float]:
        return self.pallet_master.get(
            pallet_id,
            {
                "max_weight_kg": DEFAULT_MAX_WEIGHT_KG,
                "max_height_cm": DEFAULT_MAX_HEIGHT_CM,
                "max_volume_cm3": DEFAULT_MAX_VOLUME_CM3,
                "max_boxes": DEFAULT_MAX_BOXES,
            },
        )

    # --------------------------------------------------------
    # Events
    # --------------------------------------------------------

    def log_event(
        self,
        processed_index: int,
        event_type: str,
        description: str,
        box_id: str = "",
        pallet_id: str = "",
        from_pallet: str = "",
        to_pallet: str = "",
        severity: str = "INFO",
    ) -> None:
        video_seconds = processed_index * FRAME_SKIP / self.source_fps

        event = {
            "processed_index": processed_index,
            "video_time": format_video_time(video_seconds),
            "timestamp_sec": round(video_seconds, 3),
            "event_type": event_type,
            "severity": severity,
            "box_id": box_id,
            "pallet_id": pallet_id,
            "from_pallet": from_pallet,
            "to_pallet": to_pallet,
            "description": description,
            "created_at": datetime.now().isoformat(timespec="seconds"),
        }

        self.events.append(event)
        self.recent_events.appendleft(event)
        self.latest_banner = {
            **event,
            "expires": time.perf_counter() + 2.5,
        }

        print(f"[{event['video_time']}] {event_type}: {description}")

    # --------------------------------------------------------
    # YOLO parsing
    # --------------------------------------------------------

    def parse_result(self, result: Any) -> dict[str, list[dict[str, Any]]]:
        grouped = {"pallet": [], "box": [], "qr_code": []}

        if result.boxes is None or len(result.boxes) == 0:
            return grouped

        xyxy = result.boxes.xyxy.detach().cpu().numpy()
        cls = result.boxes.cls.detach().cpu().numpy().astype(int)
        conf = result.boxes.conf.detach().cpu().numpy()

        if result.boxes.id is not None:
            track_ids = result.boxes.id.detach().cpu().numpy().astype(int)
        else:
            track_ids = np.full(len(xyxy), -1, dtype=int)

        for bbox_values, class_id, confidence, track_id in zip(
            xyxy, cls, conf, track_ids
        ):
            class_name = canonical_class_name(str(self.model.names[int(class_id)]))
            if class_name not in grouped:
                continue

            if float(confidence) < CLASS_CONFIDENCE.get(class_name, YOLO_CONF):
                continue

            x1, y1, x2, y2 = [int(round(value)) for value in bbox_values]

            grouped[class_name].append(
                {
                    "bbox": (x1, y1, x2, y2),
                    "confidence": float(confidence),
                    "track_id": None if int(track_id) < 0 else int(track_id),
                    "resolved_id": None,
                }
            )

        return grouped

    # --------------------------------------------------------
    # Fast QR decoding
    # --------------------------------------------------------

    def decode_qr_from_original(
        self,
        original_frame: np.ndarray,
        process_bbox: tuple[int, int, int, int],
        scale_x: float,
        scale_y: float,
    ) -> Optional[str]:
        frame_height, frame_width = original_frame.shape[:2]
        x1, y1, x2, y2 = process_bbox

        ox1 = int(x1 * scale_x)
        oy1 = int(y1 * scale_y)
        ox2 = int(x2 * scale_x)
        oy2 = int(y2 * scale_y)

        width = max(1, ox2 - ox1)
        height = max(1, oy2 - oy1)

        pad_x = int(width * 0.18)
        pad_y = int(height * 0.18)

        ox1 = max(0, ox1 - pad_x)
        oy1 = max(0, oy1 - pad_y)
        ox2 = min(frame_width, ox2 + pad_x)
        oy2 = min(frame_height, oy2 + pad_y)

        crop = original_frame[oy1:oy2, ox1:ox2]
        if crop.size == 0:
            return None

        # Only a few variants. The old version used many rotations and
        # threshold variants for every QR on every frame, which was slow.
        variants: list[np.ndarray] = [crop]

        gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)

        if max(gray.shape[:2]) < 240:
            gray = cv2.resize(
                gray,
                None,
                fx=2.5,
                fy=2.5,
                interpolation=cv2.INTER_CUBIC,
            )

        variants.append(gray)

        _, binary = cv2.threshold(
            gray,
            0,
            255,
            cv2.THRESH_BINARY + cv2.THRESH_OTSU,
        )
        variants.append(binary)

        for variant in variants:
            decoded, _, _ = self.qr_detector.detectAndDecode(variant)
            normalized = normalize_qr_text(decoded)
            if normalized:
                return normalized

        return None

    @staticmethod
    def associate_qr(
        qr_bbox: tuple[int, int, int, int],
        objects: list[dict[str, Any]],
    ) -> Optional[dict[str, Any]]:
        if not objects:
            return None

        qr_center = bbox_center(qr_bbox)

        inside = [
            item
            for item in objects
            if point_inside(qr_center, item["bbox"], margin=10)
        ]

        if inside:
            return min(inside, key=lambda item: bbox_area(item["bbox"]))

        return min(
            objects,
            key=lambda item: math.dist(
                qr_center,
                bbox_center(item["bbox"]),
            ),
        )

    def resolve_identities(
        self,
        original_frame: np.ndarray,
        grouped: dict[str, list[dict[str, Any]]],
        processed_index: int,
        scale_x: float,
        scale_y: float,
    ) -> list[dict[str, Any]]:
        decoded_records: list[dict[str, Any]] = []
        should_scan_unknown = processed_index % QR_SCAN_EVERY == 0

        for qr_detection in grouped["qr_code"]:
            qr_track_id = qr_detection["track_id"]
            decoded_text: Optional[str] = None

            if qr_track_id is not None and qr_track_id in self.qr_cache:
                decoded_text = self.qr_cache[qr_track_id]["text"]
                self.qr_cache[qr_track_id]["last_seen"] = processed_index

            elif should_scan_unknown:
                decoded_text = self.decode_qr_from_original(
                    original_frame,
                    qr_detection["bbox"],
                    scale_x,
                    scale_y,
                )

                if decoded_text and qr_track_id is not None:
                    self.qr_cache[qr_track_id] = {
                        "text": decoded_text,
                        "last_seen": processed_index,
                    }

            if not decoded_text:
                continue

            decoded_records.append(
                {
                    "text": decoded_text,
                    "bbox": qr_detection["bbox"],
                }
            )

            if decoded_text.startswith("PALLET_"):
                parent = self.associate_qr(
                    qr_detection["bbox"],
                    grouped["pallet"],
                )
                if parent is not None:
                    parent["resolved_id"] = decoded_text
                    if parent["track_id"] is not None:
                        self.pallet_track_to_id[parent["track_id"]] = decoded_text

            elif decoded_text.startswith("BOX_"):
                parent = self.associate_qr(
                    qr_detection["bbox"],
                    grouped["box"],
                )
                if parent is not None:
                    parent["resolved_id"] = decoded_text
                    if parent["track_id"] is not None:
                        self.box_track_to_id[parent["track_id"]] = decoded_text

        for detection in grouped["pallet"]:
            track_id = detection["track_id"]
            if detection["resolved_id"] is None and track_id in self.pallet_track_to_id:
                detection["resolved_id"] = self.pallet_track_to_id[track_id]

        for detection in grouped["box"]:
            track_id = detection["track_id"]
            if detection["resolved_id"] is None and track_id in self.box_track_to_id:
                detection["resolved_id"] = self.box_track_to_id[track_id]

        stale_qr_tracks = [
            track_id
            for track_id, record in self.qr_cache.items()
            if processed_index - int(record["last_seen"]) > QR_CACHE_MAX_AGE
        ]
        for track_id in stale_qr_tracks:
            self.qr_cache.pop(track_id, None)

        return decoded_records

    # --------------------------------------------------------
    # Pallets
    # --------------------------------------------------------

    def update_pallets(
        self,
        pallet_detections: list[dict[str, Any]],
        processed_index: int,
    ) -> None:
        seen_ids: set[str] = set()

        for detection in pallet_detections:
            pallet_id = detection["resolved_id"]
            if not pallet_id:
                continue

            seen_ids.add(pallet_id)

            state = self.pallets.get(pallet_id)
            if state is None:
                state = PalletState(pallet_id=pallet_id)
                self.pallets[pallet_id] = state

            state.track_id = detection["track_id"]
            state.bbox = smooth_bbox(state.bbox, detection["bbox"], 0.65)
            state.present_count += 1
            state.missing_count = 0
            state.last_seen_processed = processed_index

            if not state.active and state.present_count >= PALLET_ADD_CONFIRM:
                state.active = True
                self.log_event(
                    processed_index,
                    "PALLET_ADDED",
                    f"{pallet_id} activated",
                    pallet_id=pallet_id,
                    severity="SUCCESS",
                )

        for pallet_id, state in self.pallets.items():
            if not state.active or pallet_id in seen_ids:
                continue

            state.missing_count += 1

            if state.missing_count >= PALLET_REMOVE_CONFIRM:
                state.active = False
                state.present_count = 0

                self.log_event(
                    processed_index,
                    "PALLET_REMOVED",
                    f"{pallet_id} removed",
                    pallet_id=pallet_id,
                    severity="WARNING",
                )

                for box_state in self.boxes.values():
                    if box_state.current_pallet == pallet_id:
                        box_state.current_pallet = None
                        box_state.slot_no = None
                        box_state.layer_no = None
                        box_state.status = "PALLET_REMOVED"

    # --------------------------------------------------------
    # Box-to-pallet association
    # --------------------------------------------------------

    def find_pallet_for_box(
        self,
        box_bbox: tuple[int, int, int, int],
    ) -> Optional[str]:
        box_center = bbox_center(box_bbox)
        box_area_value = max(1, bbox_area(box_bbox))

        best_pallet: Optional[str] = None
        best_score = 0.0

        for pallet_id, pallet_state in self.pallets.items():
            if not pallet_state.active or pallet_state.bbox is None:
                continue

            overlap_ratio = (
                intersection_area(box_bbox, pallet_state.bbox)
                / box_area_value
            )

            score = overlap_ratio

            if point_inside(
                box_center,
                pallet_state.bbox,
                margin=PALLET_MARGIN,
            ):
                score += 1.0

            if score > best_score:
                best_score = score
                best_pallet = pallet_id

        return best_pallet if best_score >= 0.15 else None

    def slot_number(
        self,
        pallet_id: str,
        box_bbox: tuple[int, int, int, int],
    ) -> Optional[int]:
        pallet = self.pallets.get(pallet_id)
        if pallet is None or pallet.bbox is None:
            return None

        px1, py1, px2, py2 = pallet.bbox
        cx, cy = bbox_center(box_bbox)

        relative_x = min(max((cx - px1) / max(1, px2 - px1), 0.0), 0.999)
        relative_y = min(max((cy - py1) / max(1, py2 - py1), 0.0), 0.999)

        column = int(relative_x * GRID_COLS)
        row = int(relative_y * GRID_ROWS)

        return row * GRID_COLS + column + 1

    def recompute_layers(self, pallet_id: Optional[str]) -> None:
        if not pallet_id:
            return

        slot_groups: dict[int, list[BoxState]] = defaultdict(list)

        for state in self.boxes.values():
            if state.current_pallet == pallet_id and state.slot_no is not None:
                slot_groups[state.slot_no].append(state)

        for slot_boxes in slot_groups.values():
            slot_boxes.sort(key=lambda state: state.assigned_processed)

            for layer, state in enumerate(slot_boxes, start=1):
                state.layer_no = layer

    # --------------------------------------------------------
    # Capacity
    # --------------------------------------------------------

    def boxes_on_pallet(self, pallet_id: str) -> list[BoxState]:
        return [
            state
            for state in self.boxes.values()
            if state.current_pallet == pallet_id
        ]

    def pallet_stats(self, pallet_id: str) -> dict[str, Any]:
        capacity = self.pallet_capacity(pallet_id)
        states = self.boxes_on_pallet(pallet_id)

        total_weight = 0.0
        total_volume = 0.0
        slot_heights: dict[int, float] = defaultdict(float)

        for state in states:
            metadata = self.box_metadata(state.box_id)

            total_weight += metadata["weight_kg"]
            total_volume += metadata["volume_cm3"]

            if state.slot_no is not None:
                slot_heights[state.slot_no] += metadata["height_cm"]

        max_stack_height = max(slot_heights.values(), default=0.0)

        max_weight = max(float(capacity["max_weight_kg"]), 0.001)
        max_volume = max(float(capacity["max_volume_cm3"]), 0.001)
        max_height = max(float(capacity["max_height_cm"]), 0.001)
        max_boxes = max(int(capacity["max_boxes"]), 1)

        weight_pct = total_weight / max_weight * 100.0
        volume_pct = total_volume / max_volume * 100.0
        height_pct = max_stack_height / max_height * 100.0
        count_pct = len(states) / max_boxes * 100.0

        bottleneck_pct = max(weight_pct, volume_pct, height_pct, count_pct)

        reasons: list[str] = []

        if weight_pct > 100:
            reasons.append("weight exceeded")
        if volume_pct > 100:
            reasons.append("volume exceeded")
        if height_pct > 100:
            reasons.append("height exceeded")
        if count_pct > 100:
            reasons.append("box count exceeded")

        if reasons:
            status = "OVERLOAD"
        elif bottleneck_pct >= CRITICAL_PERCENT:
            status = "CRITICAL"
        elif bottleneck_pct >= WARNING_PERCENT:
            status = "WARNING"
        else:
            status = "SAFE"

        return {
            "pallet_id": pallet_id,
            "box_count": len(states),
            "max_boxes": max_boxes,
            "weight_kg": total_weight,
            "max_weight_kg": max_weight,
            "volume_cm3": total_volume,
            "max_volume_cm3": max_volume,
            "stack_height_cm": max_stack_height,
            "max_height_cm": max_height,
            "weight_pct": weight_pct,
            "volume_pct": volume_pct,
            "height_pct": height_pct,
            "count_pct": count_pct,
            "space_efficiency_pct": volume_pct,
            "bottleneck_pct": bottleneck_pct,
            "status": status,
            "reasons": reasons,
        }

    def recommend_pallet(self, box_id: str) -> Optional[str]:
        metadata = self.box_metadata(box_id)

        best_pallet: Optional[str] = None
        best_projected_bottleneck = float("inf")

        for pallet_id, pallet_state in self.pallets.items():
            if not pallet_state.active:
                continue

            stats = self.pallet_stats(pallet_id)

            projected_weight = (
                stats["weight_kg"] + metadata["weight_kg"]
            ) / stats["max_weight_kg"] * 100.0

            projected_volume = (
                stats["volume_cm3"] + metadata["volume_cm3"]
            ) / stats["max_volume_cm3"] * 100.0

            projected_count = (
                stats["box_count"] + 1
            ) / stats["max_boxes"] * 100.0

            projected_bottleneck = max(
                projected_weight,
                projected_volume,
                projected_count,
            )

            if projected_bottleneck <= 100.0 and projected_bottleneck < best_projected_bottleneck:
                best_projected_bottleneck = projected_bottleneck
                best_pallet = pallet_id

        return best_pallet

    def capacity_alert_if_needed(
        self,
        pallet_id: str,
        processed_index: int,
    ) -> None:
        stats = self.pallet_stats(pallet_id)

        if stats["status"] == "OVERLOAD":
            description = (
                f"{pallet_id} overload: "
                + ", ".join(stats["reasons"])
            )
            self.log_event(
                processed_index,
                "CAPACITY_ALERT",
                description,
                pallet_id=pallet_id,
                severity="CRITICAL",
            )

    # --------------------------------------------------------
    # Box events
    # --------------------------------------------------------

    def detect_swap(
        self,
        box_id: str,
        from_pallet: str,
        to_pallet: str,
        processed_index: int,
    ) -> None:
        for previous in reversed(self.recent_moves):
            if processed_index - previous["processed_index"] > SWAP_WINDOW_PROCESSED_FRAMES:
                break

            reciprocal = (
                previous["from_pallet"] == to_pallet
                and previous["to_pallet"] == from_pallet
                and previous["box_id"] != box_id
            )

            if not reciprocal:
                continue

            key = tuple(
                sorted((box_id, previous["box_id"]))
                + sorted((from_pallet, to_pallet))
            )

            if key not in self.logged_swaps:
                self.logged_swaps.add(key)
                self.log_event(
                    processed_index,
                    "BOX_SWAP",
                    (
                        f"{box_id} and {previous['box_id']} swapped "
                        f"between {from_pallet} and {to_pallet}"
                    ),
                    severity="SUCCESS",
                )
            break

        self.recent_moves.append(
            {
                "box_id": box_id,
                "from_pallet": from_pallet,
                "to_pallet": to_pallet,
                "processed_index": processed_index,
            }
        )

    def assign_box(
        self,
        state: BoxState,
        pallet_id: str,
        processed_index: int,
    ) -> None:
        old_pallet = state.current_pallet
        previous_status = state.status

        state.current_pallet = pallet_id
        state.candidate_pallet = None
        state.candidate_count = 0
        state.outside_count = 0
        state.missing_count = 0
        state.assigned_processed = processed_index
        state.slot_no = (
            self.slot_number(pallet_id, state.bbox)
            if state.bbox is not None
            else None
        )

        if old_pallet is None:
            if state.ever_assigned or previous_status in {
                "REMOVED",
                "COMPLETE_REMOVED",
                "PALLET_REMOVED",
            }:
                event_type = "BOX_RE_ADDED"
                description = f"{state.box_id} re-added to {pallet_id}"
            else:
                event_type = "BOX_ADDED"
                description = f"{state.box_id} added to {pallet_id}"

            state.status = "ON_PALLET"
            state.ever_assigned = True

            self.log_event(
                processed_index,
                event_type,
                description,
                box_id=state.box_id,
                to_pallet=pallet_id,
                severity="SUCCESS",
            )

        elif old_pallet != pallet_id:
            state.status = "MOVED"

            self.log_event(
                processed_index,
                "BOX_MOVED",
                f"{state.box_id}: {old_pallet} -> {pallet_id}",
                box_id=state.box_id,
                from_pallet=old_pallet,
                to_pallet=pallet_id,
                severity="SUCCESS",
            )

            self.detect_swap(
                state.box_id,
                old_pallet,
                pallet_id,
                processed_index,
            )

        self.recompute_layers(old_pallet)
        self.recompute_layers(pallet_id)

        if state.layer_no is not None and state.layer_no > 1:
            self.log_event(
                processed_index,
                "BOX_STACKED",
                (
                    f"{state.box_id} stacked on {pallet_id}, "
                    f"slot {state.slot_no}, layer {state.layer_no}"
                ),
                box_id=state.box_id,
                pallet_id=pallet_id,
                severity="INFO",
            )

        self.capacity_alert_if_needed(pallet_id, processed_index)

    def remove_visible_box(
        self,
        state: BoxState,
        processed_index: int,
    ) -> None:
        old_pallet = state.current_pallet
        if old_pallet is None:
            return

        state.current_pallet = None
        state.slot_no = None
        state.layer_no = None
        state.status = "REMOVED"
        state.outside_count = 0
        state.candidate_count = 0
        state.candidate_pallet = None

        self.log_event(
            processed_index,
            "BOX_REMOVED",
            f"{state.box_id} removed from {old_pallet}",
            box_id=state.box_id,
            from_pallet=old_pallet,
            severity="WARNING",
        )

        self.recompute_layers(old_pallet)

    def complete_remove_missing_box(
        self,
        state: BoxState,
        processed_index: int,
    ) -> None:
        old_pallet = state.current_pallet
        if old_pallet is None:
            return

        state.current_pallet = None
        state.slot_no = None
        state.layer_no = None
        state.status = "COMPLETE_REMOVED"
        state.candidate_pallet = None
        state.candidate_count = 0

        self.log_event(
            processed_index,
            "BOX_COMPLETE_REMOVED",
            f"{state.box_id} completely removed from {old_pallet}",
            box_id=state.box_id,
            from_pallet=old_pallet,
            severity="WARNING",
        )

        self.recompute_layers(old_pallet)

    def update_boxes(
        self,
        box_detections: list[dict[str, Any]],
        processed_index: int,
    ) -> None:
        seen_ids: set[str] = set()

        for detection in box_detections:
            box_id = detection["resolved_id"]
            if not box_id:
                continue

            seen_ids.add(box_id)

            state = self.boxes.get(box_id)
            if state is None:
                state = BoxState(box_id=box_id)
                self.boxes[box_id] = state

            state.track_id = detection["track_id"]
            state.bbox = smooth_bbox(state.bbox, detection["bbox"], 0.45)
            state.last_seen_processed = processed_index
            state.missing_count = 0

            if DRAW_TRAILS and state.bbox is not None:
                self.trails[box_id].append(bbox_center(state.bbox))

            detected_pallet = self.find_pallet_for_box(detection["bbox"])

            if state.current_pallet is None:
                if detected_pallet is None:
                    state.status = (
                        state.status
                        if state.status in {
                            "REMOVED",
                            "COMPLETE_REMOVED",
                            "PALLET_REMOVED",
                        }
                        else "UNASSIGNED"
                    )
                    state.candidate_pallet = None
                    state.candidate_count = 0
                    state.recommended_pallet = self.recommend_pallet(box_id)
                    continue

                if state.candidate_pallet == detected_pallet:
                    state.candidate_count += 1
                else:
                    state.candidate_pallet = detected_pallet
                    state.candidate_count = 1

                state.status = "PLACING"

                if state.candidate_count >= BOX_PLACE_CONFIRM:
                    self.assign_box(state, detected_pallet, processed_index)

                continue

            if detected_pallet == state.current_pallet:
                state.status = "ON_PALLET"
                state.outside_count = 0
                state.candidate_count = 0
                state.candidate_pallet = None
                continue

            if detected_pallet is None:
                state.status = "MOVING"
                state.outside_count += 1

                if state.outside_count >= BOX_REMOVE_CONFIRM:
                    self.remove_visible_box(state, processed_index)

                continue

            state.outside_count = 0

            if state.candidate_pallet == detected_pallet:
                state.candidate_count += 1
            else:
                state.candidate_pallet = detected_pallet
                state.candidate_count = 1

            state.status = "MOVING"

            if state.candidate_count >= BOX_MOVE_CONFIRM:
                self.assign_box(state, detected_pallet, processed_index)

        for box_id, state in self.boxes.items():
            if box_id in seen_ids:
                continue

            state.missing_count += 1

            if state.current_pallet is not None:
                state.status = "TEMP_MISSING"

                if state.missing_count >= BOX_COMPLETE_REMOVE_CONFIRM:
                    self.complete_remove_missing_box(state, processed_index)

    # --------------------------------------------------------
    # Drawing
    # --------------------------------------------------------

    @staticmethod
    def put_text(
        image: np.ndarray,
        text: str,
        position: tuple[int, int],
        scale: float = 0.5,
        colour: tuple[int, int, int] = TEXT,
        thickness: int = 1,
    ) -> None:
        cv2.putText(
            image,
            text,
            position,
            cv2.FONT_HERSHEY_SIMPLEX,
            scale,
            colour,
            thickness,
            cv2.LINE_AA,
        )

    def draw_frame(
        self,
        frame: np.ndarray,
        grouped: dict[str, list[dict[str, Any]]],
        decoded_qrs: list[dict[str, Any]],
    ) -> np.ndarray:
        output = frame.copy()

        for detection in grouped["pallet"]:
            x1, y1, x2, y2 = detection["bbox"]
            pallet_id = detection["resolved_id"]

            if pallet_id:
                stats = self.pallet_stats(pallet_id)
                colour = risk_colour(stats["status"])
                label = (
                    f"{pallet_id} | Boxes {stats['box_count']} | "
                    f"Space {stats['space_efficiency_pct']:.1f}%"
                )
            else:
                colour = UNKNOWN
                label = "UNIDENTIFIED PALLET"

            cv2.rectangle(output, (x1, y1), (x2, y2), colour, 3)
            self.put_text(
                output,
                label,
                (x1, max(20, y1 - 7)),
                0.50,
                colour,
                2,
            )

        for detection in grouped["box"]:
            x1, y1, x2, y2 = detection["bbox"]
            box_id = detection["resolved_id"]

            cv2.rectangle(output, (x1, y1), (x2, y2), BOX_COLOUR, 2)

            if box_id and box_id in self.boxes:
                state = self.boxes[box_id]
                pallet_text = state.current_pallet or "OUTSIDE"

                label = f"{box_id} | {pallet_text} | {state.status}"

                if state.slot_no is not None:
                    label += f" | S{state.slot_no}/L{state.layer_no or 1}"

                if state.current_pallet is None and state.recommended_pallet:
                    label += f" | Suggest {state.recommended_pallet}"
            else:
                label = "UNKNOWN BOX"

            self.put_text(
                output,
                label,
                (x1, max(18, y1 - 5)),
                0.42,
                BOX_COLOUR,
                2,
            )

            if DRAW_TRAILS and box_id:
                trail = list(self.trails[box_id])
                for index in range(1, len(trail)):
                    cv2.line(
                        output,
                        trail[index - 1],
                        trail[index],
                        BOX_COLOUR,
                        2,
                    )

        for record in decoded_qrs:
            x1, y1, x2, y2 = record["bbox"]
            cv2.rectangle(output, (x1, y1), (x2, y2), QR_COLOUR, 2)
            self.put_text(
                output,
                record["text"],
                (x1, max(18, y1 - 4)),
                0.40,
                QR_COLOUR,
                2,
            )

        if (
            self.latest_banner is not None
            and time.perf_counter() <= self.latest_banner["expires"]
        ):
            severity = self.latest_banner["severity"]
            colour = (
                DANGER
                if severity == "CRITICAL"
                else WARNING
                if severity == "WARNING"
                else SAFE
            )

            cv2.rectangle(
                output,
                (12, 12),
                (min(output.shape[1] - 12, 760), 60),
                colour,
                -1,
            )

            self.put_text(
                output,
                self.latest_banner["description"],
                (25, 44),
                0.62,
                (20, 20, 20),
                2,
            )

        return output

    @staticmethod
    def progress_bar(
        panel: np.ndarray,
        x: int,
        y: int,
        width: int,
        percent: float,
        label: str,
        colour: tuple[int, int, int],
    ) -> None:
        cv2.putText(
            panel,
            f"{label} {percent:.1f}%",
            (x, y - 4),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.37,
            TEXT,
            1,
            cv2.LINE_AA,
        )

        cv2.rectangle(panel, (x, y), (x + width, y + 11), (75, 79, 90), -1)
        fill = int(width * min(max(percent, 0.0), 100.0) / 100.0)
        cv2.rectangle(panel, (x, y), (x + fill, y + 11), colour, -1)

    def build_dashboard(
        self,
        height: int,
        processed_index: int,
        processing_fps: float,
        elapsed_processing_sec: float,
        eta_sec: float,
    ) -> np.ndarray:
        panel = np.full((height, DASHBOARD_WIDTH, 3), PANEL, dtype=np.uint8)

        active = sorted(
            (
                state
                for state in self.pallets.values()
                if state.active
            ),
            key=lambda state: state.pallet_id,
        )

        assigned_boxes = sum(
            state.current_pallet is not None
            for state in self.boxes.values()
        )

        move_count = sum(
            event["event_type"] == "BOX_MOVED"
            for event in self.events
        )

        self.put_text(
            panel,
            "DYNAMIC PALLET DASHBOARD",
            (18, 30),
            0.64,
            TEXT,
            2,
        )

        video_seconds = processed_index * FRAME_SKIP / self.source_fps

        self.put_text(
            panel,
            (
                f"Video {format_video_time(video_seconds)} | "
                f"Speed {processing_fps:.1f} FPS"
            ),
            (18, 55),
            0.42,
            (190, 195, 208),
        )

        self.put_text(
            panel,
            (
                f"Elapsed {elapsed_processing_sec:.1f}s | "
                f"ETA {max(0.0, eta_sec):.1f}s"
            ),
            (18, 78),
            0.42,
            (190, 195, 208),
        )

        self.put_text(
            panel,
            (
                f"Pallets {len(active)} | Boxes {assigned_boxes} | "
                f"Moves {move_count}"
            ),
            (18, 101),
            0.42,
            (190, 195, 208),
        )

        y = 118
        card_height = 100

        for pallet_state in active[:4]:
            stats = self.pallet_stats(pallet_state.pallet_id)
            colour = risk_colour(stats["status"])

            cv2.rectangle(
                panel,
                (10, y),
                (DASHBOARD_WIDTH - 10, y + card_height),
                CARD,
                -1,
            )
            cv2.rectangle(
                panel,
                (10, y),
                (DASHBOARD_WIDTH - 10, y + card_height),
                colour,
                2,
            )

            self.put_text(
                panel,
                f"{pallet_state.pallet_id} [{stats['status']}]",
                (20, y + 22),
                0.49,
                colour,
                2,
            )

            self.put_text(
                panel,
                (
                    f"Boxes {stats['box_count']}/{stats['max_boxes']} | "
                    f"Weight {stats['weight_kg']:.1f}/"
                    f"{stats['max_weight_kg']:.1f} kg"
                ),
                (20, y + 42),
                0.35,
                TEXT,
            )

            self.progress_bar(
                panel,
                20,
                y + 58,
                175,
                stats["weight_pct"],
                "Weight",
                colour,
            )
            self.progress_bar(
                panel,
                220,
                y + 58,
                175,
                stats["volume_pct"],
                "Volume",
                colour,
            )
            self.progress_bar(
                panel,
                20,
                y + 86,
                175,
                stats["height_pct"],
                "Height",
                colour,
            )
            self.progress_bar(
                panel,
                220,
                y + 86,
                175,
                stats["space_efficiency_pct"],
                "Efficiency",
                colour,
            )

            y += card_height + 7

        event_y = max(y + 7, height - 112)

        self.put_text(panel, "LATEST EVENTS", (18, event_y), 0.44, TEXT, 2)

        line_y = event_y + 22

        for event in list(self.recent_events)[:3]:
            description = event["description"]
            if len(description) > 48:
                description = description[:45] + "..."

            self.put_text(
                panel,
                f"{event['video_time']} {description}",
                (18, line_y),
                0.33,
                (195, 199, 210),
            )
            line_y += 19

        self.put_text(
            panel,
            "SPACE Pause | D Dashboard | S Shot | Q Quit",
            (14, height - 10),
            0.31,
            (150, 155, 168),
        )

        return panel

    # --------------------------------------------------------
    # Reports
    # --------------------------------------------------------

    def save_reports(
        self,
        events_csv: Path,
        final_boxes_csv: Path,
        final_pallets_csv: Path,
    ) -> None:
        pd.DataFrame(self.events).to_csv(events_csv, index=False)

        box_rows: list[dict[str, Any]] = []

        for box_id, state in sorted(self.boxes.items()):
            metadata = self.box_metadata(box_id)

            box_rows.append(
                {
                    "box_id": box_id,
                    "status": state.status,
                    "pallet_id": state.current_pallet or "",
                    "slot_no": state.slot_no or "",
                    "layer_no": state.layer_no or "",
                    "length_cm": metadata["length_cm"],
                    "width_cm": metadata["width_cm"],
                    "height_cm": metadata["height_cm"],
                    "weight_kg": metadata["weight_kg"],
                    "volume_cm3": metadata["volume_cm3"],
                }
            )

        pd.DataFrame(box_rows).to_csv(final_boxes_csv, index=False)

        pallet_rows: list[dict[str, Any]] = []

        for pallet_id, state in sorted(self.pallets.items()):
            stats = self.pallet_stats(pallet_id)

            pallet_rows.append(
                {
                    "pallet_id": pallet_id,
                    "active": state.active,
                    "status": stats["status"],
                    "box_count": stats["box_count"],
                    "max_boxes": stats["max_boxes"],
                    "total_weight_kg": round(stats["weight_kg"], 3),
                    "max_weight_kg": stats["max_weight_kg"],
                    "weight_utilization_percent": round(stats["weight_pct"], 2),
                    "total_volume_cm3": round(stats["volume_cm3"], 3),
                    "max_volume_cm3": stats["max_volume_cm3"],
                    "volume_efficiency_percent": round(
                        stats["space_efficiency_pct"],
                        2,
                    ),
                    "max_stack_height_cm": round(stats["stack_height_cm"], 3),
                    "max_height_cm": stats["max_height_cm"],
                    "height_utilization_percent": round(stats["height_pct"], 2),
                    "bottleneck_utilization_percent": round(
                        stats["bottleneck_pct"],
                        2,
                    ),
                    "reasons": "; ".join(stats["reasons"]),
                }
            )

        pd.DataFrame(pallet_rows).to_csv(final_pallets_csv, index=False)


# ============================================================
# TERMINAL ARGUMENTS
# ============================================================

def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "High-speed dynamic pallet processing. "
            "Pass the input-video number, filename, or full path."
        ),
        formatter_class=argparse.RawTextHelpFormatter,
        epilog=(
            "Examples:\n"
            "  python main1.py 1\n"
            "  python main1.py input_video1\n"
            "  python main1.py input_video1.mp4\n"
            "  python main1.py 2 --no-display\n"
            "  python main1.py 3 --no-display --no-save\n"
        ),
    )

    parser.add_argument(
        "video",
        type=str,
        help=(
            "Video number, filename, or path.\n"
            "Examples: 1, input_video1, input_video1.mp4"
        ),
    )

    parser.add_argument(
        "--model",
        type=str,
        default=str(MODEL_PATH),
        help=(
            "Optional YOLO model path.\n"
            f"Default: {MODEL_PATH}"
        ),
    )

    parser.add_argument(
        "--input-dir",
        type=str,
        default=str(INPUT_VIDEO_DIR),
        help=(
            "Folder containing input videos.\n"
            f"Default: {INPUT_VIDEO_DIR}"
        ),
    )

    parser.add_argument(
        "--no-display",
        action="store_true",
        help="Do not open the live OpenCV processing window.",
    )

    parser.add_argument(
        "--no-save",
        action="store_true",
        help="Do not save the annotated output video.",
    )
    parser.add_argument(
        "--db",
        action="store_true",
        help="Store run results into the MySQL warehouse database.",
    )

    parser.add_argument(
        "--db-host",
        type=str,
        default=os.getenv("DB_HOST", "localhost"),
        help="MySQL host. Default: localhost or DB_HOST environment variable.",
    )

    parser.add_argument(
        "--db-port",
        type=int,
        default=int(os.getenv("DB_PORT", "3306")),
        help="MySQL port. Default: 3306 or DB_PORT environment variable.",
    )

    parser.add_argument(
        "--db-user",
        type=str,
        default=os.getenv("DB_USER", "root"),
        help="MySQL username. Default: root or DB_USER environment variable.",
    )

    parser.add_argument(
        "--db-password",
        type=str,
        default=os.getenv("DB_PASSWORD", ""),
        help="MySQL password. Default: DB_PASSWORD environment variable or empty.",
    )

    parser.add_argument(
        "--db-name",
        type=str,
        default=os.getenv("DB_NAME", "warehouse"),
        help="MySQL database name. Default: warehouse or DB_NAME environment variable.",
    )

    return parser.parse_args()


def resolve_input_video(
    video_argument: str,
    input_directory: Path,
) -> Path:
    value = video_argument.strip().strip('"').strip("'")

    if not value:
        raise ValueError("The input-video argument cannot be empty.")

    # Allows: python main1.py 1
    if value.isdigit():
        value = f"input_video{int(value)}.mp4"

    candidate = Path(value).expanduser()

    # Allows: python main1.py input_video1
    if candidate.suffix == "":
        candidate = candidate.with_suffix(".mp4")

    # Use an explicit existing relative or absolute path first.
    if candidate.exists() and candidate.is_file():
        return candidate.resolve()

    # Otherwise search inside dp_input_videos.
    folder_candidate = input_directory / candidate.name

    if folder_candidate.exists() and folder_candidate.is_file():
        return folder_candidate.resolve()

    available_files = []

    if input_directory.exists():
        available_files = sorted(
            file.name
            for file in input_directory.iterdir()
            if file.is_file()
            and file.suffix.lower() in {".mp4", ".avi", ".mov", ".mkv"}
        )

    available_text = (
        "\n".join(f"  - {name}" for name in available_files)
        if available_files
        else "  No supported videos were found."
    )

    raise FileNotFoundError(
        "\nInput video was not found.\n"
        f"Argument received : {video_argument}\n"
        f"Input directory   : {input_directory.resolve()}\n"
        f"Checked filename  : {candidate.name}\n\n"
        f"Available videos:\n{available_text}\n"
    )


def build_output_paths(video_stem: str) -> dict[str, Path]:
    safe_stem = re.sub(r"[^A-Za-z0-9_.-]+", "_", video_stem)

    return {
        "output_video": (
            OUTPUT_VIDEO_DIR
            / f"{safe_stem}_high_speed_output.mp4"
        ),
        "events_csv": (
            OUTPUT_DATA_DIR
            / f"{safe_stem}_all_events.csv"
        ),
        "final_boxes_csv": (
            OUTPUT_DATA_DIR
            / f"{safe_stem}_final_box_locations.csv"
        ),
        "final_pallets_csv": (
            OUTPUT_DATA_DIR
            / f"{safe_stem}_final_pallet_summary.csv"
        ),
        "performance_csv": (
            OUTPUT_DATA_DIR
            / f"{safe_stem}_processing_metrics.csv"
        ),
    }


# ============================================================
# MAIN
# ============================================================

def main() -> None:
    args = parse_arguments()

    ensure_project_files()

    input_directory = Path(args.input_dir).expanduser().resolve()
    input_directory.mkdir(parents=True, exist_ok=True)

    input_video = resolve_input_video(
        video_argument=args.video,
        input_directory=input_directory,
    )

    model_path = Path(args.model).expanduser().resolve()

    if not model_path.exists():
        raise FileNotFoundError(
            f"Model was not found:\n{model_path}"
        )

    show_window = SHOW_WINDOW and not args.no_display
    save_output_video = SAVE_OUTPUT_VIDEO and not args.no_save

    output_paths = build_output_paths(input_video.stem)

    output_video = output_paths["output_video"]
    events_csv = output_paths["events_csv"]
    final_boxes_csv = output_paths["final_boxes_csv"]
    final_pallets_csv = output_paths["final_pallets_csv"]
    performance_csv = output_paths["performance_csv"]

    cv2.setUseOptimized(True)
    cv2.setNumThreads(2)

    if torch.cuda.is_available():
        torch.backends.cudnn.benchmark = True
        torch.set_float32_matmul_precision("high")
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()

    model = YOLO(str(model_path))

    try:
        model.fuse()
    except Exception:
        pass

    model_classes = {
        canonical_class_name(str(name))
        for name in model.names.values()
    }

    required = {"pallet", "box", "qr_code"}
    missing = required - model_classes

    if missing:
        raise ValueError(
            f"Model is missing classes: {sorted(missing)}. "
            f"Available classes: {sorted(model_classes)}"
        )

    capture = cv2.VideoCapture(str(input_video))
    capture.set(cv2.CAP_PROP_BUFFERSIZE, 2)

    if not capture.isOpened():
        raise RuntimeError(f"Cannot open video: {input_video}")

    source_fps = capture.get(cv2.CAP_PROP_FPS)
    if source_fps <= 0:
        source_fps = 30.0

    source_width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
    source_height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
    video_duration_sec = total_frames / source_fps if source_fps > 0 else 0.0

    process_height = int(source_height * PROCESS_WIDTH / source_width)
    process_height = max(32, (process_height // 2) * 2)

    scale_x = source_width / PROCESS_WIDTH
    scale_y = source_height / process_height

    display_video_width = int(PROCESS_WIDTH * DISPLAY_HEIGHT / process_height)
    display_video_width = max(2, (display_video_width // 2) * 2)

    output_width = display_video_width + DASHBOARD_WIDTH
    output_fps = source_fps / FRAME_SKIP

    writer = None

    if save_output_video:
        writer = cv2.VideoWriter(
            str(output_video),
            cv2.VideoWriter_fourcc(*"mp4v"),
            output_fps,
            (output_width, DISPLAY_HEIGHT),
        )

        if not writer.isOpened():
            raise RuntimeError(f"Cannot create output video: {output_video}")

    system = FastDynamicPalletSystem(
        model=model,
        box_master=load_box_master(),
        pallet_master=load_pallet_master(),
        source_fps=source_fps,
    )

    print("=" * 76)
    print("HIGH-SPEED DYNAMIC PALLET PROCESSING")
    print("=" * 76)
    print(f"Model                : {model_path}")
    print(f"Input argument       : {args.video}")
    print(f"Input                : {input_video}")
    print(f"Device               : {DEVICE}")
    print(
        f"GPU                  : "
        f"{torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU'}"
    )
    print(f"Source resolution    : {source_width}x{source_height}")
    print(f"Video duration       : {video_duration_sec:.2f} seconds")
    print(f"Processing size      : {PROCESS_WIDTH}x{process_height}")
    print(f"YOLO image size      : {YOLO_IMGSZ}")
    print(f"Frame skip           : {FRAME_SKIP}")
    print(f"QR scan interval     : {QR_SCAN_EVERY}")
    print(f"Output FPS           : {output_fps:.2f}")
    print(f"Total source frames  : {total_frames}")
    print("=" * 76)

    warmup_time_sec = 0.0

    if WARMUP_MODEL:
        warmup_frame = np.zeros(
            (process_height, PROCESS_WIDTH, 3),
            dtype=np.uint8,
        )

        warmup_start = time.perf_counter()

        with torch.inference_mode():
            model.predict(
                source=warmup_frame,
                conf=YOLO_CONF,
                iou=YOLO_IOU,
                imgsz=YOLO_IMGSZ,
                device=DEVICE,
                half=USE_HALF,
                max_det=MAX_DETECTIONS,
                verbose=False,
            )

        if torch.cuda.is_available():
            torch.cuda.synchronize()

        warmup_time_sec = time.perf_counter() - warmup_start

        print(
            f"Model warm-up time   : "
            f"{warmup_time_sec:.2f} seconds"
        )
        print("=" * 76)

    source_frame_index = 0
    processed_index = 0
    dashboard_enabled = True
    end_of_video = False

    cached_dashboard: Optional[np.ndarray] = None

    fps_counter = 0
    fps_timer = time.perf_counter()
    live_processing_fps = 0.0

    timings: dict[str, float] = defaultdict(float)
    ultralytics_preprocess_ms = 0.0
    ultralytics_inference_ms = 0.0
    ultralytics_postprocess_ms = 0.0

    start_time = time.perf_counter()

    try:
        while not end_of_video:
            frame_cycle_start = time.perf_counter()

            read_start = time.perf_counter()
            success, original_frame = capture.read()
            timings["read_sec"] += time.perf_counter() - read_start

            if not success:
                break

            source_frame_index += 1
            processed_index += 1

            resize_start = time.perf_counter()
            process_frame = cv2.resize(
                original_frame,
                (PROCESS_WIDTH, process_height),
                interpolation=cv2.INTER_AREA,
            )
            timings["resize_sec"] += time.perf_counter() - resize_start

            yolo_start = time.perf_counter()

            with torch.inference_mode():
                result = model.track(
                    source=process_frame,
                    persist=True,
                    tracker=TRACKER,
                    conf=YOLO_CONF,
                    iou=YOLO_IOU,
                    imgsz=YOLO_IMGSZ,
                    device=DEVICE,
                    half=USE_HALF,
                    max_det=MAX_DETECTIONS,
                    verbose=False,
                )[0]

            timings["yolo_wall_sec"] += time.perf_counter() - yolo_start

            if isinstance(result.speed, dict):
                ultralytics_preprocess_ms += float(result.speed.get("preprocess", 0.0))
                ultralytics_inference_ms += float(result.speed.get("inference", 0.0))
                ultralytics_postprocess_ms += float(result.speed.get("postprocess", 0.0))

            parse_start = time.perf_counter()
            grouped = system.parse_result(result)
            timings["parse_sec"] += time.perf_counter() - parse_start

            qr_start = time.perf_counter()
            decoded_qrs = system.resolve_identities(
                original_frame=original_frame,
                grouped=grouped,
                processed_index=processed_index,
                scale_x=scale_x,
                scale_y=scale_y,
            )
            timings["qr_sec"] += time.perf_counter() - qr_start

            state_start = time.perf_counter()
            system.update_pallets(grouped["pallet"], processed_index)
            system.update_boxes(grouped["box"], processed_index)
            timings["state_sec"] += time.perf_counter() - state_start

            render_start = time.perf_counter()
            annotated = system.draw_frame(
                process_frame,
                grouped,
                decoded_qrs,
            )

            video_panel = cv2.resize(
                annotated,
                (display_video_width, DISPLAY_HEIGHT),
                interpolation=cv2.INTER_AREA,
            )

            fps_counter += 1
            elapsed_fps = time.perf_counter() - fps_timer

            if elapsed_fps >= 1.0:
                live_processing_fps = fps_counter / elapsed_fps
                fps_counter = 0
                fps_timer = time.perf_counter()

            elapsed_processing = time.perf_counter() - start_time
            progress_ratio = (
                source_frame_index / total_frames
                if total_frames > 0
                else 0.0
            )
            estimated_total = (
                elapsed_processing / progress_ratio
                if progress_ratio > 0
                else 0.0
            )
            eta_sec = max(0.0, estimated_total - elapsed_processing)

            if (
                cached_dashboard is None
                or processed_index % DASHBOARD_EVERY == 0
            ):
                cached_dashboard = system.build_dashboard(
                    DISPLAY_HEIGHT,
                    processed_index,
                    live_processing_fps,
                    elapsed_processing,
                    eta_sec,
                )

            if dashboard_enabled and cached_dashboard is not None:
                combined = np.hstack((video_panel, cached_dashboard))
            else:
                combined = np.full(
                    (DISPLAY_HEIGHT, output_width, 3),
                    PANEL,
                    dtype=np.uint8,
                )
                combined[:, :display_video_width] = video_panel

            timings["render_sec"] += time.perf_counter() - render_start

            output_start = time.perf_counter()

            if writer is not None:
                writer.write(combined)

            if show_window:
                cv2.imshow(
                    "High-Speed Dynamic Pallet Optimization",
                    fit_display(combined),
                )

                key = cv2.waitKey(1) & 0xFF

                if key in (ord("q"), 27):
                    break

                if key == ord("d"):
                    dashboard_enabled = not dashboard_enabled

                if key == ord("s"):
                    screenshot_path = (
                        OUTPUT_VIDEO_DIR
                        / f"{input_video.stem}_screenshot_{processed_index:06d}.jpg"
                    )
                    cv2.imwrite(str(screenshot_path), combined)
                    print(f"\nScreenshot saved: {screenshot_path}")

                if key == 32:
                    while True:
                        pause_key = cv2.waitKey(100) & 0xFF
                        if pause_key == 32:
                            break
                        if pause_key in (ord("q"), 27):
                            raise KeyboardInterrupt

            timings["output_sec"] += time.perf_counter() - output_start
            timings["frame_cycle_sec"] += time.perf_counter() - frame_cycle_start

            # Skip compressed source frames with grab(), avoiding full decoding.
            skip_start = time.perf_counter()
            for _ in range(FRAME_SKIP - 1):
                if not capture.grab():
                    end_of_video = True
                    break
                source_frame_index += 1
            timings["skip_grab_sec"] += time.perf_counter() - skip_start

            if (
                processed_index % PRINT_PROGRESS_EVERY == 0
                or end_of_video
            ):
                percent = (
                    min(100.0, source_frame_index / total_frames * 100.0)
                    if total_frames > 0
                    else 0.0
                )
                print(
                    f"\rProgress {percent:6.2f}% | "
                    f"processed {processed_index} | "
                    f"elapsed {elapsed_processing:7.2f}s | "
                    f"ETA {eta_sec:7.2f}s | "
                    f"speed {live_processing_fps:5.1f} FPS",
                    end="",
                    flush=True,
                )

    except KeyboardInterrupt:
        print("\nProcessing stopped by user.")

    finally:
        capture.release()

        if writer is not None:
            writer.release()

        cv2.destroyAllWindows()

    if torch.cuda.is_available():
        torch.cuda.synchronize()

    elapsed = time.perf_counter() - start_time
    print()

    system.save_reports(
        events_csv=events_csv,
        final_boxes_csv=final_boxes_csv,
        final_pallets_csv=final_pallets_csv,
    )

    covered_video_duration_sec = (
        source_frame_index / source_fps
        if source_fps > 0
        else 0.0
    )

    processed_fps = processed_index / elapsed if elapsed > 0 else 0.0
    effective_source_fps = source_frame_index / elapsed if elapsed > 0 else 0.0
    realtime_factor = (
        elapsed / covered_video_duration_sec
        if covered_video_duration_sec > 0
        else 0.0
    )
    speed_vs_realtime = (
        covered_video_duration_sec / elapsed
        if elapsed > 0
        else 0.0
    )

    avg_yolo_wall_ms = (
        timings["yolo_wall_sec"] / processed_index * 1000.0
        if processed_index > 0
        else 0.0
    )
    avg_qr_ms = (
        timings["qr_sec"] / processed_index * 1000.0
        if processed_index > 0
        else 0.0
    )
    avg_render_ms = (
        timings["render_sec"] / processed_index * 1000.0
        if processed_index > 0
        else 0.0
    )
    avg_output_ms = (
        timings["output_sec"] / processed_index * 1000.0
        if processed_index > 0
        else 0.0
    )

    peak_gpu_mb = (
        torch.cuda.max_memory_allocated(0) / (1024 ** 2)
        if torch.cuda.is_available()
        else 0.0
    )

    metrics = {
        "input_video": str(input_video),
        "model": str(model_path),
        "device": str(DEVICE),
        "gpu_name": (
            torch.cuda.get_device_name(0)
            if torch.cuda.is_available()
            else "CPU"
        ),
        "source_width": source_width,
        "source_height": source_height,
        "source_fps": round(source_fps, 3),
        "video_duration_sec": round(video_duration_sec, 3),
        "model_warmup_time_sec": round(warmup_time_sec, 3),
        "end_to_end_time_sec": round(warmup_time_sec + elapsed, 3),
        "video_argument": args.video,
        "input_directory": str(input_directory),
        "covered_video_duration_sec": round(covered_video_duration_sec, 3),
        "source_frames_covered": source_frame_index,
        "processed_frames": processed_index,
        "frame_skip": FRAME_SKIP,
        "process_width": PROCESS_WIDTH,
        "process_height": process_height,
        "yolo_imgsz": YOLO_IMGSZ,
        "total_processing_time_sec": round(elapsed, 3),
        "processed_fps": round(processed_fps, 3),
        "effective_source_fps": round(effective_source_fps, 3),
        "realtime_factor": round(realtime_factor, 3),
        "speed_vs_realtime": round(speed_vs_realtime, 3),
        "avg_yolo_wall_ms": round(avg_yolo_wall_ms, 3),
        "avg_qr_ms": round(avg_qr_ms, 3),
        "avg_render_ms": round(avg_render_ms, 3),
        "avg_output_ms": round(avg_output_ms, 3),
        "ultralytics_preprocess_avg_ms": round(
            ultralytics_preprocess_ms / max(1, processed_index),
            3,
        ),
        "ultralytics_inference_avg_ms": round(
            ultralytics_inference_ms / max(1, processed_index),
            3,
        ),
        "ultralytics_postprocess_avg_ms": round(
            ultralytics_postprocess_ms / max(1, processed_index),
            3,
        ),
        "peak_gpu_memory_mb": round(peak_gpu_mb, 3),
        "save_output_video": save_output_video,
        "show_window": show_window,
        "event_count": len(system.events),
        "tracked_boxes": len(system.boxes),
        "known_pallets": len(system.pallets),
    }

    pd.DataFrame([metrics]).to_csv(performance_csv, index=False)

    database_run_id: Optional[int] = None

    if args.db:
        try:
            db_config = DatabaseConfig(
                host=args.db_host,
                port=args.db_port,
                user=args.db_user,
                password=args.db_password,
                database=args.db_name,
            )
            database = DatabaseManager(db_config)
            database.initialize()
            database.save_master_data(system.box_master, system.pallet_master)

            final_boxes_rows = (
                pd.read_csv(final_boxes_csv).fillna("").to_dict("records")
                if final_boxes_csv.exists()
                else []
            )
            final_pallets_rows = (
                pd.read_csv(final_pallets_csv).fillna("").to_dict("records")
                if final_pallets_csv.exists()
                else []
            )

            database_run_id = database.create_completed_run(metrics)
            database.save_run_outputs(
                run_id=database_run_id,
                metrics=metrics,
                events=system.events,
                final_boxes=final_boxes_rows,
                final_pallets=final_pallets_rows,
            )

            print(f"Database saved successfully. Run ID: {database_run_id}")

        except Exception as database_error:
            print("Database save failed. CSV files were still saved correctly.")
            print(f"Database error: {database_error}")


    print("=" * 76)
    print("HIGH-SPEED PROCESSING COMPLETED")
    print("=" * 76)
    print(f"Full video duration      : {video_duration_sec:.2f} seconds")
    print(f"Covered video duration   : {covered_video_duration_sec:.2f} seconds")
    print(f"Total processing time    : {elapsed:.2f} seconds")
    print(f"Source frames covered    : {source_frame_index}")
    print(f"Processed frames         : {processed_index}")
    print(f"Processed-frame speed    : {processed_fps:.2f} FPS")
    print(f"Effective source speed   : {effective_source_fps:.2f} FPS")
    print(f"Real-time factor         : {realtime_factor:.2f}x")
    print(f"Speed versus real time   : {speed_vs_realtime:.2f}x")
    print(f"Average YOLO wall time   : {avg_yolo_wall_ms:.2f} ms/frame")
    print(f"Average QR time          : {avg_qr_ms:.2f} ms/frame")
    print(f"Average render time      : {avg_render_ms:.2f} ms/frame")
    print(f"Average output time      : {avg_output_ms:.2f} ms/frame")
    print(f"YOLO inference reported  : {metrics['ultralytics_inference_avg_ms']:.2f} ms/frame")
    print(f"Peak GPU memory          : {peak_gpu_mb:.2f} MB")
    print("-" * 76)
    print(f"Output video             : {output_video if save_output_video else 'Not saved (--no-save)' }")
    print(f"Events CSV               : {events_csv}")
    print(f"Final boxes CSV          : {final_boxes_csv}")
    print(f"Final pallets CSV        : {final_pallets_csv}")
    print(f"Performance metrics CSV  : {performance_csv}")
    print(f"Database run ID          : {database_run_id if database_run_id is not None else 'Not saved (--db disabled or failed)'}")
    print("=" * 76)


if __name__ == "__main__":
    main()
