from __future__ import annotations

import os
from pathlib import Path
from typing import Any

from visiontrack.config.model_registry import ModelProfile
from visiontrack.perception.class_mapper import canonical_class_name
from visiontrack.perception.raw_observation import RawObservation


class Detector:
    def __init__(self, profile: ModelProfile):
        self.profile = profile
        if not os.getenv("YOLO_CONFIG_DIR"):
            config_directory = (
                Path(__file__).resolve().parents[3]
                / "outputs"
                / "ultralytics"
            )
            config_directory.mkdir(parents=True, exist_ok=True)
            os.environ["YOLO_CONFIG_DIR"] = str(config_directory)
        try:
            from ultralytics import YOLO
        except Exception as exc:  # pragma: no cover
            raise RuntimeError("ultralytics is required for detection.") from exc

        self.model = YOLO(str(profile.path))
        self.names = self.model.names
        self.device = self._select_device(profile.device)
        self.device_label = "cuda:0" if self.device == "0" else self.device

    def track_objects(self, frame: Any) -> list[RawObservation]:
        frame_height, frame_width = frame.shape[:2]
        frame_area = max(float(frame_width * frame_height), 1.0)
        results = self.model.track(
            frame,
            persist=True,
            conf=self.profile.conf,
            iou=self.profile.iou,
            imgsz=self.profile.imgsz,
            tracker=self.profile.tracker,
            device=self.device,
            verbose=False,
        )
        observations: list[RawObservation] = []

        if not results:
            return observations

        result = results[0]
        boxes = getattr(result, "boxes", None)
        if boxes is None:
            return observations

        for box in boxes:
            class_id = int(box.cls[0])
            raw_name = str(self.names.get(class_id, class_id))
            class_name = canonical_class_name(raw_name, self.profile.class_aliases)
            confidence = float(box.conf[0])
            x1, y1, x2, y2 = [int(v) for v in box.xyxy[0].tolist()]
            x1 = max(0, min(frame_width - 1, x1))
            y1 = max(0, min(frame_height - 1, y1))
            x2 = max(0, min(frame_width - 1, x2))
            y2 = max(0, min(frame_height - 1, y2))

            class_min_conf = self.profile.class_confidence.get(class_name, self.profile.conf)
            if confidence < class_min_conf:
                continue

            bbox_area = max(0, x2 - x1) * max(0, y2 - y1)
            max_ratio = self.profile.max_area_ratio.get(class_name)
            if max_ratio is not None and (bbox_area / frame_area) > max_ratio:
                continue

            track_id = None
            if getattr(box, "id", None) is not None:
                track_id = int(box.id[0])

            observations.append(
                RawObservation(
                    class_name=class_name,
                    display_class_name=raw_name,
                    confidence=confidence,
                    bbox=(x1, y1, x2, y2),
                    track_id=track_id,
                )
            )

        return observations

    @staticmethod
    def _select_device(requested_device: str | None) -> str:
        requested = (requested_device or os.getenv("VISIONTRACK_DEVICE") or "auto").strip().lower()
        if requested in {"", "auto"}:
            try:
                import torch
            except Exception:
                return "cpu"
            return "0" if torch.cuda.is_available() else "cpu"

        if requested in {"cuda", "gpu"}:
            requested = "0"
        elif requested.startswith("cuda:"):
            requested = requested.removeprefix("cuda:")

        if requested.isdigit():
            try:
                import torch
            except Exception as exc:  # pragma: no cover
                raise RuntimeError("GPU device requested, but PyTorch is not installed.") from exc
            if not torch.cuda.is_available():
                raise RuntimeError(
                    "GPU device requested, but this Python environment cannot use CUDA. "
                    "Install a CUDA-enabled torch/torchvision build."
                )
            return requested

        return requested
