from __future__ import annotations

from typing import Any, Optional

import cv2
import numpy as np

from visiontrack.identity import canonicalize_identity


class QRDecoder:
    def __init__(self, qr_config: dict[str, Any] | None = None):
        self.qr_config = qr_config or {}
        self.detector = cv2.QRCodeDetector()
        self.use_pyzbar = bool(self.qr_config.get("use_pyzbar", False))
        self.pyzbar_decode = None
        if self.use_pyzbar:
            try:
                from pyzbar.pyzbar import decode as pyzbar_decode
            except Exception:  # pragma: no cover
                pyzbar_decode = None
            self.pyzbar_decode = pyzbar_decode

    def normalize_value(self, value: str) -> Optional[str]:
        return canonicalize_identity(value)

    def decode_frame(
        self,
        frame,
        qr_bboxes: list[tuple[int, int, int, int]] | None = None,
    ) -> list[dict[str, Any]]:
        decoded: list[dict[str, Any]] = []

        for bbox in qr_bboxes or []:
            value = self._decode_crop(frame, bbox)
            if value:
                decoded.append({"value": value, "bbox": bbox})

        return self._dedupe(decoded)

    def _decode_crop(self, frame, bbox: tuple[int, int, int, int]) -> str | None:
        height, width = frame.shape[:2]
        raw_x1, raw_y1, raw_x2, raw_y2 = bbox
        crop_width = max(1, raw_x2 - raw_x1)
        crop_height = max(1, raw_y2 - raw_y1)
        pad = max(14, int(max(crop_width, crop_height) * 0.35))
        x1, y1, x2, y2 = raw_x1, raw_y1, raw_x2, raw_y2
        x1 = max(0, x1 - pad)
        y1 = max(0, y1 - pad)
        x2 = min(width, x2 + pad)
        y2 = min(height, y2 + pad)
        crop = frame[y1:y2, x1:x2]
        if crop.size == 0:
            return None

        for variant in self._crop_variants(crop):
            value = self._decode_variant(variant)
            if value:
                return value

        return self._decode_crop_with_pyzbar(crop)

    def _decode_variant(self, image) -> str | None:
        try:
            raw, _, _ = self.detector.detectAndDecode(image)
        except cv2.error:
            raw = ""
        value = self.normalize_value(raw)
        if value:
            return value

        return None

    @staticmethod
    def _crop_variants(crop) -> list:
        variants = []

        def add_variant(image) -> None:
            variants.append(image)

        add_variant(crop)

        border = max(16, int(min(crop.shape[:2]) * 0.18))
        bordered = cv2.copyMakeBorder(
            crop,
            border,
            border,
            border,
            border,
            cv2.BORDER_CONSTANT,
            value=(255, 255, 255),
        )
        add_variant(bordered)

        for image in list(variants):
            min_side = min(image.shape[:2])
            if min_side < 420:
                scale = max(2, int(round(420 / max(min_side, 1))))
                add_variant(cv2.resize(image, None, fx=scale, fy=scale, interpolation=cv2.INTER_CUBIC))

        gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
        add_variant(cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR))

        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8)).apply(gray)
        add_variant(cv2.cvtColor(clahe, cv2.COLOR_GRAY2BGR))

        sharpen_kernel = np.array(((0, -1, 0), (-1, 5, -1), (0, -1, 0)), dtype="float32")
        sharpened = cv2.filter2D(crop, -1, sharpen_kernel)
        add_variant(sharpened)

        blur = cv2.GaussianBlur(gray, (3, 3), 0)
        _, threshold = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        add_variant(cv2.cvtColor(threshold, cv2.COLOR_GRAY2BGR))

        adaptive = cv2.adaptiveThreshold(
            gray,
            255,
            cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            cv2.THRESH_BINARY,
            31,
            5,
        )
        add_variant(cv2.cvtColor(adaptive, cv2.COLOR_GRAY2BGR))
        return variants

    def _decode_crop_with_pyzbar(self, crop) -> str | None:
        if self.pyzbar_decode is None:
            return None

        for item in self.pyzbar_decode(crop):
            raw = item.data.decode("utf-8", errors="ignore")
            value = self.normalize_value(raw)
            if value:
                return value
        return None

    @staticmethod
    def _dedupe(items: list[dict[str, Any]]) -> list[dict[str, Any]]:
        indexes_by_value: dict[str, int] = {}
        result: list[dict[str, Any]] = []
        for item in items:
            value = item["value"]
            existing_index = indexes_by_value.get(value)
            if existing_index is None:
                indexes_by_value[value] = len(result)
                result.append(item)
                continue

            existing = result[existing_index]
            if QRDecoder._is_better_qr_item(item, existing):
                result[existing_index] = item
        return result

    @staticmethod
    def _is_better_qr_item(candidate: dict[str, Any], current: dict[str, Any]) -> bool:
        candidate_has_points = bool(candidate.get("points"))
        current_has_points = bool(current.get("points"))
        if candidate_has_points != current_has_points:
            return candidate_has_points
        return QRDecoder._bbox_area(candidate["bbox"]) < QRDecoder._bbox_area(current["bbox"])

    @staticmethod
    def _bbox_area(bbox: tuple[int, int, int, int]) -> int:
        x1, y1, x2, y2 = bbox
        return max(0, x2 - x1) * max(0, y2 - y1)
