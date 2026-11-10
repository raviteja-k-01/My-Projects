from __future__ import annotations

from collections import Counter, deque
from dataclasses import dataclass
from pathlib import Path
import shutil
import subprocess
import tempfile

import cv2
import numpy as np


@dataclass(frozen=True)
class _PanelEvent:
    frame_number: int
    event_type: str
    box_code: str
    source: str
    destination: str


class OutputPanel:
    """Render a separate telemetry panel without resizing the video frame."""

    BACKGROUND = (28, 31, 38)
    CARD = (40, 45, 54)
    BORDER = (72, 82, 96)
    TEXT = (240, 242, 245)
    MUTED = (165, 174, 187)
    ACCENT = (235, 181, 67)
    GREEN = (98, 210, 145)
    ORANGE = (75, 165, 245)
    RED = (95, 105, 235)

    def __init__(self, max_events: int = 6, max_locations: int = 12):
        self.max_locations = max(1, max_locations)
        self.recent_events: deque[_PanelEvent] = deque(maxlen=max(1, max_events))
        self.frame_number = 0
        self.current_counts: Counter[str] = Counter()
        self.current_qrs: list[str] = []
        self.box_locations: list[tuple[str, str]] = []
        self.visible_box_location_count = 0
        self.pallet_locations: list[tuple[str, str]] = []
        self.total_events = 0

    @staticmethod
    def panel_width_for(video_width: int) -> int:
        panel_width = max(2, int(round(video_width * 0.25)))
        if (video_width + panel_width) % 2:
            panel_width += 1
        return panel_width

    def update(
        self,
        frame_number: int,
        observations,
        qrs,
        events,
        engine,
    ) -> None:
        self.frame_number = frame_number
        self.current_counts = Counter(item.class_name for item in observations)
        self.current_qrs = sorted(
            {
                str(qr["value"])
                for qr in qrs
                if qr.get("value")
            }
        )

        visible_boxes = []
        for code, state in engine.boxes.items():
            if state.missing_frames != 0:
                continue
            visible_boxes.append(
                (
                    str(code),
                    self._location(
                        state.current_pallet_code,
                        state.current_zone_code,
                    ),
                )
            )
        self.visible_box_location_count = len(visible_boxes)
        self.box_locations = sorted(visible_boxes)[: self.max_locations]

        visible_pallets = []
        for code, state in engine.pallets.items():
            if not state.visible:
                continue
            visible_pallets.append(
                (
                    str(code),
                    str(state.current_zone_code or "UNASSIGNED"),
                )
            )
        self.pallet_locations = sorted(visible_pallets)[:4]

        for event in events:
            self.total_events += 1
            self.recent_events.appendleft(
                _PanelEvent(
                    frame_number=frame_number,
                    event_type=str(event.event_type.value),
                    box_code=str(event.box_code),
                    source=self._location(
                        event.from_pallet_code,
                        event.from_zone_code,
                    ),
                    destination=self._location(
                        event.to_pallet_code,
                        event.to_zone_code,
                    ),
                )
            )

    def compose(self, video_frame, total_frames: int):
        video_height, video_width = video_frame.shape[:2]
        panel_width = self.panel_width_for(video_width)
        panel = np.full(
            (video_height, panel_width, 3),
            self.BACKGROUND,
            dtype=np.uint8,
        )

        scale = min(
            max(panel_width / 960.0, 0.45),
            max(video_height / 2160.0, 0.45),
        )
        margin = max(14, int(round(30 * scale)))
        line_height = max(22, int(round(42 * scale)))
        small_scale = max(0.43, 0.82 * scale)
        body_scale = max(0.48, 0.92 * scale)
        heading_scale = max(0.60, 1.10 * scale)
        title_scale = max(0.75, 1.35 * scale)
        thin = max(1, int(round(2 * scale)))
        thick = max(1, int(round(3 * scale)))
        content_width = panel_width - margin * 2

        cv2.line(
            panel,
            (0, 0),
            (0, video_height - 1),
            self.ACCENT,
            max(2, thick),
        )

        y = margin + int(40 * scale)
        self._text(
            panel,
            "VISIONTRACK",
            margin,
            y,
            title_scale,
            self.TEXT,
            thick,
            content_width,
        )
        y += int(42 * scale)
        self._text(
            panel,
            "LIVE WAREHOUSE INTELLIGENCE",
            margin,
            y,
            small_scale,
            self.ACCENT,
            thin,
            content_width,
        )

        y += int(45 * scale)
        progress = (
            min(max(self.frame_number / total_frames, 0.0), 1.0)
            if total_frames > 0
            else 0.0
        )
        bar_height = max(8, int(round(12 * scale)))
        cv2.rectangle(
            panel,
            (margin, y),
            (panel_width - margin, y + bar_height),
            self.BORDER,
            -1,
        )
        cv2.rectangle(
            panel,
            (margin, y),
            (margin + int(content_width * progress), y + bar_height),
            self.ACCENT,
            -1,
        )
        y += bar_height + int(34 * scale)
        progress_text = (
            f"Frame {self.frame_number}/{total_frames}  |  {progress * 100:5.1f}%"
            if total_frames > 0
            else f"Frame {self.frame_number}"
        )
        self._text(
            panel,
            progress_text,
            margin,
            y,
            small_scale,
            self.MUTED,
            thin,
            content_width,
        )

        y += int(42 * scale)
        y = self._section_title(
            panel,
            "VISIBLE DETECTIONS",
            y,
            margin,
            heading_scale,
            thick,
            content_width,
        )

        count_rows = [
            (
                f"Boxes  {self.current_counts.get('box', 0)}",
                f"Pallets  {self.current_counts.get('pallet', 0)}",
            ),
            (
                f"QR  {self.current_counts.get('qr_code', 0)}",
                f"Conveyor  {self.current_counts.get('conveyor_belt', 0)}",
            ),
            (
                f"Dock  {self.current_counts.get('dock', 0)}",
                f"Zones  {self.current_counts.get('zone', 0)}",
            ),
        ]
        card_gap = max(8, int(round(12 * scale)))
        card_width = (content_width - card_gap) // 2
        card_height = max(36, int(round(58 * scale)))
        for left_text, right_text in count_rows:
            for column, text in enumerate((left_text, right_text)):
                x1 = margin + column * (card_width + card_gap)
                x2 = x1 + card_width
                cv2.rectangle(
                    panel,
                    (x1, y),
                    (x2, y + card_height),
                    self.CARD,
                    -1,
                )
                cv2.rectangle(
                    panel,
                    (x1, y),
                    (x2, y + card_height),
                    self.BORDER,
                    thin,
                )
                self._text(
                    panel,
                    text,
                    x1 + int(13 * scale),
                    y + int(38 * scale),
                    body_scale,
                    self.TEXT,
                    thin,
                    card_width - int(24 * scale),
                )
            y += card_height + card_gap

        y += int(18 * scale)
        y = self._section_title(
            panel,
            (
                "CURRENT BOX LOCATIONS "
                f"({len(self.box_locations)}/{self.visible_box_location_count})"
            ),
            y,
            margin,
            heading_scale,
            thick,
            content_width,
        )
        if not self.box_locations:
            y = self._simple_line(
                panel,
                "No confirmed box locations",
                y,
                margin,
                body_scale,
                self.MUTED,
                thin,
                content_width,
                line_height,
            )
        else:
            for code, location in self.box_locations:
                y = self._simple_line(
                    panel,
                    f"{code}  ->  {location}",
                    y,
                    margin,
                    small_scale,
                    self.GREEN,
                    thin,
                    content_width,
                    line_height,
                )

        y += int(16 * scale)
        y = self._section_title(
            panel,
            "CURRENT PALLETS",
            y,
            margin,
            heading_scale,
            thick,
            content_width,
        )
        if not self.pallet_locations:
            y = self._simple_line(
                panel,
                "No visible pallets",
                y,
                margin,
                body_scale,
                self.MUTED,
                thin,
                content_width,
                line_height,
            )
        else:
            for code, location in self.pallet_locations:
                y = self._simple_line(
                    panel,
                    f"{code}  |  {location}",
                    y,
                    margin,
                    small_scale,
                    self.GREEN,
                    thin,
                    content_width,
                    line_height,
                )

        y += int(16 * scale)
        y = self._section_title(
            panel,
            f"RECENT EVENTS  ({self.total_events})",
            y,
            margin,
            heading_scale,
            thick,
            content_width,
        )
        if not self.recent_events:
            y = self._simple_line(
                panel,
                "Waiting for a confirmed movement...",
                y,
                margin,
                body_scale,
                self.MUTED,
                thin,
                content_width,
                line_height,
            )
        else:
            for event in self.recent_events:
                if y + line_height * 2 > video_height - margin * 4:
                    break
                color = (
                    self.RED
                    if event.event_type in {"REMOVE", "DISPATCH"}
                    else self.ORANGE
                )
                y = self._simple_line(
                    panel,
                    (
                        f"F{event.frame_number:04d}  "
                        f"{event.event_type}  {event.box_code}"
                    ),
                    y,
                    margin,
                    small_scale,
                    color,
                    thick,
                    content_width,
                    line_height,
                )
                y = self._simple_line(
                    panel,
                    f"  {event.source} -> {event.destination}",
                    y,
                    margin,
                    small_scale,
                    self.MUTED,
                    thin,
                    content_width,
                    line_height,
                )

        footer_y = video_height - margin - int(20 * scale)
        qr_values = ", ".join(self.current_qrs[:5]) or "-"
        self._text(
            panel,
            f"QR: {qr_values}",
            margin,
            footer_y - line_height,
            small_scale,
            self.MUTED,
            thin,
            content_width,
        )
        qr_text = (
            f"Decoded QR now: {len(self.current_qrs)}"
            if self.current_qrs
            else "Decoded QR now: 0"
        )
        self._text(
            panel,
            qr_text,
            margin,
            footer_y,
            small_scale,
            self.MUTED,
            thin,
            content_width,
        )

        return np.concatenate((video_frame, panel), axis=1)

    def _section_title(
        self,
        panel,
        title: str,
        y: int,
        margin: int,
        font_scale: float,
        thickness: int,
        content_width: int,
    ) -> int:
        self._text(
            panel,
            title,
            margin,
            y,
            font_scale,
            self.ACCENT,
            thickness,
            content_width,
        )
        return y + max(28, int(round(48 * font_scale)))

    def _simple_line(
        self,
        panel,
        text: str,
        y: int,
        x: int,
        font_scale: float,
        color,
        thickness: int,
        max_width: int,
        line_height: int,
    ) -> int:
        self._text(
            panel,
            text,
            x,
            y,
            font_scale,
            color,
            thickness,
            max_width,
        )
        return y + line_height

    @staticmethod
    def _text(
        panel,
        text: str,
        x: int,
        y: int,
        font_scale: float,
        color,
        thickness: int,
        max_width: int,
    ) -> None:
        font = cv2.FONT_HERSHEY_SIMPLEX
        visible = OutputPanel._fit_text(
            str(text),
            font,
            font_scale,
            thickness,
            max_width,
        )
        cv2.putText(
            panel,
            visible,
            (x, y),
            font,
            font_scale,
            color,
            thickness,
            cv2.LINE_AA,
        )

    @staticmethod
    def _fit_text(
        text: str,
        font: int,
        font_scale: float,
        thickness: int,
        max_width: int,
    ) -> str:
        if cv2.getTextSize(text, font, font_scale, thickness)[0][0] <= max_width:
            return text
        suffix = "..."
        visible = text
        while visible:
            candidate = visible.rstrip() + suffix
            width = cv2.getTextSize(
                candidate,
                font,
                font_scale,
                thickness,
            )[0][0]
            if width <= max_width:
                return candidate
            visible = visible[:-1]
        return suffix

    @staticmethod
    def _location(
        pallet_code: str | None,
        region_code: str | None,
    ) -> str:
        if pallet_code and region_code:
            return f"{pallet_code} / {region_code}"
        return str(pallet_code or region_code or "UNASSIGNED")


class HighQualityVideoWriter:
    """Stream BGR frames to FFmpeg using high-quality H.264 encoding."""

    def __init__(
        self,
        path: str | Path,
        width: int,
        height: int,
        fps: float,
        crf: int = 17,
    ):
        ffmpeg = shutil.which("ffmpeg")
        if ffmpeg is None:
            raise RuntimeError("FFmpeg was not found in this environment.")
        if width % 2 or height % 2:
            raise ValueError("H.264 output width and height must be even.")

        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.width = int(width)
        self.height = int(height)
        self.stderr_file = tempfile.TemporaryFile(mode="w+b")
        command = [
            ffmpeg,
            "-hide_banner",
            "-loglevel",
            "error",
            "-y",
            "-f",
            "rawvideo",
            "-pix_fmt",
            "bgr24",
            "-s:v",
            f"{self.width}x{self.height}",
            "-r",
            f"{fps:.6f}",
            "-i",
            "pipe:0",
            "-an",
            "-c:v",
            "libx264",
            "-preset",
            "veryfast",
            "-crf",
            str(max(0, min(int(crf), 51))),
            "-pix_fmt",
            "yuv420p",
            "-movflags",
            "+faststart",
            str(self.path),
        ]
        try:
            self.process = subprocess.Popen(
                command,
                stdin=subprocess.PIPE,
                stdout=subprocess.DEVNULL,
                stderr=self.stderr_file,
            )
        except OSError:
            self.stderr_file.close()
            raise

    def write(self, frame) -> None:
        if frame.shape[:2] != (self.height, self.width):
            raise ValueError(
                "Output frame has shape "
                f"{frame.shape[1]}x{frame.shape[0]}, expected "
                f"{self.width}x{self.height}."
            )
        if self.process.stdin is None:
            raise RuntimeError("FFmpeg input pipe is unavailable.")
        try:
            self.process.stdin.write(
                memoryview(np.ascontiguousarray(frame, dtype=np.uint8)).cast("B")
            )
        except BrokenPipeError as exc:
            raise RuntimeError(self._error_message()) from exc

    def release(self) -> None:
        if self.stderr_file.closed:
            return
        try:
            if self.process.stdin is not None and not self.process.stdin.closed:
                try:
                    self.process.stdin.close()
                except BrokenPipeError:
                    pass
            if self.process.wait(timeout=60) != 0:
                raise RuntimeError(self._error_message())
        except subprocess.TimeoutExpired as exc:
            self.process.kill()
            self.process.wait()
            raise RuntimeError("FFmpeg timed out while finalizing the video.") from exc
        finally:
            self.stderr_file.close()

    def _error_message(self) -> str:
        self.stderr_file.seek(0)
        details = self.stderr_file.read().decode("utf-8", errors="replace").strip()
        return f"FFmpeg video writer failed: {details or 'unknown error'}"
