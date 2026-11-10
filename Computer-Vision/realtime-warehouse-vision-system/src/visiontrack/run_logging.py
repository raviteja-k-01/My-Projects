from __future__ import annotations

import logging
from datetime import datetime
from pathlib import Path


class RunLogger:
    """Writes one plain-text log file for each inference run."""

    def __init__(self, directory: str | Path, video_name: str):
        log_dir = Path(directory)
        log_dir.mkdir(parents=True, exist_ok=True)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.path = log_dir / f"{Path(video_name).stem}_{timestamp}.log"
        self.logger = logging.getLogger(f"visiontrack.run.{timestamp}.{id(self)}")
        self.logger.setLevel(logging.INFO)
        self.logger.propagate = False
        handler = logging.FileHandler(self.path, encoding="utf-8")
        handler.setFormatter(logging.Formatter(
            "%(asctime)s | %(levelname)s | %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",
        ))
        self.logger.addHandler(handler)

    def info(self, message: str) -> None:
        self.logger.info(message)

    def exception(self, message: str) -> None:
        self.logger.exception(message)

    def close(self) -> None:
        for handler in list(self.logger.handlers):
            handler.close()
            self.logger.removeHandler(handler)
