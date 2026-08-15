from __future__ import annotations

from dataclasses import dataclass, fields
from datetime import datetime
from enum import Enum
import math
from typing import Any, Mapping

from visiontrack.identity import canonicalize_identity
from visiontrack.lifecycle.errors import InvalidEventError


class LowConfidencePolicy(str, Enum):
    IGNORE = "IGNORE"
    ERROR = "ERROR"


class ProcessingOutcome(str, Enum):
    PROMOTED = "PROMOTED"
    IGNORED = "IGNORED"
    ERROR = "ERROR"


@dataclass(frozen=True)
class LifecycleProcessorSettings:
    batch_size: int = 200
    poll_ms: int = 250
    min_confidence: float = 0.0
    low_confidence_policy: LowConfidencePolicy = LowConfidencePolicy.IGNORE
    recorded_by: int | None = None

    def __post_init__(self) -> None:
        if self.batch_size < 1:
            raise ValueError("batch_size must be at least 1")
        if self.poll_ms < 50:
            raise ValueError("poll_ms must be at least 50")
        if not math.isfinite(float(self.min_confidence)):
            raise ValueError("min_confidence must be finite")
        if not 0.0 <= float(self.min_confidence) <= 100.0:
            raise ValueError("min_confidence must be between 0 and 100")
        if self.recorded_by is not None and self.recorded_by < 1:
            raise ValueError("recorded_by must be a positive database ID")

        policy = self.low_confidence_policy
        if not isinstance(policy, LowConfidencePolicy):
            try:
                policy = LowConfidencePolicy(str(policy).strip().upper())
            except ValueError as exc:
                raise ValueError(
                    "low_confidence_policy must be IGNORE or ERROR"
                ) from exc
            object.__setattr__(self, "low_confidence_policy", policy)


def clean_text(value: Any) -> str | None:
    if value is None:
        return None
    text = str(value).strip()
    if not text or text.lower() in {"none", "null", "nan", "<na>"}:
        return None
    return text


def normalize_code(value: Any) -> str | None:
    text = clean_text(value)
    if text is None:
        return None
    return canonicalize_identity(text)


def confidence_percent(value: Any) -> float | None:
    """Validate the database's explicit 0..100 confidence contract.

    ``video_events.confidence_score`` is already a percentage. In particular,
    1.0 means one percent; it must never be guessed to mean 100 percent.
    """

    if value is None:
        return None
    try:
        number = float(value)
    except (TypeError, ValueError, OverflowError) as exc:
        raise InvalidEventError(f"Invalid confidence_score: {value!r}") from exc
    if not math.isfinite(number) or not 0.0 <= number <= 100.0:
        raise InvalidEventError(
            f"confidence_score must be a finite 0..100 percentage: {value!r}"
        )
    return number


def event_datetime(value: Any, *, field_name: str) -> datetime:
    if isinstance(value, datetime):
        return value
    text = clean_text(value)
    if text is not None:
        try:
            return datetime.fromisoformat(text.replace("Z", "+00:00"))
        except ValueError:
            pass
    raise InvalidEventError(f"Missing or invalid {field_name}")


def optional_bool(value: Any) -> bool | None:
    if value is None:
        return None
    if isinstance(value, bool):
        return value
    if isinstance(value, int) and value in {0, 1}:
        return bool(value)
    text = clean_text(value)
    if text is not None:
        normalized = text.lower()
        if normalized in {"1", "true", "yes", "on"}:
            return True
        if normalized in {"0", "false", "no", "off"}:
            return False
    raise InvalidEventError(f"Invalid qr_read_success value: {value!r}")


def normalize_severity(value: Any) -> str:
    raw = (clean_text(value) or "NORMAL").upper()
    return {
        "INFO": "NORMAL",
        "SUCCESS": "NORMAL",
        "NORMAL": "NORMAL",
        "WARNING": "WARNING",
        "WARN": "WARNING",
        "ERROR": "CRITICAL",
        "CRITICAL": "CRITICAL",
    }.get(raw, "NORMAL")


@dataclass(frozen=True)
class SourceVideoEvent:
    video_event_id: int
    run_id: int
    processed_index: int | None
    video_time: str | None
    timestamp_sec: float | None
    event_type: str
    severity: str
    box_code: str | None
    pallet_code: str | None
    from_pallet_code: str | None
    to_pallet_code: str | None
    conveyor_code: str | None
    from_zone_code: str | None
    to_zone_code: str | None
    dock_code: str | None
    confidence_score: float | None
    qr_read_success: bool | None
    description: str | None
    event_time: datetime

    @classmethod
    def from_row(cls, row: Mapping[str, Any]) -> "SourceVideoEvent":
        try:
            video_event_id = int(row["video_event_id"])
            run_id = int(row["run_id"])
        except (KeyError, TypeError, ValueError, OverflowError) as exc:
            raise InvalidEventError(
                "video_event_id and run_id must be valid integers"
            ) from exc

        event_type = (clean_text(row.get("event_type")) or "").upper()
        if not event_type:
            raise InvalidEventError("Missing event_type")

        raw_processed_index = row.get("processed_index")
        raw_timestamp = row.get("timestamp_sec")
        try:
            processed_index = (
                None if raw_processed_index is None else int(raw_processed_index)
            )
            timestamp_sec = (
                None if raw_timestamp is None else float(raw_timestamp)
            )
        except (TypeError, ValueError, OverflowError) as exc:
            raise InvalidEventError(
                "processed_index or timestamp_sec is invalid"
            ) from exc

        raw_event_time = row.get("event_created_at") or row.get("inserted_at")
        return cls(
            video_event_id=video_event_id,
            run_id=run_id,
            processed_index=processed_index,
            video_time=clean_text(row.get("video_time")),
            timestamp_sec=timestamp_sec,
            event_type=event_type,
            severity=normalize_severity(row.get("severity")),
            box_code=normalize_code(row.get("box_code")),
            pallet_code=normalize_code(row.get("pallet_code")),
            from_pallet_code=normalize_code(row.get("from_pallet_code")),
            to_pallet_code=normalize_code(row.get("to_pallet_code")),
            conveyor_code=normalize_code(row.get("conveyor_code")),
            from_zone_code=normalize_code(row.get("from_zone_code")),
            to_zone_code=normalize_code(row.get("to_zone_code")),
            dock_code=normalize_code(row.get("dock_code")),
            confidence_score=confidence_percent(row.get("confidence_score")),
            qr_read_success=optional_bool(row.get("qr_read_success")),
            description=clean_text(row.get("description")),
            event_time=event_datetime(
                raw_event_time,
                field_name="event_created_at/inserted_at",
            ),
        )


@dataclass(frozen=True)
class LocationCodes:
    conveyor_code: str | None = None
    pallet_code: str | None = None
    zone_code: str | None = None
    dock_code: str | None = None


@dataclass(frozen=True)
class LifecycleTarget:
    lifecycle_status: str
    business_event_type: str
    compare_fields: tuple[str, ...]
    reason: str


@dataclass(frozen=True)
class RuleEvaluation:
    target: LifecycleTarget | None
    location_codes: LocationCodes
    ignored_reason: str | None = None


@dataclass(frozen=True)
class OrderContext:
    order_id: int
    order_item_product_id: int


@dataclass(frozen=True)
class LocationContext:
    conveyor_id: int | None = None
    pallet_id: int | None = None
    zone_id: int | None = None
    dock_id: int | None = None
    trailer_id: int | None = None
    truck_id: int | None = None
    trailer_dock_assignment_id: int | None = None
    trailer_truck_assignment_id: int | None = None

    def updated(self, **values: int | None) -> "LocationContext":
        current = {field.name: getattr(self, field.name) for field in fields(self)}
        current.update(values)
        return LocationContext(**current)


@dataclass(frozen=True)
class LifecycleState(LocationContext):
    lifecycle_status: str = ""
    event_time: datetime | None = None


@dataclass(frozen=True)
class ProcessingResult:
    video_event_id: int
    outcome: ProcessingOutcome
    reason: str
    inserted_box_rows: int = 0
    inserted_order_rows: int = 0


@dataclass
class BatchStats:
    fetched: int = 0
    promoted: int = 0
    ignored: int = 0
    errors: int = 0
    inserted_box_rows: int = 0
    inserted_order_rows: int = 0

    def add(self, result: ProcessingResult) -> None:
        self.fetched += 1
        if result.outcome is ProcessingOutcome.PROMOTED:
            self.promoted += 1
        elif result.outcome is ProcessingOutcome.IGNORED:
            self.ignored += 1
        elif result.outcome is ProcessingOutcome.ERROR:
            self.errors += 1
        self.inserted_box_rows += result.inserted_box_rows
        self.inserted_order_rows += result.inserted_order_rows

    def merge(self, other: "BatchStats") -> None:
        for field in fields(self):
            setattr(
                self,
                field.name,
                getattr(self, field.name) + getattr(other, field.name),
            )

    def describe(self) -> str:
        return (
            f"fetched={self.fetched} promoted={self.promoted} "
            f"ignored={self.ignored} errors={self.errors} "
            f"box_lifecycle={self.inserted_box_rows} "
            f"order_lifecycle={self.inserted_order_rows}"
        )
