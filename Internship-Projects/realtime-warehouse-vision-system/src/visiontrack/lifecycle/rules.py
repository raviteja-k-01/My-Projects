from __future__ import annotations

import re

from visiontrack.lifecycle.errors import (
    InvalidEventError,
    UnsupportedEventError,
)
from visiontrack.lifecycle.models import (
    LifecycleTarget,
    LocationCodes,
    RuleEvaluation,
    SourceVideoEvent,
    normalize_code,
)


_ZONE_TRANSITION_RE = re.compile(
    r"\bzone\s*:\s*(?P<from>[A-Za-z0-9_-]+|NONE)\s*->\s*"
    r"(?P<to>[A-Za-z0-9_-]+|NONE)",
    re.IGNORECASE,
)
_DOCK_CODE_RE = re.compile(r"\b(DOCK[_-][A-Za-z0-9_-]+)\b", re.IGNORECASE)


class LocationCodeExtractor:
    """Build destination codes before business rules validate an event."""

    _PALLET_DESTINATION_EVENTS = frozenset(
        {
            "ADD",
            "BOX_ADDED",
            "STACK",
            "BOX_STACKED",
            "RE_ADD",
            "BOX_RE_ADDED",
            "SWAP",
            "DISPATCH",
            "DOCK_ARRIVAL",
            "LOADED_TO_TRAILER",
            "TRAILER_DEPARTED",
        }
    )

    def extract(self, event: SourceVideoEvent) -> LocationCodes:
        conveyor_code = event.conveyor_code
        zone_code = event.to_zone_code
        dock_code = event.dock_code

        description = event.description or ""
        if description and (zone_code is None or dock_code is None):
            match = _ZONE_TRANSITION_RE.search(description)
            if match:
                destination = normalize_code(match.group("to"))
                if destination and destination.upper() != "NONE":
                    if self._is_dock(destination):
                        dock_code = dock_code or destination
                    elif self._is_conveyor(destination):
                        conveyor_code = conveyor_code or destination
                    elif zone_code is None:
                        zone_code = destination

        if zone_code and self._is_dock(zone_code):
            dock_code = dock_code or zone_code
            zone_code = None
        elif zone_code and self._is_conveyor(zone_code):
            conveyor_code = conveyor_code or zone_code
            zone_code = None

        if dock_code is None and description:
            dock_match = _DOCK_CODE_RE.search(description)
            if dock_match:
                dock_code = normalize_code(dock_match.group(1))

        pallet_code = event.to_pallet_code
        if (
            pallet_code is None
            and event.from_pallet_code is None
            and event.event_type in self._PALLET_DESTINATION_EVENTS
        ):
            pallet_code = event.pallet_code

        return LocationCodes(
            conveyor_code=conveyor_code,
            pallet_code=pallet_code,
            zone_code=zone_code,
            dock_code=dock_code,
        )

    @staticmethod
    def _is_dock(code: str) -> bool:
        return code.upper().startswith("DOCK")

    @staticmethod
    def _is_conveyor(code: str) -> bool:
        return code.upper().startswith(("CONVEYOR", "BELT"))


class LifecycleRuleEngine:
    """Translate explicit CV semantics into canonical business states."""

    _IGNORED_EVENTS = frozenset({"PALLET_ADDED", "PALLET_REMOVED"})
    _REMOVE_EVENTS = frozenset(
        {"REMOVE", "BOX_REMOVED", "BOX_COMPLETE_REMOVED"}
    )
    _LOCATION_EVENTS = frozenset(
        {
            "ADD",
            "BOX_ADDED",
            "MOVE",
            "BOX_MOVED",
            "RE_ADD",
            "BOX_RE_ADDED",
            "SWAP",
            "CONVEYOR_EXIT",
        }
    )
    _DEPARTURE_EVENTS = frozenset(
        {"TRAILER_DEPARTED", "TRAILER_DISPATCH", "WAREHOUSE_DISPATCH"}
    )

    def __init__(self, extractor: LocationCodeExtractor | None = None) -> None:
        self.extractor = extractor or LocationCodeExtractor()

    def evaluate(self, event: SourceVideoEvent) -> RuleEvaluation:
        codes = self.extractor.extract(event)
        event_type = event.event_type

        if event_type in self._IGNORED_EVENTS:
            return RuleEvaluation(
                target=None,
                location_codes=codes,
                ignored_reason="explicitly configured as non-box noise",
            )

        self._require_box(event)

        if event_type in self._REMOVE_EVENTS:
            return RuleEvaluation(
                target=LifecycleTarget(
                    "BOX_EXCEPTION",
                    event_type,
                    (),
                    "confirmed box removal requires operational review",
                ),
                location_codes=codes,
            )

        if event_type in {"STACK", "BOX_STACKED"}:
            self._require(codes.pallet_code, "destination pallet")
            return RuleEvaluation(
                self._target("ON_PALLET", "STACK", "stacked on pallet"),
                codes,
            )

        if event_type == "CONVEYOR_ENTRY":
            self._require(codes.conveyor_code, "conveyor_code")
            return RuleEvaluation(
                self._target(
                    "ON_CONVEYOR",
                    "CONVEYOR_ENTRY",
                    "entered conveyor",
                ),
                codes,
            )

        if event_type == "ZONE_TRANSFER":
            if codes.dock_code:
                return RuleEvaluation(
                    self._target("AT_DOCK", "DOCK_ARRIVAL", "zone resolved as dock"),
                    codes,
                )
            self._require(codes.zone_code, "destination zone")
            return RuleEvaluation(
                self._target("IN_ZONE", "ZONE_TRANSFER", "transferred to zone"),
                codes,
            )

        if event_type == "DOCK_ARRIVAL":
            self._require(codes.dock_code, "dock_code")
            return RuleEvaluation(
                self._target("AT_DOCK", "DOCK_ARRIVAL", "arrived at dock"),
                codes,
            )

        if event_type == "DISPATCH":
            # The current warehouse emitter creates DISPATCH when a tracked box
            # enters a DOCK_* region. That observation proves dock arrival, not
            # trailer departure. Keep the business state semantically honest.
            self._require(codes.dock_code, "dock destination")
            return RuleEvaluation(
                self._target(
                    "AT_DOCK",
                    "DOCK_ARRIVAL",
                    "current CV DISPATCH means entered dock",
                ),
                codes,
            )

        if event_type == "LOADED_TO_TRAILER":
            return RuleEvaluation(
                self._target(
                    "LOADED_TO_TRAILER",
                    event_type,
                    "explicit trailer loading event",
                ),
                codes,
            )

        if event_type in self._DEPARTURE_EVENTS:
            return RuleEvaluation(
                self._target(
                    "DISPATCHED",
                    "ORDER_DISPATCHED",
                    "explicit trailer departure event",
                ),
                codes,
            )

        if event_type in self._LOCATION_EVENTS:
            return RuleEvaluation(
                self._target_for_location(event_type, codes),
                codes,
            )

        raise UnsupportedEventError(
            f"Unsupported event_type {event_type!r}; add an explicit rule"
        )

    @staticmethod
    def _require_box(event: SourceVideoEvent) -> None:
        if not event.box_code:
            raise InvalidEventError("Missing required box_code")

    @staticmethod
    def _require(value: str | None, label: str) -> None:
        if not value:
            raise InvalidEventError(f"Missing required {label}")

    def _target_for_location(
        self,
        event_type: str,
        codes: LocationCodes,
    ) -> LifecycleTarget:
        if codes.dock_code:
            return self._target(
                "AT_DOCK",
                "DOCK_ARRIVAL",
                f"{event_type} resolved to dock",
            )
        if codes.pallet_code:
            return self._target(
                "ON_PALLET",
                event_type,
                f"{event_type} resolved to pallet",
            )
        if codes.conveyor_code:
            return self._target(
                "ON_CONVEYOR",
                event_type,
                f"{event_type} resolved to conveyor",
            )
        if codes.zone_code:
            return self._target(
                "IN_ZONE",
                event_type,
                f"{event_type} resolved to zone",
            )
        raise InvalidEventError(
            f"{event_type} has no resolvable destination location"
        )

    @staticmethod
    def _target(
        status: str,
        event_type: str,
        reason: str,
    ) -> LifecycleTarget:
        compare_fields = {
            "ON_CONVEYOR": ("conveyor_id", "pallet_id"),
            "IN_ZONE": ("zone_id", "pallet_id"),
            "ON_PALLET": ("pallet_id", "zone_id"),
            "AT_DOCK": ("dock_id", "pallet_id"),
            "LOADED_TO_TRAILER": (
                "dock_id",
                "pallet_id",
                "trailer_id",
                "truck_id",
            ),
            "DISPATCHED": (
                "dock_id",
                "pallet_id",
                "trailer_id",
                "truck_id",
            ),
        }.get(status, ())
        return LifecycleTarget(status, event_type, compare_fields, reason)
