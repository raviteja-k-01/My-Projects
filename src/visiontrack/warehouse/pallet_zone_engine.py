from __future__ import annotations

from dataclasses import dataclass, replace
import math

from visiontrack.association.location_resolver import (
    REGION_CLASSES,
    DynamicLocationResolver,
)
from visiontrack.perception.raw_observation import RawObservation
from visiontrack.warehouse.entities import BoxState, PalletState
from visiontrack.warehouse.events import EventType, WarehouseEvent


Location = tuple[str | None, str | None]


@dataclass
class _Candidate:
    location: Location
    frames: int = 1
    confidence_total: float = 0.0
    confidence_samples: int = 0

    def add_confidence(self, confidence_score: float | None) -> None:
        if confidence_score is None:
            return
        self.confidence_total += confidence_score
        self.confidence_samples += 1

    @property
    def confidence_score(self) -> float | None:
        if self.confidence_samples == 0:
            return None
        return self.confidence_total / self.confidence_samples


@dataclass
class _PendingEvent:
    event: WarehouseEvent
    queued_frame: int


@dataclass
class _IdentityChangeCandidate:
    code: str
    frames: int
    last_frame: int


class PalletZoneEngine:
    """Create location events only from tracked YOLO detections."""

    def __init__(self, config: dict):
        dynamic = config.get("dynamic_location", {})
        self.confirmation_frames = max(
            1,
            int(dynamic.get("confirmation_frames", 5)),
        )
        self.startup_baseline_frames = max(
            self.confirmation_frames,
            int(dynamic.get("startup_baseline_frames", 30)),
        )
        self.identity_resolution_timeout_frames = max(
            1,
            int(dynamic.get("identity_resolution_timeout_frames", 300)),
        )
        self.identity_change_confirmation_frames = max(
            2,
            int(dynamic.get("identity_change_confirmation_frames", 3)),
        )
        self.semantic_region_codes = {
            str(class_name).strip().lower(): str(region_code).strip()
            for class_name, region_code in dynamic.get(
                "semantic_region_codes",
                {"conveyor_belt": "conveyor_belt"},
            ).items()
            if str(class_name).strip() and str(region_code).strip()
        }
        self.location_resolver = DynamicLocationResolver(
            min_pallet_overlap_ratio=float(
                dynamic.get("min_pallet_overlap_ratio", 0.10)
            ),
            min_region_overlap_ratio=float(
                dynamic.get("min_region_overlap_ratio", 0.50)
            ),
        )
        default_pallet = config.get("capacity", {}).get("default_pallet", {})
        self.default_pallet_max_boxes = max(
            1,
            int(default_pallet.get("max_boxes", 18)),
        )

        self.boxes: dict[str, BoxState] = {}
        self.pallets: dict[str, PalletState] = {}
        self.box_candidates: dict[str, _Candidate] = {}
        self.pallet_candidates: dict[str, _Candidate] = {}
        self.initialized_boxes: set[str] = set()
        self.initialized_pallets: set[str] = set()
        self.track_identities: dict[tuple[str, int], str] = {}
        self.identity_change_candidates: dict[
            tuple[str, int], _IdentityChangeCandidate
        ] = {}
        self.pending_events: list[_PendingEvent] = []
        self.dropped_unresolved_event_count = 0
        self.frame_number = 0

    def update(self, observations: list[RawObservation]) -> list[WarehouseEvent]:
        self.frame_number += 1
        events: list[WarehouseEvent] = []
        boxes = [item for item in observations if item.class_name == "box"]
        pallets = [item for item in observations if item.class_name == "pallet"]
        regions = [
            item for item in observations if item.class_name in REGION_CLASSES
        ]

        for state in self.pallets.values():
            state.visible = False

        visible_pallets: set[str] = set()
        for observation in pallets:
            pallet_code = self._identity(observation, "PALLET")
            if pallet_code is None:
                continue

            visible_pallets.add(pallet_code)
            state = self.pallets.setdefault(
                pallet_code,
                PalletState(
                    pallet_code=pallet_code,
                    max_boxes=self.default_pallet_max_boxes,
                ),
            )
            state.visible = True
            state.status = "IN_USE"

            resolved = self.location_resolver.resolve_pallet(
                observation,
                regions,
            )
            region_code = self._region_identity(resolved.region)
            self._consider_pallet_region(
                state,
                region_code,
                self._location_evidence(observation, resolved.region),
            )

        for pallet_code in list(self.pallet_candidates):
            if pallet_code not in visible_pallets:
                self.pallet_candidates.pop(pallet_code, None)

        visible_boxes: set[str] = set()
        for observation in boxes:
            box_code = self._identity(observation, "BOX")
            if box_code is None:
                continue

            visible_boxes.add(box_code)
            state = self.boxes.setdefault(
                box_code,
                BoxState(box_code=box_code),
            )
            state.missing_frames = 0
            if (
                observation.qr_read_success
                and observation.qr_value == box_code
            ):
                state.identity_from_qr = True

            resolved = self.location_resolver.resolve_box(
                observation,
                pallets,
                regions,
            )
            pallet_code = self._identity(resolved.pallet, "PALLET")
            region_code = self._region_identity(resolved.region)
            event = self._consider_box_location(
                state,
                pallet_code,
                region_code,
                confidence_score=self._location_evidence(
                    observation,
                    resolved.pallet,
                    resolved.region,
                ),
            )
            self._queue_or_emit(event, events)

        for box_code, state in self.boxes.items():
            if box_code in visible_boxes:
                continue

            pallet_code = state.current_pallet_code
            if pallet_code and pallet_code in visible_pallets:
                # A hidden box moves with its still-visible parent pallet.
                state.missing_frames = 0
                event = self._consider_box_location(
                    state,
                    pallet_code,
                    self.pallets[pallet_code].current_zone_code,
                    location_is_confirmed=True,
                    confidence_score=(
                        self.pallets[pallet_code]
                        .current_location_confidence_score
                    ),
                )
                self._queue_or_emit(event, events)
                continue

            self.box_candidates.pop(box_code, None)
            state.missing_frames += 1

        events.extend(self._release_resolved_pending_events())
        self._expire_unresolved_pending_events()
        return events

    def _queue_or_emit(
        self,
        event: WarehouseEvent | None,
        emitted: list[WarehouseEvent],
    ) -> None:
        if event is None:
            return
        if self._event_has_temporary_identity(event):
            self.pending_events.append(
                _PendingEvent(
                    event=event,
                    queued_frame=self.frame_number,
                )
            )
            return
        emitted.append(event)

    def _release_resolved_pending_events(self) -> list[WarehouseEvent]:
        ready: list[WarehouseEvent] = []
        waiting: list[_PendingEvent] = []
        for pending in self.pending_events:
            if self._event_has_temporary_identity(pending.event):
                waiting.append(pending)
            else:
                ready.append(pending.event)
        self.pending_events = waiting
        return ready

    def _expire_unresolved_pending_events(self) -> None:
        cutoff = self.frame_number - self.identity_resolution_timeout_frames
        waiting: list[_PendingEvent] = []
        for pending in self.pending_events:
            if pending.queued_frame <= cutoff:
                self.dropped_unresolved_event_count += 1
            else:
                waiting.append(pending)
        self.pending_events = waiting

    @classmethod
    def _event_has_temporary_identity(cls, event: WarehouseEvent) -> bool:
        return any(
            cls._is_temporary_identity(value)
            for value in (
                event.box_code,
                event.from_pallet_code,
                event.to_pallet_code,
                event.from_zone_code,
                event.to_zone_code,
            )
        )

    @staticmethod
    def _is_temporary_identity(value: str | None) -> bool:
        return bool(value and "_TRACK_" in value.upper())

    def _consider_pallet_region(
        self,
        pallet: PalletState,
        region_code: str | None,
        confidence_score: float | None,
    ) -> None:
        key = pallet.pallet_code
        if region_code is None:
            self.pallet_candidates.pop(key, None)
            return
        if (
            key in self.initialized_pallets
            and pallet.current_zone_code == region_code
        ):
            self.pallet_candidates.pop(key, None)
            return

        confirmed = self._confirmed_candidate(
            self.pallet_candidates,
            key,
            (None, region_code),
            confidence_score,
        )
        if confirmed is None:
            return

        pallet.current_zone_code = region_code
        pallet.current_location_confidence_score = (
            confirmed.confidence_score
        )
        self.initialized_pallets.add(key)

    def _consider_box_location(
        self,
        box: BoxState,
        pallet_code: str | None,
        region_code: str | None,
        location_is_confirmed: bool = False,
        confidence_score: float | None = None,
    ) -> WarehouseEvent | None:
        key = box.box_code

        # A missing region detection is not proof that the object moved.
        # Preserve the last confirmed region while the parent pallet is unchanged.
        if (
            region_code is None
            and key in self.initialized_boxes
            and box.current_pallet_code == pallet_code
        ):
            region_code = box.current_zone_code

        location = (pallet_code, region_code)

        # No detected relationship means "unknown", not a fabricated location.
        if location == (None, None):
            self.box_candidates.pop(key, None)
            return None

        current = (box.current_pallet_code, box.current_zone_code)
        if key in self.initialized_boxes and current == location:
            self.box_candidates.pop(key, None)
            return None

        if location_is_confirmed:
            self.box_candidates.pop(key, None)
            confirmed_confidence = confidence_score
        else:
            confirmed = self._confirmed_candidate(
                self.box_candidates,
                key,
                location,
                confidence_score,
            )
            if confirmed is None:
                return None
            confirmed_confidence = confirmed.confidence_score

        if key not in self.initialized_boxes:
            self._set_box_location(
                box,
                pallet_code,
                region_code,
                confirmed_confidence,
            )
            self.initialized_boxes.add(key)
            if self.frame_number > self.startup_baseline_frames:
                event_region_code = self._event_region_code(region_code)
                destination = self._format_location(
                    pallet_code,
                    event_region_code,
                )
                return WarehouseEvent(
                    event_type=self._transition_event_type(
                        previous_pallet=None,
                        previous_region=None,
                        previous_status="DETECTED_UNLOCATED",
                        current_pallet=pallet_code,
                        current_region=event_region_code,
                    ),
                    box_code=box.box_code,
                    to_pallet_code=pallet_code,
                    to_zone_code=event_region_code,
                    remarks=f"{box.box_code}: detected on {destination}",
                    confidence_score=confirmed_confidence,
                    qr_read_success=box.identity_from_qr,
                )
            return None

        previous_pallet = box.current_pallet_code
        previous_region = box.current_zone_code
        previous_status = box.status
        self._set_box_location(
            box,
            pallet_code,
            region_code,
            confirmed_confidence,
        )
        return self._build_event(
            box,
            previous_pallet,
            previous_region,
            previous_status,
            pallet_code,
            region_code,
            confirmed_confidence,
        )

    def _confirmed_candidate(
        self,
        candidates: dict[str, _Candidate],
        key: str,
        location: Location,
        confidence_score: float | None,
    ) -> _Candidate | None:
        candidate = candidates.get(key)
        if candidate is not None and candidate.location == location:
            candidate.frames += 1
        else:
            candidate = _Candidate(location=location)
            candidates[key] = candidate
        candidate.add_confidence(confidence_score)

        if candidate.frames < self.confirmation_frames:
            return None

        candidates.pop(key, None)
        return candidate

    @staticmethod
    def _set_box_location(
        box: BoxState,
        pallet_code: str | None,
        region_code: str | None,
        confidence_score: float | None,
    ) -> None:
        box.current_pallet_code = pallet_code
        box.current_zone_code = region_code
        box.current_location_confidence_score = confidence_score
        if pallet_code:
            box.status = "ON_PALLET"
        else:
            box.status = "IN_TRANSIT" if region_code else "REMOVED"

        # The six-class detector does not infer physical slot/row/layer values.
        box.slot_no = None
        box.row_no = None
        box.column_no = None
        box.layer_no = None

    def _build_event(
        self,
        box: BoxState,
        previous_pallet: str | None,
        previous_region: str | None,
        previous_status: str,
        current_pallet: str | None,
        current_region: str | None,
        confidence_score: float | None,
    ) -> WarehouseEvent:
        event_previous_region = self._event_region_code(previous_region)
        event_current_region = self._event_region_code(current_region)
        event_type = self._transition_event_type(
            previous_pallet=previous_pallet,
            previous_region=event_previous_region,
            previous_status=previous_status,
            current_pallet=current_pallet,
            current_region=event_current_region,
        )
        source = self._format_location(previous_pallet, event_previous_region)
        destination = self._format_location(current_pallet, event_current_region)
        if (
            event_type == EventType.ADD
            and previous_pallet is None
            and event_previous_region is None
        ):
            remarks = f"{box.box_code}: detected on {destination}"
        else:
            remarks = f"{box.box_code}: {source} -> {destination}"
        return WarehouseEvent(
            event_type=event_type,
            box_code=box.box_code,
            from_pallet_code=previous_pallet,
            to_pallet_code=current_pallet,
            from_zone_code=event_previous_region,
            to_zone_code=event_current_region,
            remarks=remarks,
            confidence_score=confidence_score,
            qr_read_success=box.identity_from_qr,
        )

    @classmethod
    def _transition_event_type(
        cls,
        *,
        previous_pallet: str | None,
        previous_region: str | None,
        previous_status: str,
        current_pallet: str | None,
        current_region: str | None,
    ) -> EventType:
        """Describe the observed transition without guessing a later milestone."""
        if cls._is_dock(current_region) and not cls._is_dock(previous_region):
            return EventType.DOCK_ARRIVAL
        if cls._is_conveyor(current_region) and not cls._is_conveyor(previous_region):
            return EventType.CONVEYOR_ENTRY
        if previous_status == "REMOVED":
            return EventType.RE_ADD
        if current_pallet and previous_pallet != current_pallet:
            return EventType.STACK if previous_pallet is None else EventType.SWAP
        if previous_pallet is not None and current_pallet is None:
            return EventType.REMOVE
        if current_region and previous_region != current_region:
            return EventType.ZONE_TRANSFER
        return EventType.MOVE

    @staticmethod
    def _validated_confidence(value: float) -> float | None:
        """Return a finite detector probability without clamping or guessing."""
        try:
            confidence = float(value)
        except (TypeError, ValueError):
            return None
        if not math.isfinite(confidence) or not 0.0 <= confidence <= 1.0:
            return None
        return confidence

    @classmethod
    def _location_evidence(
        cls,
        *observations: RawObservation | None,
    ) -> float | None:
        confidences = [
            confidence
            for observation in observations
            if observation is not None
            for confidence in [
                cls._validated_confidence(observation.confidence)
            ]
            if confidence is not None
        ]
        return min(confidences) if confidences else None

    @classmethod
    def _event_region_code(cls, region_code: str | None) -> str | None:
        """Keep tracker-only region names out of business events and the DB."""
        return None if cls._is_temporary_identity(region_code) else region_code

    def _identity(
        self,
        observation: RawObservation | None,
        prefix: str,
    ) -> str | None:
        if observation is None:
            return None

        if observation.qr_value is None:
            semantic_region_code = self.semantic_region_codes.get(
                observation.class_name.strip().lower()
            )
            if semantic_region_code:
                return semantic_region_code

        if observation.track_id is None:
            return observation.qr_value

        key = (prefix, observation.track_id)
        temporary_code = f"{prefix}_TRACK_{observation.track_id}"
        existing_code = self.track_identities.get(key)

        if existing_code is None:
            resolved_code = observation.qr_value or temporary_code
            self.track_identities[key] = resolved_code
            return resolved_code

        # Promote the temporary tracking identity when QR becomes readable.
        if existing_code == temporary_code and observation.qr_value:
            resolved_code = observation.qr_value
            self.identity_change_candidates.pop(key, None)
            self._promote_identity(
                prefix=prefix,
                temporary_code=temporary_code,
                resolved_code=resolved_code,
            )
            self.track_identities[key] = resolved_code
            return resolved_code

        if prefix == "BOX":
            observed_code = observation.qr_value
            if not observed_code or observed_code == existing_code:
                self.identity_change_candidates.pop(key, None)
                return existing_code

            # A tracker may remain on the same pallet footprint when another
            # physical box is stacked over the previous one. A stable new QR
            # therefore starts a new box identity; it must not rename or erase
            # the hidden box underneath.
            if self._box_identity_change_confirmed(key, observed_code):
                self.track_identities[key] = observed_code
                return observed_code

        return existing_code

    def _box_identity_change_confirmed(
        self,
        key: tuple[str, int],
        observed_code: str,
    ) -> bool:
        candidate = self.identity_change_candidates.get(key)
        is_consecutive = bool(
            candidate is not None
            and candidate.code == observed_code
            and candidate.last_frame == self.frame_number - 1
        )

        if is_consecutive:
            candidate.frames += 1
            candidate.last_frame = self.frame_number
        else:
            candidate = _IdentityChangeCandidate(
                code=observed_code,
                frames=1,
                last_frame=self.frame_number,
            )
            self.identity_change_candidates[key] = candidate

        if candidate.frames < self.identity_change_confirmation_frames:
            return False

        self.identity_change_candidates.pop(key, None)
        return True

    def _promote_identity(
    self,
    prefix: str,
    temporary_code: str,
    resolved_code: str,
) -> None:
        if temporary_code == resolved_code:
            return

        self.pending_events = [
            _PendingEvent(
                event=self._replace_event_identity(
                    pending.event,
                    temporary_code,
                    resolved_code,
                    box_qr_resolved=(prefix == "BOX"),
                ),
                queued_frame=pending.queued_frame,
            )
            for pending in self.pending_events
        ]

        if prefix == "BOX":
            temporary_state = self.boxes.pop(temporary_code, None)

            if temporary_state is not None:
                temporary_state.box_code = resolved_code
                temporary_state.identity_from_qr = True
                self.boxes.setdefault(resolved_code, temporary_state)

            candidate = self.box_candidates.pop(temporary_code, None)
            if candidate is not None:
                self.box_candidates.setdefault(resolved_code, candidate)

            if temporary_code in self.initialized_boxes:
                self.initialized_boxes.remove(temporary_code)
                self.initialized_boxes.add(resolved_code)

        elif prefix == "PALLET":
            temporary_state = self.pallets.pop(temporary_code, None)

            if temporary_state is not None:
                temporary_state.pallet_code = resolved_code
                self.pallets.setdefault(resolved_code, temporary_state)

            candidate = self.pallet_candidates.pop(temporary_code, None)
            if candidate is not None:
                self.pallet_candidates.setdefault(resolved_code, candidate)

            if temporary_code in self.initialized_pallets:
                self.initialized_pallets.remove(temporary_code)
                self.initialized_pallets.add(resolved_code)

            # Update boxes already associated with the temporary pallet.
            for box in self.boxes.values():
                if box.current_pallet_code == temporary_code:
                    box.current_pallet_code = resolved_code

    @staticmethod
    def _replace_event_identity(
        event: WarehouseEvent,
        temporary_code: str,
        resolved_code: str,
        *,
        box_qr_resolved: bool = False,
    ) -> WarehouseEvent:
        return replace(
            event,
            box_code=(
                resolved_code
                if event.box_code == temporary_code
                else event.box_code
            ),
            from_pallet_code=(
                resolved_code
                if event.from_pallet_code == temporary_code
                else event.from_pallet_code
            ),
            to_pallet_code=(
                resolved_code
                if event.to_pallet_code == temporary_code
                else event.to_pallet_code
            ),
            from_zone_code=(
                resolved_code
                if event.from_zone_code == temporary_code
                else event.from_zone_code
            ),
            to_zone_code=(
                resolved_code
                if event.to_zone_code == temporary_code
                else event.to_zone_code
            ),
            remarks=event.remarks.replace(temporary_code, resolved_code),
            qr_read_success=(
                True
                if box_qr_resolved and event.box_code == temporary_code
                else event.qr_read_success
            ),
        )
    def _region_identity(
        self,
        observation: RawObservation | None,
    ) -> str | None:
        if observation is None:
            return None
        return self._identity(observation, observation.class_name.upper())

    @staticmethod
    def _is_dock(region_code: str | None) -> bool:
        return bool(region_code and region_code.upper().startswith("DOCK"))

    @staticmethod
    def _is_conveyor(region_code: str | None) -> bool:
        return bool(
            region_code and region_code.upper().startswith("CONVEYOR")
        )

    @staticmethod
    def _format_location(
        pallet_code: str | None,
        region_code: str | None,
    ) -> str:
        if pallet_code and region_code:
            return f"{pallet_code} in {region_code}"
        return pallet_code or region_code or "UNKNOWN"
