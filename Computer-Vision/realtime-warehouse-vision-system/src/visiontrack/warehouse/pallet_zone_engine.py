from __future__ import annotations

from dataclasses import dataclass, field, replace
import math

from visiontrack.association.location_resolver import (
    REGION_CLASSES,
    DynamicLocationResolver,
)
from visiontrack.perception.raw_observation import RawObservation
from visiontrack.warehouse.entities import BoxState, PalletState
from visiontrack.warehouse.events import EventType, WarehouseEvent


Location = tuple[str | None, str | None]


# =====================================================================
# INTERNAL STATE
# =====================================================================


@dataclass
class _Candidate:
    location: Location
    frames: int = 1
    confidence_total: float = 0.0
    confidence_samples: int = 0

    def add_confidence(
        self,
        confidence_score: float | None,
    ) -> None:
        if confidence_score is None:
            return

        self.confidence_total += confidence_score
        self.confidence_samples += 1

    @property
    def confidence_score(self) -> float | None:
        if self.confidence_samples == 0:
            return None

        return (
            self.confidence_total
            / self.confidence_samples
        )


@dataclass
class _PendingEvent:
    event: WarehouseEvent
    queued_frame: int


@dataclass
class _IdentityChangeCandidate:
    code: str
    frames: int
    last_frame: int


@dataclass
class _BoxMotionState:
    """
    Short-term motion and floor-inference state for one real box.

    Motion is measured from bbox-center displacement and normalized
    by the box bbox diagonal so movement thresholds are not tied
    directly to raw pixel distances.
    """

    track_id: int | None = None

    # (engine_frame, center_x, center_y, bbox_diagonal)
    samples: list[
        tuple[int, float, float, float]
    ] = field(default_factory=list)

    moving_frames: int = 0
    stationary_frames: int = 0
    is_moving: bool = False

    last_displacement_ratio: float | None = None

    # Consecutive frames where all positive floor conditions hold.
    floor_candidate_frames: int = 0


# =====================================================================
# PALLET / ZONE ENGINE
# =====================================================================


class PalletZoneEngine:
    """
    Maintain warehouse box/pallet state and produce location events
    from tracked YOLO observations.

    Important separation:

    REGISTERED
        Confirms that a real box participated in this video run.

    ADD/MOVE/REMOVE/RE_ADD/DISPATCH
        Describe warehouse location transitions.

    IN_TRANSIT / DETECTED_UNLOCATED / ON_FLOOR_INFERRED
        Are CV state classifications and are not fabricated as events.
    """

    def __init__(
        self,
        config: dict,
    ):
        dynamic = config.get(
            "dynamic_location",
            {},
        )
        self.record_region_arrivals = dynamic.get("record_region_arrivals", False)

        # -------------------------------------------------------------
        # LOCATION CONFIRMATION
        # -------------------------------------------------------------

        self.confirmation_frames = max(
            1,
            int(
                dynamic.get(
                    "confirmation_frames",
                    5,
                )
            ),
        )

        self.startup_baseline_frames = max(
            self.confirmation_frames,
            int(
                dynamic.get(
                    "startup_baseline_frames",
                    30,
                )
            ),
        )

        self.identity_resolution_timeout_frames = max(
            1,
            int(
                dynamic.get(
                    "identity_resolution_timeout_frames",
                    300,
                )
            ),
        )

        self.identity_change_confirmation_frames = max(
            2,
            int(
                dynamic.get(
                    "identity_change_confirmation_frames",
                    3,
                )
            ),
        )

        # -------------------------------------------------------------
        # MOTION SETTINGS
        # -------------------------------------------------------------

        self.motion_window_frames = max(
            2,
            int(
                dynamic.get(
                    "motion_window_frames",
                    5,
                )
            ),
        )

        self.motion_enter_ratio = max(
            0.0,
            float(
                dynamic.get(
                    "motion_enter_ratio",
                    0.10,
                )
            ),
        )

        self.motion_exit_ratio = max(
            0.0,
            float(
                dynamic.get(
                    "motion_exit_ratio",
                    0.03,
                )
            ),
        )

        self.motion_confirmation_frames = max(
            1,
            int(
                dynamic.get(
                    "motion_confirmation_frames",
                    3,
                )
            ),
        )

        self.stationary_confirmation_frames = max(
            1,
            int(
                dynamic.get(
                    "stationary_confirmation_frames",
                    8,
                )
            ),
        )

        if (
            self.motion_exit_ratio
            >= self.motion_enter_ratio
        ):
            raise ValueError(
                "motion_exit_ratio must be smaller "
                "than motion_enter_ratio"
            )

        # -------------------------------------------------------------
        # FLOOR INFERENCE SETTINGS
        # -------------------------------------------------------------

        floor_config = dynamic.get(
            "floor_inference",
            {},
        )

        self.floor_inference_enabled = bool(
            floor_config.get(
                "enabled",
                False,
            )
        )

        self.floor_reference_frame_width = max(
            1,
            int(
                floor_config.get(
                    "reference_frame_width",
                    3840,
                )
            ),
        )

        self.floor_reference_frame_height = max(
            1,
            int(
                floor_config.get(
                    "reference_frame_height",
                    2160,
                )
            ),
        )

        self.floor_confirmation_frames = max(
            1,
            int(
                floor_config.get(
                    "confirmation_frames",
                    30,
                )
            ),
        )

        self.floor_polygons: list[
            tuple[
                tuple[float, float],
                ...
            ]
        ] = []

        for polygon in floor_config.get(
            "polygons",
            [],
        ):
            parsed_points: list[
                tuple[float, float]
            ] = []

            for point in polygon:

                if (
                    not isinstance(
                        point,
                        (list, tuple),
                    )
                    or len(point) != 2
                ):
                    raise ValueError(
                        "Each floor polygon point must "
                        "be [x_ratio, y_ratio]"
                    )

                x_ratio = float(
                    point[0]
                )

                y_ratio = float(
                    point[1]
                )

                if not (
                    0.0 <= x_ratio <= 1.0
                    and
                    0.0 <= y_ratio <= 1.0
                ):
                    raise ValueError(
                        "Floor polygon coordinates must "
                        "be between 0.0 and 1.0"
                    )

                parsed_points.append(
                    (
                        x_ratio,
                        y_ratio,
                    )
                )

            if len(parsed_points) < 3:
                raise ValueError(
                    "Each floor polygon must contain "
                    "at least 3 points"
                )

            self.floor_polygons.append(
                tuple(
                    parsed_points
                )
            )

        if (
            self.floor_inference_enabled
            and not self.floor_polygons
        ):
            raise ValueError(
                "floor_inference is enabled but "
                "no polygons are configured"
            )

        # -------------------------------------------------------------
        # SEMANTIC REGION MAPPING
        # -------------------------------------------------------------

        self.semantic_region_codes = {
            str(class_name).strip().lower():
            str(region_code).strip()

            for class_name, region_code
            in dynamic.get(
                "semantic_region_codes",
                {
                    "conveyor_belt":
                    "conveyor_belt",
                },
            ).items()

            if str(class_name).strip()
            and str(region_code).strip()
        }

        # -------------------------------------------------------------
        # LOCATION RESOLVER
        # -------------------------------------------------------------

        self.location_resolver = (
            DynamicLocationResolver(
                min_pallet_overlap_ratio=float(
                    dynamic.get(
                        "min_pallet_overlap_ratio",
                        0.10,
                    )
                ),
                min_region_overlap_ratio=float(
                    dynamic.get(
                        "min_region_overlap_ratio",
                        0.50,
                    )
                ),
            )
        )

        # -------------------------------------------------------------
        # CAPACITY
        # -------------------------------------------------------------

        default_pallet = (
            config.get(
                "capacity",
                {},
            ).get(
                "default_pallet",
                {},
            )
        )

        self.default_pallet_max_boxes = max(
            1,
            int(
                default_pallet.get(
                    "max_boxes",
                    18,
                )
            ),
        )

        # -------------------------------------------------------------
        # WAREHOUSE STATE
        # -------------------------------------------------------------

        self.boxes: dict[
            str,
            BoxState,
        ] = {}

        self.pallets: dict[
            str,
            PalletState,
        ] = {}

        # -------------------------------------------------------------
        # LOCATION CANDIDATES
        # -------------------------------------------------------------

        self.box_candidates: dict[
            str,
            _Candidate,
        ] = {}

        self.pallet_candidates: dict[
            str,
            _Candidate,
        ] = {}

        # -------------------------------------------------------------
        # REGISTRATION
        # -------------------------------------------------------------

        self.box_registration_candidates: dict[
            str,
            _Candidate,
        ] = {}

        self.registered_boxes: set[str] = set()

        # -------------------------------------------------------------
        # MOTION / FLOOR STATE
        # -------------------------------------------------------------

        self.box_motion_states: dict[
            str,
            _BoxMotionState,
        ] = {}

        # -------------------------------------------------------------
        # INITIALIZED LOCATIONS
        # -------------------------------------------------------------

        self.initialized_boxes: set[str] = set()
        self.initialized_pallets: set[str] = set()

        # -------------------------------------------------------------
        # IDENTITY
        # -------------------------------------------------------------

        self.track_identities: dict[
            tuple[str, int],
            str,
        ] = {}

        self.identity_change_candidates: dict[
            tuple[str, int],
            _IdentityChangeCandidate,
        ] = {}

        # -------------------------------------------------------------
        # PENDING EVENTS
        # -------------------------------------------------------------

        self.pending_events: list[
            _PendingEvent
        ] = []

        self.dropped_unresolved_event_count = 0

        self.frame_number = 0

    # =================================================================
    # FRAME UPDATE
    # =================================================================

    def update(
        self,
        observations: list[RawObservation],
        *,
        blocked_identities: set[str] | None = None,
    ) -> list[WarehouseEvent]:

        self.frame_number += 1
        blocked_identities = blocked_identities or set()

        events: list[WarehouseEvent] = []

        boxes = [
            item
            for item in observations
            if item.class_name == "box"
        ]

        pallets = [
            item
            for item in observations
            if item.class_name == "pallet"
        ]

        regions = [
            item
            for item in observations
            if item.class_name in REGION_CLASSES
        ]

        # -------------------------------------------------------------
        # PALLETS
        # -------------------------------------------------------------

        for state in self.pallets.values():
            state.visible = False

        visible_pallets: set[str] = set()

        for observation in pallets:

            pallet_code = self._identity(
                observation,
                "PALLET",
            )

            if pallet_code is None:
                continue

            visible_pallets.add(
                pallet_code
            )

            state = self.pallets.setdefault(
                pallet_code,
                PalletState(
                    pallet_code=pallet_code,
                    max_boxes=(
                        self.default_pallet_max_boxes
                    ),
                ),
            )

            state.visible = True
            state.status = "IN_USE"

            resolved = (
                self.location_resolver.resolve_pallet(
                    observation,
                    regions,
                )
            )

            region_code = self._region_identity(
                resolved.region
            )

            self._consider_pallet_region(
                state,
                region_code,
                self._location_evidence(
                    observation,
                    resolved.region,
                ),
            )

        # Remove incomplete pallet candidates when pallet disappears.

        for pallet_code in list(
            self.pallet_candidates
        ):

            if pallet_code not in visible_pallets:
                self.pallet_candidates.pop(
                    pallet_code,
                    None,
                )

        # -------------------------------------------------------------
        # BOXES
        # -------------------------------------------------------------

        visible_boxes: set[str] = set()

        for observation in boxes:

            box_code = self._identity(
                observation,
                "BOX",
            )

            if box_code is None:
                continue

            visible_boxes.add(
                box_code
            )

            state = self.boxes.setdefault(
                box_code,
                BoxState(
                    box_code=box_code,
                ),
            )

            state.missing_frames = 0

            if state.current_pallet_code in blocked_identities:
                self.box_candidates.pop(box_code, None)
                continue

            if (
                observation.qr_read_success
                and observation.qr_value
                == box_code
            ):
                state.identity_from_qr = True

            # ---------------------------------------------------------
            # RESOLVE LOCATION
            # ---------------------------------------------------------

            resolved = (
                self.location_resolver.resolve_box(
                    observation,
                    pallets,
                    regions,
                )
            )

            pallet_code = self._identity(
                resolved.pallet,
                "PALLET",
            )

            region_code = self._region_identity(
                resolved.region
            )

            # ---------------------------------------------------------
            # MOTION
            # ---------------------------------------------------------

            box_is_moving = (
                self._update_box_motion(
                    box_code,
                    observation,
                )
            )

            # ---------------------------------------------------------
            # EXISTING PALLET / REGION LOCATION LOGIC
            # ---------------------------------------------------------

            location_event = (
                self._consider_box_location(
                    state,
                    pallet_code,
                    region_code,
                    confidence_score=(
                        self._location_evidence(
                            observation,
                            resolved.pallet,
                            resolved.region,
                        )
                    ),
                )
            )

            # ---------------------------------------------------------
            # UNASSOCIATED BOX FALLBACK
            # ---------------------------------------------------------
            #
            # Only use motion/floor inference when BOTH:
            #
            # 1. Current observation has no relationship.
            # 2. Box does not already hold a confirmed relationship.
            #
            # This prevents one temporary detection dropout from
            # overriding a confirmed ON_PALLET / zone relationship.
            # ---------------------------------------------------------

            if (
                pallet_code is None
                and region_code is None
                and state.current_pallet_code is None
                and state.current_zone_code is None
            ):

                self._apply_unassociated_box_state(
                    state,
                    observation,
                    box_is_moving,
                )

            else:

                # Stronger physical/location evidence exists.
                # Any previous floor candidate is invalid now.

                motion_state = (
                    self.box_motion_states.get(
                        box_code
                    )
                )

                if motion_state is not None:
                    motion_state.floor_candidate_frames = 0

            # ---------------------------------------------------------
            # ONE-TIME BOX REGISTRATION
            # ---------------------------------------------------------

            registration_event = (
                self._consider_box_registration(
                    state,
                    observation,
                )
            )

            # REGISTERED should appear before ADD/MOVE/etc. when both
            # become confirmed on exactly the same frame.

            self._queue_or_emit(
                registration_event,
                events,
            )

            self._queue_or_emit(
                location_event,
                events,
            )

        # -------------------------------------------------------------
        # BOXES NOT DIRECTLY VISIBLE THIS FRAME
        # -------------------------------------------------------------

        for box_code, state in self.boxes.items():

            if box_code in visible_boxes:
                continue

            if box_code in blocked_identities or state.current_pallet_code in blocked_identities:
                self.box_candidates.pop(box_code, None)
                self.box_registration_candidates.pop(box_code, None)
                continue

            # Registration requires consecutive direct detections.

            self.box_registration_candidates.pop(
                box_code,
                None,
            )

            # Missing direct detection invalidates current floor
            # confirmation accumulation.

            motion_state = (
                self.box_motion_states.get(
                    box_code
                )
            )

            if motion_state is not None:
                motion_state.floor_candidate_frames = 0

            pallet_code = (
                state.current_pallet_code
            )

            if (
                pallet_code
                and pallet_code in visible_pallets
            ):
                # Hidden box follows its still-visible confirmed
                # parent pallet.

                state.missing_frames = 0

                event = (
                    self._consider_box_location(
                        state,
                        pallet_code,
                        self.pallets[
                            pallet_code
                        ].current_zone_code,
                        location_is_confirmed=True,
                        confidence_score=(
                            self.pallets[
                                pallet_code
                            ]
                            .current_location_confidence_score
                        ),
                    )
                )

                self._queue_or_emit(
                    event,
                    events,
                )

                continue

            self.box_candidates.pop(
                box_code,
                None,
            )

            state.missing_frames += 1

        # -------------------------------------------------------------
        # RELEASE / EXPIRE TEMPORARY-ID EVENTS
        # -------------------------------------------------------------

        events.extend(self._release_resolved_pending_events())
        if blocked_identities:
            events = [
                event for event in events
                if not blocked_identities.intersection(
                    (event.box_code, event.from_pallet_code, event.to_pallet_code)
                )
            ]

        self._expire_unresolved_pending_events()

        return events

    # =================================================================
    # MOTION
    # =================================================================

    def _update_box_motion(
        self,
        box_code: str,
        observation: RawObservation,
    ) -> bool:
        """
        Update geometric movement state.

        True is returned only after sustained movement evidence has
        satisfied motion_confirmation_frames.
        """

        # Never build permanent motion state around BOX_TRACK_x.

        if self._is_temporary_identity(
            box_code
        ):
            return False

        x1, y1, x2, y2 = observation.bbox

        width = max(
            float(x2 - x1),
            1.0,
        )

        height = max(
            float(y2 - y1),
            1.0,
        )

        center_x = (
            float(x1)
            + float(x2)
        ) / 2.0

        center_y = (
            float(y1)
            + float(y2)
        ) / 2.0

        bbox_diagonal = math.hypot(
            width,
            height,
        )

        state = self.box_motion_states.setdefault(
            box_code,
            _BoxMotionState(),
        )

        # -------------------------------------------------------------
        # TRACKER CHANGE
        # -------------------------------------------------------------

        if (
            state.track_id is not None
            and observation.track_id is not None
            and state.track_id
            != observation.track_id
        ):
            self._reset_motion_state(
                state
            )

        state.track_id = (
            observation.track_id
        )

        # -------------------------------------------------------------
        # OBSERVATION GAP
        # -------------------------------------------------------------
        #
        # Do not calculate apparent movement across missing frames.
        # -------------------------------------------------------------

        if (
            state.samples
            and (
                self.frame_number
                - state.samples[-1][0]
                > 1
            )
        ):

            self._reset_motion_state(
                state
            )

            state.track_id = (
                observation.track_id
            )

        # -------------------------------------------------------------
        # ADD CURRENT SAMPLE
        # -------------------------------------------------------------

        state.samples.append(
            (
                self.frame_number,
                center_x,
                center_y,
                bbox_diagonal,
            )
        )

        if (
            len(state.samples)
            > self.motion_window_frames
        ):
            state.samples.pop(
                0
            )

        # Need complete time window before classifying motion.

        if (
            len(state.samples)
            < self.motion_window_frames
        ):
            return state.is_moving

        (
            _,
            old_x,
            old_y,
            old_diagonal,
        ) = state.samples[0]

        (
            _,
            new_x,
            new_y,
            new_diagonal,
        ) = state.samples[-1]

        displacement = math.hypot(
            new_x - old_x,
            new_y - old_y,
        )

        reference_size = max(
            (
                old_diagonal
                + new_diagonal
            )
            / 2.0,
            1.0,
        )

        displacement_ratio = (
            displacement
            / reference_size
        )

        state.last_displacement_ratio = (
            displacement_ratio
        )

        # -------------------------------------------------------------
        # MOVING EVIDENCE
        # -------------------------------------------------------------

        if (
            displacement_ratio
            >= self.motion_enter_ratio
        ):

            state.moving_frames += 1
            state.stationary_frames = 0
            state.floor_candidate_frames = 0

            if (
                state.moving_frames
                >= self.motion_confirmation_frames
            ):
                state.is_moving = True

        # -------------------------------------------------------------
        # STATIONARY EVIDENCE
        # -------------------------------------------------------------

        elif (
            displacement_ratio
            <= self.motion_exit_ratio
        ):

            state.stationary_frames += 1
            state.moving_frames = 0

            if (
                state.stationary_frames
                >= self.stationary_confirmation_frames
            ):
                state.is_moving = False

        # -------------------------------------------------------------
        # HYSTERESIS / UNCERTAIN BAND
        # -------------------------------------------------------------
        #
        # Neither enough movement nor enough stability.
        # Do not flip the existing motion state.
        # -------------------------------------------------------------

        else:

            state.moving_frames = 0
            state.stationary_frames = 0
            state.floor_candidate_frames = 0

        return state.is_moving

    @staticmethod
    def _reset_motion_state(
        state: _BoxMotionState,
    ) -> None:

        state.samples.clear()

        state.moving_frames = 0
        state.stationary_frames = 0

        state.is_moving = False

        state.last_displacement_ratio = None

        state.floor_candidate_frames = 0

    # =================================================================
    # UNASSOCIATED BOX STATE
    # =================================================================

    def _apply_unassociated_box_state(
        self,
        box: BoxState,
        observation: RawObservation,
        box_is_moving: bool,
    ) -> None:
        """
        Resolve a directly visible box that currently has no confirmed
        pallet, conveyor, zone, or dock relationship.

        Priority:

        confirmed moving
            -> IN_TRANSIT

        stationary + inside approved floor ROI + sustained evidence
            -> ON_FLOOR_INFERRED

        insufficient evidence
            -> DETECTED_UNLOCATED
        """

        motion_state = (
            self.box_motion_states.get(
                box.box_code
            )
        )

        if motion_state is None:

            box.status = (
                "DETECTED_UNLOCATED"
            )

            return

        # -------------------------------------------------------------
        # MOVEMENT HAS PRIORITY
        # -------------------------------------------------------------

        if box_is_moving:

            motion_state.floor_candidate_frames = 0

            box.status = (
                "IN_TRANSIT"
            )

            return

        # -------------------------------------------------------------
        # FLOOR GEOMETRY
        # -------------------------------------------------------------

        inside_floor = (
            self._box_center_in_floor_roi(
                observation
            )
        )

        # -------------------------------------------------------------
        # PRESERVE ALREADY-CONFIRMED FLOOR STATE
        # -------------------------------------------------------------
        #
        # Do not repeatedly reconfirm the same stationary box.
        #
        # This remains valid only while:
        #
        # - no stronger association exists
        # - positive movement is not detected
        # - box remains inside configured floor ROI
        # -------------------------------------------------------------

        if (
            box.status
            == "ON_FLOOR"
            and inside_floor
        ):

            motion_state.floor_candidate_frames = (
                self.floor_confirmation_frames
            )

            return

        # -------------------------------------------------------------
        # STATIONARY CONFIRMATION
        # -------------------------------------------------------------

        stationary_confirmed = (
            motion_state.stationary_frames
            >= self.stationary_confirmation_frames
        )

        if (
            self.floor_inference_enabled
            and inside_floor
            and stationary_confirmed
        ):

            motion_state.floor_candidate_frames += 1

            if (
                motion_state.floor_candidate_frames
                >= self.floor_confirmation_frames
            ):

                box.status = (
                    "ON_FLOOR"
                )

            else:

                box.status = (
                    "DETECTED_UNLOCATED"
                )

            return

        # -------------------------------------------------------------
        # NO SUFFICIENT POSITIVE FLOOR EVIDENCE
        # -------------------------------------------------------------

        motion_state.floor_candidate_frames = 0

        box.status = (
            "DETECTED_UNLOCATED"
        )

    # =================================================================
    # FLOOR ROI
    # =================================================================

    def _box_center_in_floor_roi(
        self,
        observation: RawObservation,
    ) -> bool:
        """
        Return True only when the center of the box bbox lies inside
        one of the explicitly configured normalized floor polygons.
        """

        if (
            not self.floor_inference_enabled
            or not self.floor_polygons
        ):
            return False

        x1, y1, x2, y2 = observation.bbox

        center_x = (
            float(x1)
            + float(x2)
        ) / 2.0

        center_y = (
            float(y1)
            + float(y2)
        ) / 2.0

        normalized_x = (
            center_x
            / float(
                self.floor_reference_frame_width
            )
        )

        normalized_y = (
            center_y
            / float(
                self.floor_reference_frame_height
            )
        )

        # Bbox should correspond to reference-frame coordinates.
        # Reject impossible values rather than guessing.

        if not (
            0.0
            <= normalized_x
            <= 1.0
            and
            0.0
            <= normalized_y
            <= 1.0
        ):
            return False

        return any(
            self._point_in_polygon(
                normalized_x,
                normalized_y,
                polygon,
            )
            for polygon
            in self.floor_polygons
        )

    @staticmethod
    def _point_in_polygon(
        x: float,
        y: float,
        polygon: tuple[
            tuple[float, float],
            ...
        ],
    ) -> bool:
        """
        Ray-casting point-in-polygon test.

        A point exactly on the configured polygon boundary is treated
        as being inside.
        """

        inside = False

        count = len(
            polygon
        )

        j = count - 1

        epsilon = 1e-9

        for i in range(
            count
        ):

            xi, yi = polygon[i]
            xj, yj = polygon[j]

            # ---------------------------------------------------------
            # BOUNDARY CHECK
            # ---------------------------------------------------------

            dx = xj - xi
            dy = yj - yi

            cross = (
                (x - xi) * dy
                - (y - yi) * dx
            )

            if (
                abs(cross)
                <= epsilon
            ):

                if (
                    min(xi, xj)
                    - epsilon
                    <= x
                    <= max(xi, xj)
                    + epsilon
                    and
                    min(yi, yj)
                    - epsilon
                    <= y
                    <= max(yi, yj)
                    + epsilon
                ):

                    return True

            # ---------------------------------------------------------
            # RAY CAST
            # ---------------------------------------------------------

            intersects = (
                (yi > y)
                != (yj > y)
            )

            if intersects:

                intersection_x = (
                    (xj - xi)
                    * (y - yi)
                    / (yj - yi)
                    + xi
                )

                if (
                    x
                    < intersection_x
                ):
                    inside = (
                        not inside
                    )

            j = i

        return inside

    # =================================================================
    # EVENT QUEUE
    # =================================================================

    def _queue_or_emit(
        self,
        event: WarehouseEvent | None,
        emitted: list[WarehouseEvent],
    ) -> None:

        if event is None:
            return

        if self._event_has_temporary_identity(
            event
        ):

            self.pending_events.append(
                _PendingEvent(
                    event=event,
                    queued_frame=self.frame_number,
                )
            )

            return

        emitted.append(
            event
        )

    def _release_resolved_pending_events(
        self,
    ) -> list[WarehouseEvent]:

        ready: list[
            WarehouseEvent
        ] = []

        waiting: list[
            _PendingEvent
        ] = []

        for pending in self.pending_events:

            if self._event_has_temporary_identity(
                pending.event
            ):

                waiting.append(
                    pending
                )

            else:

                ready.append(
                    pending.event
                )

        self.pending_events = (
            waiting
        )

        return ready

    def _expire_unresolved_pending_events(
        self,
    ) -> None:

        cutoff = (
            self.frame_number
            - self.identity_resolution_timeout_frames
        )

        waiting: list[
            _PendingEvent
        ] = []

        for pending in self.pending_events:

            if (
                pending.queued_frame
                <= cutoff
            ):

                self.dropped_unresolved_event_count += 1

            else:

                waiting.append(
                    pending
                )

        self.pending_events = (
            waiting
        )

    @classmethod
    def _event_has_temporary_identity(
        cls,
        event: WarehouseEvent,
    ) -> bool:

        return any(
            cls._is_temporary_identity(
                value
            )
            for value in (
                event.box_code,
                event.from_pallet_code,
                event.to_pallet_code,
                event.from_zone_code,
                event.to_zone_code,
            )
        )

    @staticmethod
    def _is_temporary_identity(
        value: str | None,
    ) -> bool:

        return bool(
            value
            and "_TRACK_"
            in value.upper()
        )

    # =================================================================
    # PALLET REGION
    # =================================================================

    def _consider_pallet_region(
        self,
        pallet: PalletState,
        region_code: str | None,
        confidence_score: float | None,
    ) -> None:

        key = (
            pallet.pallet_code
        )

        if region_code is None:

            self.pallet_candidates.pop(
                key,
                None,
            )

            return

        if (
            key
            in self.initialized_pallets
            and pallet.current_zone_code
            == region_code
        ):

            self.pallet_candidates.pop(
                key,
                None,
            )

            return

        confirmed = (
            self._confirmed_candidate(
                self.pallet_candidates,
                key,
                (
                    None,
                    region_code,
                ),
                confidence_score,
            )
        )

        if confirmed is None:
            return

        pallet.current_zone_code = (
            region_code
        )

        pallet.current_location_confidence_score = (
            confirmed.confidence_score
        )

        self.initialized_pallets.add(
            key
        )

    # =================================================================
    # BOX REGISTRATION
    # =================================================================

    def _consider_box_registration(
        self,
        box: BoxState,
        observation: RawObservation,
    ) -> WarehouseEvent | None:

        key = (
            box.box_code
        )

        # Never permanently register tracker-only identities.

        if self._is_temporary_identity(
            key
        ):

            self.box_registration_candidates.pop(
                key,
                None,
            )

            return None

        # REGISTERED exactly once per real box per run.

        if (
            key
            in self.registered_boxes
        ):

            self.box_registration_candidates.pop(
                key,
                None,
            )

            return None

        confirmed = (
            self._confirmed_candidate(
                self.box_registration_candidates,
                key,
                (
                    None,
                    None,
                ),
                self._validated_confidence(
                    observation.confidence
                ),
            )
        )

        if confirmed is None:
            return None

        self.registered_boxes.add(
            key
        )

        # REGISTERED must not depend on unresolved tracker-only
        # pallet/region identities.

        event_pallet_code = (
            None
            if self._is_temporary_identity(
                box.current_pallet_code
            )
            else box.current_pallet_code
        )

        event_region_code = (
            self._event_region_code(
                box.current_zone_code
            )
        )

        if (
            event_pallet_code
            or event_region_code
        ):

            location = (
                self._format_location(
                    event_pallet_code,
                    event_region_code,
                )
            )

            remarks = (
                f"{box.box_code}: "
                f"registered at {location}"
            )

        else:

            remarks = (
                f"{box.box_code}: "
                "registered with no confirmed location"
            )

        return WarehouseEvent(
            event_type=(
                EventType.REGISTERED
            ),
            box_code=(
                box.box_code
            ),
            to_pallet_code=(
                event_pallet_code
            ),
            to_zone_code=(
                event_region_code
            ),
            remarks=remarks,
            confidence_score=(
                confirmed.confidence_score
            ),
            qr_read_success=(
                box.identity_from_qr
            ),
        )

    # =================================================================
    # BOX LOCATION
    # =================================================================

    def _consider_box_location(
        self,
        box: BoxState,
        pallet_code: str | None,
        region_code: str | None,
        location_is_confirmed: bool = False,
        confidence_score: float | None = None,
    ) -> WarehouseEvent | None:

        key = (
            box.box_code
        )

        # Missing region detection alone does not prove a move.
        # Preserve previous region while parent pallet remains unchanged.

        if (
            region_code is None
            and key
            in self.initialized_boxes
            and box.current_pallet_code
            == pallet_code
        ):

            region_code = (
                box.current_zone_code
            )

        location = (
            pallet_code,
            region_code,
        )

        # No detected relationship means unknown.
        # Never fabricate a floor relationship here.

        if (
            location
            == (
                None,
                None,
            )
        ):

            self.box_candidates.pop(
                key,
                None,
            )

            return None

        current = (
            box.current_pallet_code,
            box.current_zone_code,
        )

        if (
            key
            in self.initialized_boxes
            and current
            == location
        ):

            self.box_candidates.pop(
                key,
                None,
            )

            return None

        # -------------------------------------------------------------
        # CONFIRM LOCATION
        # -------------------------------------------------------------

        if location_is_confirmed:

            self.box_candidates.pop(
                key,
                None,
            )

            confirmed_confidence = (
                confidence_score
            )

        else:

            confirmed = (
                self._confirmed_candidate(
                    self.box_candidates,
                    key,
                    location,
                    confidence_score,
                )
            )

            if confirmed is None:
                return None

            confirmed_confidence = (
                confirmed.confidence_score
            )

        # -------------------------------------------------------------
        # FIRST CONFIRMED LOCATION
        # -------------------------------------------------------------

        if (
            key
            not in self.initialized_boxes
        ):

            self._set_box_location(
                box,
                pallet_code,
                region_code,
                confirmed_confidence,
            )

            self.initialized_boxes.add(
                key
            )

            # During startup baseline, establish state only.
            #
            # After startup, first confirmed pallet association
            # remains an ADD event.

            if (
                self.frame_number
                > self.startup_baseline_frames
                and pallet_code
                is not None
            ):

                event_region_code = (
                    self._event_region_code(
                        region_code
                    )
                )

                destination = (
                    self._format_location(
                        pallet_code,
                        event_region_code,
                    )
                )

                return WarehouseEvent(
                    event_type=(
                        EventType.ADD
                    ),
                    box_code=(
                        box.box_code
                    ),
                    to_pallet_code=(
                        pallet_code
                    ),
                    to_zone_code=(
                        event_region_code
                    ),
                    remarks=(
                        f"{box.box_code}: "
                        f"detected on {destination}"
                    ),
                    confidence_score=(
                        confirmed_confidence
                    ),
                    qr_read_success=(
                        box.identity_from_qr
                    ),
                )

            return None

        # -------------------------------------------------------------
        # CONFIRMED LOCATION TRANSITION
        # -------------------------------------------------------------

        previous_pallet = (
            box.current_pallet_code
        )

        previous_region = (
            box.current_zone_code
        )

        previous_status = (
            box.status
        )

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

    # =================================================================
    # CANDIDATE CONFIRMATION
    # =================================================================

    def _confirmed_candidate(
        self,
        candidates: dict[
            str,
            _Candidate,
        ],
        key: str,
        location: Location,
        confidence_score: float | None,
    ) -> _Candidate | None:

        candidate = (
            candidates.get(
                key
            )
        )

        if (
            candidate is not None
            and candidate.location
            == location
        ):

            candidate.frames += 1

        else:

            candidate = (
                _Candidate(
                    location=location
                )
            )

            candidates[
                key
            ] = candidate

        candidate.add_confidence(
            confidence_score
        )

        if (
            candidate.frames
            < self.confirmation_frames
        ):

            return None

        candidates.pop(
            key,
            None,
        )

        return candidate

    # =================================================================
    # BOX LOCATION STATE
    # =================================================================
    @staticmethod
    def _set_box_location(
        box: BoxState,
        pallet_code: str | None,
        region_code: str | None,
        confidence_score: float | None,
    ) -> None:
        """
        Store a confirmed physical/location relationship.

        State priority:

        pallet       -> ON_PALLET
        conveyor     -> ON_CONVEYOR
        dock         -> AT_DOCK
        warehouse zone -> IN_ZONE

        IN_TRANSIT is intentionally NOT assigned here.
        It is reserved for the motion-based fallback when a moving box
        has no stronger confirmed pallet/region relationship.
        """

        box.current_pallet_code = pallet_code
        box.current_zone_code = region_code
        box.current_location_confidence_score = confidence_score

        if pallet_code is not None:
            box.status = "ON_PALLET"

        elif region_code is not None:
            normalized_region = region_code.strip().upper()

            if normalized_region.startswith("CONVEYOR"):
                box.status = "ON_CONVEYOR"

            elif normalized_region.startswith("DOCK"):
                box.status = "AT_DOCK"

            elif normalized_region.startswith("ZONE"):
                box.status = "IN_ZONE"

            else:
                # A region relationship exists, but its business meaning
                # is not recognized. Do not invent a physical state.
                box.status = "DETECTED_UNLOCATED"

        else:
            # Normally this branch is not reached because (None, None)
            # is rejected by _consider_box_location().
            box.status = "DETECTED_UNLOCATED"

        # The current detector does not infer physical slot geometry.
        box.slot_no = None
        box.row_no = None
        box.column_no = None
        box.layer_no = None

    # =================================================================
    # LOCATION EVENT BUILDER
    # =================================================================

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

        entered_dock = (
            self._is_dock(
                current_region
            )
            and not self._is_dock(
                previous_region
            )
        )

        if entered_dock:

            event_type = (
                EventType.DOCK_ARRIVAL if self.record_region_arrivals else EventType.DISPATCH
            )

        elif (
            self.record_region_arrivals
            and current_region
            and current_region != previous_region
            and current_region.upper().startswith("CONVEYOR")
        ):
            event_type = EventType.CONVEYOR_ENTRY

        elif (
            self.record_region_arrivals
            and current_region
            and current_region != previous_region
            and current_region.upper().startswith("ZONE")
        ):
            event_type = EventType.ZONE_TRANSFER

        elif (
            previous_status
            == "REMOVED"
        ):

            event_type = (
                EventType.RE_ADD
            )

        elif (
            previous_pallet is None
            and current_pallet
            is not None
        ):

            event_type = (
                EventType.ADD
            )

        elif (
            previous_pallet is not None
            and current_pallet
            is None
        ):

            event_type = (
                EventType.REMOVE
            )

        else:

            event_type = (
                EventType.MOVE
            )

        event_previous_region = (
            self._event_region_code(
                previous_region
            )
        )

        event_current_region = (
            self._event_region_code(
                current_region
            )
        )

        source = (
            self._format_location(
                previous_pallet,
                event_previous_region,
            )
        )

        destination = (
            self._format_location(
                current_pallet,
                event_current_region,
            )
        )

        if (
            event_type
            == EventType.ADD
            and previous_pallet
            is None
            and event_previous_region
            is None
        ):

            remarks = (
                f"{box.box_code}: "
                f"detected on {destination}"
            )

        else:

            remarks = (
                f"{box.box_code}: "
                f"{source} -> {destination}"
            )

        return WarehouseEvent(
            event_type=event_type,
            box_code=(
                box.box_code
            ),
            from_pallet_code=(
                previous_pallet
            ),
            to_pallet_code=(
                current_pallet
            ),
            from_zone_code=(
                event_previous_region
            ),
            to_zone_code=(
                event_current_region
            ),
            remarks=remarks,
            confidence_score=(
                confidence_score
            ),
            qr_read_success=(
                box.identity_from_qr
            ),
        )

    # =================================================================
    # CONFIDENCE
    # =================================================================

    @staticmethod
    def _validated_confidence(
        value: float,
    ) -> float | None:

        try:

            confidence = float(
                value
            )

        except (
            TypeError,
            ValueError,
        ):

            return None

        if (
            not math.isfinite(
                confidence
            )
            or not (
                0.0
                <= confidence
                <= 1.0
            )
        ):

            return None

        return confidence

    @classmethod
    def _location_evidence(
        cls,
        *observations: RawObservation | None,
    ) -> float | None:

        confidences = [
            confidence

            for observation
            in observations

            if observation is not None

            for confidence in [
                cls._validated_confidence(
                    observation.confidence
                )
            ]

            if confidence is not None
        ]

        return (
            min(
                confidences
            )
            if confidences
            else None
        )

    # =================================================================
    # EVENT REGION SANITIZATION
    # =================================================================

    @classmethod
    def _event_region_code(
        cls,
        region_code: str | None,
    ) -> str | None:

        return (
            None

            if cls._is_temporary_identity(
                region_code
            )

            else region_code
        )

    # =================================================================
    # IDENTITY
    # =================================================================

    def _identity(
        self,
        observation: RawObservation | None,
        prefix: str,
    ) -> str | None:

        if observation is None:
            return None

        # Semantic classes such as conveyor may have a configured
        # warehouse identity even without a QR.

        if observation.qr_value is None:

            semantic_region_code = (
                self.semantic_region_codes.get(
                    observation.class_name
                    .strip()
                    .lower()
                )
            )

            if semantic_region_code:
                return (
                    semantic_region_code
                )

        if observation.track_id is None:
            return (
                observation.qr_value
            )

        key = (
            prefix,
            observation.track_id,
        )

        temporary_code = (
            f"{prefix}_TRACK_"
            f"{observation.track_id}"
        )

        existing_code = (
            self.track_identities.get(
                key
            )
        )

        if existing_code is None:

            resolved_code = (
                observation.qr_value
                or temporary_code
            )

            self.track_identities[
                key
            ] = resolved_code

            return resolved_code

        # -------------------------------------------------------------
        # TEMPORARY TRACKER ID -> REAL QR ID
        # -------------------------------------------------------------

        if (
            existing_code
            == temporary_code
            and observation.qr_value
        ):

            resolved_code = (
                observation.qr_value
            )

            self.identity_change_candidates.pop(
                key,
                None,
            )

            self._promote_identity(
                prefix=prefix,
                temporary_code=temporary_code,
                resolved_code=resolved_code,
            )

            self.track_identities[
                key
            ] = resolved_code

            return resolved_code

        # -------------------------------------------------------------
        # SAME TRACKER, NEW BOX QR
        # -------------------------------------------------------------

        if prefix == "BOX":

            observed_code = (
                observation.qr_value
            )

            if (
                not observed_code
                or observed_code
                == existing_code
            ):

                self.identity_change_candidates.pop(
                    key,
                    None,
                )

                return existing_code

            # A tracker may remain over the same footprint while
            # another stacked physical box becomes visible.
            #
            # Require sustained QR evidence before changing identity.

            if (
                self._box_identity_change_confirmed(
                    key,
                    observed_code,
                )
            ):

                self.track_identities[
                    key
                ] = observed_code

                return observed_code

        return existing_code

    def _box_identity_change_confirmed(
        self,
        key: tuple[
            str,
            int,
        ],
        observed_code: str,
    ) -> bool:

        candidate = (
            self.identity_change_candidates.get(
                key
            )
        )

        is_consecutive = bool(
            candidate is not None
            and candidate.code
            == observed_code
            and candidate.last_frame
            == self.frame_number - 1
        )

        if is_consecutive:

            candidate.frames += 1

            candidate.last_frame = (
                self.frame_number
            )

        else:

            candidate = (
                _IdentityChangeCandidate(
                    code=(
                        observed_code
                    ),
                    frames=1,
                    last_frame=(
                        self.frame_number
                    ),
                )
            )

            self.identity_change_candidates[
                key
            ] = candidate

        if (
            candidate.frames
            < self.identity_change_confirmation_frames
        ):

            return False

        self.identity_change_candidates.pop(
            key,
            None,
        )

        return True

    # =================================================================
    # IDENTITY PROMOTION
    # =================================================================

    def _promote_identity(
        self,
        prefix: str,
        temporary_code: str,
        resolved_code: str,
    ) -> None:

        if (
            temporary_code
            == resolved_code
        ):
            return

        # -------------------------------------------------------------
        # UPDATE PENDING EVENTS
        # -------------------------------------------------------------

        self.pending_events = [
            _PendingEvent(
                event=(
                    self._replace_event_identity(
                        pending.event,
                        temporary_code,
                        resolved_code,
                        box_qr_resolved=(
                            prefix
                            == "BOX"
                        ),
                    )
                ),
                queued_frame=(
                    pending.queued_frame
                ),
            )

            for pending
            in self.pending_events
        ]

        # -------------------------------------------------------------
        # BOX
        # -------------------------------------------------------------

        if prefix == "BOX":

            temporary_state = (
                self.boxes.pop(
                    temporary_code,
                    None,
                )
            )

            if (
                temporary_state
                is not None
            ):

                temporary_state.box_code = (
                    resolved_code
                )

                temporary_state.identity_from_qr = (
                    True
                )

                self.boxes.setdefault(
                    resolved_code,
                    temporary_state,
                )

            candidate = (
                self.box_candidates.pop(
                    temporary_code,
                    None,
                )
            )

            if candidate is not None:

                self.box_candidates.setdefault(
                    resolved_code,
                    candidate,
                )

            # Tracker-only identities must never become registered.

            self.box_registration_candidates.pop(
                temporary_code,
                None,
            )

            # Do not carry geometric motion evidence from an unresolved
            # temporary identity into the permanent warehouse identity.

            self.box_motion_states.pop(
                temporary_code,
                None,
            )

            if (
                temporary_code
                in self.initialized_boxes
            ):

                self.initialized_boxes.remove(
                    temporary_code
                )

                self.initialized_boxes.add(
                    resolved_code
                )

        # -------------------------------------------------------------
        # PALLET
        # -------------------------------------------------------------

        elif prefix == "PALLET":

            temporary_state = (
                self.pallets.pop(
                    temporary_code,
                    None,
                )
            )

            if (
                temporary_state
                is not None
            ):

                temporary_state.pallet_code = (
                    resolved_code
                )

                self.pallets.setdefault(
                    resolved_code,
                    temporary_state,
                )

            candidate = (
                self.pallet_candidates.pop(
                    temporary_code,
                    None,
                )
            )

            if candidate is not None:

                self.pallet_candidates.setdefault(
                    resolved_code,
                    candidate,
                )

            if (
                temporary_code
                in self.initialized_pallets
            ):

                self.initialized_pallets.remove(
                    temporary_code
                )

                self.initialized_pallets.add(
                    resolved_code
                )

            # Update box relationships already referring to the
            # temporary pallet identity.

            for box in self.boxes.values():

                if (
                    box.current_pallet_code
                    == temporary_code
                ):

                    box.current_pallet_code = (
                        resolved_code
                    )

    # =================================================================
    # EVENT IDENTITY REPLACEMENT
    # =================================================================

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
                if event.box_code
                == temporary_code
                else event.box_code
            ),

            from_pallet_code=(
                resolved_code
                if event.from_pallet_code
                == temporary_code
                else event.from_pallet_code
            ),

            to_pallet_code=(
                resolved_code
                if event.to_pallet_code
                == temporary_code
                else event.to_pallet_code
            ),

            from_zone_code=(
                resolved_code
                if event.from_zone_code
                == temporary_code
                else event.from_zone_code
            ),

            to_zone_code=(
                resolved_code
                if event.to_zone_code
                == temporary_code
                else event.to_zone_code
            ),

            remarks=(
                event.remarks.replace(
                    temporary_code,
                    resolved_code,
                )
            ),

            qr_read_success=(
                True
                if (
                    box_qr_resolved
                    and event.box_code
                    == temporary_code
                )
                else event.qr_read_success
            ),
        )

    # =================================================================
    # REGION IDENTITY
    # =================================================================

    def _region_identity(
        self,
        observation: RawObservation | None,
    ) -> str | None:

        if observation is None:
            return None

        return self._identity(
            observation,
            observation.class_name.upper(),
        )

    # =================================================================
    # LOCATION HELPERS
    # =================================================================

    @staticmethod
    def _is_dock(
        region_code: str | None,
    ) -> bool:

        return bool(
            region_code
            and region_code
            .upper()
            .startswith(
                "DOCK"
            )
        )

    @staticmethod
    def _format_location(
        pallet_code: str | None,
        region_code: str | None,
    ) -> str:

        if (
            pallet_code
            and region_code
        ):

            return (
                f"{pallet_code} "
                f"in {region_code}"
            )

        return (
            pallet_code
            or region_code
            or "UNKNOWN"
        )
