from __future__ import annotations

from datetime import UTC, datetime

from visiontrack.lifecycle.errors import (
    InvalidEventError,
    InvalidTransitionError,
    LifecycleEventError,
)
from visiontrack.lifecycle.models import (
    LifecycleProcessorSettings,
    LifecycleState,
    LifecycleTarget,
    LocationCodes,
    LocationContext,
    LowConfidencePolicy,
    ProcessingOutcome,
    ProcessingResult,
    SourceVideoEvent,
)
from visiontrack.lifecycle.repository import (
    LifecycleRepositoryProtocol,
    LifecycleTransactionProtocol,
)
from visiontrack.lifecycle.rules import LifecycleRuleEngine


class LifecycleEventProcessor:
    """Promote one locked CV event into the operational lifecycle ledger."""

    def __init__(
        self,
        settings: LifecycleProcessorSettings,
        rule_engine: LifecycleRuleEngine | None = None,
    ) -> None:
        self.settings = settings
        self.rule_engine = rule_engine or LifecycleRuleEngine()

    def process_next(
        self,
        repository: LifecycleRepositoryProtocol,
    ) -> ProcessingResult | None:
        with repository.transaction() as transaction:
            row = transaction.lock_next_new_event()
            if row is None:
                return None

            try:
                video_event_id = int(row["video_event_id"])
            except (KeyError, TypeError, ValueError, OverflowError) as exc:
                # The primary key comes from our fixed SELECT. If this invariant
                # breaks, no row can be marked safely, so treat it as fatal.
                raise RuntimeError(
                    "Locked video_events row has no valid video_event_id"
                ) from exc

            transaction.create_event_savepoint()
            try:
                event = SourceVideoEvent.from_row(row)
                return self._process_locked(transaction, event)
            except LifecycleEventError as exc:
                transaction.rollback_event_savepoint()
                message = f"{type(exc).__name__}: {exc}"
                transaction.mark_error(video_event_id, message)
                return ProcessingResult(
                    video_event_id=video_event_id,
                    outcome=ProcessingOutcome.ERROR,
                    reason=message,
                )

    def _process_locked(
        self,
        transaction: LifecycleTransactionProtocol,
        event: SourceVideoEvent,
    ) -> ProcessingResult:
        low_confidence_reason = self._low_confidence_reason(event)
        if low_confidence_reason:
            if (
                self.settings.low_confidence_policy
                is LowConfidencePolicy.ERROR
            ):
                raise InvalidEventError(low_confidence_reason)
            transaction.mark_processed(event.video_event_id)
            return ProcessingResult(
                event.video_event_id,
                ProcessingOutcome.IGNORED,
                low_confidence_reason,
            )

        evaluation = self.rule_engine.evaluate(event)
        if evaluation.target is None:
            transaction.mark_processed(event.video_event_id)
            return ProcessingResult(
                event.video_event_id,
                ProcessingOutcome.IGNORED,
                evaluation.ignored_reason or "explicitly ignored",
            )

        if not event.box_code:
            raise InvalidEventError("Missing required box_code")
        box_id = transaction.resolve_box(event.box_code)
        order_contexts = transaction.resolve_order_contexts(box_id)
        order_ids = {context.order_id for context in order_contexts}
        if len(order_ids) != 1:
            raise InvalidEventError(
                f"Box ID {box_id} is linked to multiple orders: "
                + ", ".join(str(value) for value in sorted(order_ids))
            )

        previous = transaction.latest_box_state(box_id)
        self._validate_event_order(event, previous)
        self._validate_transition(previous, evaluation.target)
        location = self._build_location_context(
            transaction,
            evaluation.location_codes,
            evaluation.target,
            previous,
            event.event_time,
        )

        if not self._state_changed(previous, evaluation.target, location):
            transaction.mark_processed(event.video_event_id)
            return ProcessingResult(
                event.video_event_id,
                ProcessingOutcome.IGNORED,
                "no lifecycle status or meaningful location change",
            )

        remarks = (
            f"Promoted from video_events #{event.video_event_id}; "
            f"CV event={event.event_type}; rule={evaluation.target.reason}. "
            f"{event.description or ''}"
        ).strip()
        box_rows = transaction.insert_box_lifecycle(
            event=event,
            target=evaluation.target,
            box_id=box_id,
            order_contexts=order_contexts,
            location=location,
            recorded_by=self.settings.recorded_by,
            remarks=remarks,
        )

        order_rows = 0
        if evaluation.target.lifecycle_status == "DISPATCHED":
            for order_id in sorted(order_ids):
                if not transaction.order_dispatch_complete(order_id):
                    continue
                if transaction.order_already_dispatched(order_id):
                    continue
                order_rows += transaction.insert_order_dispatch(
                    event=event,
                    order_id=order_id,
                    recorded_by=self.settings.recorded_by,
                )

        transaction.mark_processed(event.video_event_id)
        if box_rows == 0 and order_rows == 0:
            return ProcessingResult(
                event.video_event_id,
                ProcessingOutcome.IGNORED,
                "database idempotency key already exists",
            )
        return ProcessingResult(
            event.video_event_id,
            ProcessingOutcome.PROMOTED,
            f"promoted to {evaluation.target.lifecycle_status}",
            inserted_box_rows=box_rows,
            inserted_order_rows=order_rows,
        )

    def _low_confidence_reason(
        self,
        event: SourceVideoEvent,
    ) -> str | None:
        threshold = float(self.settings.min_confidence)
        if threshold <= 0.0:
            return None
        if event.confidence_score is None:
            return (
                "confidence_score is missing while minimum confidence is "
                f"{threshold:.2f}"
            )
        if event.confidence_score < threshold:
            return (
                f"confidence {event.confidence_score:.2f} below minimum "
                f"{threshold:.2f}"
            )
        return None

    @classmethod
    def _validate_event_order(
        cls,
        event: SourceVideoEvent,
        previous: LifecycleState | None,
    ) -> None:
        if previous is None or previous.event_time is None:
            return
        if cls._comparable_time(event.event_time) < cls._comparable_time(
            previous.event_time
        ):
            raise InvalidEventError(
                "stale event_time precedes the current box lifecycle state"
            )

    @staticmethod
    def _validate_transition(
        previous: LifecycleState | None,
        target: LifecycleTarget,
    ) -> None:
        if previous is None:
            return
        if (
            previous.lifecycle_status.upper() == "DISPATCHED"
            and target.lifecycle_status.upper() != "DISPATCHED"
        ):
            raise InvalidTransitionError(
                "cannot regress a DISPATCHED box without an explicit reversal"
            )

    def _build_location_context(
        self,
        transaction: LifecycleTransactionProtocol,
        codes: LocationCodes,
        target: LifecycleTarget,
        previous: LifecycleState | None,
        event_time: datetime,
    ) -> LocationContext:
        context = LocationContext(
            conveyor_id=(
                transaction.resolve_location_id(
                    "conveyor", codes.conveyor_code
                )
                if codes.conveyor_code
                else None
            ),
            pallet_id=(
                transaction.resolve_location_id("pallet", codes.pallet_code)
                if codes.pallet_code
                else None
            ),
            zone_id=(
                transaction.resolve_location_id("zone", codes.zone_code)
                if codes.zone_code
                else None
            ),
            dock_id=(
                transaction.resolve_location_id("dock", codes.dock_code)
                if codes.dock_code
                else None
            ),
        )

        status = target.lifecycle_status
        if previous is not None:
            if status == "ON_PALLET" and context.zone_id is None:
                context = context.updated(zone_id=previous.zone_id)
            if status == "AT_DOCK" and context.pallet_id is None:
                context = context.updated(pallet_id=previous.pallet_id)
            if status in {"LOADED_TO_TRAILER", "DISPATCHED"}:
                context = self._carry_previous_location(context, previous)

        if status in {"AT_DOCK", "LOADED_TO_TRAILER", "DISPATCHED"}:
            if context.dock_id is None:
                raise InvalidEventError(
                    f"{status} requires a current or prior dock"
                )
            require_attached = status in {
                "LOADED_TO_TRAILER",
                "DISPATCHED",
            }
            carrier = transaction.resolve_carrier_context(
                context.dock_id,
                event_time,
                require_attached=require_attached,
            )
            context = self._merge_known(context, carrier)
            if require_attached and (
                context.trailer_id is None or context.truck_id is None
            ):
                raise InvalidEventError(
                    f"{status} requires an active dock assignment and "
                    "attached truck at event_time"
                )

        return context

    @staticmethod
    def _carry_previous_location(
        context: LocationContext,
        previous: LifecycleState,
    ) -> LocationContext:
        return context.updated(
            conveyor_id=context.conveyor_id or previous.conveyor_id,
            pallet_id=context.pallet_id or previous.pallet_id,
            zone_id=context.zone_id or previous.zone_id,
            dock_id=context.dock_id or previous.dock_id,
            trailer_id=context.trailer_id or previous.trailer_id,
            truck_id=context.truck_id or previous.truck_id,
            trailer_dock_assignment_id=(
                context.trailer_dock_assignment_id
                or previous.trailer_dock_assignment_id
            ),
            trailer_truck_assignment_id=(
                context.trailer_truck_assignment_id
                or previous.trailer_truck_assignment_id
            ),
        )

    @staticmethod
    def _merge_known(
        base: LocationContext,
        new: LocationContext,
    ) -> LocationContext:
        return base.updated(
            trailer_id=new.trailer_id or base.trailer_id,
            truck_id=new.truck_id or base.truck_id,
            trailer_dock_assignment_id=(
                new.trailer_dock_assignment_id
                or base.trailer_dock_assignment_id
            ),
            trailer_truck_assignment_id=(
                new.trailer_truck_assignment_id
                or base.trailer_truck_assignment_id
            ),
        )

    @staticmethod
    def _state_changed(
        previous: LifecycleState | None,
        target: LifecycleTarget,
        context: LocationContext,
    ) -> bool:
        if previous is None:
            return True
        if previous.lifecycle_status.upper() != target.lifecycle_status.upper():
            return True
        return any(
            getattr(previous, field_name) != getattr(context, field_name)
            for field_name in target.compare_fields
        )

    @staticmethod
    def _comparable_time(value: datetime) -> datetime:
        if value.tzinfo is None or value.utcoffset() is None:
            return value
        return value.astimezone(UTC).replace(tzinfo=None)
