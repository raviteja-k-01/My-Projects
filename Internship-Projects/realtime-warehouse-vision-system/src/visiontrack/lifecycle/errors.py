from __future__ import annotations


class LifecycleProcessorError(RuntimeError):
    """Base error for lifecycle promotion."""


class LifecycleEventError(LifecycleProcessorError):
    """A permanent problem with one source event.

    Event errors are safe to record on ``video_events``. Infrastructure and
    connection errors intentionally do not inherit from this class, so callers
    retry them instead of incorrectly dead-lettering the source row.
    """


class InvalidEventError(LifecycleEventError):
    """The source row is malformed or lacks required business context."""


class UnsupportedEventError(LifecycleEventError):
    """The source event type has no explicit lifecycle policy."""


class InvalidTransitionError(LifecycleEventError):
    """The requested lifecycle transition would regress a terminal state."""


class LifecycleSchemaError(LifecycleProcessorError):
    """The selected database cannot satisfy the lifecycle SQL contract."""
