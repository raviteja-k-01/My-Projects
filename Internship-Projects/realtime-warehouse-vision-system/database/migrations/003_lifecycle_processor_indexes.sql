-- Database changes required by the lifecycle event processor.
-- Adds idempotency protection and indexes used by the processor queries.

START TRANSACTION;

ALTER TABLE order_lifecycle_events
    ADD COLUMN idempotency_key VARCHAR(191) NULL
        AFTER source_reference,
    ADD UNIQUE INDEX uq_ole_idempotency_key (idempotency_key),
    ADD INDEX idx_ole_box_latest (
        entity_type,
        box_id,
        event_time,
        order_lifecycle_event_id
    ),
    ADD INDEX idx_ole_order_dispatch (
        entity_type,
        order_id,
        lifecycle_status
    );

ALTER TABLE trailer_dock_assignments
    ADD INDEX idx_tda_lifecycle_lookup (
        dock_id,
        assignment_status,
        scheduled_from,
        scheduled_to
    );

ALTER TABLE trailer_truck_assignments
    ADD INDEX idx_tta_lifecycle_lookup (
        trailer_id,
        assignment_status,
        assigned_at
    );

COMMIT;
