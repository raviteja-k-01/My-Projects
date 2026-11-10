-- Extend the local VisionTrack demo box master from 500 to 1,000 records.
--
-- The supplied seed already contains orders numbered 1 through 1,000, which
-- provide a stable local sequence. New boxes are registered as RECEIVED and
-- deliberately have no fabricated order or lifecycle relationship. This file
-- is idempotent: existing box IDs or codes are never overwritten.

START TRANSACTION;

INSERT INTO boxes (
    box_id,
    box_code,
    length_cm,
    breadth_cm,
    height_cm,
    weight_kg,
    box_status,
    created_by,
    created_at
)
SELECT
    orders.order_id,
    CONCAT(
        'BOX_',
        LPAD(
            orders.order_id,
            GREATEST(3, CHAR_LENGTH(orders.order_id)),
            '0'
        )
    ),
    55 + MOD(orders.order_id, 8),
    35 + MOD(orders.order_id, 6),
    30 + MOD(orders.order_id, 7),
    7.50 + (0.65 * MOD(orders.order_id, 20)),
    'RECEIVED',
    1 + MOD(orders.order_id - 1, 10),
    CURRENT_TIMESTAMP
FROM orders
LEFT JOIN boxes
    ON boxes.box_id = orders.order_id
    OR boxes.box_code = CONCAT(
        'BOX_',
        LPAD(
            orders.order_id,
            GREATEST(3, CHAR_LENGTH(orders.order_id)),
            '0'
        )
    )
WHERE orders.order_id BETWEEN 501 AND 1000
  AND boxes.box_id IS NULL;

COMMIT;

SELECT
    COUNT(*) AS master_box_count,
    MIN(box_id) AS first_box_id,
    MAX(box_id) AS last_box_id
FROM boxes;
