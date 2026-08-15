-- Add the structured CV location and worker-control fields required by the
-- local video_events -> order_lifecycle_events processor.
--
-- Apply this migration once to an existing VisionTrack database. New database
-- installations receive these columns from visiontrack_video_tables_only.sql.

START TRANSACTION;

ALTER TABLE video_events
    ADD COLUMN conveyor_code VARCHAR(50) NULL AFTER to_pallet_code,
    ADD COLUMN from_zone_code VARCHAR(50) NULL AFTER conveyor_code,
    ADD COLUMN to_zone_code VARCHAR(50) NULL AFTER from_zone_code,
    ADD COLUMN dock_code VARCHAR(30) NULL AFTER to_zone_code,
    ADD COLUMN confidence_score DECIMAL(5,2) NULL AFTER dock_code,
    ADD COLUMN qr_read_success BOOLEAN NULL AFTER confidence_score,
    ADD COLUMN processing_status ENUM('NEW','PROCESSED','ERROR')
        NOT NULL DEFAULT 'NEW' AFTER qr_read_success,
    ADD COLUMN processed_at DATETIME NULL AFTER processing_status,
    ADD COLUMN processing_error VARCHAR(500) NULL AFTER processed_at,
    ADD INDEX idx_video_events_processing (processing_status, video_event_id);

-- Rows created before this processor architecture are historical. Mark them as
-- already handled so starting the worker cannot replay old CV events into the
-- lifecycle ledger. New inference rows explicitly use processing_status=NEW.
UPDATE video_events
SET processing_status = 'PROCESSED',
    processed_at = COALESCE(processed_at, inserted_at, CURRENT_TIMESTAMP),
    processing_error = NULL
WHERE processing_status = 'NEW';

COMMIT;

SELECT
    column_name,
    column_type,
    is_nullable,
    column_default
FROM information_schema.columns
WHERE table_schema = DATABASE()
  AND table_name = 'video_events'
  AND column_name IN (
      'conveyor_code',
      'from_zone_code',
      'to_zone_code',
      'dock_code',
      'confidence_score',
      'qr_read_success',
      'processing_status',
      'processed_at',
      'processing_error'
  )
ORDER BY ordinal_position;
