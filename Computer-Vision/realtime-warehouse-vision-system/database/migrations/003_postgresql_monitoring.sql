CREATE SCHEMA IF NOT EXISTS visiontrack_monitor;

CREATE TABLE IF NOT EXISTS visiontrack_monitor.runs (
    id uuid PRIMARY KEY,
    source_name text NOT NULL,
    source_bytes bigint NOT NULL,
    source_width integer NOT NULL,
    source_height integer NOT NULL,
    source_fps double precision NOT NULL,
    source_frames integer NOT NULL,
    model_path text NOT NULL,
    device text,
    status text NOT NULL CHECK (status IN ('PROCESSING', 'COMPLETED', 'STOPPED', 'FAILED')),
    started_at timestamptz NOT NULL DEFAULT now(),
    finished_at timestamptz,
    processed_frames integer NOT NULL DEFAULT 0,
    elapsed_seconds double precision,
    output_path text,
    error text,
    unresolved integer NOT NULL DEFAULT 0,
    unresolved_movements integer NOT NULL DEFAULT 0
);

CREATE TABLE IF NOT EXISTS visiontrack_monitor.events (
    run_id uuid NOT NULL REFERENCES visiontrack_monitor.runs(id),
    frame integer NOT NULL,
    source_seconds double precision NOT NULL,
    recorded_at timestamptz NOT NULL DEFAULT now(),
    object_code text NOT NULL,
    event_type text NOT NULL,
    from_pallet text,
    to_pallet text,
    from_zone text,
    to_zone text,
    confidence double precision,
    remarks text NOT NULL,
    PRIMARY KEY (run_id, frame, object_code, event_type),
    CHECK (confidence BETWEEN 0 AND 1)
);
CREATE INDEX IF NOT EXISTS monitor_event_object ON visiontrack_monitor.events (object_code, recorded_at);

CREATE TABLE IF NOT EXISTS visiontrack_monitor.inventory (
    run_id uuid NOT NULL REFERENCES visiontrack_monitor.runs(id),
    kind text NOT NULL CHECK (kind IN ('box', 'pallet')),
    object_code text NOT NULL,
    pallet text,
    location text,
    last_seen_seconds double precision NOT NULL,
    identity_source text NOT NULL DEFAULT 'QR',
    PRIMARY KEY (run_id, kind, object_code)
);

CREATE TABLE IF NOT EXISTS visiontrack_monitor.exceptions (
    run_id uuid NOT NULL REFERENCES visiontrack_monitor.runs(id),
    object_code text NOT NULL,
    first_frame integer NOT NULL,
    last_frame integer NOT NULL,
    frames integer NOT NULL,
    PRIMARY KEY (run_id, object_code)
);
