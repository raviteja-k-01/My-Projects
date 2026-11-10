CREATE TABLE IF NOT EXISTS video_processing_runs (
    run_id BIGINT AUTO_INCREMENT PRIMARY KEY,
    video_name VARCHAR(255) NOT NULL,
    input_video_path TEXT,
    model_path TEXT,
    device VARCHAR(100),
    gpu_name VARCHAR(255),
    source_width INT,
    source_height INT,
    source_fps DECIMAL(10,3),
    video_duration_sec DECIMAL(12,3),
    frame_skip INT,
    process_width INT,
    process_height INT,
    yolo_imgsz INT,
    processed_frames INT,
    source_frames_covered INT,
    total_processing_time_sec DECIMAL(12,3),
    end_to_end_time_sec DECIMAL(12,3),
    speed_vs_realtime DECIMAL(12,3),
    realtime_factor DECIMAL(12,3),
    status ENUM('RUNNING','COMPLETED','FAILED') DEFAULT 'COMPLETED',
    started_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    completed_at TIMESTAMP NULL,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

CREATE TABLE IF NOT EXISTS video_events (
    video_event_id BIGINT AUTO_INCREMENT PRIMARY KEY,
    run_id BIGINT NOT NULL,
    processed_index INT,
    video_time VARCHAR(20),
    timestamp_sec DECIMAL(12,3),
    event_type VARCHAR(80),
    severity VARCHAR(30),
    box_code VARCHAR(50),
    pallet_code VARCHAR(50),
    from_pallet_code VARCHAR(50),
    to_pallet_code VARCHAR(50),
    conveyor_code VARCHAR(50),
    from_zone_code VARCHAR(50),
    to_zone_code VARCHAR(50),
    dock_code VARCHAR(30),
    confidence_score DECIMAL(5,2),
    qr_read_success BOOLEAN,
    processing_status ENUM('NEW','PROCESSED','ERROR') NOT NULL DEFAULT 'NEW',
    processed_at DATETIME NULL,
    processing_error VARCHAR(500),
    description TEXT,
    event_created_at DATETIME NULL,
    inserted_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    KEY idx_video_events_run_type (run_id, event_type),
    KEY idx_video_events_box (box_code),
    KEY idx_video_events_pallet (pallet_code, from_pallet_code, to_pallet_code),
    KEY idx_video_events_processing (processing_status, video_event_id),
    CONSTRAINT fk_video_events_run
        FOREIGN KEY (run_id) REFERENCES video_processing_runs(run_id)
        ON DELETE CASCADE
);

CREATE TABLE IF NOT EXISTS video_final_boxes (
    final_box_id BIGINT AUTO_INCREMENT PRIMARY KEY,
    run_id BIGINT NOT NULL,
    box_code VARCHAR(50) NOT NULL,
    status VARCHAR(80),
    pallet_code VARCHAR(50),
    slot_no INT,
    row_no INT,
    column_no INT,
    layer_no INT,
    length_cm DECIMAL(10,2),
    width_cm DECIMAL(10,2),
    height_cm DECIMAL(10,2),
    weight_kg DECIMAL(10,2),
    volume_cm3 DECIMAL(12,2),
    inserted_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    KEY idx_video_final_boxes_run (run_id, pallet_code, status),
    CONSTRAINT fk_video_final_boxes_run
        FOREIGN KEY (run_id) REFERENCES video_processing_runs(run_id)
        ON DELETE CASCADE
);

CREATE TABLE IF NOT EXISTS video_final_pallets (
    final_pallet_id BIGINT AUTO_INCREMENT PRIMARY KEY,
    run_id BIGINT NOT NULL,
    pallet_code VARCHAR(50) NOT NULL,
    active TINYINT(1) DEFAULT 0,
    status VARCHAR(80),
    box_count INT,
    max_boxes INT,
    total_weight_kg DECIMAL(12,3),
    max_weight_kg DECIMAL(12,3),
    weight_utilization_percent DECIMAL(8,2),
    total_volume_cm3 DECIMAL(14,3),
    max_volume_cm3 DECIMAL(14,3),
    volume_efficiency_percent DECIMAL(8,2),
    max_stack_height_cm DECIMAL(10,3),
    max_height_cm DECIMAL(10,3),
    height_utilization_percent DECIMAL(8,2),
    bottleneck_utilization_percent DECIMAL(8,2),
    reasons TEXT,
    inserted_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    KEY idx_video_final_pallets_run (run_id, pallet_code, status),
    CONSTRAINT fk_video_final_pallets_run
        FOREIGN KEY (run_id) REFERENCES video_processing_runs(run_id)
        ON DELETE CASCADE
);

CREATE TABLE IF NOT EXISTS video_processing_metrics (
    metric_id BIGINT AUTO_INCREMENT PRIMARY KEY,
    run_id BIGINT NOT NULL,
    metric_name VARCHAR(128) NOT NULL,
    metric_value TEXT,
    inserted_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    KEY idx_video_metrics_run (run_id, metric_name),
    CONSTRAINT fk_video_metrics_run
        FOREIGN KEY (run_id) REFERENCES video_processing_runs(run_id)
        ON DELETE CASCADE
);

CREATE OR REPLACE VIEW v_latest_video_run AS
SELECT *
FROM video_processing_runs
WHERE run_id = (SELECT MAX(run_id) FROM video_processing_runs);

CREATE OR REPLACE VIEW v_latest_video_pallets AS
SELECT p.*
FROM video_final_pallets p
JOIN v_latest_video_run r ON p.run_id = r.run_id;

CREATE OR REPLACE VIEW v_latest_video_boxes AS
SELECT b.*
FROM video_final_boxes b
JOIN v_latest_video_run r ON b.run_id = r.run_id;

CREATE OR REPLACE VIEW v_latest_video_events AS
SELECT e.*
FROM video_events e
JOIN v_latest_video_run r ON e.run_id = r.run_id;
