SHOW TABLES;

SELECT run_id, video_name, processed_frames, total_processing_time_sec, status, completed_at
FROM video_processing_runs
ORDER BY run_id DESC
LIMIT 10;

SELECT video_time, event_type, box_code, pallet_code, from_pallet_code, to_pallet_code, description
FROM video_events
ORDER BY video_event_id DESC
LIMIT 20;

SELECT box_id, box_code, current_pallet_id, box_status, length_cm, breadth_cm, height_cm, weight_kg
FROM boxes
ORDER BY box_id DESC
LIMIT 20;

SELECT p.pallet_code, b.box_code, a.slot_no, a.row_no, a.column_no, a.layer_no, a.assigned_at, a.removed_at
FROM pallet_box_assignments a
JOIN pallets p ON a.pallet_id = p.pallet_id
JOIN boxes b ON a.box_id = b.box_id
ORDER BY a.assignment_id DESC
LIMIT 20;
