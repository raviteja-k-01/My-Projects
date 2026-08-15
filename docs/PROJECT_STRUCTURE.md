# VisionTrack Project Structure

```text
visiontrack-dynamic-pallet-system/
|-- configs/                 Runtime, model, and zone configuration
|-- data/
|   |-- input_videos/        Private source videos
|   `-- master/              Legacy master-data CSV files
|-- database/schema/         VisionTrack video-history tables and views
|-- docs/                    Supporting queries and project notes
|-- models/                  Trained YOLO weights
|-- outputs/
|   |-- logs/                One diagnostic log per processing run
|   `-- videos/              Optional annotated H.264 output
|-- scripts/
|   |-- run_video.ps1        Convenience launcher for the current CLI
|   `-- run_dashboard.ps1    Profile-aware Streamlit launcher
|-- src/visiontrack/
|   |-- association/         Box-to-pallet and object-to-region assignment
|   |-- config/              YAML, environment, and model selection
|   |-- database/            MySQL connection and run persistence
|   |-- dashboard/
|   |   |-- app.py           Sarfaraz's unchanged visual interface
|   |   |-- data_access.py   Reconnecting read-only dashboard DB client
|   |   `-- launcher.py      Isolated dashboard process launcher
|   |-- geometry/            Polygon and zone calculations
|   |-- perception/          YOLO, QR decoding, and stabilization
|   |-- pipeline/            Per-frame orchestration
|   |-- warehouse/           Entity state and movement events
|   |-- cli.py               Video-processing entry point
|   `-- main.py              Application command dispatcher
|-- .env.example             Safe environment-variable template
|-- README.md                Setup, operation, and component guide
`-- requirements.txt         Python dependencies
```

## Runtime flow

1. `run.cmd` selects video inference or the isolated dashboard launcher.
2. `scripts/run_video.ps1` prepares the Python path and forwards video options.
3. `visiontrack.cli` opens the source and selects the configured YOLO model.
4. The perception pipeline detects and stabilizes warehouse objects and decodes
   QR identities.
5. The warehouse engine assigns boxes to pallets or regions and emits movement
   events.
6. The CLI updates the preview, writes the run log and optional annotated video,
   and persists results when database mode is enabled.

## Database ownership

The current processor writes only to these VisionTrack video-history tables:

- `video_processing_runs`
- `video_events`
- `video_final_boxes`
- `video_final_pallets`
- `video_processing_metrics`

They are defined in `database/schema/visiontrack_video_tables_only.sql`. The
processor does not create or modify the wider orders, customers, or products
schema. Raw event rows include real stabilized detection confidence, QR-read
provenance, structured location fields, and a `NEW` processing status. Promotion
to `order_lifecycle_events` belongs exclusively to the separate Event Processor
connected to the same database.

## Legacy compatibility files

`dynamic_pallet.py` and `db_manager.py` belong to the earlier single-file
processor. They remain available for reference, but the supported processing
path is `visiontrack.cli` through `scripts/run_video.ps1`.
