# VisionTrack Dynamic Pallet System

VisionTrack processes warehouse video to identify boxes, pallets, QR codes, and
warehouse regions. It tracks how boxes move, decides which pallet or region each
box belongs to, records movement events, and can store each processing run in a
local or cloud MySQL database.

The application dispatcher is `src/visiontrack/main.py`. Video processing stays
in `src/visiontrack/cli.py`, while the read-only dashboard is isolated under
`src/visiontrack/dashboard/`. The launch scripts contain no tracking or UI
logic.

## What the main parts are for

| Path | Purpose |
| --- | --- |
| `src/visiontrack/cli.py` | Opens the video, runs detection and tracking, displays progress, writes logs, and coordinates optional database writes. |
| `src/visiontrack/perception/` | Runs YOLO detection, QR decoding, and detection stabilization. |
| `src/visiontrack/warehouse/` | Maintains box and pallet state and produces warehouse movement events. |
| `src/visiontrack/database/` | Stores run history, events, final box state, final pallet state, and processing metrics. |
| `src/visiontrack/dashboard/app.py` | Sarfaraz's unchanged visual interface with the integrated read-only dashboard queries. |
| `src/visiontrack/dashboard/data_access.py` | Reuses one thread-safe read-only MySQL session and reconnects once after transient failures. |
| `src/visiontrack/dashboard/launcher.py` | Selects a database profile and starts the packaged dashboard without importing it into inference. |
| `configs/models.yaml` | Selects the active model and controls detection thresholds. |
| `configs/app.yaml` | Controls runtime, tracking, assignment, and capacity behavior. |
| `models/` | Contains the trained YOLO model files. The configured default is `six_class_final.pt`. |
| `data/input_videos/` | Holds local input videos. These files are intentionally ignored by Git. |
| `scripts/run_video.ps1` | Provides a short, consistent PowerShell command for starting the current CLI. |
| `scripts/run_dashboard.ps1` | Starts the packaged dashboard with the selected local or cloud profile. |
| `database/schema/visiontrack_video_tables_only.sql` | Creates only the VisionTrack video-run tables and views. It does not recreate the wider business database. |

## 1. Set up Python

Run these commands from this directory:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
pip install -e .
```

`requirements.txt` is the single dependency manifest for inference, local and
cloud MySQL access, tests, and the dashboard. It installs the CUDA 12.4 build
of PyTorch. FFmpeg must also be available on `PATH` if you want to save an
annotated H.264 video.

## 2. Configure local files

Create the private environment file:

```powershell
Copy-Item .env.example .env
```

Keep `.env` private. It is ignored by Git and is the correct place for database
passwords.

Place input videos in `data/input_videos/`. A file can then be selected by its
full path, filename, name prefix, or number. For example, `-Video 1` resolves a
single matching `1.*.mp4` file or `input_video1.mp4`.

The default model is configured in `configs/models.yaml`. Model files can also be
selected explicitly with the launcher's `-Model` option.

## 3. Process a video

From Command Prompt, the short launcher accepts the video, database mode,
device, and optional display mode as positional arguments:

```cmd
run 43 local gpu
run 43 local gpu headless
```

The first command opens the preview. The second processes without opening a
preview window. `local` targets `visiontrack_local`, `gpu` means
`cuda:0`, and the configured model remains `models/six_class_final.pt`.
Every run automatically saves its annotated MP4 under `outputs/videos/` and
its run log under `outputs/logs/`. Inference still reads the original source
frames, while the saved video canvas is capped at `1920x1080` for practical
playback, encoding time, and file size.

Preview video 1 without writing to a database:

```powershell
.\scripts\run_video.ps1 -Video 1
```

Run without opening a preview window:

```powershell
.\scripts\run_video.ps1 -Video 1 -NoDisplay
```

Save an annotated video with the live status panel:

```powershell
.\scripts\run_video.ps1 `
    -Video 1 `
    -OutputVideo "outputs/videos/video_1_annotated.mp4"
```

Run a short 300-frame check on the CPU:

```powershell
.\scripts\run_video.ps1 -Video 1 -Device cpu -MaxFrames 300
```

Press `Esc` to stop an open preview. A log file is written under
`outputs/logs/` for every run, including preview-only runs.

### Launcher options

| Option | Meaning |
| --- | --- |
| `-Video` | Input path, stream URL, filename, prefix, or video number. |
| `-Database off\|local\|cloud` | Selects where results are stored. The safe default is `off`. |
| `-DatabaseName` | Optionally overrides the selected profile's database name for this run. |
| `-Model` | Model key, `.pt` filename, or model path. The default is `six_class_final`, which loads `models/six_class_final.pt`. |
| `-Device` | Inference device such as `auto`, `cpu`, `cuda`, `cuda:0`, or `0`. |
| `-NoDisplay` | Processes without opening the OpenCV preview window. |
| `-OutputVideo` | Writes an annotated H.264 video to the given path. |
| `-MaxFrames` | Stops after a limited number of frames for a quick check. |
| `-EstimateMissingBoxes` | Creates approximate box observations from package QR codes when YOLO misses the box. |
| `-HideConveyorBelt` | Hides conveyor-belt detections in the preview and saved video. |

The equivalent direct command is:

```powershell
python -m visiontrack.cli --video 1 --db-mode off
```

## 4. Use a database

VisionTrack supports three modes:

- `off` performs detection and tracking without database writes.
- `local` uses the `LOCAL_DB_*` values in `.env`.
- `cloud` uses the `CLOUD_DB_*` values in `.env`.

Before using `local` or `cloud`, apply
`database/schema/visiontrack_video_tables_only.sql` to the selected database.
This creates the following VisionTrack-owned tables:

- `video_processing_runs`
- `video_events`
- `video_final_boxes`
- `video_final_pallets`
- `video_processing_metrics`

For a database created with an older schema, apply
`database/migrations/002_add_event_processor_fields.sql` once instead. Do not
reapply that migration to a database where those columns already exist.

Each emitted `video_events` row contains structured zone/dock fields and starts
with `processing_status='NEW'`. `confidence_score` is a 0–100 percentage built
only from real detections: for each confirmation frame the engine takes the
weakest required box/pallet/region confidence, then averages those values over
the configured confirmation window. `qr_read_success` is true only when that
box identity originated from a successful QR decode; the provenance survives
tracking, temporary-identity resolution, and pallet-driven occlusion.

Inference does not promote rows into operational lifecycle tables. A separate
Event Processor connected to the same database owns that step and changes each
source row from `NEW` to `PROCESSED` or `ERROR`. This single-writer boundary
prevents final CV snapshots from overwriting later lifecycle states.

Then process a video with the selected destination:

```powershell
.\scripts\run_video.ps1 -Video 1 -Database local -NoDisplay
.\scripts\run_video.ps1 -Video 1 -Database cloud -NoDisplay
```

### Load the full seed into an isolated local database

The seed uses explicit primary keys and cannot be inserted twice into a populated
database. Use the guarded local command instead of running it directly in
Workbench:

```powershell
.\scripts\seed_local_database.ps1
```

It creates `visiontrack_local` by default, refuses remote hosts, and stops
if that database already contains tables. It then applies the idempotent project
migration `database/migrations/001_extend_box_master_to_1000.sql`, which
registers `BOX_501` through `BOX_1000` as received inventory without inventing
order or lifecycle history. To intentionally rebuild only that isolated local
database:

```powershell
.\scripts\seed_local_database.ps1 -Reset
```

Run inference against it without changing `.env`:

```powershell
.\scripts\run_video.ps1 `
    -Video 1 `
    -Database local `
    -DatabaseName visiontrack_local `
    -NoDisplay
```

Database-enabled runs are marked `RUNNING`, `COMPLETED`, or `FAILED`, and retain
their event history and final warehouse state.

## 5. Open the dashboard

Open Sarfaraz's dashboard against the cloud database:

```cmd
run dashboard
```

Select a database profile explicitly when needed:

```cmd
run dashboard cloud
run dashboard local
```

The launcher maps the selected `CLOUD_DB_*` or `LOCAL_DB_*` profile to the
dashboard's existing `DB_*` contract. Sarfaraz's visual interface remains
unchanged. Its read-only backend reuses one reconnecting database session and
caches the schema inventory and repeated query results, avoiding a new cloud
connection for every table. The dashboard runs in a separate Streamlit process,
so it does not load into or slow down video inference. The equivalent command is
`.\scripts\run_dashboard.ps1 -Database cloud`.

## Generated and private files

Do not commit these paths:

- `.env`
- `data/input_videos/*`
- `outputs/*`
- private or experimental model weights

The `.gitignore` file already protects the standard runtime locations.
