VisionTrack — AI-Powered Warehouse Monitoring

A computer vision application that detects, tracks, and records warehouse inventory movements from video footage.

The Problem

Warehouse cameras capture boxes moving between pallets, zones, and loading docks. But reviewing footage manually to identify objects and understand their movements is time-consuming.

VisionTrack makes warehouse footage searchable and easier to review by combining object detection, tracking, QR-code identification, movement-event processing, and a database-backed dashboard.

What VisionTrack Does

Upload a warehouse video and let the system process it frame by frame.

- Object Detection: Fine-tuned YOLOv8 model detecting boxes, pallets, QR codes, conveyor belts, docks, and zones.
- Object Tracking: ByteTrack maintains object tracks across video frames.
- QR Identification: Identifies boxes through QR codes, with caching to avoid repeated decoding.
- Pallet Association: Links detected boxes to pallets using spatial and temporal information.
- Movement Detection: Records events such as "ADD", "MOVE", "ZONE_TRANSFER", and "DOCK_ARRIVAL".
- PostgreSQL Storage: Stores processing runs, object records, movement events, and metrics.
- Interactive Dashboard: Explore inventory, movement history, exceptions, and analytics.
- Video Export: Download annotated MP4 footage for reviewing detections and movements.

Application Preview

Live Monitoring

Upload footage and monitor the annotated video while processing.

"Live Monitoring" (docs/screenshots/live-monitoring.png)

Processing Status

Track processing progress, frame counts, and measured FPS.

"Processing Status" (docs/screenshots/processing-status.png)

Analytics

View recorded inventory counts, movement events, and run-level insights.

"Analytics Dashboard" (docs/screenshots/analytics.png)

Movement Timeline

Review detected movement events at their corresponding video timestamps.

"Movement Timeline" (docs/screenshots/movement-timeline.png)

Exceptions

Identify unresolved object tracks and movements with insufficient identity evidence.

"Exceptions Dashboard" (docs/screenshots/exceptions.png)

How It Works

Warehouse Video (MP4)
         |
         v
YOLOv8 Object Detection
         |
         v
ByteTrack + QR Recognition
         |
         v
Box-to-Pallet & Location Association
         |
         v
Warehouse Movement Event Engine
         |
         +---------------------+
         |                     |
         v                     v
   PostgreSQL            Annotated Video
         |
         v
   Streamlit Dashboard

The system uses detected warehouse regions rather than relying exclusively on manually defined camera zones.

Tech Stack

Component| Technologies
Programming| Python
Object Detection| YOLOv8, PyTorch, Ultralytics
Tracking & Image Processing| ByteTrack, OpenCV
QR Recognition| Pyzbar
Database| PostgreSQL, Psycopg
Dashboard| Streamlit, Plotly
Video Processing| OpenCV, FFmpeg
Testing| pytest

Getting Started

Requirements

- Python 3.11
- PostgreSQL
- FFmpeg available on "PATH"
- Compatible YOLOv8 model checkpoint
- NVIDIA CUDA GPU recommended for faster inference

1. Install dependencies

Open Windows CMD in the project directory.

py -3.11 -m venv .venv
.venv\Scripts\python.exe -m pip install --upgrade pip
.venv\Scripts\python.exe -m pip install -r requirements.txt
.venv\Scripts\python.exe -m pip install -e .

2. Configure PostgreSQL

Create your local environment file:

copy .env.example .env

Configure the database connection in ".env":

VISIONTRACK_PG_HOST=localhost
VISIONTRACK_PG_PORT=5432
VISIONTRACK_PG_DATABASE=visiontrack
VISIONTRACK_PG_USER=your_database_user
VISIONTRACK_PG_PASSWORD=your_database_password

Create the "visiontrack" database and apply the applicable PostgreSQL migration in "database/migrations/003_postgresql_monitoring.sql".

Never commit your ".env" file.

3. Add the YOLOv8 model

Place the compatible fine-tuned checkpoint at:

models/six_class_final.pt

Model weights are not included in this repository.

4. Launch VisionTrack

set "PYTHONPATH=%CD%\src"
.venv\Scripts\python.exe -m visiontrack monitor

Open the Streamlit URL displayed in your terminal.

Upload an MP4 from Live Monitoring, start processing, and explore the recorded results.

Testing

Run the automated tests:

.venv\Scripts\python.exe -m pytest -q

The last reported test run had 44 passing tests and one failing legacy event-engine test.

Performance

VisionTrack reports measured processing FPS for each run. Throughput depends on video resolution, hardware, detection, QR decoding, tracking, and video encoding.

A previous warehouse pipeline recorded a 3.9 → 7.1 FPS improvement on an NVIDIA Tesla T4 after QR-processing optimizations. This is a historical benchmark, not a measured result for the current VisionTrack application.

Current Limitations

- The application processes prerecorded video, not a direct live CCTV feed.
- Some annotations may overlap in crowded scenes.
- QR identification and movement events can remain uncertain and require review.
- The system is a portfolio prototype, not a production-certified warehouse inventory solution.

Project Outcome

VisionTrack brings together computer vision, object tracking, warehouse movement logic, PostgreSQL, and an interactive dashboard in a single application.

Instead of presenting only model detections, it provides a workflow for processing warehouse footage, reviewing object movements, inspecting uncertain observations, and retrieving recorded events.