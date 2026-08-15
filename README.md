# Real-Time Warehouse Vision System

AI-powered warehouse monitoring system for real-time object detection, QR-based identity tracking, pallet association, movement-event detection, database logging, and dashboard visualization.

## Overview

The system processes warehouse video and tracks how boxes and pallets move through different regions such as conveyor belts, zones, and docks.

It combines YOLO-based detection, OpenCV processing, QR decoding, warehouse event logic, and MySQL storage into one end-to-end computer vision pipeline.

## Key Features

- YOLO-based warehouse object detection
- Box and pallet tracking across video frames
- QR-based box identity resolution
- Detection stabilization for noisy predictions
- Box-to-pallet association
- Zone and dock awareness
- Warehouse movement-event generation
- Local and cloud MySQL integration
- Annotated video output
- Processing logs and metrics
- Streamlit monitoring dashboard

## Detected Classes

The current model supports six warehouse classes:

- Box
- Pallet
- QR Code
- Conveyor Belt
- Dock
- Zone

## Tech Stack

**Computer Vision & ML**
- Python
- YOLO / Ultralytics
- OpenCV
- PyTorch
- Pyzbar
- NumPy

**Data & Backend**
- MySQL
- Pandas
- Python-dotenv

**Dashboard**
- Streamlit
- Plotly

## Pipeline

```text
Warehouse Video
      ↓
YOLO Detection
      ↓
Detection Stabilization
      ↓
QR Identity Resolution
      ↓
Box / Pallet Tracking
      ↓
Pallet & Region Association
      ↓
Warehouse Event Engine
      ↓
Database + Logs + Annotated Video
      ↓
Streamlit Dashboard
```

## Project Structure

```text
realtime-warehouse-vision-system/
├── configs/                 # Runtime and model configuration
├── data/
│   ├── input_videos/        # Local input videos (ignored by Git)
│   └── master/              # Box and pallet master data
├── database/
│   ├── migrations/          # Database migrations
│   ├── schema/              # VisionTrack database schema
│   └── seeds/               # Demo/local seed data
├── models/                  # Model instructions; weights are ignored by Git
├── outputs/
│   ├── csv/
│   ├── logs/
│   └── videos/
├── scripts/                 # Video, dashboard, and database launch scripts
├── src/visiontrack/
│   ├── association/         # Pallet and location association
│   ├── config/              # Configuration loading
│   ├── dashboard/           # Streamlit dashboard
│   ├── database/            # Local/cloud database integration
│   ├── geometry/            # Polygon and spatial utilities
│   ├── lifecycle/           # Lifecycle processing logic
│   ├── perception/          # Detection, QR decoding, stabilization
│   ├── pipeline/            # Video-processing pipeline
│   ├── warehouse/           # Warehouse state and event logic
│   ├── cli.py               # Video inference CLI
│   └── main.py              # Application dispatcher
├── tests/                   # Automated tests
├── .env.example
├── requirements.txt
└── pyproject.toml
```

## Installation

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1

python -m pip install --upgrade pip
pip install -r requirements.txt
pip install -e .
```

FFmpeg must be available on `PATH` when saving annotated H.264 output videos.

## Environment Configuration

Create a private `.env` file from the provided template:

```powershell
Copy-Item .env.example .env
```

Add database credentials only when local or cloud database integration is required.

The `.env` file is ignored by Git.

## Model Setup

Place the required YOLO model weight inside:

```text
models/
```

The default configuration uses:

```text
models/six_class_final.pt
```

Model weights are intentionally excluded from Git.

## Input Video

Place input videos inside:

```text
data/input_videos/
```

Video files are excluded from Git.

## Run Inference

### Basic inference without database writes

```powershell
python -m visiontrack.cli --video 1 --db-mode off
```

### PowerShell launcher

```powershell
.\scripts\run_video.ps1 -Video 1
```

### Headless processing

```powershell
.\scripts\run_video.ps1 -Video 1 -NoDisplay
```

### Quick CPU test

```powershell
.\scripts\run_video.ps1 -Video 1 -Device cpu -MaxFrames 300
```

## Database Modes

Three modes are supported:

```text
off     → inference only
local   → local MySQL database
cloud   → cloud MySQL database
```

Database schema, migrations, and seed files are available under:

```text
database/
```

Example:

```powershell
.\scripts\run_video.ps1 -Video 1 -Database local -NoDisplay
```

## Dashboard

Launch the packaged dashboard with:

```powershell
warehouse-vision dashboard
```

or:

```powershell
.\scripts\run_dashboard.ps1 -Database local
```

The dashboard is read-only and runs separately from video inference.

## Tests

Run the automated test suite:

```powershell
python -m pytest -q
```

Current verified result:

```text
37 passed, 10 subtests passed
```

## Generated and Private Files

The repository intentionally excludes:

```text
.env
.venv/
.vs/
models/*.pt
data/input_videos/*
outputs/*
__pycache__/
```

This keeps credentials, model weights, videos, generated outputs, and machine-specific files out of GitHub.

## Project Goal

This project demonstrates an end-to-end warehouse computer vision workflow combining object detection, tracking, QR identity resolution, spatial association, warehouse event processing, database integration, and visualization.
