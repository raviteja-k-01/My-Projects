# VisionTrack — AI-Powered Warehouse Monitoring

VisionTrack helps turn warehouse video into a record of **which boxes were seen, where they were observed, and how they moved**. It combines object detection, QR identification, tracking, and warehouse movement logic in a single application.

Upload an MP4, follow the annotated preview as it processes, and review the results in a Streamlit dashboard. The application works with **prerecorded footage**, not a live CCTV feed.

## What it does

- **Detects and tracks:** A fine-tuned, six-class YOLOv8 model detects boxes, pallets, QR codes, conveyors, docks, and zones. ByteTrack follows detections across frames.
- **Identifies and locates:** QR decoding helps identify boxes. Spatial and temporal logic associates them with pallets and detected warehouse regions.
- **Records movements:** The event engine records observations such as zone transfers and dock arrivals, while unresolved identities remain available for review.
- **Makes footage reviewable:** PostgreSQL stores run records and events; Streamlit shows inventory, movement history, exceptions, and analytics. An annotated MP4 can be downloaded after processing.

## Application screenshots

### Live monitoring
![Annotated warehouse footage in VisionTrack](docs/screenshots/live-monitoring.png)

### Processing progress
![Frame count, progress and processing speed](docs/screenshots/processing-status.png)

### Analytics
![Inventory and movement analytics](docs/screenshots/analytics.png)

### Movement timeline
![Movement events along the video timeline](docs/screenshots/movement-timeline.png)

### Exceptions
![Unresolved tracks and identity exceptions](docs/screenshots/exceptions.png)

## Technology

**Python · YOLOv8 · PyTorch · OpenCV · ByteTrack · Pyzbar · PostgreSQL · Streamlit · Plotly · FFmpeg**

The pipeline decodes footage, detects and tracks objects, resolves available QR identities, associates boxes with pallets and locations, then writes movement records and annotated video output.

## Run locally

Requires **Python 3.11**, **PostgreSQL**, **FFmpeg**, and the compatible YOLOv8 checkpoint (not included in the repository).

From the project directory in Windows CMD:

```cmd
py -3.11 -m venv .venv
.venv\Scripts\python.exe -m pip install -r requirements.txt
.venv\Scripts\python.exe -m pip install -e .
copy .env.example .env
```

Set the `VISIONTRACK_PG_*` connection values in `.env`, create the `visiontrack` PostgreSQL database, and apply `database/migrations/003_postgresql_monitoring.sql`. Place the authorized checkpoint at `models/six_class_final.pt`.

```cmd
set "PYTHONPATH=%CD%\src"
.venv\Scripts\python.exe -m visiontrack monitor
```

Open the Streamlit URL shown in the terminal and upload an MP4.

## Notes

Movement records reflect observations from video and may need review when tracking or QR identity is uncertain. Some labels can overlap in crowded scenes. The last reported test run also had one unresolved legacy event-engine test.
