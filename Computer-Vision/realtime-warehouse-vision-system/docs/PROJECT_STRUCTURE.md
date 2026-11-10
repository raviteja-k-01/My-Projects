realtime-warehouse-vision-system/
|-- configs/
|-- data/
|   |-- input_videos/
|   `-- master/
|-- database/
|   |-- migrations/
|   |-- schema/
|   `-- seeds/
|-- docs/
|-- models/
|-- outputs/
|-- scripts/
|   |-- run_video.ps1
|   |-- run_dashboard.ps1
|   |-- seed_local_database.ps1
|   |-- cloud_db_check.py
|   |-- cloud_run_report.py
|   `-- check_cloud_data.sql
|-- src/
|   `-- visiontrack/
|       |-- association/
|       |-- config/
|       |-- dashboard/
|       |-- database/
|       |-- geometry/
|       |-- perception/
|       |-- pipeline/
|       |-- warehouse/
|       |-- cli.py
|       |-- main.py
|       `-- __main__.py
|-- tests/
|-- .env.example
|-- .gitattributes
|-- .gitignore
|-- pyproject.toml
|-- README.md
|-- requirements.txt
`-- run.cmd

Runtime Flow

1. run.cmd starts video or dashboard mode.
2. run_video.ps1 starts video processing.
3. visiontrack.main routes the command.
4. YOLO detects warehouse objects.
5. QR codes provide identities.
6. The warehouse engine tracks state and events.
7. Results can be saved to MySQL.
8. The dashboard reads stored data.