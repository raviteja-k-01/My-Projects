from dataclasses import replace
import os
import shutil
import time
from types import SimpleNamespace
from uuid import uuid4

import cv2
import numpy as np
import pytest
import streamlit as st
from psycopg import Error as DatabaseError
from streamlit.testing.v1 import AppTest

from visiontrack.config.settings import project_path
from visiontrack.database import postgres_monitor as repository
from visiontrack.output_panel import HighQualityVideoWriter
from visiontrack.perception.pipeline import QRTrackIdentityCache, TrackContinuity, conflicting_identities
from visiontrack.perception.raw_observation import RawObservation
from visiontrack.pipeline import monitor
from visiontrack.pipeline.pallet_zone_pipeline import FrameResult
from visiontrack.warehouse.entities import BoxState
from visiontrack.warehouse.events import EventType, WarehouseEvent
from visiontrack.warehouse.pallet_zone_engine import PalletZoneEngine


def observation(kind, bbox, track, code=None):
    return RawObservation(kind, kind, 0.95, bbox, track, code, qr_read_success=bool(code))


def test_qr_cache_rechecks_discontinuous_or_old_reads():
    cache = QRTrackIdentityCache(stale_frames=4, recheck=True)
    original = observation("qr_code", (0, 0, 30, 30), 1)
    cache.remember([{"value": "BOX_001", "bbox": original.bbox, "track_id": 1}], 1)
    assert cache.partition([original], 2)[0][0]["value"] == "BOX_001"
    moved = replace(original, bbox=(60, 0, 90, 30))
    assert cache.partition([moved], 3) == ([], [moved])
    assert cache.partition([original], 5) == ([], [original])


def test_identity_conflict_and_reused_tracker():
    first = observation("box", (0, 0, 30, 30), 7, "BOX_001")
    other = replace(first, track_id=8, bbox=(40, 0, 70, 30))
    assert conflicting_identities([(first.qr_value, first.bbox), (other.qr_value, other.bbox)]) == {"BOX_001"}
    assert conflicting_identities([(first.qr_value, first.bbox)] * 2) == set()
    tracks = TrackContinuity(max_gap=2)
    original, = tracks.update([first], 1)
    assert tracks.update([first], 2)[0].track_id == original.track_id
    reused, = tracks.update([replace(first, qr_value=None)], 5)
    assert reused.track_id != original.track_id
    assert reused.qr_value is None


def test_confirmed_transfer_occlusion_and_dock_arrival():
    engine = PalletZoneEngine({"dynamic_location": {
        "record_region_arrivals": True, "confirmation_frames": 2, "startup_baseline_frames": 2,
    }})
    a = observation("pallet", (0, 0, 120, 120), 1, "PALLET_01")
    b = observation("pallet", (160, 0, 280, 120), 2, "PALLET_02")
    box = observation("box", (20, 20, 60, 60), 3, "BOX_001")
    for _ in range(4):
        engine.update([a, b, box])
    assert engine.boxes[box.qr_value].current_pallet_code == a.qr_value
    for _ in range(40):
        assert engine.update([]) == []
    assert engine.boxes[box.qr_value].current_pallet_code == a.qr_value

    transferred = replace(box, track_id=4, bbox=(170, 20, 210, 60))
    assert engine.update([a, b, transferred]) == []
    assert engine.update([a, b], blocked_identities={"BOX_001"}) == []
    assert engine.boxes[box.qr_value].current_pallet_code == a.qr_value
    events = []
    for _ in range(8):
        events.extend(engine.update([a, b, transferred]))
    assert [(event.event_type, event.from_pallet_code, event.to_pallet_code) for event in events] == [
        (EventType.MOVE, "PALLET_01", "PALLET_02"),
    ]
    assert list(engine.boxes) == ["BOX_001"]
    dock = observation("dock", (300, 0, 500, 120), 6, "DOCK_01")
    at_dock = replace(transferred, bbox=(325, 20, 365, 60))
    events = []
    for _ in range(6):
        events.extend(engine.update([a, b, dock, at_dock]))
    assert [event.event_type for event in events] == [EventType.DOCK_ARRIVAL]
    assert engine.boxes[box.qr_value].current_zone_code == "DOCK_01"


@pytest.fixture
def isolated_database(monkeypatch):
    port = os.getenv("VISIONTRACK_TEST_PG_PORT")
    if not port:
        pytest.skip("Set VISIONTRACK_TEST_PG_PORT for the disposable PostgreSQL instance.")
    assert port != "5432"
    monkeypatch.setenv("VISIONTRACK_PG_HOST", "127.0.0.1")
    monkeypatch.setenv("VISIONTRACK_PG_PORT", port)
    monkeypatch.setenv("VISIONTRACK_PG_USER", "visiontrack_test")
    monkeypatch.setenv("VISIONTRACK_PG_DATABASE", "postgres")
    repository.migrate()
    repository.migrate()


def test_postgres_deduplication_run_isolation_and_dashboard(isolated_database, tmp_path):
    one, two = uuid4(), uuid4()
    video = monitor.Video(tmp_path / "source.mp4", "test.mp4", 120, 160, 90, 25, 10)
    event = WarehouseEvent(EventType.MOVE, "BOX_TEST", "PALLET_01", "PALLET_02")
    records = [(2, .04, event)]
    inventory = [("box", "BOX_TEST", "PALLET_02", "ZONE_01", .04)]
    with repository.connection() as database:
        repository.create_run(database, one, video, "test.pt")
        repository.create_run(database, two, video, "test.pt")
        repository.persist(database, one, records, inventory, {"BOX_X": (1, 2, 2)})
        repository.persist(database, one, records, inventory, {})
        with pytest.raises(DatabaseError):
            repository.persist(database, two, records, [("invalid", *inventory[0][1:])], {})
        repository.finish_run(database, one, "COMPLETED", 10, .5, "cpu", None)
    run, events, objects, conflicts = repository.load_run(one)
    assert run["status"] == "COMPLETED"
    assert len(events) == 1 and events[0]["object_code"] == "BOX_TEST"
    assert objects[0]["pallet"] == "PALLET_02"
    assert conflicts[0]["frames"] == 2
    assert not repository.load_run(two)[1]
    assert any(row["run_id"] == one for row in repository.object_history("BOX_TEST"))

    app = AppTest.from_file(str(project_path("src/visiontrack/dashboard/monitor_app.py")),
                            default_timeout=30)
    app.run()
    assert not app.exception
    assert app.sidebar.radio[0].value == "Live Monitoring"
    app.sidebar.radio[0].set_value("Overview").run()
    app.selectbox[0].set_value(one).run()
    assert not app.exception
    assert any(metric.label == "Movements" and metric.value == "1" for metric in app.metric)
    app.sidebar.radio[0].set_value("Analytics").run()
    assert not app.exception and app.selectbox[0].value == one
    assert len(app.get("plotly_chart")) == 3
    app.sidebar.radio[0].set_value("Inventory").run()
    assert not app.exception


@pytest.mark.skipif(shutil.which("ffmpeg") is None, reason="FFmpeg required")
def test_video_output_and_timing(tmp_path):
    path = tmp_path / "annotated.mp4"
    writer = HighQualityVideoWriter(path, 160, 90, 25)
    for index in range(10):
        writer.write(np.full((90, 160, 3), index * 20, dtype=np.uint8))
    writer.release()
    writer.release()
    video = monitor.inspect_video(path, path.name)
    assert video.frames == 10
    assert video.fps == pytest.approx(25)
    capture = cv2.VideoCapture(str(path))
    means = []
    while True:
        ok, frame = capture.read()
        if not ok:
            break
        means.append(frame.mean())
    capture.release()
    assert len(means) == 10 and means == sorted(means)


def test_cancellation_and_single_active_worker(isolated_database, tmp_path, monkeypatch):
    entered = monitor.Event()
    worker = monitor.Monitor()
    video = monitor.Video(tmp_path / "source.mp4", "cancel.mp4", 1, 160, 90, 25, 100)
    actual_path = monitor.project_path
    monkeypatch.setattr(monitor, "project_path", lambda value: (
        tmp_path if value == "outputs/monitor" else actual_path(value)
    ))

    def blocking_process(*args):
        entered.set()
        assert worker.cancel.wait(5)
        return 0, None, 0, 0

    monkeypatch.setattr(worker, "_process", blocking_process)
    run_id = worker.start(video)
    assert entered.wait(5)
    assert worker.start(video) == run_id
    worker.stop()
    deadline = time.monotonic() + 5
    while worker.snapshot().status == "PROCESSING" and time.monotonic() < deadline:
        time.sleep(.01)
    assert worker.snapshot().status == "STOPPED"
    assert repository.load_run(run_id)[0]["status"] == "STOPPED"
    assert worker.snapshot().output is None


def test_model_failure_is_recorded(isolated_database, tmp_path, monkeypatch):
    worker = monitor.Monitor()
    video = monitor.Video(tmp_path / "source.mp4", "missing-model.mp4", 1, 160, 90, 25, 10)

    def missing_model(*args):
        raise FileNotFoundError("model checkpoint is missing")

    monkeypatch.setattr(monitor, "choose_model_profile", missing_model)
    run_id = worker.start(video)
    deadline = time.monotonic() + 5
    while worker.snapshot().status == "PROCESSING" and time.monotonic() < deadline:
        time.sleep(.01)
    run, events, inventory, _ = repository.load_run(run_id)
    assert worker.snapshot().status == run["status"] == "FAILED"
    assert "model checkpoint is missing" in run["error"]
    assert not events and not inventory


@pytest.mark.skipif(shutil.which("ffmpeg") is None, reason="FFmpeg required")
def test_streamlit_upload_preview_and_download(isolated_database, tmp_path, monkeypatch):
    source = tmp_path / "uploaded.mp4"
    writer = HighQualityVideoWriter(source, 160, 90, 25)
    for _ in range(10):
        writer.write(np.zeros((90, 160, 3), dtype=np.uint8))
    writer.release()

    worker = monitor.Monitor()
    monkeypatch.setattr(monitor, "Monitor", lambda: worker)
    actual_path = monitor.project_path
    monkeypatch.setattr(monitor, "project_path", lambda value: (
        tmp_path / "rendered" if value == "outputs/monitor" else actual_path(value)
    ))
    release = monitor.Event()
    box = observation("box", (20, 20, 60, 60), 1, "BOX_UI")
    event = WarehouseEvent(EventType.MOVE, box.qr_value, to_zone_code="ZONE_01",
                           qr_read_success=True)
    engine = PalletZoneEngine({})
    engine.boxes[box.qr_value] = BoxState(box.qr_value, current_zone_code="ZONE_01")
    engine.track_identities["BOX", 1] = box.qr_value

    def process_frame(frame, number):
        if number == 2:
            assert release.wait(20)
        return FrameResult(number, [box], [box], [], [event] if number == 1 else [])

    monkeypatch.setattr(monitor, "Detector", lambda profile: SimpleNamespace(
        names=dict(enumerate(sorted(monitor.CLASSES))), device_label="cpu", profile=profile,
    ))
    monkeypatch.setattr(monitor, "PalletZonePipeline", lambda *args: SimpleNamespace(
        engine=engine, perception=SimpleNamespace(identity_conflicts=set()),
        process_frame=process_frame,
    ))
    st.cache_resource.clear()
    app = AppTest.from_file(str(project_path("src/visiontrack/dashboard/monitor_app.py")),
                            default_timeout=30)
    try:
        app.run()
        app.file_uploader[0].set_value(("uploaded.mp4", source.read_bytes(), "video/mp4")).run()
        assert not app.exception
        assert any(metric.label == "Resolution" and metric.value == "160 × 90" for metric in app.metric)
        next(button for button in app.button if button.label == "Start processing").click().run()
        deadline = time.monotonic() + 10
        while not worker.snapshot().preview and time.monotonic() < deadline:
            time.sleep(.01)
        progress = worker.snapshot()
        assert progress.status == "PROCESSING" and progress.preview
        app.run()
        assert not app.exception
        assert app.get("image")
        assert worker.snapshot().run_id == progress.run_id
    finally:
        release.set()
    deadline = time.monotonic() + 10
    while worker.snapshot().status == "PROCESSING" and time.monotonic() < deadline:
        time.sleep(.01)
    assert worker.snapshot().status == "COMPLETED"
    app.run()
    assert not app.exception
    assert app.get("video")
    assert app.get("download_button")
    assert repository.load_run(progress.run_id)[1][0]["object_code"] == box.qr_value
