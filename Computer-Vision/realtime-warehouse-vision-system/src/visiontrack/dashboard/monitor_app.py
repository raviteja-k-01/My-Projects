"""VisionTrack 2.0 operator console."""

from pathlib import Path
from tempfile import TemporaryDirectory

from psycopg import Error as DatabaseError
import streamlit as st

from visiontrack.dashboard.monitor_analytics import show_analytics, show_metrics
from visiontrack.database import postgres_monitor as repository
from visiontrack.pipeline.monitor import Monitor, inspect_video

st.set_page_config(page_title="VisionTrack 2.0", page_icon="▣", layout="wide")
st.html("""
<style>
  [data-testid="stMainBlockContainer"] {
    padding: 4rem clamp(1rem, 3vw, 3rem) 2.5rem;
  }
  [data-testid="stMainBlockContainer"] h1 { font-size: 2.15rem; }
  [data-testid="stMainBlockContainer"] h2 { font-size: 1.65rem; }
  [data-testid="stMainBlockContainer"] h3 { font-size: 1.25rem; }
  [data-testid="stSidebar"] [role="radiogroup"] label p { font-size: 1rem; }
  .st-key-run_kpis { container-type: inline-size; }
  .st-key-run_kpis [data-testid="stHorizontalBlock"],
  .st-key-analytics_panels [data-testid="stHorizontalBlock"] { display: grid; }
  .st-key-run_kpis [data-testid="stHorizontalBlock"] {
    grid-template-columns: repeat(5, minmax(0, 1fr));
  }
  .st-key-analytics_panels [data-testid="stHorizontalBlock"] {
    grid-template-columns: repeat(auto-fit, minmax(min(100%, 17rem), 1fr));
  }
  .st-key-run_kpis [data-testid="stColumn"],
  .st-key-analytics_panels [data-testid="stColumn"] { width: auto; min-width: 0; }
  .st-key-run_kpis [data-testid="stMetricLabel"] p {
    white-space: normal;
  }
  .st-key-run_kpis [data-testid="stMetricValue"] { font-size: 1.7rem; }
  @container (max-width: 43rem) {
    .st-key-run_kpis [data-testid="stHorizontalBlock"] {
      grid-template-columns: repeat(3, minmax(0, 1fr));
    }
  }
  @container (max-width: 23rem) {
    .st-key-run_kpis [data-testid="stHorizontalBlock"] {
      grid-template-columns: repeat(2, minmax(0, 1fr));
    }
  }
</style>
""")
st.title("VisionTrack 2.0", anchor=False)
st.caption("Warehouse operations · Video monitoring and movement records")


@st.cache_resource
def monitor():
    return Monitor()


@st.cache_data(ttl=2, max_entries=32, show_spinner=False)
def live_events(run_id):
    return repository.recent_events(run_id)


def video_output(output):
    if not output or not Path(output).is_file():
        st.info("The annotated MP4 is unavailable on this server.")
        return
    path = Path(output)
    st.video(str(path))

    def read_download():
        return path.read_bytes()

    st.download_button(
        "Download annotated MP4", data=read_download, file_name=f"{path.parent.name}.mp4",
        mime="video/mp4", on_click="ignore", key=f"download-{path.parent.name}",
    )


@st.fragment(run_every=0.5)
def live_view():
    worker = monitor()
    current = worker.snapshot()
    if current is None:
        return
    st.subheader("Current run")
    st.caption(f"Run {current.run_id} · {current.device or 'Device pending'}")
    status, frames, speed, elapsed = st.columns(4)
    status.metric("Status", current.status.title(), border=True)
    frames.metric("Frames", f"{current.frames:,}", border=True)
    speed.metric("Processing speed", f"{current.frames / current.elapsed:.1f} FPS"
                 if current.elapsed and current.frames else "—", border=True)
    elapsed.metric("Elapsed", f"{current.elapsed:.1f} s", border=True)
    if current.total:
        st.progress(min(1.0, current.frames / current.total))
    if current.status != "PROCESSING":
        # Refresh the whole page once so completion unlocks the upload controls.
        st.rerun()
    if st.button("Stop processing", key="stop_processing"):
        worker.stop()
        st.info("Stopping after the current frame and saving committed observations.")
    if current.preview:
        st.image(current.preview, width="stretch")
    else:
        st.info("Waiting for the first annotated frame.")
    if current.frames:
        try:
            recent = live_events(current.run_id)
        except DatabaseError:
            st.warning("Recent events are temporarily unavailable from PostgreSQL.")
        else:
            if recent:
                st.caption("Recent confirmed events saved to PostgreSQL")
                st.dataframe(recent, hide_index=True, width="stretch")


def live_monitoring():
    st.caption("Simulated live monitoring of uploaded MP4 footage.")
    worker = monitor()
    current = worker.snapshot()
    if current and current.status == "PROCESSING":
        live_view()
        return
    upload = st.file_uploader("Warehouse video", type=["mp4"])
    if upload is None:
        st.session_state.pop("prepared_upload", None)
    else:
        prepared = st.session_state.get("prepared_upload")
        if prepared is None or prepared[0] != upload.file_id:
            try:
                owner = TemporaryDirectory(prefix="visiontrack-upload-", dir=upload_directory())
                path = Path(owner.name) / "source.mp4"
                with path.open("wb") as destination:
                    destination.write(upload.getbuffer())
                video = inspect_video(path, upload.name)
            except (OSError, ValueError) as error:
                st.error(str(error))
                return
            prepared = (upload.file_id, video, owner)
            st.session_state.prepared_upload = prepared
        _, video, owner = prepared
        resolution, fps, duration, size = st.columns(4)
        resolution.metric("Resolution", f"{video.width} × {video.height}", border=True)
        fps.metric("Source speed", f"{video.fps:.2f} FPS", border=True)
        duration.metric("Duration", f"{video.frames / video.fps:.1f} s" if video.frames else "—",
                        border=True)
        size.metric("File size", f"{video.size / 1_048_576:.1f} MB", border=True)
        if st.button("Start processing", type="primary"):
            worker.start(video, source_owner=owner)
            st.rerun()
    if current:
        st.caption(f"Last run {current.run_id} · {current.status.title()}")
        if current.error:
            st.error(current.error)
        if current.output:
            video_output(current.output)


def upload_directory():
    from visiontrack.config.settings import project_path

    path = project_path("outputs/uploads")
    path.mkdir(parents=True, exist_ok=True)
    return path


def matches(rows, search):
    return [row for row in rows if search.casefold() in row["object_code"].casefold()]


def history(section):
    if section == "Movement History":
        lookup = st.text_input("Exact object ID across all runs")
        if lookup.strip():
            from visiontrack.identity import canonicalize_identity

            rows = repository.object_history(canonicalize_identity(lookup) or lookup.strip())
            if rows:
                st.dataframe(rows, hide_index=True, width="stretch")
            else:
                st.info("No movement history for this ID.")
    runs = repository.list_runs()
    if not runs:
        st.info("Process an MP4 to populate the warehouse records.")
        return
    labels = {
        run["id"]: f'{run["source_name"]} · {run["status"].title()} · {run["started_at"]:%Y-%m-%d %H:%M} · {str(run["id"])[:8]}'
        for run in runs
    }
    previous = st.session_state.get("selected_processing_run")
    if previous not in labels:
        st.session_state.selected_processing_run = next(iter(labels))
    selected = st.selectbox(
        "Processing run", list(labels), format_func=labels.get, key="run_selector",
        index=list(labels).index(st.session_state.selected_processing_run),
    )
    st.session_state.selected_processing_run = selected
    run, events, inventory, exceptions = repository.load_run(selected)
    movements = [event for event in events if event["event_type"] != "REGISTERED"]
    if run["status"] != "COMPLETED":
        st.info(f'{run["status"].title()} run · Showing evidence saved so far.')
    if section == "Overview":
        show_metrics(run, movements, inventory)
        st.subheader("Processing performance")
        elapsed = run["elapsed_seconds"] or 0
        rate = f'{run["processed_frames"] / elapsed:.2f} FPS' if elapsed else "Pending"
        st.caption(f'{run["source_width"]} × {run["source_height"]} · Source {run["source_fps"]:.2f} FPS · '
                   f'{run["processed_frames"]:,} frames · {elapsed:.1f} s · {rate} · {run["device"] or "Device pending"}')
        st.subheader("Recent movements")
        show_table(movements[:8], "No confirmed movements in this run.")
        if run["output_path"]:
            video_output(run["output_path"])
    elif section == "Inventory":
        search = st.text_input("Search box or pallet ID")
        show_table(matches(inventory, search), "No identified inventory matches this search.")
    elif section == "Movement History":
        search = st.text_input("Filter this run by object ID")
        types = st.multiselect("Event type", sorted({row["event_type"] for row in events}))
        rows = [row for row in matches(events, search) if not types or row["event_type"] in types]
        show_table(rows, "No records match these filters.")
    elif section == "Exceptions":
        st.metric("Unresolved object tracks", run["unresolved"], border=True)
        st.metric("Movements awaiting or lacking identity evidence", run["unresolved_movements"], border=True)
        if run["error"]:
            st.error(run["error"])
        st.subheader("Conflicting identity claims")
        show_table(exceptions, "No identity conflicts recorded.")
    elif section == "Analytics":
        show_analytics(run, movements, inventory)


def show_table(rows, empty):
    if rows:
        st.dataframe(rows, hide_index=True, width="stretch")
    else:
        st.info(empty)


st.sidebar.header("Operations console", anchor=False)
section = st.sidebar.radio(
    "Operations", ["Live Monitoring", "Overview", "Inventory", "Movement History", "Exceptions", "Analytics"],
)
st.header(section, anchor=False)
if section == "Live Monitoring":
    live_monitoring()
else:
    try:
        history(section)
    except DatabaseError as error:
        st.error(f"PostgreSQL records are unavailable. {error}")
