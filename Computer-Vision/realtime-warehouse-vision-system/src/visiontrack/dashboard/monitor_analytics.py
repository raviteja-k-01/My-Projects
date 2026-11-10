"""Read-only analytics for one selected PostgreSQL processing run."""

from collections import Counter
from datetime import timezone

import pandas as pd
import plotly.graph_objects as go
import streamlit as st


TEAL = "#45d8c6"
BLUE = "#8ab8ff"
AMBER = "#ffc578"
MUTED = "#bdcde0"
GRID = "#294158"
EVENT_COLORS = {
    "ADD": TEAL, "MOVE": BLUE, "ZONE_TRANSFER": "#b8a2ff",
    "DOCK_ARRIVAL": AMBER, "CONVEYOR_ENTRY": "#75dc95", "REMOVE": "#ff969a",
}


def run_metrics(run, movements, inventory):
    elapsed = run["elapsed_seconds"]
    frames = run["processed_frames"]
    unresolved = run["unresolved"] if run["status"] in {"COMPLETED", "STOPPED"} else None
    return [
        ("Boxes", f'{sum(row["kind"] == "box" for row in inventory):,}', "QR-identified boxes"),
        ("Pallets", f'{sum(row["kind"] == "pallet" for row in inventory):,}', "QR-identified pallets"),
        ("Movements", f"{len(movements):,}", "Confirmed movements, excluding registration"),
        ("Unresolved", f"{unresolved:,}" if unresolved is not None else "—", "Unresolved object identities"),
        ("Processing FPS", f"{frames / elapsed:.2f}" if elapsed and elapsed > 0 and frames else "—",
         "Processed frames divided by measured run duration"),
    ]


def show_metrics(run, movements, inventory):
    with st.container(key="run_kpis"):
        columns = st.columns(5, gap="small")
        for column, (label, value, help_text) in zip(columns, run_metrics(run, movements, inventory)):
            column.metric(label, value, help=help_text, border=True)


def chart(figure, height, key):
    figure.update_layout(
        height=height, font=dict(size=15, color=MUTED), hoverlabel=dict(font_size=15),
        paper_bgcolor="#0b1320", plot_bgcolor="#0b1320", margin=dict(l=8, r=22, t=12, b=8),
        legend=dict(orientation="h", y=1, yanchor="bottom", font_size=14),
    )
    figure.update_xaxes(automargin=True, tickfont_size=14, gridcolor=GRID, zeroline=False)
    figure.update_yaxes(automargin=True, tickfont_size=14, gridcolor=GRID, zeroline=False)
    st.plotly_chart(figure, width="stretch", theme=None, key=key,
                    config={"displaylogo": False, "scrollZoom": False, "responsive": True})


def event_label(event_type):
    return event_type.replace("_", " ").title()


def event_color(event_type):
    return EVENT_COLORS.get(event_type, "#a9c5df")


def movement_counts(movements):
    counts = sorted(Counter(row["event_type"] for row in movements).items(), key=lambda pair: (pair[1], pair[0]))
    labels = [event_label(kind) for kind, _ in counts]
    values = [count for _, count in counts]
    figure = go.Figure(go.Bar(
        x=values, y=labels, orientation="h", marker_color=[event_color(kind) for kind, _ in counts],
        hovertemplate="%{y}<br>%{x} confirmed events<extra></extra>",
    ))
    figure.update_xaxes(rangemode="tozero", dtick=1 if max(values) <= 8 else None, title_text="Events")
    figure.update_yaxes(showgrid=False)
    chart(figure, 265, "movement-counts")


def location_rows(inventory):
    return [
        (next((value.strip() for value in (row["pallet"], row["location"])
               if value and value.strip() and "_TRACK_" not in value.upper()), None), row["kind"])
        for row in inventory
    ]


def inventory_locations(inventory):
    rows = location_rows(inventory)
    totals = Counter(location for location, _ in rows if location is not None)
    shown = [location for location, _ in sorted(totals.items(), key=lambda pair: (-pair[1], pair[0]))[:5]]
    # Use internal category IDs so reserved labels cannot collide with saved locations.
    groups = {f"location-{i}": name for i, name in enumerate(shown)}
    if len(totals) > len(shown):
        groups["other"] = "Other confirmed"
    if any(location is None for location, _ in rows):
        groups["unknown"] = "Location unknown"
    by_location = {name: key for key, name in groups.items() if key.startswith("location-")}
    counts = Counter((by_location.get(location, "other") if location is not None else "unknown", kind)
                     for location, kind in rows)
    figure = go.Figure()
    for kind, name, color in [("box", "Boxes", TEAL), ("pallet", "Pallets", BLUE)]:
        figure.add_bar(
            name=name, orientation="h", y=list(groups), customdata=list(groups.values()),
            x=[counts[key, kind] for key in groups],
            marker=dict(color=[AMBER if key == "unknown" else color for key in groups],
                        pattern_shape=["/" if key == "unknown" and kind == "pallet" else "" for key in groups]),
            hovertemplate=f"%{{customdata}}<br>%{{x}} {name.lower()}<extra></extra>",
        )
    figure.update_layout(barmode="stack")
    figure.update_xaxes(rangemode="tozero", title_text="Identified objects",
                        dtick=1 if max(sum(counts[key, kind] for kind in ("box", "pallet")) for key in groups) <= 8 else None)
    figure.update_yaxes(showgrid=False, tickmode="array", tickvals=list(groups),
                        ticktext=[label if len(label) <= 23 else label[:20] + "…" for label in groups.values()],
                        autorange="reversed")
    chart(figure, 265, "inventory-locations")


def video_time(seconds):
    minutes, remainder = divmod(int(round(seconds * 1000)), 60_000)
    return f"{minutes:02d}:{remainder / 1000:06.3f}"


def movement_timeline(movements):
    ordered = sorted(movements, key=lambda row: (row["source_seconds"], row["frame"]))
    event_types = sorted({row["event_type"] for row in ordered})
    selected = st.multiselect("Event types", event_types, format_func=event_label,
                              placeholder="All event types")
    if selected:
        ordered = [row for row in ordered if row["event_type"] in selected]
    object_ids = sorted({row["object_code"] for row in ordered})
    object_lanes = len(object_ids) <= 10
    figure = go.Figure()
    details = []
    for row in ordered:
        details.append({
            "Video timestamp": video_time(row["source_seconds"]),
            "Event": event_label(row["event_type"]),
            "Object": row["object_code"],
            "Saved at (UTC)": row["recorded_at"].astimezone(timezone.utc).isoformat(timespec="milliseconds"),
        })
    for event_type in selected or event_types:
        points = [(row, detail) for row, detail in zip(ordered, details) if row["event_type"] == event_type]
        label = event_label(event_type)
        figure.add_scatter(
            mode="markers", name=label, x=[row["source_seconds"] for row, _ in points],
            y=[detail["Object"] if object_lanes else label for _, detail in points],
            marker=dict(color=event_color(event_type), size=10),
            customdata=[[detail["Object"], detail["Video timestamp"], detail["Saved at (UTC)"]]
                        for _, detail in points],
            hovertemplate=f"<b>%{{customdata[0]}}</b><br>{label}<br>Video %{{customdata[1]}}"
                          "<br>Saved %{customdata[2]}<extra></extra>",
        )
    figure.update_layout(showlegend=object_lanes)
    figure.update_xaxes(title_text="Video time (seconds)", rangemode="tozero")
    figure.update_yaxes(type="category", categoryorder="array",
                        categoryarray=object_ids if object_lanes else [event_label(t) for t in event_types])
    chart(figure, max(265, 115 + 22 * len(object_ids)) if object_lanes else 265, "movement-timeline")
    with st.expander("View movement timestamps and object IDs"):
        st.dataframe(pd.DataFrame(details), hide_index=True, width="stretch")


def show_analytics(run, movements, inventory):
    show_metrics(run, movements, inventory)
    with st.container(key="analytics_panels"):
        left, right = st.columns(2, gap="medium")
    with left.container(border=True):
        st.subheader("Movement mix")
        st.caption("Confirmed events by type")
        if movements:
            movement_counts(movements)
        else:
            st.info("No confirmed movements in this run.")
    with right.container(border=True):
        st.subheader("Inventory locations")
        st.caption("Last confirmed location per object. Amber marks unknown locations.")
        if inventory:
            inventory_locations(inventory)
        else:
            st.info("No identified inventory in this run.")
    with st.container(border=True):
        st.subheader("Movement timeline")
        st.caption("Events are positioned by their timestamp in the source video. Hover for object ID and the UTC save time.")
        if movements:
            movement_timeline(movements)
        else:
            st.info("No confirmed movements to plot.")
