
import os
import math
from datetime import datetime
from typing import Any, Optional

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from visiontrack.dashboard.data_access import (
    DashboardDatabaseConfig,
    MySQLReadClient,
)

try:
    from dotenv import load_dotenv
    load_dotenv()
except Exception:
    pass

# ============================================================
# PAGE SETUP
# ============================================================
st.set_page_config(
    page_title="VisionTrack Warehouse Dashboard",
    page_icon="📦",
    layout="wide",
    initial_sidebar_state="collapsed",
)

st.markdown(
    """
    <style>
        :root {
            --vt-primary: #0b3b60;
            --vt-primary-2: #155e95;
            --vt-blue: #2563eb;
            --vt-cyan: #0ea5e9;
            --vt-green: #16a34a;
            --vt-teal: #0f766e;
            --vt-amber: #f59e0b;
            --vt-red: #dc2626;
            --vt-purple: #7c3aed;
            --vt-brown: #9a6a3a;
            --vt-text: #172033;
            --vt-muted: #64748b;
            --vt-border: #d8e2ee;
            --vt-surface: #ffffff;
            --vt-background: #f4f7fb;
            --vt-soft-blue: #eaf2ff;
            --vt-shadow: 0 8px 24px rgba(15, 49, 82, 0.10);
        }

        .stApp {
            background:
                radial-gradient(circle at 8% 0%, rgba(37, 99, 235, 0.10), transparent 30%),
                radial-gradient(circle at 96% 12%, rgba(14, 165, 233, 0.08), transparent 26%),
                linear-gradient(180deg, #f8fbff 0%, var(--vt-background) 100%);
            color: var(--vt-text);
        }

        .block-container {
            max-width: 1580px;
            padding-top: 3.4rem;
            padding-bottom: 2rem;
        }

        .main-title {
            text-align: center;
            color: var(--vt-primary);
            font-size: 38px;
            font-weight: 900;
            line-height: 1.35;
            letter-spacing: 0.7px;
            padding: 12px 0 10px 0;
            margin: 0 0 24px 0;
            overflow: visible;
            text-shadow: 0 2px 10px rgba(11, 59, 96, 0.08);
        }

        .main-title::after {
            content: "";
            display: block;
            width: 104px;
            height: 4px;
            margin: 10px auto 0 auto;
            border-radius: 999px;
            background: linear-gradient(90deg, var(--vt-blue), var(--vt-cyan), var(--vt-green));
        }

        .section-title {
            text-align: center;
            color: var(--vt-primary-2);
            font-size: 30px;
            font-weight: 900;
            letter-spacing: 0.4px;
            margin: 24px 0 14px 0;
        }

        .kpi-label {
            min-height: 30px;
            text-align: center;
            font-size: 12px;
            line-height: 1.05;
            color: #334155;
            font-weight: 700;
            margin-bottom: 6px;
        }

        .kpi-card {
            --accent: var(--vt-blue);
            position: relative;
            min-height: 84px;
            border: 1px solid var(--vt-border);
            border-radius: 14px;
            background: linear-gradient(145deg, #ffffff 0%, #f8fbff 100%);
            padding: 11px 6px 9px 6px;
            display: flex;
            flex-direction: column;
            justify-content: center;
            align-items: center;
            box-sizing: border-box;
            overflow: hidden;
            box-shadow: 0 4px 14px rgba(15, 49, 82, 0.06);
            transition: transform 180ms ease, box-shadow 180ms ease, border-color 180ms ease;
        }

        .kpi-card::before {
            content: "";
            position: absolute;
            inset: 0 0 auto 0;
            height: 4px;
            background: var(--accent);
        }

        .kpi-card:hover {
            transform: translateY(-4px);
            border-color: var(--accent);
            box-shadow: 0 12px 28px rgba(15, 49, 82, 0.15);
        }

        .kpi-value {
            color: var(--accent);
            font-size: 25px;
            font-weight: 750;
            white-space: nowrap;
        }

        .kpi-note {
            color: var(--vt-muted);
            font-size: 9px;
            margin-top: 6px;
            text-align: center;
        }

        .db-ok, .db-warning {
            padding: 10px 12px;
            border-radius: 11px;
            font-weight: 700;
            box-shadow: 0 3px 10px rgba(15, 49, 82, 0.06);
        }

        .db-ok {
            border: 1px solid #a7d8b4;
            background: linear-gradient(135deg, #e8f8ee, #f4fff7);
            color: #146c2e;
        }

        .db-warning {
            border: 1px solid #f3c96d;
            background: linear-gradient(135deg, #fff5dc, #fffaf0);
            color: #7d5200;
        }

        section[data-testid="stSidebar"] {
            background: linear-gradient(180deg, #eef5ff 0%, #f8fbff 52%, #edf8f4 100%);
            border-right: 1px solid var(--vt-border);
        }

        section[data-testid="stSidebar"] h1,
        section[data-testid="stSidebar"] h2,
        section[data-testid="stSidebar"] h3 {
            color: var(--vt-primary);
        }

        div[data-baseweb="select"] > div,
        div[data-baseweb="input"] > div,
        div[data-testid="stNumberInput"] input,
        div[data-testid="stTextInput"] input {
            border-color: #bfd0e5 !important;
            border-radius: 10px !important;
            background: rgba(255, 255, 255, 0.94) !important;
        }

        div[data-baseweb="select"] > div:hover,
        div[data-testid="stTextInput"] input:hover {
            border-color: var(--vt-blue) !important;
        }

        .stButton > button {
            border-radius: 10px;
            border: 1px solid var(--vt-blue);
            color: white;
            font-weight: 750;
            background: linear-gradient(135deg, var(--vt-primary-2), var(--vt-blue));
            box-shadow: 0 5px 14px rgba(37, 99, 235, 0.20);
            transition: transform 160ms ease, box-shadow 160ms ease;
        }

        .stButton > button:hover {
            transform: translateY(-2px);
            border-color: var(--vt-cyan);
            color: white;
            box-shadow: 0 9px 20px rgba(37, 99, 235, 0.28);
        }

        div[data-testid="stMetric"] {
            border: 1px solid var(--vt-border);
            border-radius: 14px;
            padding: 12px 14px;
            background: linear-gradient(145deg, white, #f8fbff);
            box-shadow: 0 4px 14px rgba(15, 49, 82, 0.06);
            transition: transform 180ms ease, box-shadow 180ms ease;
        }

        div[data-testid="stMetric"]:hover {
            transform: translateY(-3px);
            box-shadow: var(--vt-shadow);
        }

        div[data-testid="stExpander"] {
            border: 1px solid var(--vt-border);
            border-radius: 12px;
            background: rgba(255, 255, 255, 0.88);
            overflow: hidden;
        }

        div[data-testid="stDataFrame"] {
            border: 1px solid var(--vt-border);
            border-radius: 12px;
            overflow: hidden;
            box-shadow: 0 4px 14px rgba(15, 49, 82, 0.06);
        }

        div[data-testid="stPlotlyChart"] {
            border: 1px solid var(--vt-border);
            border-radius: 16px;
            background: rgba(255, 255, 255, 0.94);
            box-shadow: var(--vt-shadow);
            overflow: hidden;
            transition: box-shadow 180ms ease, border-color 180ms ease;
        }

        div[data-testid="stPlotlyChart"]:hover {
            border-color: #a9c7ed;
            box-shadow: 0 14px 32px rgba(15, 49, 82, 0.16);
        }

        header[data-testid="stHeader"] {
            background: transparent;
        }

        div[data-testid="stToolbar"] {
            top: 0.4rem;
        }

        @media (max-width: 900px) {
            .main-title { font-size: 30px; }
            .section-title { font-size: 24px; }
            .kpi-value { font-size: 21px; }
        }
    </style>
    """,
    unsafe_allow_html=True,
)


# ============================================================
# CONFIGURATION
# ============================================================
def get_env_config() -> dict[str, Any]:
    return {
        "host": os.getenv("DB_HOST", ""),
        "port": int(os.getenv("DB_PORT", "3306") or 3306),
        "user": os.getenv("DB_USER", ""),
        "password": os.getenv("DB_PASSWORD", ""),
        "database": os.getenv("DB_NAME", "u906725174_TrackVision"),
    }


cfg = get_env_config()
DATABASE_TARGET = os.getenv(
    "VISIONTRACK_DASHBOARD_DATABASE",
    "local",
).strip().lower()


def _bounded_refresh_seconds() -> int:
    try:
        configured = int(os.getenv("DASHBOARD_REFRESH_SECONDS", "5"))
    except ValueError:
        configured = 5
    return min(max(configured, 1), 60)


DASHBOARD_QUERY_TTL_SECONDS = _bounded_refresh_seconds()


def _database_config(
    host: str,
    port: int,
    user: str,
    password: str,
    database: str,
) -> DashboardDatabaseConfig:
    return DashboardDatabaseConfig(
        host=host,
        port=int(port),
        user=user,
        password=password,
        database=database,
    )


# ============================================================
# SIDEBAR
# ============================================================
with st.sidebar:
    st.title("VisionTrack")

    page = st.radio(
        "Dashboard page",
        [
            "Live Operations",
            "Order Control Center",
            "Exceptions & Events",
        ],
    )

    st.divider()
    st.subheader("Database")

    env_ready = _database_config(
        cfg["host"],
        int(cfg["port"]),
        cfg["user"],
        cfg["password"],
        cfg["database"],
    ).is_ready(DATABASE_TARGET)

    if env_ready:
        st.markdown(
            '<div class="db-ok">✅ Credentials loaded from .env</div>',
            unsafe_allow_html=True,
        )
    else:
        st.markdown(
            '<div class="db-warning">⚠️ Enter credentials or create .env</div>',
            unsafe_allow_html=True,
        )

    with st.expander("Connection settings", expanded=not env_ready):
        DB_HOST = st.text_input("Host", value=cfg["host"])
        DB_PORT = st.number_input(
            "Port",
            value=int(cfg["port"]),
            min_value=1,
            max_value=65535,
        )
        DB_USER = st.text_input("User", value=cfg["user"])
        DB_PASSWORD = st.text_input(
            "Password",
            value=cfg["password"],
            type="password",
        )
        DB_NAME = st.text_input("Database", value=cfg["database"])

    if st.button("Refresh dashboard", width="stretch"):
        # Force fresh database results only when manually requested.
        st.cache_data.clear()
        st.rerun()

    st.caption(
        f"Database reads are cached for at most {DASHBOARD_QUERY_TTL_SECONDS} "
        "seconds. Use Refresh dashboard for an immediate reload."
    )


# ============================================================
# DATABASE ACCESS
# One cached, reconnecting read-only connection avoids repeated cloud
# authentication handshakes while preserving the existing query cache.
# ============================================================
def database_ready() -> bool:
    return _database_config(
        DB_HOST,
        int(DB_PORT),
        DB_USER,
        DB_PASSWORD,
        DB_NAME,
    ).is_ready(DATABASE_TARGET)


@st.cache_data(ttl=DASHBOARD_QUERY_TTL_SECONDS, show_spinner=False)
def _cached_database_query(
    sql: str,
    params: tuple,
    host: str,
    port: int,
    user: str,
    password: str,
    database: str,
) -> pd.DataFrame:
    """
    Cache identical database queries for a short, bounded live-data window.

    Streamlit reruns the complete script whenever the user changes pages.
    Without this cache, every page switch opens many new remote MySQL
    connections. The cache makes page switching much faster while the
    Refresh dashboard button still allows immediate fresh data.
    """
    config = _database_config(host, port, user, password, database)
    if not config.is_ready(DATABASE_TARGET):
        return pd.DataFrame()

    client = _database_client(config)
    return client.query(sql, params)


@st.cache_resource(show_spinner=False)
def _database_client(
    config: DashboardDatabaseConfig,
) -> MySQLReadClient:
    return MySQLReadClient(config)


def fetch_dataframe(
    sql: str,
    params: Optional[tuple] = None,
) -> pd.DataFrame:
    return _cached_database_query(
        sql=sql,
        params=params or (),
        host=DB_HOST,
        port=int(DB_PORT),
        user=DB_USER,
        password=DB_PASSWORD,
        database=DB_NAME,
    )


_QUERY_FAILURES: dict[str, str] = {}
_schema_inventory_snapshot: Optional[pd.DataFrame] = None


def _record_query_failure(label: str, exc: Exception) -> None:
    """Record a safe error summary without exposing credentials or SQL."""
    _QUERY_FAILURES[label] = type(exc).__name__


def safe_fetch(
    sql: str,
    params: Optional[tuple] = None,
    label: str = "data",
    show_error: bool = True,
) -> pd.DataFrame:
    try:
        return fetch_dataframe(sql, params)
    except Exception as exc:
        _record_query_failure(label, exc)
        if show_error:
            st.error(
                f"Could not load {label}. Check the database connection and schema "
                f"({type(exc).__name__})."
            )
        return pd.DataFrame()


def _current_schema_inventory() -> pd.DataFrame:
    global _schema_inventory_snapshot

    if _schema_inventory_snapshot is not None:
        return _schema_inventory_snapshot

    try:
        _schema_inventory_snapshot = _schema_inventory(
            DB_HOST,
            int(DB_PORT),
            DB_USER,
            DB_PASSWORD,
            DB_NAME,
        )
    except Exception as exc:
        _record_query_failure("database schema", exc)
        _schema_inventory_snapshot = pd.DataFrame()

    return _schema_inventory_snapshot


def table_exists(table_name: str) -> bool:
    inventory = _current_schema_inventory()
    if inventory.empty:
        return False

    normalized_columns = {
        str(col).lower(): col for col in inventory.columns
    }
    table_name_column = normalized_columns.get("table_name")
    if table_name_column is None:
        return False

    return table_name.lower() in set(
        inventory[table_name_column]
        .astype(str)
        .map(lambda value: str(value).lower())
    )


@st.cache_data(ttl=300, show_spinner=False)
def _schema_inventory(
    host: str,
    port: int,
    user: str,
    password: str,
    database: str,
) -> pd.DataFrame:
    return _cached_database_query(
        """
        SELECT
            table_name,
            column_name,
            ordinal_position
        FROM information_schema.columns
        WHERE table_schema = %s
        ORDER BY table_name, ordinal_position;
        """,
        (database,),
        host,
        int(port),
        user,
        password,
        database,
    )


def table_columns(table_name: str) -> list[str]:
    inventory = _current_schema_inventory()
    if inventory.empty:
        return []

    normalized_columns = {
        str(col).lower(): col for col in inventory.columns
    }
    table_name_column = normalized_columns.get("table_name")
    column_name_column = normalized_columns.get("column_name")
    if table_name_column is None or column_name_column is None:
        return []

    table_rows = inventory[
        inventory[table_name_column]
        .astype(str)
        .map(lambda value: str(value).lower())
        .eq(table_name.lower())
    ]
    return table_rows[column_name_column].astype(str).tolist()


def as_number(value: Any, default: float = 0.0) -> float:
    try:
        if value is None or pd.isna(value):
            return default
        return float(value)
    except Exception:
        return default


def format_value(
    value: Any,
    decimals: int = 0,
    suffix: str = "",
    unavailable: bool = False,
) -> str:
    if unavailable:
        return "N/A"

    try:
        if value is None or pd.isna(value):
            return f"0{suffix}"
        return f"{float(value):,.{decimals}f}{suffix}"
    except Exception:
        return f"0{suffix}"



# ============================================================
# CANONICAL BOX LIFECYCLE CLASSIFICATION
# Current location is derived from the latest order_lifecycle_events row.
# The removed boxes.current_zone_id/current_pallet_id columns are not used.
# ============================================================
def classify_box_stage(
    box_status: Any,
    lifecycle_status: Any,
    pallet_code: Any,
) -> str:
    """Map every physical box to one of the five dashboard states."""
    raw_box_status = str(box_status or "").strip().upper()
    lifecycle = str(lifecycle_status or "").strip().upper()

    has_pallet = (
        pallet_code is not None
        and not pd.isna(pallet_code)
        and str(pallet_code).strip() != ""
    )

    if "DISPATCH" in lifecycle or raw_box_status == "DISPATCHED":
        return "DISPATCHED"

    if "CONVEYOR" in lifecycle:
        return "ON_CONVEYOR"

    # Missing/removed describes condition, not a verified physical location.
    if raw_box_status in {"REMOVED", "MISSING"} or lifecycle in {
        "REMOVED",
        "MISSING",
    }:
        return "UNLOCATED"

    # AT_DOCK and LOADED_TO_TRAILER remain part of the pallet flow because
    # the current schema still records their pallet_id in the latest event.
    if has_pallet or "PALLET" in lifecycle or "DOCK" in lifecycle or "LOADED" in lifecycle:
        return "ON_PALLET"

    if lifecycle in {"RECEIVED", "ON_FLOOR", "IN_ZONE"}:
        return "ON_FLOOR"

    return "UNLOCATED"


def add_box_stage(frame: pd.DataFrame) -> pd.DataFrame:
    """Return a copy with one mutually exclusive operational_stage column."""
    if frame.empty:
        output = frame.copy()
        output["operational_stage"] = pd.Series(dtype="object")
        return output

    output = frame.copy()

    for column in ["box_status", "lifecycle_status", "pallet_code"]:
        if column not in output.columns:
            output[column] = None

    output["operational_stage"] = output.apply(
        lambda row: classify_box_stage(
            row.get("box_status"),
            row.get("lifecycle_status"),
            row.get("pallet_code"),
        ),
        axis=1,
    )

    return output


def box_exception_mask(frame: pd.DataFrame) -> pd.Series:
    """Return current box exceptions from master or canonical lifecycle state."""
    if frame.empty:
        return pd.Series(False, index=frame.index, dtype=bool)

    box_status = frame.get(
        "box_status",
        pd.Series(index=frame.index, dtype="object"),
    ).fillna("").astype(str).str.upper()
    lifecycle_status = frame.get(
        "lifecycle_status",
        pd.Series(index=frame.index, dtype="object"),
    ).fillna("").astype(str).str.upper()
    return box_status.isin({"MISSING", "REMOVED"}) | lifecycle_status.isin(
        {"BOX_EXCEPTION", "MISSING", "REMOVED"}
    )


# ============================================================
# CORE WAREHOUSE DATA — CURRENT SCHEMA
# customers -> orders -> order_items -> order_item_products
# -> order_boxes -> boxes
# Current box position comes from the latest lifecycle event.
# ============================================================
if not database_ready():
    password_note = (
        " A password is required for the cloud profile."
        if DATABASE_TARGET == "cloud"
        else " A blank password is valid for a local MySQL profile."
    )
    st.error(
        "Database settings are incomplete. Host, user, and database are required."
        + password_note
    )
    st.stop()


def load_boxes() -> pd.DataFrame:
    return safe_fetch(
        """
        SELECT
            b.box_id,
            b.box_code,
            b.box_status,
            b.weight_kg,
            b.volume_cm3,
            b.updated_at,

            latest.lifecycle_status,
            latest.event_type,
            latest.severity,
            latest.event_source,
            latest.confidence_score,
            latest.qr_read_success,
            latest.event_time,
            latest.remarks AS latest_remarks,
            latest.pallet_id AS current_pallet_id,
            latest.dispatch_pallet_id,
            latest.dispatch_trailer_id AS trailer_id,

            z.zone_name AS current_zone,
            conv.conveyor_code,
            pal.pallet_code,
            dispatch_pal.pallet_code AS dispatch_pallet_code,
            d.dock_code,

            GROUP_CONCAT(
                DISTINCT o.order_code
                ORDER BY o.order_code
                SEPARATOR ', '
            ) AS order_code,

            GROUP_CONCAT(
                DISTINCT c.customer_name
                ORDER BY c.customer_name
                SEPARATOR ', '
            ) AS customer_name,

            GROUP_CONCAT(
                DISTINCT pr.product_code
                ORDER BY pr.product_code
                SEPARATOR ', '
            ) AS product_code,

            GROUP_CONCAT(
                DISTINCT CONCAT(
                    pr.product_name,
                    ' × ',
                    ob.packed_quantity
                )
                ORDER BY pr.product_name
                SEPARATOR ', '
            ) AS product_name

        FROM boxes b

        LEFT JOIN (
            SELECT
                ole.*,
                COALESCE(
                    ole.pallet_id,
                    (
                        SELECT history.pallet_id
                        FROM order_lifecycle_events history
                        WHERE history.box_id = ole.box_id
                          AND history.pallet_id IS NOT NULL
                          AND history.entity_type = 'BOX'
                          AND history.lifecycle_status IN (
                              'RECEIVED', 'ON_FLOOR', 'ON_CONVEYOR', 'IN_ZONE',
                              'ON_PALLET', 'AT_DOCK', 'LOADED_TO_TRAILER',
                              'DISPATCHED', 'BOX_EXCEPTION', 'REMOVED', 'MISSING'
                          )
                          AND COALESCE(history.source_reference, '') NOT LIKE 'CV:RUN:%'
                          AND COALESCE(history.source_reference, '') NOT LIKE 'box-status-%'
                          AND COALESCE(history.source_reference, '') NOT LIKE 'video-final-box-%'
                          AND COALESCE(history.event_type, '') NOT IN (
                              'CV_FINAL_STATE', 'BOX_STATUS_CHANGED', 'VIDEO_FINAL_BOX_SYNC'
                          )
                          AND history.order_lifecycle_event_id <= ole.order_lifecycle_event_id
                        ORDER BY history.order_lifecycle_event_id DESC
                        LIMIT 1
                    )
                ) AS dispatch_pallet_id,
                COALESCE(
                    ole.trailer_id,
                    (
                        SELECT history.trailer_id
                        FROM order_lifecycle_events history
                        WHERE history.box_id = ole.box_id
                          AND history.trailer_id IS NOT NULL
                          AND history.entity_type = 'BOX'
                          AND history.lifecycle_status IN (
                              'RECEIVED', 'ON_FLOOR', 'ON_CONVEYOR', 'IN_ZONE',
                              'ON_PALLET', 'AT_DOCK', 'LOADED_TO_TRAILER',
                              'DISPATCHED', 'BOX_EXCEPTION', 'REMOVED', 'MISSING'
                          )
                          AND COALESCE(history.source_reference, '') NOT LIKE 'CV:RUN:%'
                          AND COALESCE(history.source_reference, '') NOT LIKE 'box-status-%'
                          AND COALESCE(history.source_reference, '') NOT LIKE 'video-final-box-%'
                          AND COALESCE(history.event_type, '') NOT IN (
                              'CV_FINAL_STATE', 'BOX_STATUS_CHANGED', 'VIDEO_FINAL_BOX_SYNC'
                          )
                          AND history.order_lifecycle_event_id <= ole.order_lifecycle_event_id
                        ORDER BY history.order_lifecycle_event_id DESC
                        LIMIT 1
                    )
                ) AS dispatch_trailer_id
            FROM order_lifecycle_events ole
            JOIN (
                SELECT
                    box_id,
                    MAX(order_lifecycle_event_id) AS latest_event_id
                FROM order_lifecycle_events
                WHERE box_id IS NOT NULL
                  AND entity_type = 'BOX'
                  AND lifecycle_status IN (
                      'RECEIVED', 'ON_FLOOR', 'ON_CONVEYOR', 'IN_ZONE',
                      'ON_PALLET', 'AT_DOCK', 'LOADED_TO_TRAILER',
                      'DISPATCHED', 'BOX_EXCEPTION', 'REMOVED', 'MISSING'
                  )
                  AND COALESCE(source_reference, '') NOT LIKE 'CV:RUN:%'
                  AND COALESCE(source_reference, '') NOT LIKE 'box-status-%'
                  AND COALESCE(source_reference, '') NOT LIKE 'video-final-box-%'
                  AND COALESCE(event_type, '') NOT IN (
                      'CV_FINAL_STATE', 'BOX_STATUS_CHANGED', 'VIDEO_FINAL_BOX_SYNC'
                  )
                GROUP BY box_id
            ) selected
                ON selected.latest_event_id = ole.order_lifecycle_event_id
        ) latest
            ON latest.box_id = b.box_id

        LEFT JOIN zones z
            ON z.zone_id = latest.zone_id
        LEFT JOIN conveyors conv
            ON conv.conveyor_id = latest.conveyor_id
        LEFT JOIN pallets pal
            ON pal.pallet_id = latest.pallet_id
        LEFT JOIN pallets dispatch_pal
            ON dispatch_pal.pallet_id = latest.dispatch_pallet_id
        LEFT JOIN docks d
            ON d.dock_id = latest.dock_id

        LEFT JOIN order_boxes ob
            ON ob.box_id = b.box_id
        LEFT JOIN order_item_products oip
            ON oip.order_item_product_id = ob.order_item_product_id
        LEFT JOIN order_items oi
            ON oi.order_item_id = oip.order_item_id
        LEFT JOIN orders o
            ON o.order_id = oi.order_id
        LEFT JOIN customers c
            ON c.customer_id = o.customer_id
        LEFT JOIN products pr
            ON pr.product_id = oip.product_id

        GROUP BY
            b.box_id,
            b.box_code,
            b.box_status,
            b.weight_kg,
            b.volume_cm3,
            b.updated_at,
            latest.lifecycle_status,
            latest.event_type,
            latest.severity,
            latest.event_source,
            latest.confidence_score,
            latest.qr_read_success,
            latest.event_time,
            latest.remarks,
            latest.pallet_id,
            latest.dispatch_pallet_id,
            latest.dispatch_trailer_id,
            z.zone_name,
            conv.conveyor_code,
            pal.pallet_code,
            dispatch_pal.pallet_code,
            d.dock_code

        ORDER BY b.box_code;
        """,
        label="boxes and latest lifecycle",
    )


def load_trailer_lookup() -> pd.DataFrame:
    """Load trailer identifiers without assuming one fixed code-column name."""
    if not table_exists("trailers"):
        return pd.DataFrame(columns=["trailer_id", "trailer_code", "trailer_status"])

    columns = table_columns("trailers")
    if "trailer_id" not in columns:
        return pd.DataFrame(columns=["trailer_id", "trailer_code", "trailer_status"])

    code_column = next(
        (
            column
            for column in [
                "trailer_code",
                "trailer_number",
                "trailer_no",
                "registration_no",
            ]
            if column in columns
        ),
        None,
    )
    status_column = next(
        (
            column
            for column in ["trailer_status", "status", "assignment_status"]
            if column in columns
        ),
        None,
    )

    code_expression = (
        f"t.`{code_column}`" if code_column else "CONCAT('TRAILER-', t.trailer_id)"
    )
    status_expression = f"t.`{status_column}`" if status_column else "NULL"

    return safe_fetch(
        f"""
        SELECT
            t.trailer_id,
            {code_expression} AS trailer_code,
            {status_expression} AS trailer_status
        FROM trailers t
        ORDER BY t.trailer_id;
        """,
        label="trailers",
        show_error=False,
    )


boxes = load_boxes()
trailer_lookup = load_trailer_lookup()

if "trailer_id" not in boxes.columns:
    boxes["trailer_id"] = pd.NA

for df in (boxes, trailer_lookup):
    if "trailer_id" in df.columns:
        df["trailer_id"] = pd.to_numeric(
            df["trailer_id"],
            errors="coerce",
        ).astype("Int64")

if not boxes.empty and not trailer_lookup.empty:
    boxes = boxes.merge(
        trailer_lookup.drop_duplicates("trailer_id"),
        on="trailer_id",
        how="left",
    )
else:
    boxes["trailer_code"] = pd.NA
    boxes["trailer_status"] = pd.NA

# Keep a readable identifier even when the trailers table has no code column.
missing_trailer_code = boxes["trailer_code"].isna() & boxes["trailer_id"].notna()
boxes.loc[missing_trailer_code, "trailer_code"] = boxes.loc[
    missing_trailer_code, "trailer_id"
].apply(lambda value: f"TRAILER-{int(value)}" if pd.notna(value) else None)

boxes = add_box_stage(boxes)


# ============================================================
# PALLET LAYER DATA
# The current lifecycle remains the source of truth for the pallet.
# row_no, column_no, slot_no and layer_no are taken from the latest
# matching video_final_boxes row. Missing or stale coordinates are omitted
# rather than presented as measured physical positions.
# ============================================================
def load_pallet_layer_boxes(box_frame: pd.DataFrame) -> pd.DataFrame:
    if box_frame.empty:
        return pd.DataFrame()

    active = box_frame[
        box_frame["operational_stage"].eq("ON_PALLET")
        & box_frame["pallet_code"].notna()
    ].copy()

    if active.empty:
        return pd.DataFrame()

    keep_columns = [
        column
        for column in [
            "box_id",
            "box_code",
            "pallet_code",
            "box_status",
            "order_code",
            "product_name",
            "weight_kg",
            "current_zone",
            "dock_code",
            "lifecycle_status",
        ]
        if column in active.columns
    ]

    active = active[keep_columns].drop_duplicates("box_code")
    active = active.rename(columns={"pallet_code": "current_pallet_code"})

    video_positions = pd.DataFrame()

    if table_exists("video_final_boxes"):
        video_positions = safe_fetch(
            """
            SELECT
                vfb.final_box_id,
                vfb.run_id,
                vfb.box_code,
                vfb.pallet_code AS video_pallet_code,
                vfb.slot_no,
                vfb.row_no,
                vfb.column_no,
                vfb.layer_no,
                vfb.status AS video_status,
                vfb.length_cm,
                vfb.width_cm,
                vfb.height_cm,
                vfb.weight_kg AS video_weight_kg
            FROM video_final_boxes vfb
            JOIN (
                SELECT
                    box_code,
                    MAX(final_box_id) AS latest_final_box_id
                FROM video_final_boxes
                GROUP BY box_code
            ) latest
                ON latest.latest_final_box_id = vfb.final_box_id
            ORDER BY vfb.pallet_code, vfb.layer_no,
                     vfb.row_no, vfb.column_no;
            """,
            label="pallet layer positions",
            show_error=False,
        )

    if video_positions.empty:
        merged = active.copy()
        for column in [
            "run_id",
            "video_pallet_code",
            "slot_no",
            "row_no",
            "column_no",
            "layer_no",
            "video_status",
            "length_cm",
            "width_cm",
            "height_cm",
            "video_weight_kg",
        ]:
            merged[column] = pd.NA
    else:
        merged = active.merge(
            video_positions.drop_duplicates("box_code"),
            on="box_code",
            how="left",
        )

    # Current lifecycle pallet is authoritative. Old video placement data
    # from another pallet is ignored and replaced with a safe fallback.
    same_pallet = merged["video_pallet_code"].notna() & merged[
        "video_pallet_code"
    ].astype(str).eq(merged["current_pallet_code"].astype(str))

    for column in ["slot_no", "row_no", "column_no", "layer_no"]:
        merged[column] = pd.to_numeric(merged[column], errors="coerce")

    verified_position = same_pallet
    for column in ["row_no", "column_no", "layer_no"]:
        verified_position &= merged[column].notna() & merged[column].ge(1)

    # A missing or stale coordinate is omitted. The dashboard must not invent
    # a physical stack position simply to fill the drawing.
    merged = merged[verified_position].copy()
    if merged.empty:
        return pd.DataFrame()

    merged["pallet_code"] = merged["current_pallet_code"]
    merged["status"] = merged["video_status"].where(
        merged["video_status"].notna(),
        merged["box_status"],
    )
    missing_slot = merged["slot_no"].isna()
    merged.loc[missing_slot, "slot_no"] = (
        (merged.loc[missing_slot, "row_no"] - 1) * 3
        + merged.loc[missing_slot, "column_no"]
    )
    for column in ["slot_no", "row_no", "column_no", "layer_no"]:
        merged[column] = merged[column].astype(int)
    merged["position_source"] = "Computer vision"

    return merged.sort_values(
        ["pallet_code", "layer_no", "row_no", "column_no", "box_code"]
    ).reset_index(drop=True)


pallet_layer_boxes = load_pallet_layer_boxes(boxes)

# Attach the layer coordinates to the main box dataframe so the existing
# warehouse hover and optional labels also show the detected stack layer.
if not pallet_layer_boxes.empty:
    layer_lookup = pallet_layer_boxes[
        [
            "box_code",
            "slot_no",
            "row_no",
            "column_no",
            "layer_no",
            "position_source",
        ]
    ].drop_duplicates("box_code")

    boxes = boxes.merge(layer_lookup, on="box_code", how="left")
else:
    for column in [
        "slot_no",
        "row_no",
        "column_no",
        "layer_no",
        "position_source",
    ]:
        boxes[column] = pd.NA


pallets = safe_fetch(
    """
    SELECT
        p.pallet_id,
        p.pallet_code,
        p.pallet_status,
        p.max_weight_kg,
        p.max_volume_cm3,
        p.max_height_cm,
        NULL AS current_zone,
        0 AS total_boxes,
        0 AS linked_boxes,
        0.0 AS total_weight_kg,
        0.0 AS total_volume_cm3
    FROM pallets p
    ORDER BY p.pallet_code;
    """,
    label="pallets",
)


# Recalculate live pallet totals from latest lifecycle records.
if not pallets.empty:
    active_pallet_boxes = boxes[
        boxes["operational_stage"].eq("ON_PALLET")
        & boxes["pallet_code"].notna()
    ].copy()

    if active_pallet_boxes.empty:
        pallets["total_boxes"] = 0
        pallets["total_weight_kg"] = 0.0
        pallets["total_volume_cm3"] = 0.0
    else:
        active_pallet_summary = (
            active_pallet_boxes.groupby("pallet_code", dropna=True)
            .agg(
                total_boxes=("box_id", "nunique"),
                total_weight_kg=("weight_kg", "sum"),
                total_volume_cm3=("volume_cm3", "sum"),
            )
            .reset_index()
        )

        pallets = pallets.drop(
            columns=[
                "total_boxes",
                "total_weight_kg",
                "total_volume_cm3",
            ],
            errors="ignore",
        ).merge(
            active_pallet_summary,
            on="pallet_code",
            how="left",
        )

        pallets["total_boxes"] = (
            pd.to_numeric(pallets["total_boxes"], errors="coerce")
            .fillna(0)
            .astype(int)
        )
        pallets["total_weight_kg"] = pd.to_numeric(
            pallets["total_weight_kg"], errors="coerce"
        ).fillna(0.0)
        pallets["total_volume_cm3"] = pd.to_numeric(
            pallets["total_volume_cm3"], errors="coerce"
        ).fillna(0.0)


orders = safe_fetch(
    """
    SELECT
        o.order_id,
        o.order_code,
        c.customer_name,
        o.order_status,
        COALESCE(items.total_quantity, 0) AS total_quantity,
        COALESCE(items.total_products, 0) AS total_products,
        COALESCE(box_data.total_boxes, 0) AS total_boxes,
        COALESCE(box_data.on_pallet_boxes, 0) AS on_pallet_boxes,
        COALESCE(box_data.in_process_boxes, 0) AS in_process_boxes,
        COALESCE(box_data.dispatched_boxes, 0) AS dispatched_boxes
    FROM orders o
    LEFT JOIN customers c
        ON c.customer_id = o.customer_id

    LEFT JOIN (
        SELECT
            oi.order_id,
            COALESCE(SUM(oip.ordered_quantity), 0) AS total_quantity,
            COUNT(DISTINCT oip.product_id) AS total_products
        FROM order_items oi
        LEFT JOIN order_item_products oip
            ON oip.order_item_id = oi.order_item_id
        GROUP BY oi.order_id
    ) items
        ON items.order_id = o.order_id

    LEFT JOIN (
        SELECT
            oi.order_id,
            COUNT(DISTINCT ob.box_id) AS total_boxes,

            COUNT(DISTINCT CASE
                WHEN latest.pallet_id IS NOT NULL
                 AND COALESCE(b.box_status, '') NOT IN ('DISPATCHED', 'MISSING', 'REMOVED')
                 AND COALESCE(latest.lifecycle_status, '') NOT LIKE '%DISPATCH%'
                THEN ob.box_id
            END) AS on_pallet_boxes,

            COUNT(DISTINCT CASE
                WHEN COALESCE(latest.lifecycle_status, '') NOT LIKE '%DISPATCH%'
                 AND latest.pallet_id IS NULL
                THEN ob.box_id
            END) AS in_process_boxes,

            COUNT(DISTINCT CASE
                WHEN COALESCE(latest.lifecycle_status, '') LIKE '%DISPATCH%'
                  OR b.box_status = 'DISPATCHED'
                THEN ob.box_id
            END) AS dispatched_boxes

        FROM order_boxes ob
        JOIN order_item_products oip
            ON oip.order_item_product_id = ob.order_item_product_id
        JOIN order_items oi
            ON oi.order_item_id = oip.order_item_id
        JOIN boxes b
            ON b.box_id = ob.box_id

        LEFT JOIN (
            SELECT ole.*
            FROM order_lifecycle_events ole
            JOIN (
                SELECT
                    box_id,
                    MAX(order_lifecycle_event_id) AS latest_event_id
                FROM order_lifecycle_events
                WHERE box_id IS NOT NULL
                  AND entity_type = 'BOX'
                  AND lifecycle_status IN (
                      'RECEIVED', 'ON_FLOOR', 'ON_CONVEYOR', 'IN_ZONE',
                      'ON_PALLET', 'AT_DOCK', 'LOADED_TO_TRAILER',
                      'DISPATCHED', 'BOX_EXCEPTION', 'REMOVED', 'MISSING'
                  )
                  AND COALESCE(source_reference, '') NOT LIKE 'CV:RUN:%'
                  AND COALESCE(source_reference, '') NOT LIKE 'box-status-%'
                  AND COALESCE(source_reference, '') NOT LIKE 'video-final-box-%'
                  AND COALESCE(event_type, '') NOT IN (
                      'CV_FINAL_STATE', 'BOX_STATUS_CHANGED', 'VIDEO_FINAL_BOX_SYNC'
                  )
                GROUP BY box_id
            ) selected
                ON selected.latest_event_id = ole.order_lifecycle_event_id
        ) latest
            ON latest.box_id = b.box_id

        GROUP BY oi.order_id
    ) box_data
        ON box_data.order_id = o.order_id

    ORDER BY o.order_code;
    """,
    label="orders",
)

events = safe_fetch(
    """
    SELECT
        ole.order_lifecycle_event_id AS event_id,
        ole.entity_type,
        ole.lifecycle_status,
        ole.event_type,
        ole.event_time,
        ole.severity,
        ole.event_source,
        ole.confidence_score,
        ole.qr_read_success,
        ole.remarks,
        b.box_code,
        p.pallet_code,
        z.zone_name,
        d.dock_code
    FROM order_lifecycle_events ole
    LEFT JOIN boxes b
        ON b.box_id = ole.box_id
    LEFT JOIN pallets p
        ON p.pallet_id = ole.pallet_id
    LEFT JOIN zones z
        ON z.zone_id = ole.zone_id
    LEFT JOIN docks d
        ON d.dock_id = ole.dock_id
    WHERE COALESCE(ole.source_reference, '') NOT LIKE 'CV:RUN:%'
      AND COALESCE(ole.source_reference, '') NOT LIKE 'box-status-%'
      AND COALESCE(ole.source_reference, '') NOT LIKE 'video-final-box-%'
      AND COALESCE(ole.source_reference, '') NOT LIKE 'video-final-pallet-%'
      AND COALESCE(ole.event_type, '') NOT IN (
          'CV_FINAL_STATE', 'BOX_STATUS_CHANGED',
          'VIDEO_FINAL_BOX_SYNC', 'VIDEO_FINAL_PALLET_SYNC'
      )
    ORDER BY ole.event_time DESC, ole.order_lifecycle_event_id DESC;
    """,
    label="order lifecycle events",
)


if not orders.empty:
    orders["completion_percentage"] = orders.apply(
        lambda row: round(
            (
                as_number(row["on_pallet_boxes"])
                + as_number(row["dispatched_boxes"])
            )
            / max(as_number(row["total_boxes"], 1), 1)
            * 100,
            2,
        ),
        axis=1,
    )


# ============================================================
# PERFORMANCE METRIC SNAPSHOTS
# This table stores summarized dashboard KPIs. It does not need a
# foreign key because each row is a time-based snapshot.
# ============================================================
performance_history = pd.DataFrame()
latest_performance: dict[str, Any] = {}

if table_exists("performance_metrics"):
    performance_history = safe_fetch(
        """
        SELECT *
        FROM performance_metrics
        ORDER BY recorded_at DESC, metric_id DESC
        LIMIT 200;
        """,
        label="performance metrics snapshots",
        show_error=False,
    )

    if not performance_history.empty:
        latest_performance = (
            performance_history.iloc[0].to_dict()
        )


def performance_value(column: str) -> Optional[float]:
    if not latest_performance or column not in latest_performance:
        return None

    try:
        value = latest_performance[column]
        if value is None or pd.isna(value):
            return None
        return float(value)
    except Exception:
        return None


# ============================================================
# TRUCK DATA
# The trucks table is connected to the existing zones table through
# trucks.current_zone_id -> zones.zone_id. Docking stations are reused
# from zones, so a separate docks table is not required.
# ============================================================
def load_trucks() -> tuple[pd.DataFrame, Optional[int], Optional[str]]:
    if table_exists("trucks"):
        columns = table_columns("trucks")

        code_column = next(
            (
                column
                for column in [
                    "truck_number",
                    "truck_code",
                    "vehicle_number",
                    "registration_no",
                ]
                if column in columns
            ),
            None,
        )
        status_column = next(
            (
                column
                for column in ["status", "truck_status", "queue_status"]
                if column in columns
            ),
            None,
        )

        if code_column is None or status_column is None:
            return pd.DataFrame(), None, "trucks"

        carrier_expression = (
            "t.carrier_name" if "carrier_name" in columns else "NULL"
        )
        arrival_expression = (
            "t.arrival_time" if "arrival_time" in columns else "NULL"
        )
        dock_expression = (
            "t.dock_time" if "dock_time" in columns else "NULL"
        )
        departure_expression = (
            "t.departure_time" if "departure_time" in columns else "NULL"
        )

        if "current_zone_id" in columns and table_exists("zones"):
            zone_join = (
                "LEFT JOIN zones z "
                "ON t.current_zone_id = z.zone_id"
            )
            zone_expression = "z.zone_name"
        else:
            zone_join = ""
            zone_expression = "NULL"

        truck_data = safe_fetch(
            f"""
            SELECT
                t.`{code_column}` AS truck_code,
                {carrier_expression} AS carrier_name,
                t.`{status_column}` AS truck_status,
                {zone_expression} AS current_zone,
                {arrival_expression} AS arrival_time,
                {dock_expression} AS dock_time,
                {departure_expression} AS departure_time
            FROM trucks t
            {zone_join}
            ORDER BY
                CASE UPPER(t.`{status_column}`)
                    WHEN 'UNLOADING' THEN 1
                    WHEN 'LOADING' THEN 2
                    WHEN 'AT_DOCK' THEN 3
                    WHEN 'WAITING' THEN 4
                    WHEN 'EXPECTED' THEN 5
                    WHEN 'DISPATCHED' THEN 6
                    ELSE 7
                END,
                t.`{code_column}`;
            """,
            label="trucks",
            show_error=False,
        )

        if truck_data.empty:
            return truck_data, 0, "trucks"

        waiting_count = int(
            truck_data["truck_status"]
            .fillna("")
            .astype(str)
            .str.upper()
            .isin(["WAITING", "QUEUED"])
            .sum()
        )
        return truck_data, waiting_count, "trucks"

    # Backward-compatible fallback if a truck_queue table is used later.
    if table_exists("truck_queue"):
        columns = table_columns("truck_queue")
        code_column = next(
            (
                column
                for column in [
                    "truck_number",
                    "truck_code",
                    "vehicle_number",
                    "registration_no",
                ]
                if column in columns
            ),
            None,
        )
        status_column = next(
            (
                column
                for column in ["queue_status", "truck_status", "status"]
                if column in columns
            ),
            None,
        )
        arrival_column = next(
            (
                column
                for column in [
                    "arrival_time",
                    "created_at",
                    "registered_at",
                ]
                if column in columns
            ),
            None,
        )

        if code_column is None or status_column is None:
            return pd.DataFrame(), None, "truck_queue"

        arrival_expression = (
            f"`{arrival_column}`" if arrival_column else "NULL"
        )
        truck_data = safe_fetch(
            f"""
            SELECT
                `{code_column}` AS truck_code,
                NULL AS carrier_name,
                `{status_column}` AS truck_status,
                NULL AS current_zone,
                {arrival_expression} AS arrival_time,
                NULL AS dock_time,
                NULL AS departure_time
            FROM truck_queue;
            """,
            label="truck queue",
            show_error=False,
        )

        waiting_count = int(
            truck_data["truck_status"]
            .fillna("")
            .astype(str)
            .str.upper()
            .isin(["WAITING", "QUEUED"])
            .sum()
        ) if not truck_data.empty else 0

        return truck_data, waiting_count, "truck_queue"

    return pd.DataFrame(), None, None


trucks, trucks_waiting, truck_table_name = load_trucks()


def calculate_active_docks() -> tuple[Optional[int], Optional[str]]:
    if table_exists("docks"):
        dock_frame = safe_fetch(
            """
            SELECT dock_id, dock_status, is_active
            FROM docks;
            """,
            label="active docks",
            show_error=False,
        )
        if dock_frame.empty:
            return 0, "docks table"

        active_flags = pd.to_numeric(
            dock_frame["is_active"], errors="coerce"
        ).fillna(0).astype(int)
        active_count = int((active_flags == 1).sum())
        return active_count, "docks.is_active"

    return None, None


active_docks, dock_table_name = calculate_active_docks()


# ============================================================
# OPTIONAL VIDEO PERFORMANCE DATA
# ============================================================
video_runs = pd.DataFrame()
metric_map: dict[str, str] = {}
video_events = pd.DataFrame()
selected_run: dict[str, Any] = {}
selected_run_id: Optional[int] = None

if table_exists("video_processing_runs"):
    video_runs = safe_fetch(
        """
        SELECT *
        FROM video_processing_runs
        ORDER BY run_id DESC;
        """,
        label="video processing runs",
        show_error=False,
    )

if not video_runs.empty:
    run_labels = {
        f"Run {int(row.run_id)} | {getattr(row, 'video_name', '')}":
            int(row.run_id)
        for row in video_runs.itertuples(index=False)
    }

    selected_run_label = st.sidebar.selectbox(
        "Performance run",
        list(run_labels.keys()),
    )

    selected_run_id = run_labels[selected_run_label]
    selected_run_frame = video_runs[
        video_runs["run_id"] == selected_run_id
    ]

    if not selected_run_frame.empty:
        selected_run = selected_run_frame.iloc[0].to_dict()

    if table_exists("video_processing_metrics"):
        metrics = safe_fetch(
            """
            SELECT
                metric_name,
                metric_value
            FROM video_processing_metrics
            WHERE run_id = %s
            ORDER BY metric_name;
            """,
            (selected_run_id,),
            label="performance metrics",
            show_error=False,
        )

        if not metrics.empty:
            metric_map = dict(
                zip(
                    metrics["metric_name"].astype(str),
                    metrics["metric_value"].astype(str),
                )
            )

    if table_exists("video_events"):
        video_events = safe_fetch(
            """
            SELECT *
            FROM video_events
            WHERE run_id = %s
            ORDER BY video_event_id DESC;
            """,
            (selected_run_id,),
            label="video events",
            show_error=False,
        )


def metric_value(*names: str) -> Optional[float]:
    for name in names:
        if name not in metric_map:
            continue

        try:
            return float(
                str(metric_map[name])
                .replace("%", "")
                .strip()
            )
        except Exception:
            continue

    return None


def normalize_percentage(value: Optional[float]) -> Optional[float]:
    if value is None:
        return None
    if 0 <= value <= 1:
        return value * 100
    return value


stored_boxes_per_min = performance_value("boxes_per_minute")
stored_vision_accuracy = performance_value("vision_accuracy")
stored_average_confidence = performance_value(
    "average_confidence_score"
)
stored_qr_read_rate = performance_value("qr_read_rate")
stored_elapsed_minutes = performance_value("elapsed_minutes")

boxes_per_min = stored_boxes_per_min
if boxes_per_min is None:
    boxes_per_min = metric_value(
        "boxes_per_min",
        "boxes_min",
        "box_rate_per_min",
        "boxes_processed_per_min",
        "throughput_boxes_per_min",
    )

processing_fps = metric_value(
    "average_processing_fps",
    "processed_fps",
    "effective_source_fps",
)

if processing_fps is None and selected_run:
    processing_time = as_number(
        selected_run.get("total_processing_time_sec")
    )
    processed_frames = as_number(
        selected_run.get("processed_frames")
    )

    if processing_time > 0:
        processing_fps = processed_frames / processing_time

speed_vs_realtime = metric_value("speed_vs_realtime")

if speed_vs_realtime is None and selected_run:
    speed_vs_realtime = as_number(
        selected_run.get("speed_vs_realtime")
    )

vision_accuracy = stored_vision_accuracy
if vision_accuracy is None:
    vision_accuracy = normalize_percentage(
        metric_value(
            "vision_accuracy",
            "average_detection_confidence",
            "avg_detection_confidence",
            "avg_yolo_confidence",
            "confidence_score",
            "mean_confidence",
        )
    )

average_confidence_score = stored_average_confidence
if average_confidence_score is None:
    average_confidence_score = normalize_percentage(
        metric_value(
            "average_detection_confidence",
            "avg_detection_confidence",
            "avg_yolo_confidence",
            "confidence_score",
            "mean_confidence",
        )
    )

qr_read_rate = stored_qr_read_rate
if qr_read_rate is None:
    qr_read_rate = normalize_percentage(
        metric_value(
            "qr_read_rate",
            "qr_success_rate",
            "qr_read_success_rate",
        )
    )

if qr_read_rate is None:
    qr_success = metric_value(
        "qr_success_count",
        "successful_qr_reads",
    )
    qr_attempts = metric_value(
        "qr_attempt_count",
        "total_qr_attempts",
    )

    if (
        qr_success is not None
        and qr_attempts is not None
        and qr_attempts > 0
    ):
        qr_read_rate = qr_success / qr_attempts * 100


# ============================================================
# KPI CALCULATIONS
# Every box is assigned to exactly one of five operational states.
# Therefore:
# Dispatched + On Pallets + On Conveyor + On Floor + Unlocated = processed.
# ============================================================
boxes_processed = (
    int(boxes["box_id"].nunique())
    if not boxes.empty and "box_id" in boxes.columns
    else 0
)

if boxes.empty:
    stage_counts: dict[str, int] = {}
else:
    stage_counts = (
        boxes.drop_duplicates(subset=["box_id"])
        ["operational_stage"]
        .fillna("UNLOCATED")
        .astype(str)
        .value_counts()
        .to_dict()
    )

dispatched_boxes = int(stage_counts.get("DISPATCHED", 0))
boxes_on_pallets = int(stage_counts.get("ON_PALLET", 0))
boxes_on_conveyor = int(stage_counts.get("ON_CONVEYOR", 0))
boxes_on_floor = int(stage_counts.get("ON_FLOOR", 0))
boxes_unlocated = int(stage_counts.get("UNLOCATED", 0))

reconciled_box_total = (
    dispatched_boxes
    + boxes_on_pallets
    + boxes_on_conveyor
    + boxes_on_floor
    + boxes_unlocated
)


def _clean_dispatch_identifier(value: Any) -> str:
    if value is None or pd.isna(value):
        return ""
    cleaned = str(value).strip()
    return "" if cleaned.lower() in {"", "nan", "none", "<na>"} else cleaned


def build_dispatch_assets(box_frame: pd.DataFrame) -> pd.DataFrame:
    """Represent dispatch using trailers/pallets, never individual box markers."""
    columns = [
        "asset_type",
        "asset_code",
        "pallet_count",
        "pallet_codes",
        "box_count",
    ]
    if box_frame.empty:
        return pd.DataFrame(columns=columns)

    dispatched = add_box_stage(box_frame)
    dispatched = dispatched[
        dispatched["operational_stage"].eq("DISPATCHED")
    ].drop_duplicates("box_id").copy()

    if dispatched.empty:
        return pd.DataFrame(columns=columns)

    for column in [
        "trailer_code",
        "trailer_id",
        "pallet_code",
        "dispatch_pallet_code",
    ]:
        if column not in dispatched.columns:
            dispatched[column] = pd.NA

    dispatched["resolved_trailer_code"] = dispatched["trailer_code"].apply(
        _clean_dispatch_identifier
    )
    missing_code = dispatched["resolved_trailer_code"].eq("") & dispatched[
        "trailer_id"
    ].notna()
    dispatched.loc[missing_code, "resolved_trailer_code"] = dispatched.loc[
        missing_code, "trailer_id"
    ].apply(lambda value: f"TRAILER-{int(value)}")
    dispatched["resolved_pallet_code"] = dispatched[
        "dispatch_pallet_code"
    ].apply(_clean_dispatch_identifier)
    missing_dispatch_pallet = dispatched["resolved_pallet_code"].eq("")
    dispatched.loc[missing_dispatch_pallet, "resolved_pallet_code"] = dispatched.loc[
        missing_dispatch_pallet, "pallet_code"
    ].apply(_clean_dispatch_identifier)

    assets: list[dict[str, Any]] = []

    # A linked trailer is the highest-level dispatch carrier. Its label shows
    # how many unique pallets it is carrying; box totals remain hover-only.
    trailer_rows = dispatched[dispatched["resolved_trailer_code"].ne("")]
    for trailer_code, group in trailer_rows.groupby("resolved_trailer_code"):
        pallet_codes = sorted(
            {
                code
                for code in group["resolved_pallet_code"].tolist()
                if code
            }
        )
        assets.append(
            {
                "asset_type": "TRAILER",
                "asset_code": trailer_code,
                "pallet_count": len(pallet_codes),
                "pallet_codes": ", ".join(pallet_codes),
                "box_count": int(group["box_id"].nunique()),
            }
        )

    # If a dispatched pallet has not yet been linked to a trailer, show the
    # pallet itself rather than reverting to hundreds of individual boxes.
    unlinked = dispatched[dispatched["resolved_trailer_code"].eq("")]
    linked_pallets = {
        code
        for code in trailer_rows["resolved_pallet_code"].tolist()
        if code
    }
    for pallet_code, group in unlinked[
        unlinked["resolved_pallet_code"].ne("")
        & ~unlinked["resolved_pallet_code"].isin(linked_pallets)
    ].groupby("resolved_pallet_code"):
        assets.append(
            {
                "asset_type": "PALLET",
                "asset_code": pallet_code,
                "pallet_count": 1,
                "pallet_codes": pallet_code,
                "box_count": int(group["box_id"].nunique()),
            }
        )

    # Keep the data issue visible without drawing those boxes as dispatch units.
    unresolved = unlinked[unlinked["resolved_pallet_code"].eq("")]
    if not unresolved.empty:
        assets.append(
            {
                "asset_type": "PENDING",
                "asset_code": "Carrier assignment pending",
                "pallet_count": 0,
                "pallet_codes": "",
                "box_count": int(unresolved["box_id"].nunique()),
            }
        )

    return pd.DataFrame(assets, columns=columns)


def dispatch_totals(box_frame: pd.DataFrame) -> tuple[int, int]:
    """Return unique dispatched pallets and trailers from DB-backed box rows."""
    if box_frame.empty:
        return 0, 0

    dispatched = add_box_stage(box_frame)
    dispatched = dispatched[
        dispatched["operational_stage"].eq("DISPATCHED")
    ].drop_duplicates("box_id")

    if dispatched.empty:
        return 0, 0

    dispatch_pallet_series = dispatched.get(
        "dispatch_pallet_code",
        pd.Series(index=dispatched.index, dtype="object"),
    )
    pallet_values = dispatch_pallet_series.apply(_clean_dispatch_identifier)
    if "pallet_code" in dispatched.columns:
        missing_pallet = pallet_values.eq("")
        pallet_values.loc[missing_pallet] = dispatched.loc[
            missing_pallet, "pallet_code"
        ].apply(_clean_dispatch_identifier)
    trailer_values = (
        dispatched.get("trailer_code", pd.Series(dtype="object"))
        .apply(_clean_dispatch_identifier)
    )

    # Use the lifecycle trailer_id when the master table has no readable code.
    if "trailer_id" in dispatched.columns:
        missing = trailer_values.eq("") & dispatched["trailer_id"].notna()
        trailer_values.loc[missing] = dispatched.loc[missing, "trailer_id"].apply(
            lambda value: f"TRAILER-{int(value)}"
        )

    return (
        int(pallet_values[pallet_values.ne("")].nunique()),
        int(trailer_values[trailer_values.ne("")].nunique()),
    )


dispatch_assets = build_dispatch_assets(boxes)
dispatched_pallets, dispatched_trailers = dispatch_totals(boxes)

# Dock is a physical-location detail derived only from the latest lifecycle.
# The warehouse visual keeps both AT_DOCK and LOADED_TO_TRAILER boxes in the
# dock areas, so this KPI must use exactly the same rule. RECEIVED boxes and
# already DISPATCHED boxes are not counted at the docks.
if boxes.empty:
    boxes_at_dock = 0
else:
    latest_lifecycle = (
        boxes["lifecycle_status"]
        .fillna("")
        .astype(str)
        .str.upper()
    )
    boxes_at_dock = int(
        latest_lifecycle.isin(
            ["AT_DOCK", "LOADED_TO_TRAILER"]
        ).sum()
    )

active_pallets = (
    int(
        (
            pd.to_numeric(
                pallets["total_boxes"],
                errors="coerce",
            )
            .fillna(0)
            > 0
        ).sum()
    )
    if not pallets.empty
    else 0
)

# Incoming Boxes matches the boxes currently visible on the conveyor.
incoming_boxes = boxes_on_conveyor

# Live throughput from unique physical boxes and the event time span.
elapsed_minutes_live: Optional[float] = None
if not events.empty and "event_time" in events.columns:
    box_event_rows = events
    if "box_code" in events.columns:
        box_event_rows = events[events["box_code"].notna()]

    event_times = pd.to_datetime(
        box_event_rows["event_time"],
        errors="coerce",
    ).dropna()

    if len(event_times) >= 2:
        elapsed_minutes_live = (
            event_times.max() - event_times.min()
        ).total_seconds() / 60.0

if (
    elapsed_minutes_live is not None
    and elapsed_minutes_live > 0
):
    boxes_per_min = boxes_processed / elapsed_minutes_live
elif stored_boxes_per_min is not None:
    boxes_per_min = stored_boxes_per_min

database_exceptions = int(box_exception_mask(boxes).sum())

video_exception_count = 0
if (
    not video_events.empty
    and "severity" in video_events.columns
):
    video_exception_count = int(
        video_events["severity"]
        .fillna("")
        .astype(str)
        .str.upper()
        .isin(["WARNING", "CRITICAL", "ERROR"])
        .sum()
    )

exception_count = database_exceptions + video_exception_count


# ============================================================
# KPI CARDS
# ============================================================
def render_kpis() -> None:
    cards = [
        (
            "Active Docks",
            format_value(
                active_docks,
                unavailable=active_docks is None,
            ),
            dock_table_name
            if active_docks is not None
            else "not available",
        ),
        (
            "Active Pallets",
            format_value(active_pallets),
            "pallets + latest lifecycle",
        ),
        (
            "Boxes<br>Processed",
            format_value(boxes_processed),
            "boxes table · unique box_id",
        ),
        (
            "Boxes/min",
            format_value(
                boxes_per_min,
                1,
                unavailable=boxes_per_min is None,
            ),
            "order_lifecycle_events / DB metric",
        ),
        (
            "Incoming Boxes",
            format_value(incoming_boxes),
            "latest lifecycle conveyor",
        ),
        (
            "Vision Accuracy",
            format_value(
                vision_accuracy,
                2,
                "%",
                unavailable=vision_accuracy is None,
            ),
            "DB metric tables",
        ),
        (
            "QR Read Rate",
            format_value(
                qr_read_rate,
                1,
                "%",
                unavailable=qr_read_rate is None,
            ),
            "DB metric tables",
        ),
        (
            "Exception<br>Count",
            format_value(exception_count),
            "boxes + video_events",
        ),
        (
            "Trucks Waiting",
            format_value(
                trucks_waiting,
                unavailable=trucks_waiting is None,
            ),
            "trucks.status"
            if trucks_waiting is not None
            else "not available",
        ),
    ]

    columns = st.columns(len(cards))
    accent_colors = [
        DASHBOARD_COLORS["teal"],
        DASHBOARD_COLORS["brown"],
        DASHBOARD_COLORS["blue"],
        DASHBOARD_COLORS["purple"],
        DASHBOARD_COLORS["amber"],
        DASHBOARD_COLORS["cyan"],
        DASHBOARD_COLORS["green"],
        DASHBOARD_COLORS["red"],
        DASHBOARD_COLORS["slate"],
    ]

    for column, (label, value, note), accent in zip(
        columns, cards, accent_colors
    ):
        column.markdown(
            f'<div class="kpi-label">{label}</div>',
            unsafe_allow_html=True,
        )
        column.markdown(
            (
                f'<div class="kpi-card" style="--accent:{accent};">'
                f'<div class="kpi-value">{value}</div>'
                f'<div class="kpi-note">{note}</div>'
                '</div>'
            ),
            unsafe_allow_html=True,
        )


def _selected_run_number(*names: str) -> Optional[float]:
    for name in names:
        value = selected_run.get(name)
        try:
            if value is not None and not pd.isna(value):
                return float(value)
        except (TypeError, ValueError):
            continue
    return None


def render_inference_run_kpis() -> None:
    """Render only values tied to the selected computer-vision run."""
    if selected_run_id is None:
        st.info("No computer-vision run is available yet.")
        return

    run_fps = metric_value("average_processing_fps")
    if run_fps is None:
        run_fps = processing_fps

    unique_boxes = metric_value("unique_boxes_seen", "final_box_count")
    unique_pallets = metric_value("unique_pallets_seen", "final_pallet_count")
    matched_boxes = metric_value("database_matched_box_count")
    matched_pallets = metric_value("database_matched_pallet_count")
    unique_qrs = metric_value("unique_qrs_seen")
    total_events = metric_value("total_event_count")
    add_events = metric_value("event_count_ADD", "event_count_add")
    dispatch_events = metric_value("event_count_DISPATCH", "event_count_dispatch")
    pending_identities = metric_value("pending_identity_event_count")
    dropped_identities = metric_value("dropped_unresolved_event_count")

    processed_frames = _selected_run_number("processed_frames")
    expected_frames = metric_value("expected_processed_frames")
    frame_coverage = (
        f"{int(processed_frames):,} / {int(expected_frames):,}"
        if processed_frames is not None and expected_frames is not None
        else "N/A"
    )
    run_speed = _selected_run_number("speed_vs_realtime")
    realtime_factor = _selected_run_number("realtime_factor")
    run_status = str(selected_run.get("status") or "UNKNOWN")

    cards = [
        ("Run Status", run_status, "video_processing_runs.status"),
        ("Processing FPS", format_value(run_fps, 2, unavailable=run_fps is None), "run average"),
        ("Speed vs Real Time", format_value(run_speed, 2, "x", unavailable=run_speed is None), "video duration / elapsed"),
        ("Realtime Factor", format_value(realtime_factor, 3, "x", unavailable=realtime_factor is None), "elapsed / video duration"),
        ("Frame Coverage", frame_coverage, "processed / expected"),
        ("Unique Boxes Seen", format_value(unique_boxes, unavailable=unique_boxes is None), "CV identities"),
        ("DB Matched Boxes", format_value(matched_boxes, unavailable=matched_boxes is None), "master-data matches"),
        ("Unique Pallets Seen", format_value(unique_pallets, unavailable=unique_pallets is None), "CV identities"),
        ("DB Matched Pallets", format_value(matched_pallets, unavailable=matched_pallets is None), "master-data matches"),
        ("Unique QR Values", format_value(unique_qrs, unavailable=unique_qrs is None), "decoded identities"),
        ("Run Events", format_value(total_events, unavailable=total_events is None), "persisted CV events"),
        ("ADD Events", format_value(add_events, unavailable=add_events is None), "selected run"),
        ("DISPATCH Events", format_value(dispatch_events, unavailable=dispatch_events is None), "selected run"),
        ("Pending Identities", format_value(pending_identities, unavailable=pending_identities is None), "unresolved at run end"),
        ("Dropped Unresolved", format_value(dropped_identities, unavailable=dropped_identities is None), "not persisted as warehouse state"),
    ]

    st.markdown(
        '<div class="section-title">SELECTED INFERENCE RUN KPIs</div>',
        unsafe_allow_html=True,
    )
    st.caption(f"Run {selected_run_id}: values below are run-scoped database evidence.")
    per_row = 5
    for start in range(0, len(cards), per_row):
        columns = st.columns(per_row)
        for column, (label, value, help_text) in zip(columns, cards[start:start + per_row]):
            column.metric(label, value, help=help_text)


# ============================================================
# INTERACTIVE WAREHOUSE DIGITAL TWIN
# Simplified guide-approved layout: two docks remain on the right,
# pallets remain visible, and dispatch sits outside the warehouse.
# Truck drawings and animations are intentionally excluded.
# ============================================================
DASHBOARD_COLORS = {
    "primary": "#0b3b60",
    "primary_2": "#155e95",
    "blue": "#2563eb",
    "cyan": "#0ea5e9",
    "green": "#16a34a",
    "teal": "#0f766e",
    "amber": "#f59e0b",
    "red": "#dc2626",
    "purple": "#7c3aed",
    "brown": "#9a6a3a",
    "slate": "#64748b",
    "text": "#172033",
    "muted": "#64748b",
    "border": "#d8e2ee",
    "surface": "#ffffff",
    "background": "#f4f7fb",
}

CHART_COLOR_SEQUENCE = [
    DASHBOARD_COLORS["blue"],
    DASHBOARD_COLORS["green"],
    DASHBOARD_COLORS["amber"],
    DASHBOARD_COLORS["purple"],
    DASHBOARD_COLORS["cyan"],
    DASHBOARD_COLORS["red"],
    DASHBOARD_COLORS["teal"],
    DASHBOARD_COLORS["brown"],
]


STATUS_STYLE = {
    "NORMAL": {
        "color": DASHBOARD_COLORS["blue"],
        "name": "On Pallet / Normal",
    },
    "ATTENTION": {
        "color": DASHBOARD_COLORS["amber"],
        "name": "In Process / Unassigned",
    },
    "CRITICAL": {
        "color": DASHBOARD_COLORS["red"],
        "name": "Exception Box",
    },
    "DISPATCHED": {
        "color": DASHBOARD_COLORS["green"],
        "name": "Dispatched Box",
    },
}


def visual_status_for_row(row: pd.Series) -> str:
    raw_status = str(row.get("box_status") or "").upper()
    lifecycle_status = str(row.get("lifecycle_status") or "").upper()
    stage = str(row.get("operational_stage") or "ON_FLOOR").upper()

    if raw_status in {"MISSING", "REMOVED"} or lifecycle_status in {
        "BOX_EXCEPTION",
        "MISSING",
        "REMOVED",
    }:
        return "CRITICAL"

    if stage == "DISPATCHED":
        return "DISPATCHED"

    if stage == "ON_PALLET":
        return "NORMAL"

    return "ATTENTION"


# ============================================================
# LARGE, DENSITY-AWARE WAREHOUSE DIGITAL TWIN
# The visual uses the latest lifecycle record for every box and places
# each marker inside its actual operational area without overlap.
# ============================================================
WAREHOUSE_BOUNDS = (0.40, 0.65, 15.25, 9.20)
FLOOR_TOP_BOUNDS = (0.85, 5.80, 3.20, 8.60)
FLOOR_BOTTOM_BOUNDS = (0.85, 1.20, 3.20, 4.00)
CONVEYOR_BOUNDS = (0.85, 4.35, 12.25, 5.45)
DOCK_1_BOUNDS = (12.70, 5.35, 15.00, 8.60)
DOCK_2_BOUNDS = (12.70, 1.20, 15.00, 4.10)
DISPATCH_BOUNDS = (15.80, 2.45, 19.35, 7.25)

# The warehouse zone footprints stay unchanged. Only the pallet drawings
# inside those zones are smaller, as requested.
PALLET_ZONE_BOUNDS = [
    (3.55, 5.80, 6.30, 8.60),
    (6.50, 5.80, 9.25, 8.60),
    (9.45, 5.80, 12.20, 8.60),
    (3.55, 1.20, 6.30, 4.00),
    (6.50, 1.20, 9.25, 4.00),
    (9.45, 1.20, 12.20, 4.00),
]

PALLET_BOUNDS = [
    (3.90, 6.18, 5.95, 8.22),
    (6.85, 6.18, 8.90, 8.22),
    (9.80, 6.18, 11.85, 8.22),
    (3.90, 1.58, 5.95, 3.62),
    (6.85, 1.58, 8.90, 3.62),
    (9.80, 1.58, 11.85, 3.62),
]


def _grid_points(
    bounds: tuple[float, float, float, float],
    count: int,
    pad_x: float = 0.12,
    pad_y: float = 0.12,
    top_label_space: float = 0.0,
) -> list[tuple[float, float]]:
    """Return evenly spaced, non-overlapping grid coordinates in a rectangle."""
    if count <= 0:
        return []

    x0, y0, x1, y1 = bounds
    x0 += pad_x
    x1 -= pad_x
    y0 += pad_y
    y1 -= pad_y
    y1 -= top_label_space

    usable_width = max(x1 - x0, 0.10)
    usable_height = max(y1 - y0, 0.10)

    # Shape the grid to match the available rectangle instead of forcing
    # a fixed number of columns. This prevents 500-box datasets from piling up.
    aspect = usable_width / usable_height
    columns = max(1, math.ceil(math.sqrt(count * aspect)))
    rows = max(1, math.ceil(count / columns))

    x_step = usable_width / columns
    y_step = usable_height / rows

    points: list[tuple[float, float]] = []
    for index in range(count):
        row = index // columns
        column = index % columns
        x = x0 + (column + 0.5) * x_step
        # Fill from top to bottom so the layout reads naturally.
        y = y1 - (row + 0.5) * y_step
        points.append((x, y))

    return points


def _marker_size_for_region(count: int, minimum: int = 6, maximum: int = 13) -> int:
    """Choose a readable marker size while keeping dense groups inside bounds."""
    if count <= 20:
        return maximum
    if count <= 45:
        return min(maximum, 11)
    if count <= 80:
        return 9
    if count <= 140:
        return 8
    return minimum


def add_warehouse_layout(fig: go.Figure) -> None:
    """Draw a large warehouse map matching the user's reference layout.

    Layout rules:
    - Two vertically aligned docks remain on the right.
    - The conveyor has no lines connecting it to either dock.
    - Pallets occupy six large, clearly separated warehouse zones.
    - On-floor boxes have dedicated north and south floor areas.
    - Dispatch is outside the warehouse boundary and shows carriers, not boxes.
    - No packing, staging, loaded, truck drawing, or truck animation section.
    """
    wx0, wy0, wx1, wy1 = WAREHOUSE_BOUNDS

    shapes: list[dict[str, Any]] = [
        # Warehouse shell.
        dict(
            type="rect",
            x0=wx0,
            y0=wy0,
            x1=wx1,
            y1=wy1,
            line=dict(width=2.5, color=DASHBOARD_COLORS["primary"]),
            fillcolor="rgba(239,246,255,0.72)",
            layer="below",
        ),
        # Two dedicated floor areas. These are operational floor zones,
        # not packing or staging sections.
        dict(
            type="rect",
            x0=FLOOR_TOP_BOUNDS[0],
            y0=FLOOR_TOP_BOUNDS[1],
            x1=FLOOR_TOP_BOUNDS[2],
            y1=FLOOR_TOP_BOUNDS[3],
            line=dict(width=1.6, color=DASHBOARD_COLORS["cyan"], dash="dot"),
            fillcolor="rgba(14,165,233,0.08)",
            layer="below",
        ),
        dict(
            type="rect",
            x0=FLOOR_BOTTOM_BOUNDS[0],
            y0=FLOOR_BOTTOM_BOUNDS[1],
            x1=FLOOR_BOTTOM_BOUNDS[2],
            y1=FLOOR_BOTTOM_BOUNDS[3],
            line=dict(width=1.6, color=DASHBOARD_COLORS["cyan"], dash="dot"),
            fillcolor="rgba(14,165,233,0.08)",
            layer="below",
        ),
        # Central conveyor belt.
        dict(
            type="rect",
            x0=CONVEYOR_BOUNDS[0],
            y0=CONVEYOR_BOUNDS[1],
            x1=CONVEYOR_BOUNDS[2],
            y1=CONVEYOR_BOUNDS[3],
            line=dict(width=2.2, color=DASHBOARD_COLORS["amber"]),
            fillcolor="rgba(245,158,11,0.18)",
            layer="below",
        ),
        # Two right-side docks.
        dict(
            type="rect",
            x0=DOCK_1_BOUNDS[0],
            y0=DOCK_1_BOUNDS[1],
            x1=DOCK_1_BOUNDS[2],
            y1=DOCK_1_BOUNDS[3],
            line=dict(width=2.3, color=DASHBOARD_COLORS["teal"]),
            fillcolor="rgba(15,118,110,0.12)",
            layer="below",
        ),
        dict(
            type="rect",
            x0=DOCK_2_BOUNDS[0],
            y0=DOCK_2_BOUNDS[1],
            x1=DOCK_2_BOUNDS[2],
            y1=DOCK_2_BOUNDS[3],
            line=dict(width=2.3, color=DASHBOARD_COLORS["teal"]),
            fillcolor="rgba(15,118,110,0.12)",
            layer="below",
        ),
        # Physical dock-door faces.
        dict(
            type="line",
            x0=DOCK_1_BOUNDS[2],
            y0=DOCK_1_BOUNDS[1],
            x1=DOCK_1_BOUNDS[2],
            y1=DOCK_1_BOUNDS[3],
            line=dict(width=6, color=DASHBOARD_COLORS["teal"]),
            layer="below",
        ),
        dict(
            type="line",
            x0=DOCK_2_BOUNDS[2],
            y0=DOCK_2_BOUNDS[1],
            x1=DOCK_2_BOUNDS[2],
            y1=DOCK_2_BOUNDS[3],
            line=dict(width=6, color=DASHBOARD_COLORS["teal"]),
            layer="below",
        ),
        # External dispatch block.
        dict(
            type="rect",
            x0=DISPATCH_BOUNDS[0],
            y0=DISPATCH_BOUNDS[1],
            x1=DISPATCH_BOUNDS[2],
            y1=DISPATCH_BOUNDS[3],
            line=dict(width=2.5, color=DASHBOARD_COLORS["green"]),
            fillcolor="rgba(22,163,74,0.12)",
            layer="below",
        ),
    ]

    # Conveyor rollers.
    roller_x = CONVEYOR_BOUNDS[0] + 0.22
    while roller_x <= CONVEYOR_BOUNDS[2] - 0.18:
        shapes.append(
            dict(
                type="line",
                x0=roller_x,
                y0=CONVEYOR_BOUNDS[1] + 0.08,
                x1=roller_x,
                y1=CONVEYOR_BOUNDS[3] - 0.08,
                line=dict(width=1, color="rgba(95,75,50,0.38)"),
                layer="below",
            )
        )
        roller_x += 0.34

    # Dock-to-dispatch paths only. There are intentionally no conveyor-to-dock lines.
    shapes.extend(
        [
            dict(
                type="line",
                x0=DOCK_1_BOUNDS[2] + 0.08,
                y0=(DOCK_1_BOUNDS[1] + DOCK_1_BOUNDS[3]) / 2,
                x1=DISPATCH_BOUNDS[0],
                y1=5.65,
                line=dict(width=2.0, color=DASHBOARD_COLORS["green"], dash="dash"),
                layer="below",
            ),
            dict(
                type="line",
                x0=DOCK_2_BOUNDS[2] + 0.08,
                y0=(DOCK_2_BOUNDS[1] + DOCK_2_BOUNDS[3]) / 2,
                x1=DISPATCH_BOUNDS[0],
                y1=4.05,
                line=dict(width=2.0, color=DASHBOARD_COLORS["green"], dash="dash"),
                layer="below",
            ),
        ]
    )

    fig.update_layout(shapes=shapes)

    labels = [
        (7.80, 9.48, "WAREHOUSE DIGITAL TWIN", 15, DASHBOARD_COLORS["primary"]),
        (2.02, 8.86, "ON FLOOR", 11, DASHBOARD_COLORS["blue"]),
        (2.02, 4.16, "ON FLOOR", 11, DASHBOARD_COLORS["blue"]),
        (6.55, 5.66, "CONVEYOR BELT", 12, DASHBOARD_COLORS["amber"]),
        (13.85, 8.88, "DOCK 1", 12, DASHBOARD_COLORS["teal"]),
        (13.85, 4.32, "DOCK 2", 12, DASHBOARD_COLORS["teal"]),
        (17.58, 7.50, "DISPATCH", 13, DASHBOARD_COLORS["green"]),
    ]

    for x, y, label, size, color in labels:
        fig.add_annotation(
            x=x,
            y=y,
            text=f"<b>{label}</b>",
            showarrow=False,
            font=dict(size=size, color=color),
        )

    # Zone labels for the six pallet locations.
    for index, bounds in enumerate(PALLET_ZONE_BOUNDS, start=1):
        x0, y0, x1, y1 = bounds
        fig.add_annotation(
            x=(x0 + x1) / 2,
            y=y1 + 0.16,
            text=f"<b>ZONE {index}</b>",
            showarrow=False,
            font=dict(size=10, color=DASHBOARD_COLORS["brown"]),
        )

    # Conveyor direction arrows.
    arrow_count = 9
    arrow_width = CONVEYOR_BOUNDS[2] - CONVEYOR_BOUNDS[0]
    for index in range(arrow_count):
        x = CONVEYOR_BOUNDS[0] + (index + 0.7) * arrow_width / arrow_count
        fig.add_annotation(
            x=x,
            y=(CONVEYOR_BOUNDS[1] + CONVEYOR_BOUNDS[3]) / 2,
            text="➜",
            showarrow=False,
            font=dict(size=19, color=DASHBOARD_COLORS["amber"]),
        )

    # Arrows from docks toward dispatch.
    fig.add_annotation(
        x=DISPATCH_BOUNDS[0],
        y=5.65,
        ax=DOCK_1_BOUNDS[2] + 0.08,
        ay=(DOCK_1_BOUNDS[1] + DOCK_1_BOUNDS[3]) / 2,
        text="",
        showarrow=True,
        arrowhead=3,
        arrowsize=1.1,
        arrowwidth=2,
        arrowcolor=DASHBOARD_COLORS["green"],
    )
    fig.add_annotation(
        x=DISPATCH_BOUNDS[0],
        y=4.05,
        ax=DOCK_2_BOUNDS[2] + 0.08,
        ay=(DOCK_2_BOUNDS[1] + DOCK_2_BOUNDS[3]) / 2,
        text="",
        showarrow=True,
        arrowhead=3,
        arrowsize=1.1,
        arrowwidth=2,
        arrowcolor=DASHBOARD_COLORS["green"],
    )


def draw_pallets(
    fig: go.Figure,
    pallet_frame: pd.DataFrame,
) -> dict[str, tuple[float, float, float, float]]:
    """Draw smaller pallets inside the unchanged warehouse zone footprints."""
    positions: dict[str, tuple[float, float, float, float]] = {}

    if pallet_frame.empty:
        return positions

    for index, row in pallet_frame.reset_index(drop=True).iterrows():
        if index >= len(PALLET_BOUNDS):
            break

        x0, y0, x1, y1 = PALLET_BOUNDS[index]
        code = str(row["pallet_code"])
        count = int(as_number(row.get("total_boxes", 0)))
        positions[code] = (x0, y0, x1, y1)

        base_fill = (
            "rgba(154,106,58,0.22)"
            if count > 0
            else "rgba(148,163,184,0.12)"
        )

        fig.add_shape(
            type="rect",
            x0=x0,
            y0=y0,
            x1=x1,
            y1=y1,
            line=dict(width=2, color=DASHBOARD_COLORS["brown"]),
            fillcolor=base_fill,
            layer="below",
        )

        # Subtle pallet slats stay behind the box markers.
        slat_y = y0 + 0.13
        for slat_index in range(5):
            slat_x0 = x0 + 0.20 + slat_index * ((x1 - x0 - 0.40) / 5)
            fig.add_shape(
                type="rect",
                x0=slat_x0,
                y0=slat_y,
                x1=slat_x0 + 0.28,
                y1=slat_y + 0.16,
                line=dict(width=0.7, color="#7c4f2a"),
                fillcolor="rgba(214,155,88,0.30)",
                layer="below",
            )

        fig.add_annotation(
            x=(x0 + x1) / 2,
            y=y1 - 0.20,
            text=f"<b>{code}</b> · {count} boxes",
            showarrow=False,
            font=dict(size=10, color="#5b3a20"),
            bgcolor="rgba(255,255,255,0.78)",
            borderpad=2,
        )

        # Transparent interaction target: hovering a pallet reveals its live load.
        fig.add_trace(
            go.Scatter(
                x=[(x0 + x1) / 2],
                y=[(y0 + y1) / 2],
                mode="markers",
                marker=dict(
                    size=62,
                    color="rgba(154,106,58,0.01)",
                    line=dict(width=0),
                ),
                customdata=[[
                    code,
                    count,
                    str(row.get("pallet_status", "")),
                    as_number(row.get("total_weight_kg", 0.0)),
                    as_number(row.get("total_volume_cm3", 0.0)),
                ]],
                hovertemplate=(
                    "<b>%{customdata[0]}</b><br>"
                    "Live boxes: %{customdata[1]}<br>"
                    "Status: %{customdata[2]}<br>"
                    "Weight: %{customdata[3]:.2f} kg<br>"
                    "Volume: %{customdata[4]:.2f} cm³"
                    "<extra></extra>"
                ),
                showlegend=False,
            )
        )

    return positions


def _assign_grid_to_indices(
    output: pd.DataFrame,
    indices: list[int],
    bounds: tuple[float, float, float, float],
    location_name: str,
    top_label_space: float = 0.0,
) -> None:
    if not indices:
        return

    points = _grid_points(
        bounds,
        len(indices),
        pad_x=0.13,
        pad_y=0.13,
        top_label_space=top_label_space,
    )
    marker_size = _marker_size_for_region(len(indices))

    for row_index, (x, y) in zip(indices, points):
        output.at[row_index, "x"] = x
        output.at[row_index, "y"] = y
        output.at[row_index, "display_size"] = marker_size
        output.at[row_index, "visual_location"] = location_name


def _known_dock_slot(value: Any) -> Optional[str]:
    """Map only explicit Dock 1/2 identifiers; never guess an unknown dock."""
    code = str(value or "").strip().upper().replace("-", "_").replace(" ", "_")
    if code in {"DOCK_1", "DOCK_01"}:
        return "DOCK 1"
    if code in {"DOCK_2", "DOCK_02"}:
        return "DOCK 2"
    return None


def position_boxes(
    box_frame: pd.DataFrame,
    pallet_positions: dict[str, tuple[float, float, float, float]],
) -> pd.DataFrame:
    """Place every box according to its latest lifecycle status.

    Visual priority:
    1. Dispatched boxes -> external dispatch block.
    2. AT_DOCK / LOADED_TO_TRAILER -> the recorded dock.
    3. Conveyor and conveyor exceptions -> conveyor belt.
    4. Pallet and pallet exceptions -> the recorded pallet.
    5. All remaining boxes -> north/south on-floor areas.

    This priority fixes the earlier problem where dock boxes were drawn on
    their pallets simply because the latest event still carried a pallet_id.
    """
    if box_frame.empty:
        return box_frame.copy()

    output = add_box_stage(box_frame).reset_index(drop=True)
    output["visual_status"] = output.apply(visual_status_for_row, axis=1)
    output["x"] = pd.NA
    output["y"] = pd.NA
    output["display_size"] = 9
    output["visual_location"] = ""

    lifecycle = (
        output.get("lifecycle_status", pd.Series(index=output.index, dtype="object"))
        .fillna("")
        .astype(str)
        .str.upper()
    )
    dock_codes = (
        output.get("dock_code", pd.Series(index=output.index, dtype="object"))
        .fillna("")
        .astype(str)
        .str.upper()
    )
    pallet_codes = (
        output.get("pallet_code", pd.Series(index=output.index, dtype="object"))
        .fillna("")
        .astype(str)
    )

    # Dispatched boxes are represented by their unique pallets/trailers in the
    # dispatch area. Mark them assigned here so they are not redrawn as boxes.
    dispatch_indices = output.index[
        lifecycle.str.contains("DISPATCH", regex=False)
        | output["box_status"].fillna("").astype(str).str.upper().eq("DISPATCHED")
    ].tolist()
    if dispatch_indices:
        output.loc[dispatch_indices, "visual_location"] = "DISPATCHED VIA CARRIER"

    assigned = set(dispatch_indices)

    # Dock and loaded boxes are drawn at their dock, even though they remain
    # part of the broad ON_PALLET KPI category.
    dock_mask = (
        lifecycle.str.contains("AT_DOCK", regex=False)
        | lifecycle.str.contains("LOADED", regex=False)
        | dock_codes.ne("")
    )
    dock_candidates = [index for index in output.index[dock_mask].tolist() if index not in assigned]

    dock_slots = {index: _known_dock_slot(dock_codes.iloc[index]) for index in dock_candidates}
    dock_1_indices = [index for index, slot in dock_slots.items() if slot == "DOCK 1"]
    dock_2_indices = [index for index, slot in dock_slots.items() if slot == "DOCK 2"]
    unresolved_dock_indices = [
        index for index, slot in dock_slots.items() if slot is None
    ]
    if unresolved_dock_indices:
        output.loc[unresolved_dock_indices, "visual_location"] = "UNRESOLVED DOCK"

    _assign_grid_to_indices(
        output,
        dock_1_indices,
        DOCK_1_BOUNDS,
        "DOCK 1",
        top_label_space=0.36,
    )
    _assign_grid_to_indices(
        output,
        dock_2_indices,
        DOCK_2_BOUNDS,
        "DOCK 2",
        top_label_space=0.36,
    )
    assigned.update(dock_1_indices)
    assigned.update(dock_2_indices)
    assigned.update(unresolved_dock_indices)

    # Conveyor.
    conveyor_indices = [
        index
        for index in output.index[
            lifecycle.str.contains("CONVEYOR", regex=False)
            | output["operational_stage"].eq("ON_CONVEYOR")
        ].tolist()
        if index not in assigned
    ]
    _assign_grid_to_indices(
        output,
        conveyor_indices,
        CONVEYOR_BOUNDS,
        "CONVEYOR",
        top_label_space=0.05,
    )
    assigned.update(conveyor_indices)

    # Pallets, including pallet exceptions. Each pallet gets its own independent
    # grid, so markers never spill into a neighbouring pallet zone.
    pallet_candidate_indices = [
        index
        for index in output.index[
            output["operational_stage"].eq("ON_PALLET")
            | lifecycle.str.contains("PALLET", regex=False)
        ].tolist()
        if index not in assigned
    ]

    for pallet_code, bounds in pallet_positions.items():
        indices = [
            index
            for index in pallet_candidate_indices
            if pallet_codes.iloc[index] == pallet_code
        ]
        _assign_grid_to_indices(
            output,
            indices,
            bounds,
            pallet_code,
            top_label_space=0.42,
        )
        assigned.update(indices)

    # Do not invent a pallet placement when the current pallet is unknown or is
    # not represented by one of the visible pallet records.
    unmatched_pallet_indices = [
        index for index in pallet_candidate_indices if index not in assigned
    ]
    if unmatched_pallet_indices:
        output.loc[unmatched_pallet_indices, "visual_location"] = "UNRESOLVED PALLET"
        assigned.update(unmatched_pallet_indices)

    # Keep boxes with no verified physical state out of the warehouse map.
    remaining_indices = [index for index in output.index if index not in assigned]
    unlocated_indices = [
        index
        for index in remaining_indices
        if output.at[index, "operational_stage"] == "UNLOCATED"
    ]
    if unlocated_indices:
        output.loc[unlocated_indices, "visual_location"] = "UNLOCATED"
        assigned.update(unlocated_indices)

    # Verified floor boxes share two display regions only to keep markers
    # readable; this split is visual layout, not an inferred warehouse location.
    remaining_indices = [index for index in output.index if index not in assigned]
    north_floor = [
        index for index in remaining_indices
        if lifecycle.iloc[index].startswith("RECEIVED")
    ]
    south_floor = [index for index in remaining_indices if index not in set(north_floor)]

    # Balance either floor area when one side is empty or heavily overloaded.
    combined_floor = north_floor + south_floor
    if len(north_floor) > len(combined_floor) * 0.72 or not south_floor:
        split = math.ceil(len(combined_floor) / 2)
        north_floor = combined_floor[:split]
        south_floor = combined_floor[split:]
    elif len(south_floor) > len(combined_floor) * 0.72 or not north_floor:
        split = math.ceil(len(combined_floor) / 2)
        north_floor = combined_floor[:split]
        south_floor = combined_floor[split:]

    _assign_grid_to_indices(
        output,
        north_floor,
        FLOOR_TOP_BOUNDS,
        "ON FLOOR — NORTH",
        top_label_space=0.32,
    )
    _assign_grid_to_indices(
        output,
        south_floor,
        FLOOR_BOTTOM_BOUNDS,
        "ON FLOOR — SOUTH",
        top_label_space=0.32,
    )

    return output


def add_region_counts(
    fig: go.Figure,
    positioned_boxes: pd.DataFrame,
    dispatch_frame: pd.DataFrame,
) -> None:
    """Show box counts in warehouse regions and carrier counts at dispatch."""
    counts: dict[str, int] = {}
    if not positioned_boxes.empty and "visual_location" in positioned_boxes.columns:
        counts = (
            positioned_boxes["visual_location"]
            .fillna("")
            .value_counts()
            .to_dict()
        )

    count_labels = [
        (2.02, 8.48, f"{counts.get('ON FLOOR — NORTH', 0)} boxes"),
        (2.02, 3.88, f"{counts.get('ON FLOOR — SOUTH', 0)} boxes"),
        (6.55, 4.52, f"{counts.get('CONVEYOR', 0)} boxes"),
        (13.85, 8.47, f"{counts.get('DOCK 1', 0)} boxes"),
        (13.85, 3.97, f"{counts.get('DOCK 2', 0)} boxes"),
    ]

    for x, y, value in count_labels:
        fig.add_annotation(
            x=x,
            y=y,
            text=f"<b>{value}</b>",
            showarrow=False,
            bgcolor="rgba(255,255,255,0.82)",
            bordercolor="rgba(80,80,80,0.25)",
            borderpad=3,
            font=dict(size=9, color="#333333"),
        )

    trailer_count = 0
    pallet_count = 0
    if not dispatch_frame.empty:
        trailer_count = int(
            dispatch_frame["asset_type"].astype(str).eq("TRAILER").sum()
        )
        pallet_count = int(
            pd.to_numeric(
                dispatch_frame["pallet_count"], errors="coerce"
            ).fillna(0).sum()
        )

    fig.add_annotation(
        x=17.58,
        y=7.05,
        text=f"<b>{pallet_count} pallets · {trailer_count} trailers</b>",
        showarrow=False,
        bgcolor="rgba(255,255,255,0.86)",
        bordercolor="rgba(22,163,74,0.38)",
        borderpad=3,
        font=dict(size=9, color=DASHBOARD_COLORS["green"]),
    )



def add_region_hover_targets(
    fig: go.Figure,
    positioned_boxes: pd.DataFrame,
    dispatch_frame: pd.DataFrame,
) -> None:
    """Add transparent hover hotspots without changing the warehouse data."""
    counts: dict[str, int] = {}
    if not positioned_boxes.empty and "visual_location" in positioned_boxes.columns:
        counts = (
            positioned_boxes["visual_location"]
            .fillna("")
            .value_counts()
            .to_dict()
        )

    trailer_count = 0
    pallet_count = 0
    if not dispatch_frame.empty:
        trailer_count = int(
            dispatch_frame["asset_type"].astype(str).eq("TRAILER").sum()
        )
        pallet_count = int(
            pd.to_numeric(
                dispatch_frame["pallet_count"], errors="coerce"
            ).fillna(0).sum()
        )

    targets = [
        (2.02, 7.20, "ON FLOOR — NORTH", counts.get("ON FLOOR — NORTH", 0), "Warehouse floor"),
        (2.02, 2.55, "ON FLOOR — SOUTH", counts.get("ON FLOOR — SOUTH", 0), "Warehouse floor"),
        (6.55, 4.90, "CONVEYOR", counts.get("CONVEYOR", 0), "Material movement"),
        (13.85, 6.90, "DOCK 1", counts.get("DOCK 1", 0), "Loading dock"),
        (13.85, 2.60, "DOCK 2", counts.get("DOCK 2", 0), "Loading dock"),
        (17.58, 4.85, "DISPATCH", pallet_count, f"{trailer_count} trailers"),
    ]

    fig.add_trace(
        go.Scatter(
            x=[item[0] for item in targets],
            y=[item[1] for item in targets],
            mode="markers",
            marker=dict(
                size=[74, 74, 44, 78, 78, 92],
                color="rgba(37,99,235,0.01)",
                line=dict(width=0),
            ),
            customdata=[
                [item[2], item[3], item[4]] for item in targets
            ],
            hovertemplate=(
                "<b>%{customdata[0]}</b><br>"
                "Current count: %{customdata[1]}<br>"
                "%{customdata[2]}"
                "<extra></extra>"
            ),
            showlegend=False,
            name="Warehouse regions",
        )
    )

def draw_dispatch_assets(fig: go.Figure, dispatch_frame: pd.DataFrame) -> None:
    """Draw unique trailers or unlinked pallets instead of dispatched boxes."""
    if dispatch_frame.empty:
        fig.add_annotation(
            x=(DISPATCH_BOUNDS[0] + DISPATCH_BOUNDS[2]) / 2,
            y=(DISPATCH_BOUNDS[1] + DISPATCH_BOUNDS[3]) / 2,
            text="No dispatched pallets or trailers",
            showarrow=False,
            font=dict(size=11, color="#6b6b6b"),
        )
        return

    assets = dispatch_frame.reset_index(drop=True).copy()
    points = _grid_points(
        DISPATCH_BOUNDS,
        len(assets),
        pad_x=0.30,
        pad_y=0.36,
        top_label_space=0.65,
    )

    x_values: list[float] = []
    y_values: list[float] = []
    symbols: list[str] = []
    sizes: list[int] = []
    colors: list[str] = []
    labels: list[str] = []
    custom_rows: list[list[Any]] = []

    for row, (x, y) in zip(assets.itertuples(index=False), points):
        asset_type = str(getattr(row, "asset_type", "PENDING"))
        asset_code = str(getattr(row, "asset_code", "Dispatch asset"))
        pallet_count = int(as_number(getattr(row, "pallet_count", 0)))
        pallet_codes = str(getattr(row, "pallet_codes", ""))
        box_count = int(as_number(getattr(row, "box_count", 0)))

        x_values.append(x)
        y_values.append(y)
        custom_rows.append(
            [asset_type, asset_code, pallet_count, pallet_codes, box_count]
        )

        if asset_type == "TRAILER":
            symbols.append("square")
            sizes.append(34)
            colors.append(DASHBOARD_COLORS["green"])
            labels.append(f"<b>{asset_code}</b><br>{pallet_count} pallets")
        elif asset_type == "PALLET":
            symbols.append("diamond")
            sizes.append(26)
            colors.append(DASHBOARD_COLORS["brown"])
            labels.append(f"<b>{asset_code}</b><br>1 pallet")
        else:
            symbols.append("square-open")
            sizes.append(26)
            colors.append(DASHBOARD_COLORS["slate"])
            labels.append("<b>Carrier pending</b>")

    fig.add_trace(
        go.Scatter(
            x=x_values,
            y=y_values,
            mode="markers+text",
            name="Dispatch carriers",
            marker=dict(
                symbol=symbols,
                size=sizes,
                color=colors,
                opacity=0.94,
                line=dict(color="white", width=1.2),
            ),
            text=labels,
            textposition="top center",
            textfont=dict(size=9, color="#14532d"),
            customdata=custom_rows,
            hovertemplate=(
                "<b>%{customdata[1]}</b><br>"
                "Type: %{customdata[0]}<br>"
                "Pallets: %{customdata[2]}<br>"
                "Pallet codes: %{customdata[3]}<br>"
                "Carrier assignment: database lifecycle"
                "<extra></extra>"
            ),
        )
    )


def add_box_markers(
    fig: go.Figure,
    positioned_boxes: pd.DataFrame,
    show_labels: bool,
) -> None:
    if positioned_boxes.empty:
        return

    positioned_boxes = positioned_boxes[
        positioned_boxes["x"].notna()
        & positioned_boxes["y"].notna()
    ].copy()

    if positioned_boxes.empty:
        return

    hover_columns = [
        "box_code",
        "order_code",
        "product_name",
        "box_status",
        "current_zone",
        "pallet_code",
        "dock_code",
        "lifecycle_status",
        "operational_stage",
        "visual_location",
        "layer_no",
        "slot_no",
        "row_no",
        "column_no",
        "position_source",
    ]

    for column in hover_columns:
        if column not in positioned_boxes.columns:
            positioned_boxes[column] = ""

    # Showing all 500 labels would hide the geometry. Labels are automatically
    # enabled only when filters reduce the visible set to a readable amount.
    can_show_all_labels = show_labels and len(positioned_boxes) <= 70
    if can_show_all_labels:
        positioned_boxes["display_text"] = positioned_boxes.apply(
            lambda row: (
                f"{row['box_code']}<br>L{int(row['layer_no'])}"
                if pd.notna(row.get("layer_no"))
                else str(row["box_code"])
            ),
            axis=1,
        )
    else:
        positioned_boxes["display_text"] = ""

    for status, style in STATUS_STYLE.items():
        subset = positioned_boxes[
            positioned_boxes["visual_status"] == status
        ]

        if subset.empty:
            continue

        fig.add_trace(
            go.Scatter(
                x=pd.to_numeric(subset["x"], errors="coerce"),
                y=pd.to_numeric(subset["y"], errors="coerce"),
                mode="markers+text" if can_show_all_labels else "markers",
                name=style["name"],
                text=subset["display_text"],
                textposition="top center",
                textfont=dict(size=8),
                customdata=subset[hover_columns],
                marker=dict(
                    size=pd.to_numeric(
                        subset["display_size"], errors="coerce"
                    ).fillna(8).tolist(),
                    color=style["color"],
                    symbol="square",
                    opacity=0.94,
                    line=dict(color="white", width=1.0),
                ),
                hovertemplate=(
                    "<b>%{customdata[0]}</b><br>"
                    "Orders: %{customdata[1]}<br>"
                    "Products: %{customdata[2]}<br>"
                    "Box status: %{customdata[3]}<br>"
                    "Current zone: %{customdata[4]}<br>"
                    "Pallet: %{customdata[5]}<br>"
                    "Dock: %{customdata[6]}<br>"
                    "Lifecycle: %{customdata[7]}<br>"
                    "KPI stage: %{customdata[8]}<br>"
                    "Visual location: %{customdata[9]}<br>"
                    "Layer: %{customdata[10]}<br>"
                    "Slot: %{customdata[11]} · Row: %{customdata[12]} · Column: %{customdata[13]}<br>"
                    "Layer source: %{customdata[14]}"
                    "<extra></extra>"
                ),
            )
        )


@st.cache_data(ttl=60, show_spinner=False)
def create_warehouse_figure(
    box_frame: pd.DataFrame,
    pallet_frame: pd.DataFrame,
    show_labels: bool,
) -> go.Figure:
    figure = go.Figure()
    add_warehouse_layout(figure)

    pallet_positions = draw_pallets(figure, pallet_frame)
    positioned = position_boxes(box_frame, pallet_positions)
    unresolved_count = int(
        positioned.get("visual_location", pd.Series(dtype="object"))
        .isin({"UNLOCATED", "UNRESOLVED DOCK", "UNRESOLVED PALLET"})
        .sum()
    )
    if unresolved_count:
        figure.add_annotation(
            x=0.55,
            y=9.48,
            xanchor="left",
            text=f"<b>{unresolved_count} box(es) omitted: no verified map location</b>",
            showarrow=False,
            bgcolor="rgba(254,243,199,0.95)",
            bordercolor=DASHBOARD_COLORS["amber"],
            borderpad=4,
            font=dict(size=10, color="#7c4a03"),
        )
    filtered_dispatch_assets = build_dispatch_assets(box_frame)
    add_region_counts(figure, positioned, filtered_dispatch_assets)
    add_region_hover_targets(figure, positioned, filtered_dispatch_assets)
    draw_dispatch_assets(figure, filtered_dispatch_assets)
    add_box_markers(figure, positioned, show_labels)

    figure.update_xaxes(
        range=[0.15, 19.65],
        visible=False,
        showgrid=False,
        zeroline=False,
        showticklabels=False,
        fixedrange=False,
    )
    figure.update_yaxes(
        range=[0.40, 9.70],
        visible=False,
        showgrid=False,
        zeroline=False,
        showticklabels=False,
        fixedrange=False,
    )

    figure.update_layout(
        height=930,
        margin=dict(l=8, r=8, t=36, b=8),
        hovermode="closest",
        dragmode="pan",
        plot_bgcolor=DASHBOARD_COLORS["surface"],
        paper_bgcolor=DASHBOARD_COLORS["surface"],
        font=dict(color=DASHBOARD_COLORS["text"]),
        hoverlabel=dict(
            bgcolor="#ffffff",
            bordercolor=DASHBOARD_COLORS["blue"],
            font=dict(color=DASHBOARD_COLORS["text"], size=12),
        ),
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.005,
            xanchor="right",
            x=1,
            font=dict(size=10, color=DASHBOARD_COLORS["text"]),
            bgcolor="rgba(255,255,255,0.88)",
            bordercolor=DASHBOARD_COLORS["border"],
            borderwidth=1,
            itemclick="toggle",
            itemdoubleclick="toggleothers",
        ),
        uirevision="visiontrack-large-organized-layout",
    )

    return figure


# ============================================================
# PALLET LAYER VISUALIZATION
# This keeps the original dashboard design and adds only the stacking view.
# Layer 1 is placed on the pallet base; higher layers are drawn above it.
# ============================================================
LAYER_COLORS = {
    1: DASHBOARD_COLORS["blue"],
    2: DASHBOARD_COLORS["purple"],
    3: DASHBOARD_COLORS["amber"],
    4: DASHBOARD_COLORS["green"],
    5: "#db2777",
}


def _layer_color(layer_no: int) -> str:
    if layer_no in LAYER_COLORS:
        return LAYER_COLORS[layer_no]

    palette = [DASHBOARD_COLORS["blue"], DASHBOARD_COLORS["purple"], DASHBOARD_COLORS["amber"], DASHBOARD_COLORS["green"], "#db2777"]
    return palette[(max(layer_no, 1) - 1) % len(palette)]


def _hex_to_rgba(hex_color: str, alpha: float) -> str:
    value = hex_color.lstrip("#")
    red = int(value[0:2], 16)
    green = int(value[2:4], 16)
    blue = int(value[4:6], 16)
    return f"rgba({red},{green},{blue},{alpha})"


@st.cache_data(ttl=60, show_spinner=False)
def create_pallet_layer_figure(
    layer_frame: pd.DataFrame,
    pallet_code: str,
    show_box_labels: bool,
) -> go.Figure:
    figure = go.Figure()

    if layer_frame.empty:
        figure.add_annotation(
            x=0.5,
            y=0.5,
            text="No pallet layer data is available.",
            showarrow=False,
            font=dict(size=18, color="#666666"),
        )
        figure.update_xaxes(visible=False, range=[0, 1])
        figure.update_yaxes(visible=False, range=[0, 1])
        figure.update_layout(height=460, paper_bgcolor=DASHBOARD_COLORS["surface"], plot_bgcolor=DASHBOARD_COLORS["surface"], font=dict(color=DASHBOARD_COLORS["text"]))
        return figure

    pallet_boxes = layer_frame[
        layer_frame["pallet_code"].astype(str).eq(str(pallet_code))
    ].copy()

    if pallet_boxes.empty:
        figure.add_annotation(
            x=0.5,
            y=0.5,
            text=f"No boxes are currently assigned to {pallet_code}.",
            showarrow=False,
            font=dict(size=18, color="#666666"),
        )
        figure.update_xaxes(visible=False, range=[0, 1])
        figure.update_yaxes(visible=False, range=[0, 1])
        figure.update_layout(height=460, paper_bgcolor=DASHBOARD_COLORS["surface"], plot_bgcolor=DASHBOARD_COLORS["surface"], font=dict(color=DASHBOARD_COLORS["text"]))
        return figure

    pallet_boxes = pallet_boxes.sort_values(
        ["layer_no", "row_no", "column_no", "box_code"]
    )

    max_layer = max(int(pallet_boxes["layer_no"].max()), 1)
    max_column = max(int(pallet_boxes["column_no"].max()), 3)

    box_width = 1.12
    box_height = 0.68
    column_gap = 0.20
    layer_gap = 0.30
    depth_x = 0.18
    depth_y = 0.13

    total_width = max_column * (box_width + column_gap) + 1.0
    max_y = max_layer * (box_height + layer_gap) + 1.65

    # Pallet slats and base.
    base_x0 = 0.42
    base_x1 = total_width - 0.30
    base_y0 = 0.20
    base_y1 = 0.47

    figure.add_shape(
        type="rect",
        x0=base_x0,
        y0=base_y0,
        x1=base_x1,
        y1=base_y1,
        line=dict(color="#75502d", width=2),
        fillcolor="#b27a42",
        layer="below",
    )

    slat_count = max(3, max_column)
    slat_width = (base_x1 - base_x0) / (slat_count * 1.55)
    for slat_index in range(slat_count):
        slat_x0 = base_x0 + 0.14 + slat_index * ((base_x1 - base_x0 - 0.28) / slat_count)
        figure.add_shape(
            type="rect",
            x0=slat_x0,
            y0=base_y1,
            x1=slat_x0 + slat_width,
            y1=base_y1 + 0.13,
            line=dict(color="#8c6239", width=1),
            fillcolor="#d3a16a",
            layer="below",
        )

    # Layer guide lines make vertical stacking immediately visible.
    for layer_no in range(1, max_layer + 1):
        band_y = 0.68 + (layer_no - 1) * (box_height + layer_gap)
        color = _layer_color(layer_no)

        figure.add_shape(
            type="line",
            x0=0.18,
            y0=band_y - 0.10,
            x1=total_width - 0.05,
            y1=band_y - 0.10,
            line=dict(color=_hex_to_rgba(color, 0.45), width=1.4, dash="dot"),
            layer="below",
        )
        figure.add_annotation(
            x=0.06,
            y=band_y + box_height / 2,
            text=f"<b>LAYER {layer_no}</b>",
            showarrow=False,
            textangle=-90,
            font=dict(size=10, color=color),
        )

    hover_x: list[float] = []
    hover_y: list[float] = []
    custom_rows: list[list[Any]] = []

    for row in pallet_boxes.itertuples(index=False):
        layer_no = max(int(getattr(row, "layer_no", 1)), 1)
        row_no = max(int(getattr(row, "row_no", 1)), 1)
        column_no = max(int(getattr(row, "column_no", 1)), 1)
        slot_no = max(int(getattr(row, "slot_no", 1)), 1)

        # Row 2 is projected slightly up/right to create visible depth.
        depth_level = max(row_no - 1, 0)
        x0 = 0.63 + (column_no - 1) * (box_width + column_gap) + depth_level * depth_x
        y0 = 0.68 + (layer_no - 1) * (box_height + layer_gap) + depth_level * depth_y
        x1 = x0 + box_width
        y1 = y0 + box_height

        color = _layer_color(layer_no)
        front_fill = _hex_to_rgba(color, 0.86)
        top_fill = _hex_to_rgba(color, 0.58)
        side_fill = _hex_to_rgba(color, 0.70)

        # Front face.
        figure.add_shape(
            type="rect",
            x0=x0,
            y0=y0,
            x1=x1,
            y1=y1,
            line=dict(color="white", width=1.4),
            fillcolor=front_fill,
            layer="below",
        )

        # Top and right faces provide the box-like 3D effect.
        figure.add_shape(
            type="path",
            path=(
                f"M {x0},{y1} L {x0 + 0.13},{y1 + 0.11} "
                f"L {x1 + 0.13},{y1 + 0.11} L {x1},{y1} Z"
            ),
            line=dict(color="white", width=0.8),
            fillcolor=top_fill,
            layer="below",
        )
        figure.add_shape(
            type="path",
            path=(
                f"M {x1},{y0} L {x1 + 0.13},{y0 + 0.11} "
                f"L {x1 + 0.13},{y1 + 0.11} L {x1},{y1} Z"
            ),
            line=dict(color="white", width=0.8),
            fillcolor=side_fill,
            layer="below",
        )

        box_code = str(getattr(row, "box_code", "BOX"))
        if show_box_labels:
            figure.add_annotation(
                x=(x0 + x1) / 2,
                y=(y0 + y1) / 2,
                text=f"<b>{box_code}</b><br>L{layer_no} · S{slot_no}",
                showarrow=False,
                font=dict(size=9, color="white"),
                align="center",
            )

        hover_x.append((x0 + x1) / 2)
        hover_y.append((y0 + y1) / 2)
        custom_rows.append(
            [
                box_code,
                str(getattr(row, "status", "")),
                layer_no,
                slot_no,
                row_no,
                column_no,
                str(getattr(row, "order_code", "")),
                str(getattr(row, "product_name", "")),
                as_number(getattr(row, "weight_kg", 0.0)),
                str(getattr(row, "position_source", "")),
            ]
        )

    # Invisible hover layer over the drawn box faces.
    figure.add_trace(
        go.Scatter(
            x=hover_x,
            y=hover_y,
            mode="markers",
            marker=dict(size=42, opacity=0.01, color="white"),
            customdata=custom_rows,
            hovertemplate=(
                "<b>%{customdata[0]}</b><br>"
                "Status: %{customdata[1]}<br>"
                "Layer: %{customdata[2]}<br>"
                "Slot: %{customdata[3]} · Row: %{customdata[4]} · Column: %{customdata[5]}<br>"
                "Orders: %{customdata[6]}<br>"
                "Products: %{customdata[7]}<br>"
                "Weight: %{customdata[8]:.2f} kg<br>"
                "Position source: %{customdata[9]}"
                "<extra></extra>"
            ),
            showlegend=False,
        )
    )

    # Compact layer legend matching the dashboard palette.
    for layer_no in range(1, max_layer + 1):
        figure.add_trace(
            go.Scatter(
                x=[None],
                y=[None],
                mode="markers",
                marker=dict(
                    size=11,
                    symbol="square",
                    color=_layer_color(layer_no),
                ),
                name=f"Layer {layer_no}",
                hoverinfo="skip",
            )
        )

    figure.add_annotation(
        x=(base_x0 + base_x1) / 2,
        y=0.04,
        text=f"<b>{pallet_code}</b> · Layer 1 is the bottom layer",
        showarrow=False,
        font=dict(size=12, color="#4f3420"),
    )

    figure.update_xaxes(
        visible=False,
        range=[-0.15, total_width + 0.35],
        fixedrange=False,
    )
    figure.update_yaxes(
        visible=False,
        range=[-0.05, max_y],
        fixedrange=False,
    )
    figure.update_layout(
        height=max(480, 320 + max_layer * 95),
        margin=dict(l=10, r=10, t=25, b=10),
        plot_bgcolor=DASHBOARD_COLORS["surface"],
        paper_bgcolor=DASHBOARD_COLORS["surface"],
        hovermode="closest",
        dragmode="pan",
        font=dict(color=DASHBOARD_COLORS["text"]),
        hoverlabel=dict(
            bgcolor="#ffffff",
            bordercolor=DASHBOARD_COLORS["purple"],
            font=dict(color=DASHBOARD_COLORS["text"], size=12),
        ),
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.005,
            xanchor="right",
            x=1,
            font=dict(size=10, color=DASHBOARD_COLORS["text"]),
            bgcolor="rgba(255,255,255,0.88)",
            bordercolor=DASHBOARD_COLORS["border"],
            borderwidth=1,
            itemclick="toggle",
            itemdoubleclick="toggleothers",
        ),
        uirevision=f"visiontrack-pallet-layers-{pallet_code}",
    )

    return figure


# ============================================================
# PAGE: LIVE OPERATIONS
# ============================================================
if _QUERY_FAILURES:
    failed_sources = ", ".join(
        f"{label} ({error_type})"
        for label, error_type in sorted(_QUERY_FAILURES.items())
    )
    st.error(
        "Dashboard data is incomplete, so operational KPIs and location maps "
        "are withheld instead of showing false zero or healthy states."
    )
    st.caption("Failed data sources: " + failed_sources)
    st.stop()


if page == "Live Operations":
    st.markdown(
        '<div class="main-title">VISIONTRACK LIVE OPERATIONS</div>',
        unsafe_allow_html=True,
    )

    render_kpis()
    render_inference_run_kpis()

    # Calculation consistency check.
    if reconciled_box_total != boxes_processed:
        st.error(
            "Box calculation mismatch: "
            f"{reconciled_box_total} categorized but "
            f"{boxes_processed} unique boxes exist."
        )

    # Exception-first visibility requested by the guide.
    if exception_count > 0:
        st.error(
            f"⚠️ {exception_count} active exception(s) require review. "
            "Open Exceptions & Events for details."
        )
    else:
        st.success(
            "✅ Operations normal — no active missing, removed, warning, "
            "critical, or error cases."
        )

    filter_1, filter_2, filter_3, filter_4 = st.columns(
        [2, 2, 2, 1]
    )

    with filter_1:
        order_options = ["All Orders"]

        if not boxes.empty:
            order_options += sorted(
                boxes["order_code"]
                .dropna()
                .astype(str)
                .unique()
                .tolist()
            )

        selected_order = st.selectbox(
            "Order",
            order_options,
        )

    with filter_2:
        status_options = (
            sorted(
                boxes["box_status"]
                .dropna()
                .astype(str)
                .unique()
                .tolist()
            )
            if not boxes.empty
            else []
        )

        selected_statuses = st.multiselect(
            "Box status",
            status_options,
            default=status_options,
        )

    with filter_3:
        zone_options = (
            sorted(
                boxes["current_zone"]
                .dropna()
                .astype(str)
                .unique()
                .tolist()
            )
            if not boxes.empty
            else []
        )

        selected_zones = st.multiselect(
            "Zone",
            zone_options,
            default=zone_options,
        )

    with filter_4:
        show_labels = st.toggle(
            "Show labels",
            value=False,
        )

    visual_boxes = boxes.copy()

    if selected_order != "All Orders":
        visual_boxes = visual_boxes[
            visual_boxes["order_code"] == selected_order
        ]

    if selected_statuses:
        visual_boxes = visual_boxes[
            visual_boxes["box_status"].isin(selected_statuses)
        ]

    if selected_zones:
        visual_boxes = visual_boxes[
            visual_boxes["current_zone"].isin(selected_zones)
        ]

    st.markdown(
        '<div class="section-title">LIVE WAREHOUSE VISUALIZATION</div>',
        unsafe_allow_html=True,
    )

    st.plotly_chart(
        create_warehouse_figure(
            visual_boxes,
            pallets,
            show_labels,
        ),
        width="stretch",
        config={
            "displaylogo": False,
            "scrollZoom": True,
            "displayModeBar": "hover",
            "doubleClick": "reset",
            "modeBarButtonsToRemove": ["lasso2d", "select2d"],
            "responsive": True,
        },
    )

    # Pallet stack layer visual. The rest of the dashboard remains unchanged.
    st.markdown(
        '<div class="section-title">PALLET LAYER VISUALIZATION</div>',
        unsafe_allow_html=True,
    )

    if pallet_layer_boxes.empty:
        st.info(
            "No live pallet boxes are available. Layer positions will appear "
            "when boxes are assigned to a pallet or video_final_boxes receives data."
        )
    else:
        visible_layer_data = pallet_layer_boxes.copy()

        # Respect the Live Operations filters where possible.
        visible_box_codes = set(visual_boxes["box_code"].astype(str).tolist())
        visible_layer_data = visible_layer_data[
            visible_layer_data["box_code"].astype(str).isin(visible_box_codes)
        ]

        if visible_layer_data.empty:
            st.info("No pallet boxes match the selected dashboard filters.")
        else:
            pallet_options = sorted(
                visible_layer_data["pallet_code"]
                .dropna()
                .astype(str)
                .unique()
                .tolist()
            )

            layer_filter_1, layer_filter_2 = st.columns([2, 1])
            with layer_filter_1:
                selected_layer_pallet = st.selectbox(
                    "Pallet stack",
                    pallet_options,
                    key="selected_layer_pallet",
                )
            with layer_filter_2:
                show_layer_box_labels = st.toggle(
                    "Show box and layer labels",
                    value=True,
                    key="show_layer_box_labels",
                )

            selected_layer_frame = visible_layer_data[
                visible_layer_data["pallet_code"].astype(str).eq(
                    str(selected_layer_pallet)
                )
            ].copy()

            layer_count = (
                int(selected_layer_frame["layer_no"].max())
                if not selected_layer_frame.empty
                else 0
            )
            layer_box_count = int(selected_layer_frame["box_code"].nunique())
            layer_weight = pd.to_numeric(
                selected_layer_frame.get(
                    "weight_kg",
                    pd.Series(dtype=float),
                ),
                errors="coerce",
            ).fillna(0).sum()
            computer_vision_positions = int(
                selected_layer_frame["position_source"]
                .fillna("")
                .astype(str)
                .eq("Computer vision")
                .sum()
            )

            layer_kpis = st.columns(4)
            layer_kpis[0].markdown(
                (
                    '<div class="kpi-label">Boxes on Pallet</div>'
                    '<div class="kpi-card">'
                    f'<div class="kpi-value">{layer_box_count}</div>'
                    '<div class="kpi-note">unique physical boxes</div>'
                    '</div>'
                ),
                unsafe_allow_html=True,
            )
            layer_kpis[1].markdown(
                (
                    '<div class="kpi-label">Stack Layers</div>'
                    '<div class="kpi-card">'
                    f'<div class="kpi-value">{layer_count}</div>'
                    '<div class="kpi-note">Layer 1 is the base</div>'
                    '</div>'
                ),
                unsafe_allow_html=True,
            )
            layer_kpis[2].markdown(
                (
                    '<div class="kpi-label">Total Weight</div>'
                    '<div class="kpi-card">'
                    f'<div class="kpi-value">{layer_weight:,.1f} kg</div>'
                    '<div class="kpi-note">current pallet load</div>'
                    '</div>'
                ),
                unsafe_allow_html=True,
            )
            layer_kpis[3].markdown(
                (
                    '<div class="kpi-label">Vision Positions</div>'
                    '<div class="kpi-card">'
                    f'<div class="kpi-value">{computer_vision_positions}</div>'
                    '<div class="kpi-note">from video_final_boxes</div>'
                    '</div>'
                ),
                unsafe_allow_html=True,
            )

            st.plotly_chart(
                create_pallet_layer_figure(
                    visible_layer_data,
                    selected_layer_pallet,
                    show_layer_box_labels,
                ),
                width="stretch",
                config={
                    "displaylogo": False,
                    "scrollZoom": True,
                    "displayModeBar": "hover",
                    "doubleClick": "reset",
                    "modeBarButtonsToRemove": ["lasso2d", "select2d"],
                    "responsive": True,
                },
            )

            with st.expander("View layer placement data"):
                display_columns = [
                    column
                    for column in [
                        "box_code",
                        "pallet_code",
                        "layer_no",
                        "slot_no",
                        "row_no",
                        "column_no",
                        "status",
                        "weight_kg",
                        "position_source",
                    ]
                    if column in selected_layer_frame.columns
                ]
                st.dataframe(
                    selected_layer_frame[display_columns],
                    width="stretch",
                    hide_index=True,
                )


    # Guide-aligned 30-second situational-awareness layer.
    flow_left, exception_right = st.columns([1.65, 1])

    with flow_left:
        st.markdown(
            '<div class="section-title">SHIPMENT FLOW PIPELINE</div>',
            unsafe_allow_html=True,
        )

        flow_stages = [
            ("At Docks (Boxes)", boxes_at_dock),
            ("On Conveyor (Boxes)", boxes_on_conveyor),
            ("On Pallets (Boxes)", boxes_on_pallets),
            (
                "Dispatch Carriers",
                f"{dispatched_pallets} pallets / {dispatched_trailers} trailers",
            ),
        ]

        flow_columns = st.columns(len(flow_stages))
        for index, (stage_name, stage_value) in enumerate(flow_stages):
            with flow_columns[index]:
                st.markdown(
                    (
                        '<div class="kpi-card">'
                        f'<div class="kpi-label">{stage_name}</div>'
                        f'<div class="kpi-value">{stage_value}</div>'
                        '</div>'
                    ),
                    unsafe_allow_html=True,
                )

    with exception_right:
        st.markdown(
            '<div class="section-title">ACTIVE EXCEPTIONS</div>',
            unsafe_allow_html=True,
        )

        live_exception_rows = []

        if not boxes.empty:
            for row in (
                boxes[
                    boxes["box_status"]
                    .fillna("")
                    .astype(str)
                    .str.upper()
                    .isin(["MISSING", "REMOVED"])
                ]
                .head(4)
                .itertuples(index=False)
            ):
                live_exception_rows.append(
                    {
                        "time": "Current",
                        "type": (
                            f"{getattr(row, 'box_code', 'Box')} — "
                            f"{getattr(row, 'box_status', 'Exception')}"
                        ),
                        "severity": "Critical",
                    }
                )

        if (
            not video_events.empty
            and "severity" in video_events.columns
        ):
            event_time_column = next(
                (
                    column
                    for column in [
                        "event_time",
                        "timestamp",
                        "created_at",
                    ]
                    if column in video_events.columns
                ),
                None,
            )
            event_type_column = next(
                (
                    column
                    for column in [
                        "event_type",
                        "event_name",
                        "description",
                    ]
                    if column in video_events.columns
                ),
                None,
            )

            warning_frame = video_events[
                video_events["severity"]
                .fillna("")
                .astype(str)
                .str.upper()
                .isin(["WARNING", "CRITICAL", "ERROR"])
            ].head(max(0, 5 - len(live_exception_rows)))

            for _, event_row in warning_frame.iterrows():
                raw_time = (
                    event_row.get(event_time_column)
                    if event_time_column
                    else "Current"
                )
                raw_type = (
                    event_row.get(event_type_column)
                    if event_type_column
                    else "Vision exception"
                )
                raw_severity = str(
                    event_row.get("severity", "Attention")
                ).title()
                live_exception_rows.append(
                    {
                        "time": str(raw_time),
                        "type": str(raw_type),
                        "severity": raw_severity,
                    }
                )

        if not live_exception_rows:
            st.success("No active exceptions.")
        else:
            for item in live_exception_rows[:5]:
                severity_upper = item["severity"].upper()
                icon = (
                    "🔴"
                    if severity_upper in {"CRITICAL", "ERROR"}
                    else "🟡"
                )
                st.markdown(
                    f"**{icon} {item['type']}**  \n"
                    f"{item['time']} · {item['severity']}"
                )

    st.caption(
        "The warehouse map follows the guide reference style: open pallet "
        "zones above and below the conveyor, two right-side docks beside the belt, and a dispatch area showing pallets/trailers outside the warehouse. Packing, "
        "staging, loaded sections, and truck drawings are removed."
    )


# ============================================================
# PAGE: ORDER CONTROL CENTER
# ============================================================
elif page == "Order Control Center":
    st.markdown(
        '<div class="main-title">ORDER CONTROL CENTER</div>',
        unsafe_allow_html=True,
    )

    total_orders_count = int(len(orders))
    dispatched_count = f"{dispatched_pallets} P / {dispatched_trailers} T"
    on_pallet_count = boxes_on_pallets
    on_conveyor_count = boxes_on_conveyor
    on_floor_count = boxes_on_floor

    order_snapshot_cards = [
        ("Total Orders", total_orders_count),
        ("Dispatched Pallets / Trailers", dispatched_count),
        ("On Pallets", on_pallet_count),
        ("On Conveyor", on_conveyor_count),
        ("On Floor", on_floor_count),
    ]

    snapshot_columns = st.columns(5)
    for column, (label, value) in zip(
        snapshot_columns,
        order_snapshot_cards,
    ):
        column.markdown(
            f'<div class="kpi-label">{label}</div>',
            unsafe_allow_html=True,
        )
        column.markdown(
            (
                '<div class="kpi-card">'
                f'<div class="kpi-value">{value if isinstance(value, str) else f"{value:,}"}</div>'
                '</div>'
            ),
            unsafe_allow_html=True,
        )

    st.markdown(
        '<div class="section-title">ORDER STATUS DISTRIBUTION</div>',
        unsafe_allow_html=True,
    )

    if orders.empty or "order_status" not in orders.columns:
        st.info("No order-status data is available.")
    else:
        order_status_distribution = (
            orders["order_status"]
            .fillna("UNKNOWN")
            .astype(str)
            .str.replace("_", " ", regex=False)
            .str.title()
            .value_counts()
            .rename_axis("Order Status")
            .reset_index(name="Orders")
        )

        status_figure = px.pie(
            order_status_distribution,
            names="Order Status",
            values="Orders",
            hole=0.46,
            title=f"Order Status Distribution ({total_orders_count:,} orders)",
            color_discrete_sequence=CHART_COLOR_SEQUENCE,
        )
        status_figure.update_traces(
            textposition="inside",
            textinfo="label+percent+value",
            hovertemplate=(
                "<b>%{label}</b><br>"
                "Orders: %{value}<br>"
                "Share: %{percent}"
                "<extra></extra>"
            ),
        )
        status_figure.update_layout(
            height=620,
            margin=dict(l=20, r=20, t=70, b=20),
            paper_bgcolor=DASHBOARD_COLORS["surface"],
            plot_bgcolor=DASHBOARD_COLORS["surface"],
            font=dict(color=DASHBOARD_COLORS["text"]),
            hoverlabel=dict(
                bgcolor="#ffffff",
                bordercolor=DASHBOARD_COLORS["blue"],
                font=dict(color=DASHBOARD_COLORS["text"], size=12),
            ),
            legend=dict(
                orientation="h",
                yanchor="bottom",
                y=-0.08,
                xanchor="center",
                x=0.5,
                bgcolor="rgba(255,255,255,0.88)",
                bordercolor=DASHBOARD_COLORS["border"],
                borderwidth=1,
                itemclick="toggle",
                itemdoubleclick="toggleothers",
            ),
        )
        st.plotly_chart(
            status_figure,
            width="stretch",
            config={"displaylogo": False, "displayModeBar": "hover", "responsive": True},
        )


elif page == "Exceptions & Events":
    st.markdown(
        '<div class="main-title">EXCEPTIONS & EVENTS</div>',
        unsafe_allow_html=True,
    )

    box_exceptions = (
        boxes[
            boxes["box_status"].isin(["MISSING", "REMOVED"])
        ]
        if not boxes.empty
        else pd.DataFrame()
    )

    warning_events = pd.DataFrame()

    if (
        not video_events.empty
        and "severity" in video_events.columns
    ):
        warning_events = video_events[
            video_events["severity"]
            .fillna("")
            .astype(str)
            .str.upper()
            .isin(["WARNING", "CRITICAL", "ERROR"])
        ]

    exception_1, exception_2, exception_3 = st.columns(3)
    exception_1.metric(
        "Missing/Removed Boxes",
        len(box_exceptions),
    )
    exception_2.metric(
        "Video Warnings",
        len(warning_events),
    )
    exception_3.metric(
        "Total Exceptions",
        exception_count,
    )

    tab_1, tab_2, tab_3 = st.tabs(
        [
            "Current Box Exceptions",
            "Video Exceptions",
            "Warehouse Events",
        ]
    )

    with tab_1:
        if box_exceptions.empty:
            st.success("No missing or removed boxes.")
        else:
            st.dataframe(
                box_exceptions,
                width="stretch",
                hide_index=True,
            )

    with tab_2:
        if warning_events.empty:
            st.success("No warning or critical video events.")
        else:
            st.dataframe(
                warning_events,
                width="stretch",
                hide_index=True,
            )

    with tab_3:
        st.dataframe(
            events,
            width="stretch",
            hide_index=True,
        )


st.caption(
    "Last refreshed: "
    + datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    + " | Stable manual-refresh mode"
)
