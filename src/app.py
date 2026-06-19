from pathlib import Path
from textwrap import dedent
from html import escape
import math
import sys

SRC_DIR = Path(__file__).resolve().parent
PROJECT_DIR = SRC_DIR.parent
DATA_DIR = PROJECT_DIR / "data"
MODEL_DIR = PROJECT_DIR / "models"

# Makes local modules import correctly whether the command is executed
# from the project root or from the src directory.
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

import folium
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from streamlit_folium import st_folium

from feedback import PostEventLogger
from live_monitor import LiveTrafficInput, LiveTrafficMonitor
from model_predict import DurationPredictor
from mongodb_store import MongoStore
from recommend import IncidentRecommender
from scenario_simulator import ScenarioPlan, ScenarioSimulator

st.set_page_config(
    page_title="Event Congestion Ops Console",
    page_icon="🚦",
    layout="wide",
)
def build_resource_plan(latest):
    """
    Converts model recommendations into a practical field-deployment plan.

    The recommender output is treated as the baseline. Minimum operational
    resources are then added according to congestion risk, event type,
    disruption duration, road closure and diversion requirements.
    """

    result = latest["result"]
    duration = latest["duration"]

    risk_score = int(latest.get("risk_score", 0))
    p50 = float(duration.get("p50", 0))
    p90 = float(duration.get("p90", p50))

    cause = str(latest.get("cause", "others"))
    event_type = str(latest.get("event_type", "unplanned"))

    model_officers = max(
        0,
        int(result.get("recommend_traffic_constables", 0))
    )

    model_barricades = max(
        0,
        int(result.get("recommend_barricades", 0))
    )

    road_closure = bool(
        result.get("recommend_road_closure", False)
    )

    diversion = bool(
        result.get("recommend_diversion", False)
    )

    # Minimum deployment according to threat level
    if risk_score >= 80:
        minimum_officers = 24
        minimum_barricades = 12
        command_level = "CRITICAL RESPONSE"

    elif risk_score >= 60:
        minimum_officers = 16
        minimum_barricades = 8
        command_level = "HIGH RESPONSE"

    elif risk_score >= 35:
        minimum_officers = 8
        minimum_barricades = 4
        command_level = "MODERATE RESPONSE"

    else:
        minimum_officers = 4
        minimum_barricades = 2
        command_level = "ROUTINE RESPONSE"

    officers = max(
        model_officers,
        minimum_officers
    )

    barricades = max(
        model_barricades,
        minimum_barricades
    )

    # Cause-based adjustment
    major_public_events = {
        "protest",
        "procession",
        "public_event",
        "vip_movement",
    }

    if cause in major_public_events:
        officers += 4
        barricades += 2

    # Planned events require advance perimeter deployment
    if event_type == "planned":
        officers += 2

    # Long-duration congestion requires shift support
    if p50 >= 180:
        officers += 4

    if p90 >= 300:
        officers += 4

    # Road-control adjustments
    if road_closure:
        officers += 6
        barricades += 6

    if diversion:
        officers += 4
        barricades += 4

    # Deployment distribution
    junction_officers = max(
        2,
        round(officers * 0.35)
    )

    entry_exit_officers = max(
        2,
        round(officers * 0.25)
    )

    mobile_patrol_officers = max(
        1,
        round(officers * 0.20)
    )

    control_reserve_officers = max(
        1,
        officers
        - junction_officers
        - entry_exit_officers
        - mobile_patrol_officers
    )

    entry_barricades = max(
        1,
        round(barricades * 0.40)
    )

    exit_barricades = max(
        1,
        round(barricades * 0.25)
    )

    closure_barricades = max(
        0,
        barricades
        - entry_barricades
        - exit_barricades
    )

    # Operational action
    if risk_score >= 80:
        response_time = "Immediate · within 5 minutes"
        monitoring_interval = "Every 2 minutes"

    elif risk_score >= 60:
        response_time = "Urgent · within 10 minutes"
        monitoring_interval = "Every 5 minutes"

    elif risk_score >= 35:
        response_time = "Priority · within 20 minutes"
        monitoring_interval = "Every 10 minutes"

    else:
        response_time = "Routine · within 30 minutes"
        monitoring_interval = "Every 15 minutes"

    actions = [
        "Notify the nearest traffic police station",
        "Deploy field officers at entry and exit junctions",
        "Start CCTV and traffic-speed monitoring",
        "Prepare public traffic advisory",
    ]

    if diversion:
        actions.append(
            "Activate the approved diversion corridor"
        )

    if road_closure:
        actions.append(
            "Establish temporary road-closure perimeter"
        )

    if risk_score >= 60:
        actions.append(
            "Enable manual traffic-signal supervision"
        )

    if cause in major_public_events:
        actions.append(
            "Create pedestrian and crowd-control boundary"
        )

    return {
        "model_officers": model_officers,
        "model_barricades": model_barricades,

        "officers": officers,
        "barricades": barricades,

        "road_closure": road_closure,
        "diversion": diversion,

        "command_level": command_level,
        "response_time": response_time,
        "monitoring_interval": monitoring_interval,

        "junction_officers": junction_officers,
        "entry_exit_officers": entry_exit_officers,
        "mobile_patrol_officers": mobile_patrol_officers,
        "control_reserve_officers": control_reserve_officers,

        "entry_barricades": entry_barricades,
        "exit_barricades": exit_barricades,
        "closure_barricades": closure_barricades,

        "actions": actions,
    }

def ui_html(content):
    """
    Prevents indented HTML from being rendered as a code block.
    """
    st.markdown(
        dedent(content).strip(),
        unsafe_allow_html=True
    )


def safe_number(value, default=0.0):
    try:
        value = float(value)

        if math.isnan(value) or math.isinf(value):
            return default

        return value

    except (TypeError, ValueError):
        return default


def calculate_operational_risk(
    duration,
    recommendation,
    hour,
    named_corridor
):
    """
    Operational risk index.

    This is not a model probability. It combines model duration,
    historical severity and recommended intervention requirements.
    """

    p50 = safe_number(duration.get("p50"))

    duration_component = min(
        55,
        (p50 / 180) * 55
    )

    corridor_component = 10 if named_corridor else 0

    peak_hour_component = (
        10 if 7 <= hour <= 10 or 17 <= hour <= 20 else 0
    )

    diversion_component = (
        12 if recommendation.get("recommend_diversion") else 0
    )

    closure_component = (
        8 if recommendation.get("recommend_road_closure") else 0
    )

    historical_priority_component = (
        safe_number(
            recommendation.get(
                "pct_similar_high_priority"
            )
        ) * 5
    )

    score = (
        duration_component
        + corridor_component
        + peak_hour_component
        + diversion_component
        + closure_component
        + historical_priority_component
    )

    return int(
        round(
            min(
                100,
                max(5, score)
            )
        )
    )


def get_risk_style(risk_score):
    if risk_score >= 80:
        return {
            "class": "alert-critical",
            "badge": "badge-red",
            "label": "CRITICAL CONGESTION THREAT",
            "short_label": "CRITICAL",
        }

    if risk_score >= 60:
        return {
            "class": "alert-critical",
            "badge": "badge-red",
            "label": "HIGH CONGESTION THREAT",
            "short_label": "HIGH",
        }

    if risk_score >= 35:
        return {
            "class": "alert-medium",
            "badge": "badge-yellow",
            "label": "MODERATE CONGESTION THREAT",
            "short_label": "MODERATE",
        }

    return {
        "class": "alert-low",
        "badge": "badge-green",
        "label": "LOW CONGESTION THREAT",
        "short_label": "LOW",
    }

# ===================== CSS =====================
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=JetBrains+Mono:wght@400;600;700;800&display=swap');

html, body, [class*="css"] {
    font-family: 'JetBrains Mono', monospace;
}

.stApp {
    background: #07090f;
    color: #e5e7eb;
}

.block-container {
    padding-top: 1rem;
    padding-bottom: 2rem;
}

div[data-testid="stToolbar"] {
    visibility: hidden;
}

.ops-header {
    background: #10131c;
    border: 1px solid #252a38;
    border-radius: 18px;
    padding: 20px 26px;
    margin-bottom: 18px;
    display: flex;
    justify-content: space-between;
    align-items: center;
}

.logo-box {
    display: flex;
    align-items: center;
    gap: 16px;
}

.logo-icon {
    width: 52px;
    height: 52px;
    border-radius: 14px;
    background: #241805;
    border: 1px solid #f59e0b;
    display: flex;
    align-items: center;
    justify-content: center;
    font-size: 26px;
}

.main-title {
    color: #f8fafc;
    font-size: 19px;
    font-weight: 900;
    letter-spacing: 0.02em;
}

.tag {
    background: #f59e0b;
    color: #111827;
    padding: 4px 9px;
    font-size: 10px;
    border-radius: 5px;
    margin-left: 8px;
    font-weight: 900;
}

.sub-title {
    color: #64748b;
    font-size: 11px;
    margin-top: 5px;
}

.header-stats {
    display: flex;
    gap: 34px;
    align-items: center;
}

.stat-label {
    color: #64748b;
    font-size: 10px;
    font-weight: 700;
}

.stat-value {
    color: #f8fafc;
    font-size: 13px;
    font-weight: 800;
    margin-top: 4px;
}

.online {
    color: #22c55e;
}

.ops-card {
    background: #10131c;
    border: 1px solid #252a38;
    border-radius: 16px;
    padding: 20px;
    margin-bottom: 18px;
}

.card-title {
    color: #f8fafc;
    font-size: 14px;
    font-weight: 900;
    margin-bottom: 5px;
}

.card-sub {
    color: #64748b;
    font-size: 11px;
    margin-bottom: 16px;
}

.metric-card {
    background: #0b0e15;
    border: 1px solid #252a38;
    border-radius: 14px;
    padding: 18px;
    text-align: center;
}

.metric-value {
    color: #f59e0b;
    font-size: 30px;
    font-weight: 900;
}

.metric-label {
    color: #64748b;
    font-size: 10px;
    margin-top: 6px;
}

.alert-critical {
    background: rgba(239, 68, 68, 0.14);
    border-left: 5px solid #ef4444;
    border-radius: 12px;
    padding: 17px;
}

.alert-medium {
    background: rgba(245, 158, 11, 0.14);
    border-left: 5px solid #f59e0b;
    border-radius: 12px;
    padding: 17px;
}

.alert-low {
    background: rgba(34, 197, 94, 0.14);
    border-left: 5px solid #22c55e;
    border-radius: 12px;
    padding: 17px;
}

.badge-red {
    background: #3b1114;
    color: #f87171;
    border: 1px solid #ef4444;
    padding: 5px 10px;
    border-radius: 5px;
    font-size: 10px;
    font-weight: 900;
}

.badge-yellow {
    background: #3b2a0b;
    color: #fbbf24;
    border: 1px solid #f59e0b;
    padding: 5px 10px;
    border-radius: 5px;
    font-size: 10px;
    font-weight: 900;
}

.badge-green {
    background: #0f2f1c;
    color: #4ade80;
    border: 1px solid #22c55e;
    padding: 5px 10px;
    border-radius: 5px;
    font-size: 10px;
    font-weight: 900;
}

.hot-row {
    background: #0b0e15;
    border: 1px solid #252a38;
    border-radius: 12px;
    padding: 13px;
    margin-bottom: 10px;
}

.hot-title {
    color: #f1f5f9;
    font-size: 12px;
    font-weight: 900;
}

.hot-sub {
    color: #64748b;
    font-size: 10px;
    margin-top: 5px;
}

.rationale {
    background: #0b0e15;
    border-left: 4px solid #f59e0b;
    border-radius: 10px;
    padding: 15px;
    color: #cbd5e1;
    font-size: 12px;
    line-height: 1.7;
}

button[data-baseweb="tab"] {
    font-family: 'JetBrains Mono', monospace;
    font-size: 12px;
    font-weight: 800;
}

button[data-baseweb="tab"][aria-selected="true"] {
    color: #f59e0b;
}

div[data-baseweb="tab-highlight"] {
    background-color: #f59e0b;
}

.stButton > button {
    background: #0b0e15;
    border: 1px solid #374151;
    color: #d1d5db;
    border-radius: 10px;
    font-size: 12px;
    font-weight: 800;
}

.stButton > button:hover {
    border-color: #f59e0b;
    color: #f59e0b;
}

.stDownloadButton > button {
    background: #f59e0b;
    color: #111827;
    border: none;
    border-radius: 10px;
    font-size: 12px;
    font-weight: 900;
}

div[data-testid="stMetric"] {
    background: #0b0e15;
    border: 1px solid #252a38;
    border-radius: 14px;
    padding: 15px;
}
</style>
""", unsafe_allow_html=True)
st.markdown("""
<style>
div[data-testid="stVerticalBlockBorderWrapper"] {
    background: #10131c;
    border: 1px solid #252a38 !important;
    border-radius: 16px;
    padding: 10px;
}

.forecast-summary {
    background: linear-gradient(
        135deg,
        rgba(17, 24, 39, 0.95),
        rgba(15, 23, 42, 0.95)
    );
    border: 1px solid #2f3748;
    border-radius: 14px;
    padding: 16px;
}

.result-row {
    display: flex;
    justify-content: space-between;
    align-items: center;
    background: #0b0e15;
    border: 1px solid #252a38;
    border-radius: 10px;
    padding: 12px 14px;
    margin-bottom: 8px;
}

.result-row-label {
    color: #94a3b8;
    font-size: 11px;
    text-transform: uppercase;
}

.result-row-value {
    color: #f8fafc;
    font-size: 13px;
    font-weight: 800;
}

.forecast-note {
    background: #0b0e15;
    border-left: 4px solid #38bdf8;
    border-radius: 10px;
    padding: 14px;
    color: #cbd5e1;
    font-size: 12px;
    line-height: 1.6;
}
</style>
""", unsafe_allow_html=True)


# ===================== LOAD DATA =====================
@st.cache_resource
def load_recommender():
    return IncidentRecommender()


@st.cache_resource
def load_predictor():
    return DurationPredictor()

@st.cache_resource
def load_feedback_logger():
    return PostEventLogger()


feedback_logger = load_feedback_logger() 

@st.cache_resource
def load_live_monitor():
    return LiveTrafficMonitor()


live_monitor = load_live_monitor()

@st.cache_resource
def load_scenario_simulator():
    return ScenarioSimulator()


scenario_simulator = load_scenario_simulator()

@st.cache_resource
def load_mongodb():
    return MongoStore()


try:
    mongo = load_mongodb()
    mongo_status = mongo.health_check()
except Exception as error:
    mongo = None
    mongo_status = {
        "connected": False,
        "message": str(error),
    }
@st.cache_data
def load_hotspots():
    return pd.read_csv(DATA_DIR / "hotspots.csv")


@st.cache_data
def load_events():
    return pd.read_csv(DATA_DIR / "events_clean.csv")


@st.cache_data
def load_importance():
    try:
        return pd.read_csv(
            MODEL_DIR / "feature_importance.csv"
        )
    except FileNotFoundError:
        return None


rec = load_recommender()
predictor = load_predictor()
hotspots = load_hotspots()
events = load_events()
importance = load_importance()


# ===================== HEADER =====================
st.markdown(f"""
<div class="ops-header">
    <div class="logo-box">
        <div class="logo-icon">🚦</div>
        <div>
            <div class="main-title">
                EVENT CONGESTION OPERATIONS PROTOCOL
                <span class="tag">OPS CONSOLE</span>
            </div>
            <div class="sub-title">
                MUNICIPAL TRAFFIC INTELLIGENCE & RESOURCE PLANNING SYSTEM
            </div>
        </div>
    </div>
</div>
            
""", unsafe_allow_html=True)
if mongo_status["connected"]:
    st.success("MongoDB: Connected")
else:
    st.error(
        f"MongoDB connection failed: "
        f"{mongo_status['message']}"
    )



# ===================== METRICS =====================
m1, m2, m3, m4 = st.columns(4)

with m1:
    st.markdown(f"""
    <div class="metric-card">
        <div class="metric-value">{len(hotspots)}</div>
        <div class="metric-label">HOTSPOT ZONES</div>
    </div>
    """, unsafe_allow_html=True)

with m2:
    st.markdown(f"""
    <div class="metric-card">
        <div class="metric-value">{int(hotspots["n_incidents"].sum())}</div>
        <div class="metric-label">HISTORICAL INCIDENTS</div>
    </div>
    """, unsafe_allow_html=True)

with m3:
    st.markdown(f"""
    <div class="metric-card">
        <div class="metric-value">{int(hotspots["median_duration_min"].median())}</div>
        <div class="metric-label">MEDIAN DISRUPTION MIN</div>
    </div>
    """, unsafe_allow_html=True)

with m4:
    st.markdown(f"""
    <div class="metric-card">
        <div class="metric-value">{int(hotspots["risk_score"].max())}</div>
        <div class="metric-label">MAX THREAT INDEX</div>
    </div>
    """, unsafe_allow_html=True)

st.write("")


tabs = st.tabs([
    "📍 OPERATIONS MAP",
    "🚨 INCIDENT PLANNER",
    "👮 RESOURCE DEPLOYMENT",
    "📡 LIVE MONITOR",
    "🧪 SCENARIO LAB",
    "📊 INTELLIGENCE CENTER",
    "🔁 POST-EVENT REVIEW",
])

# ===================== TAB 0 =====================
with tabs[0]:
    left, right = st.columns([2.3, 1])

    with left:
        st.markdown("""
        <div class="ops-card">
            <div class="card-title">LIVE CONGESTION RISK MAP</div>
            <div class="card-sub">
                Clustered historical hotspots. Circle size and color indicate risk score.
            </div>
        """, unsafe_allow_html=True)

        center_lat = hotspots["lat"].mean()
        center_lon = hotspots["lon"].mean()

        fmap = folium.Map(
            location=[center_lat, center_lon],
            zoom_start=11,
            tiles="CartoDB dark_matter"
        )

        for _, row in hotspots.iterrows():
            risk = float(row["risk_score"])

            if risk >= 70:
                color = "#ef4444"
            elif risk >= 40:
                color = "#f59e0b"
            else:
                color = "#22c55e"

            folium.CircleMarker(
                location=[row["lat"], row["lon"]],
                radius=4 + risk / 8,
                color=color,
                fill=True,
                fill_color=color,
                fill_opacity=0.65,
                weight=1,
                popup=folium.Popup(
                    f"""
                    <b>{row['top_corridor']}</b><br>
                    Cause: {row['top_cause']}<br>
                    Incidents: {int(row['n_incidents'])}<br>
                    Median Duration: {row['median_duration_min']:.0f} min<br>
                    Risk Score: {risk:.0f}
                    """,
                    max_width=250
                )
            ).add_to(fmap)

        st_folium(fmap, height=560, use_container_width=True, returned_objects=[])
        st.markdown("</div>", unsafe_allow_html=True)

    with right:
        st.markdown("""
        <div class="ops-card">
            <div class="card-title">TOP RISK REGISTER</div>
            <div class="card-sub">Highest operational risk corridors</div>
        """, unsafe_allow_html=True)

        top_hotspots = hotspots.sort_values("risk_score", ascending=False).head(10)

        for _, row in top_hotspots.iterrows():
            risk = row["risk_score"]
            badge = "badge-red" if risk >= 70 else "badge-yellow" if risk >= 40 else "badge-green"

            st.markdown(f"""
            <div class="hot-row">
                <div style="display:flex; justify-content:space-between;">
                    <div class="hot-title">{row['top_corridor']}</div>
                    <span class="{badge}">{risk:.0f}</span>
                </div>
                <div class="hot-sub">
                    {int(row['n_incidents'])} incidents · {row['top_cause']} · median {row['median_duration_min']:.0f} min
                </div>
            </div>
            """, unsafe_allow_html=True)

        st.markdown("</div>", unsafe_allow_html=True)


# ===================== TAB 1: INCIDENT PLANNER =====================
with tabs[1]:
    causes = [
        "vehicle_breakdown",
        "accident",
        "construction",
        "water_logging",
        "pot_holes",
        "tree_fall",
        "public_event",
        "procession",
        "vip_movement",
        "protest",
        "congestion",
        "others",
    ]

    presets = {
        "Political Rally": "protest",
        "Festival Procession": "procession",
        "Sports Event": "public_event",
        "Construction Work": "construction",
        "VIP Movement": "vip_movement",
        "Accident": "accident",
        "Water Logging": "water_logging",
        "Vehicle Breakdown": "vehicle_breakdown",
    }

    day_names = [
        "Monday",
        "Tuesday",
        "Wednesday",
        "Thursday",
        "Friday",
        "Saturday",
        "Sunday",
    ]

    month_names = [
        "January",
        "February",
        "March",
        "April",
        "May",
        "June",
        "July",
        "August",
        "September",
        "October",
        "November",
        "December",
    ]

    corridor_options = ["Non-corridor"]
    zone_options = ["unknown"]
    vehicle_options = ["unknown"]

    if not events.empty:
        if "corridor" in events.columns:
            available_corridors = (
                events["corridor"]
                .dropna()
                .astype(str)
                .str.strip()
                .unique()
                .tolist()
            )

            corridor_options = sorted(
                set(["Non-corridor"] + available_corridors)
            )

        if "zone" in events.columns:
            available_zones = (
                events["zone"]
                .dropna()
                .astype(str)
                .str.strip()
                .unique()
                .tolist()
            )

            zone_options = sorted(
                set(["unknown"] + available_zones)
            )

        if "veh_type" in events.columns:
            available_vehicles = (
                events["veh_type"]
                .dropna()
                .astype(str)
                .str.strip()
                .unique()
                .tolist()
            )

            vehicle_options = sorted(
                set(["unknown"] + available_vehicles)
            )

    input_col, result_col = st.columns(
        [1, 1.45],
        gap="large"
    )

    # =====================================================
    # INPUT PANEL
    # =====================================================
    with input_col:
        with st.container(border=True):
            st.markdown("### 🚨 Incident Classification Desk")
            st.caption(
                "Enter the planned or unplanned event details "
                "to generate the congestion forecast."
            )

            preset = st.selectbox(
                "Quick Scenario",
                list(presets.keys()),
                key="incident_scenario_preset"
            )

            default_cause = presets[preset]

            with st.form("incident_forecast_form"):
                lat_col, lon_col = st.columns(2)

                latitude = lat_col.number_input(
                    "Latitude",
                    min_value=-90.0,
                    max_value=90.0,
                    value=12.95500,
                    step=0.0001,
                    format="%.5f"
                )

                longitude = lon_col.number_input(
                    "Longitude",
                    min_value=-180.0,
                    max_value=180.0,
                    value=77.55200,
                    step=0.0001,
                    format="%.5f"
                )

                cause = st.selectbox(
                    "Incident Classification",
                    causes,
                    index=causes.index(default_cause),
                    format_func=lambda value: (
                        value.replace("_", " ").title()
                    )
                )

                event_type = st.radio(
                    "Event Type",
                    ["planned", "unplanned"],
                    horizontal=True,
                    format_func=lambda value: value.title()
                )

                time_col, day_col = st.columns(2)

                hour = time_col.slider(
                    "Hour of Day",
                    min_value=0,
                    max_value=23,
                    value=18
                )

                day_of_week = day_col.selectbox(
                    "Day of Week",
                    list(range(7)),
                    format_func=lambda value: day_names[value]
                )

                corridor = st.selectbox(
                    "Traffic Corridor",
                    corridor_options,
                    index=(
                        corridor_options.index("Mysore Road")
                        if "Mysore Road" in corridor_options
                        else 0
                    )
                )

                with st.expander(
                    "Advanced Model Inputs",
                    expanded=False
                ):
                    month = st.selectbox(
                        "Month",
                        list(range(1, 13)),
                        index=5,
                        format_func=lambda value: month_names[value - 1]
                    )

                    zone = st.selectbox(
                        "Traffic Zone",
                        zone_options
                    )

                    vehicle_type = st.selectbox(
                        "Affected Vehicle Type",
                        vehicle_options,
                        format_func=lambda value: (
                            value.replace("_", " ").title()
                        )
                    )

                    requires_road_closure = st.checkbox(
                        "Road Closure Already Required",
                        value=False
                    )

                submitted = st.form_submit_button(
                    "GENERATE AI FORECAST",
                    use_container_width=True,
                    type="primary"
                )

            if not (
                12.5 <= latitude <= 13.5
                and 77.0 <= longitude <= 78.2
            ):
                st.warning(
                    "The coordinates appear to be outside "
                    "the Bengaluru operating area."
                )

    # =====================================================
    # RUN PREDICTION
    # =====================================================
    # ===================== RUN PREDICTION =====================
    if submitted:
        named_corridor = corridor != "Non-corridor"
    
        try:
            with st.spinner(
                "Running duration models and searching similar historical incidents..."
            ):
                duration = predictor.predict(
                    lat=latitude,
                    lon=longitude,
                    event_cause=cause,
                    hour=hour,
                    dow=day_of_week,
                    month=month,
                    event_type=event_type,
                    corridor=corridor,
                    zone=zone,
                    veh_type=vehicle_type,
                    requires_road_closure=requires_road_closure,
                )
    
                recommendation = rec.recommend(
                    latitude,
                    longitude,
                    cause,
                    hour,
                    day_of_week,
                    is_named_corridor=named_corridor,
                )
    
                p10 = float(duration.get("p10", 0))
                p50 = float(duration.get("p50", 0))
                p90 = float(duration.get("p90", 0))
    
                ordered_duration = sorted([
                    max(0.0, p10),
                    max(0.0, p50),
                    max(0.0, p90),
                ])
    
                duration["p10"] = ordered_duration[0]
                duration["p50"] = ordered_duration[1]
                duration["p90"] = ordered_duration[2]
    
                p50 = duration["p50"]
    
                duration_component = min(
                    55,
                    (p50 / 180) * 55
                )
    
                peak_component = (
                    10
                    if 7 <= hour <= 10 or 17 <= hour <= 20
                    else 0
                )
    
                corridor_component = (
                    10 if named_corridor else 0
                )
    
                diversion_component = (
                    12
                    if recommendation.get(
                        "recommend_diversion",
                        False
                    )
                    else 0
                )
    
                closure_component = (
                    8
                    if recommendation.get(
                        "recommend_road_closure",
                        False
                    )
                    else 0
                )
    
                risk_score = int(
                    min(
                        100,
                        max(
                            5,
                            round(
                                duration_component
                                + peak_component
                                + corridor_component
                                + diversion_component
                                + closure_component
                            )
                        )
                    )
                )
    
                if risk_score >= 80:
                    risk_text = "CRITICAL CONGESTION THREAT"
                    risk_color = "#ef4444"
                    risk_background = "rgba(239,68,68,0.14)"
    
                elif risk_score >= 60:
                    risk_text = "HIGH CONGESTION THREAT"
                    risk_color = "#f97316"
                    risk_background = "rgba(249,115,22,0.14)"
    
                elif risk_score >= 35:
                    risk_text = "MODERATE CONGESTION THREAT"
                    risk_color = "#f59e0b"
                    risk_background = "rgba(245,158,11,0.14)"
    
                else:
                    risk_text = "LOW CONGESTION THREAT"
                    risk_color = "#22c55e"
                    risk_background = "rgba(34,197,94,0.14)"
    
                latest_result = {
                    "lat": latitude,
                    "lon": longitude,
                    "cause": cause,
                    "event_type": event_type,
                    "hour": hour,
                    "dow": day_of_week,
                    "month": month,
                    "corridor": corridor,
                    "zone": zone,
                    "vehicle_type": vehicle_type,
                    "requires_road_closure": requires_road_closure,
                    "named_corridor": named_corridor,
                    "duration": duration,
                    "result": recommendation,
                    "risk_score": risk_score,
                    "risk_text": risk_text,
                    "risk_color": risk_color,
                    "risk_background": risk_background,
                    "forecast_id": None,
                }
    
                if mongo is not None:
                    try:
                        forecast_id = mongo.save_forecast(
                            latest_result
                        )
    
                        latest_result["forecast_id"] = forecast_id
    
                        st.success(
                            f"Forecast saved to MongoDB: {forecast_id}"
                        )
    
                    except Exception as mongo_error:
                        st.warning(
                            "Forecast generated, but MongoDB save failed: "
                            f"{mongo_error}"
                        )
    
                st.session_state["latest_result"] = latest_result
            st.session_state.pop("scenario_result", None)
            st.session_state.pop("custom_scenario_plan", None)
            st.session_state.pop("resource_plan_id", None)
    
        except Exception as error:
            st.error(
                f"Forecast generation failed: {error}"
            )
    # =====================================================
    # RESULT PANEL
    # =====================================================
    with result_col:
        with st.container(border=True):
            st.markdown("### 🔮 Disruption Forecast Register")
            st.caption(
                "Predicted duration, uncertainty, operational risk "
                "and recommended resources."
            )

            latest = st.session_state.get("latest_result")

            if latest:
                duration = latest["duration"]
                result = latest["result"]

                risk_score = latest["risk_score"]
                risk_text = latest["risk_text"]
                risk_color = latest["risk_color"]
                risk_background = latest["risk_background"]

                cause_text = escape(
                    latest["cause"].replace("_", " ").title()
                )

                corridor_text = escape(
                    str(latest["corridor"])
                )

                day_text = escape(
                    day_names[latest["dow"]]
                )

                risk_text_safe = escape(
                    str(risk_text)
                )


                risk_card_html = f"""
<div style="background:{risk_background};border-left:5px solid {risk_color};border-radius:12px;padding:18px;margin-bottom:16px;">
    <div style="display:flex;justify-content:space-between;align-items:center;gap:15px;">
        <div>
            <div style="color:#f8fafc;font-size:18px;font-weight:800;">
                {risk_text_safe}
            </div>
            <div style="color:#cbd5e1;font-size:12px;margin-top:6px;">
                {cause_text} &nbsp;•&nbsp;
                {corridor_text} &nbsp;•&nbsp;
                {day_text} &nbsp;•&nbsp;
                {latest["hour"]:02d}:00
            </div>
        </div>
        <div style="color:{risk_color};border:1px solid {risk_color};border-radius:8px;padding:7px 12px;font-weight:800;white-space:nowrap;">
            INDEX {risk_score}/100
        </div>
    </div>
</div>
"""

                st.markdown(
                    risk_card_html.strip(),
                    unsafe_allow_html=True
                )

                st.progress(risk_score / 100)

                st.markdown("#### Predicted Disruption Window")

                p1, p2, p3 = st.columns(3)

                p1.metric(
                    "Optimistic · P10",
                    f"{duration['p10']:.1f} min"
                )

                p2.metric(
                    "Expected · P50",
                    f"{duration['p50']:.1f} min"
                )

                p3.metric(
                    "Severe Case · P90",
                    f"{duration['p90']:.1f} min"
                )

                confidence = duration.get(
                    "confidence",
                    "UNKNOWN"
                )

                confidence_score = duration.get(
                    "confidence_score",
                    0
                )

                uncertainty_width = duration.get(
                    "interval_width",
                    duration["p90"] - duration["p10"]
                )

                confidence_col, uncertainty_col, adjustment_col = (
                    st.columns(3)
                )

                confidence_col.metric(
                    "Forecast Confidence",
                    confidence,
                    f"{confidence_score}/100"
                )

                uncertainty_col.metric(
                    "Uncertainty Window",
                    f"{uncertainty_width:.1f} min"
                )

                adjustment_col.metric(
                    "P90 Adjustment",
                    (
                        "Applied"
                        if duration.get(
                            "operationally_capped",
                            False
                        )
                        else "Not Applied"
                    )
                )

                if confidence == "VERY LOW":
                    st.error(
                        "Forecast uncertainty is very high. "
                        "Use the severe-case window for contingency "
                        "planning and request manual review."
                    )

                elif confidence == "LOW":
                    st.warning(
                        "Forecast uncertainty is high. "
                        "Keep additional standby resources available."
                    )

                elif confidence == "MODERATE":
                    st.info(
                        "Forecast confidence is moderate. "
                        "Continue live traffic monitoring."
                    )

                else:
                    st.success(
                        "Forecast confidence is high based on "
                        "historical incident consistency."
                    )

                if duration.get("warning"):
                    st.warning(duration["warning"])

                st.markdown("#### Historical Evidence")

                h1, h2, h3, h4 = st.columns(4)

                h1.metric(
                    "Similar Cases",
                    result.get(
                        "n_similar_incidents_used",
                        0
                    )
                )

                h2.metric(
                    "Historical Median",
                    (
                        f"{float(result.get('expected_duration_min', 0)):.0f} min"
                    )
                )

                closure_rate = float(
                    result.get(
                        "pct_similar_required_closure",
                        0
                    )
                )

                high_priority_rate = float(
                    result.get(
                        "pct_similar_high_priority",
                        0
                    )
                )

                if closure_rate <= 1:
                    closure_rate *= 100

                if high_priority_rate <= 1:
                    high_priority_rate *= 100

                h3.metric(
                    "Closure History",
                    f"{closure_rate:.0f}%"
                )

                h4.metric(
                    "High Priority",
                    f"{high_priority_rate:.0f}%"
                )

                st.markdown("#### Resource Recommendation")

                r1, r2, r3, r4 = st.columns(4)

                r1.metric(
                    "Constables",
                    result.get(
                        "recommend_traffic_constables",
                        0
                    )
                )

                r2.metric(
                    "Barricades",
                    result.get(
                        "recommend_barricades",
                        0
                    )
                )

                r3.metric(
                    "Road Closure",
                    (
                        "YES"
                        if result.get(
                            "recommend_road_closure",
                            False
                        )
                        else "NO"
                    )
                )

                r4.metric(
                    "Diversion",
                    (
                        "YES"
                        if result.get(
                            "recommend_diversion",
                            False
                        )
                        else "NO"
                    )
                )

                p50 = duration["p50"]

                timeline = pd.DataFrame({
                    "Stage": [
                        "Event Reported",
                        "Traffic Build-Up",
                        "Peak Disruption",
                        "Control Measures",
                        "Recovery Phase",
                    ],
                    "Minute": [
                        0,
                        round(p50 * 0.20),
                        round(p50 * 0.45),
                        round(p50 * 0.65),
                        round(p50),
                    ]
                })

                st.markdown("#### Projected Operational Timeline")

                timeline_chart = px.line(
                    timeline,
                    x="Minute",
                    y="Stage",
                    markers=True,
                    text="Minute",
                    template="plotly_dark"
                )

                timeline_chart.update_traces(
                    line=dict(
                        width=3,
                        color="#f59e0b"
                    ),
                    marker=dict(
                        size=10,
                        color="#38bdf8"
                    ),
                    texttemplate="T+%{text} min",
                    textposition="top center"
                )

                timeline_chart.update_layout(
                    height=310,
                    showlegend=False,
                    paper_bgcolor="#10131c",
                    plot_bgcolor="#10131c",
                    margin=dict(
                        l=10,
                        r=10,
                        t=25,
                        b=10
                    ),
                    xaxis_title="Minutes after event report",
                    yaxis_title=""
                )

                st.plotly_chart(
                    timeline_chart,
                    use_container_width=True
                )

                st.markdown("#### AI Decision Rationale")

                st.info(
                    result.get(
                        "rationale",
                        "No rationale returned."
                    )
                )

                with st.expander(
                    "Advanced Model Diagnostics"
                ):
                    st.write(
                        "Raw P10:",
                        f"{duration.get('raw_p10', 0):.1f} min"
                    )

                    st.write(
                        "Raw P50:",
                        f"{duration.get('raw_p50', 0):.1f} min"
                    )

                    st.write(
                        "Raw P90:",
                        f"{duration.get('raw_p90', 0):.1f} min"
                    )

                    st.write(
                        "Operational duration cap:",
                        (
                            f"{duration.get('duration_cap_min', 0):.1f} min"
                        )
                    )

                    st.write(
                        "Model version:",
                        duration.get(
                            "model_version",
                            "unknown"
                        )
                    )

                report = f"""
EVENT CONGESTION OPERATIONS FORECAST

INCIDENT DETAILS
Cause: {latest['cause']}
Event Type: {latest['event_type']}
Location: {latest['lat']}, {latest['lon']}
Corridor: {latest['corridor']}
Zone: {latest['zone']}
Day: {day_names[latest['dow']]}
Hour: {latest['hour']:02d}:00
Month: {month_names[latest['month'] - 1]}

OPERATIONAL RISK
Threat Level: {latest['risk_text']}
Risk Index: {latest['risk_score']}/100

DURATION FORECAST
Optimistic P10: {duration['p10']:.1f} minutes
Expected P50: {duration['p50']:.1f} minutes
Operational P90: {duration['p90']:.1f} minutes
Confidence: {duration.get('confidence', 'UNKNOWN')}
Confidence Score: {duration.get('confidence_score', 0)}/100

RESOURCE RECOMMENDATION
Constables: {result.get('recommend_traffic_constables', 0)}
Barricades: {result.get('recommend_barricades', 0)}
Road Closure: {"YES" if result.get('recommend_road_closure') else "NO"}
Diversion: {"YES" if result.get('recommend_diversion') else "NO"}

RATIONALE
{result.get('rationale', '')}
"""

                st.download_button(
                    "DOWNLOAD FORECAST REPORT",
                    data=report,
                    file_name="event_congestion_forecast.txt",
                    mime="text/plain",
                    use_container_width=True
                )

            else:
                st.info(
                    "Enter event details and click "
                    "GENERATE AI FORECAST."
                )
# ===================== TAB 2: RESOURCE DEPLOYMENT =====================
with tabs[2]:
    latest = st.session_state.get("latest_result")

    if not latest:
        st.info(
            "Generate a forecast in Incident Planner first."
        )

    else:
        result = latest["result"]
        duration = latest["duration"]

        # Your existing resource calculation
        resource_plan = build_resource_plan(latest)

        # ================= SAVE RESOURCE PLAN TO MONGODB =================
        forecast_id = latest.get("forecast_id")

        if forecast_id:
            st.caption(f"Linked forecast ID: {forecast_id}")
        else:
            st.warning(
                "This forecast is not saved in MongoDB. "
                "Generate a new forecast first."
            )

        save_col, status_col = st.columns(2)

        with save_col:
            save_resource_clicked = st.button(
                "SAVE RESOURCE PLAN TO DATABASE",
                use_container_width=True,
                type="primary",
                key="save_resource_plan_button"
            )

        with status_col:
            if mongo is not None:
                st.success("MongoDB Connected")
            else:
                st.error("MongoDB Offline")

        if save_resource_clicked:
            if mongo is None:
                st.error(
                    "MongoDB is not connected. Check your .env file."
                )

            elif not forecast_id:
                st.warning(
                    "Forecast ID is missing. Generate a new forecast."
                )

            else:
                try:
                    resource_plan_id = mongo.save_resource_plan(
                        forecast_id=forecast_id,
                        resource_plan=resource_plan,
                    )

                    st.session_state[
                        "resource_plan_id"
                    ] = resource_plan_id

                    st.success(
                        f"Resource plan saved: {resource_plan_id}"
                    )

                except Exception as resource_error:
                    st.error(
                        f"Resource plan save failed: {resource_error}"
                    )

        saved_resource_plan_id = st.session_state.get(
            "resource_plan_id"
        )

        if saved_resource_plan_id:
            st.caption(
                f"Saved database record: "
                f"{saved_resource_plan_id}"
            )

        # Continue your existing Resource Deployment UI below
        risk_score = latest["risk_score"]

        st.markdown("### Resource Deployment Register")

        r1, r2, r3, r4 = st.columns(4)

        r1.metric(
            "Traffic Officers",
            resource_plan["officers"]
        )

        r2.metric(
            "Barricades",
            resource_plan["barricades"]
        )

        r3.metric(
            "Road Closure",
            (
                "REQUIRED"
                if resource_plan["road_closure"]
                else "NOT REQUIRED"
            )
        )

        r4.metric(
            "Diversion",
            (
                "ACTIVATE"
                if resource_plan["diversion"]
                else "STANDBY"
            )
        )

        risk_score = latest["risk_score"]
        cause_label = (
            latest["cause"]
            .replace("_", " ")
            .title()
        )

        day_names_resource = [
            "Monday",
            "Tuesday",
            "Wednesday",
            "Thursday",
            "Friday",
            "Saturday",
            "Sunday",
        ]

        if risk_score >= 80:
            status_color = "#ef4444"
            status_background = "rgba(239,68,68,0.14)"

        elif risk_score >= 60:
            status_color = "#f97316"
            status_background = "rgba(249,115,22,0.14)"

        elif risk_score >= 35:
            status_color = "#f59e0b"
            status_background = "rgba(245,158,11,0.14)"

        else:
            status_color = "#22c55e"
            status_background = "rgba(34,197,94,0.14)"

        status_html = (
            f'<div style="background:{status_background};'
            f'border-left:5px solid {status_color};'
            f'border-radius:12px;padding:18px;margin-bottom:18px;">'
            f'<div style="display:flex;justify-content:space-between;'
            f'align-items:center;gap:16px;">'
            f'<div>'
            f'<div style="color:#f8fafc;font-size:18px;font-weight:800;">'
            f'{resource_plan["command_level"]}'
            f'</div>'
            f'<div style="color:#cbd5e1;font-size:12px;margin-top:6px;">'
            f'{cause_label} &nbsp;•&nbsp; '
            f'{latest["corridor"]} &nbsp;•&nbsp; '
            f'{day_names_resource[latest["dow"]]} &nbsp;•&nbsp; '
            f'{latest["hour"]:02d}:00'
            f'</div>'
            f'</div>'
            f'<div style="color:{status_color};'
            f'border:1px solid {status_color};'
            f'border-radius:8px;padding:7px 12px;'
            f'font-weight:800;white-space:nowrap;">'
            f'RISK {risk_score}/100'
            f'</div>'
            f'</div>'
            f'</div>'
        )

        st.markdown(
            status_html,
            unsafe_allow_html=True
        )

        # =================================================
        # MAIN RESOURCE METRICS
        # =================================================
        with st.container(border=True):
            st.markdown("### Resource Deployment Register")

            st.caption(
                "Operationally adjusted manpower and barricade plan. "
                "The historical recommender output is used as the baseline."
            )

            r1, r2, r3, r4 = st.columns(4)

            r1.metric(
                "Traffic Officers",
                resource_plan["officers"],
                delta=(
                    resource_plan["officers"]
                    - resource_plan["model_officers"]
                ),
                help=(
                    "Adjusted deployment after applying minimum "
                    "operational safety requirements."
                )
            )

            r2.metric(
                "Barricades",
                resource_plan["barricades"],
                delta=(
                    resource_plan["barricades"]
                    - resource_plan["model_barricades"]
                )
            )

            r3.metric(
                "Road Closure",
                (
                    "REQUIRED"
                    if resource_plan["road_closure"]
                    else "NOT REQUIRED"
                )
            )

            r4.metric(
                "Diversion",
                (
                    "ACTIVATE"
                    if resource_plan["diversion"]
                    else "STANDBY"
                )
            )

            st.caption(
                f"Model baseline: "
                f"{resource_plan['model_officers']} officers and "
                f"{resource_plan['model_barricades']} barricades."
            )

        # =================================================
        # DEPLOYMENT DISTRIBUTION
        # =================================================
        left, right = st.columns(
            [1.2, 1],
            gap="large"
        )

        with left:
            with st.container(border=True):
                st.markdown("### Officer Deployment Distribution")

                officer_data = pd.DataFrame({
                    "Deployment Zone": [
                        "Major Junctions",
                        "Entry and Exit Points",
                        "Mobile Patrol",
                        "Control and Reserve",
                    ],
                    "Officers": [
                        resource_plan["junction_officers"],
                        resource_plan["entry_exit_officers"],
                        resource_plan["mobile_patrol_officers"],
                        resource_plan["control_reserve_officers"],
                    ],
                })

                officer_chart = px.bar(
                    officer_data,
                    x="Deployment Zone",
                    y="Officers",
                    text="Officers",
                    template="plotly_dark",
                )

                officer_chart.update_traces(
                    marker_color="#f59e0b",
                    textposition="outside"
                )

                officer_chart.update_layout(
                    height=340,
                    showlegend=False,
                    paper_bgcolor="#10131c",
                    plot_bgcolor="#10131c",
                    margin=dict(
                        l=10,
                        r=10,
                        t=20,
                        b=10
                    ),
                    xaxis_title="",
                    yaxis_title="Number of officers"
                )

                st.plotly_chart(
                    officer_chart,
                    use_container_width=True
                )

                officer_table = officer_data.copy()

                officer_table["Primary Duty"] = [
                    "Signal control and queue management",
                    "Vehicle screening and movement control",
                    "Dynamic congestion and route monitoring",
                    "Command support and emergency replacement",
                ]

                st.dataframe(
                    officer_table,
                    use_container_width=True,
                    hide_index=True
                )

        with right:
            with st.container(border=True):
                st.markdown("### Barricade Distribution")

                b1, b2, b3 = st.columns(3)

                b1.metric(
                    "Entry Points",
                    resource_plan["entry_barricades"]
                )

                b2.metric(
                    "Exit Points",
                    resource_plan["exit_barricades"]
                )

                b3.metric(
                    "Closure / Buffer",
                    resource_plan["closure_barricades"]
                )

                barricade_data = pd.DataFrame({
                    "Location": [
                        "Entry Points",
                        "Exit Points",
                        "Closure and Buffer Zone",
                    ],
                    "Barricades": [
                        resource_plan["entry_barricades"],
                        resource_plan["exit_barricades"],
                        resource_plan["closure_barricades"],
                    ],
                })

                barricade_chart = px.pie(
                    barricade_data,
                    names="Location",
                    values="Barricades",
                    hole=0.58,
                    template="plotly_dark"
                )

                barricade_chart.update_traces(
                    textinfo="label+value"
                )

                barricade_chart.update_layout(
                    height=300,
                    showlegend=False,
                    paper_bgcolor="#10131c",
                    margin=dict(
                        l=10,
                        r=10,
                        t=10,
                        b=10
                    )
                )

                st.plotly_chart(
                    barricade_chart,
                    use_container_width=True
                )

        # =================================================
        # OPERATIONAL CONTROL
        # =================================================
        operation_col, checklist_col = st.columns(
            [1, 1.2],
            gap="large"
        )

        with operation_col:
            with st.container(border=True):
                st.markdown("### Operational Control")

                o1, o2 = st.columns(2)

                o1.metric(
                    "Response Time",
                    resource_plan["response_time"]
                )

                o2.metric(
                    "Monitoring Cycle",
                    resource_plan["monitoring_interval"]
                )

                st.markdown("#### Deployment Conditions")

                closure_message = (
                    "Temporary road closure must be established."
                    if resource_plan["road_closure"]
                    else
                    "Keep road closure teams on standby."
                )

                diversion_message = (
                    "Activate diversion and place route signs."
                    if resource_plan["diversion"]
                    else
                    "Keep the diversion route prepared but inactive."
                )

                if resource_plan["road_closure"]:
                    st.error(closure_message)
                else:
                    st.info(closure_message)

                if resource_plan["diversion"]:
                    st.warning(diversion_message)
                else:
                    st.success(diversion_message)

                if duration.get("confidence") in [
                    "LOW",
                    "VERY LOW",
                ]:
                    st.warning(
                        "Prediction confidence is low. Keep at least "
                        "20% additional manpower available as reserve."
                    )

        with checklist_col:
            with st.container(border=True):
                st.markdown("### Field Deployment Checklist")

                for index, action in enumerate(
                    resource_plan["actions"],
                    start=1
                ):
                    st.checkbox(
                        f"{index:02d} · {action}",
                        key=f"resource_action_{index}_{latest['cause']}"
                    )

        # =================================================
        # RESPONSE RATIONALE
        # =================================================
        with st.container(border=True):
            st.markdown("### AI Deployment Rationale")

            st.info(
                result.get(
                    "rationale",
                    "No historical rationale was returned."
                )
            )

            st.markdown(
                f"""
**Operational adjustment applied**

- Model baseline officers: **{resource_plan["model_officers"]}**
- Final recommended officers: **{resource_plan["officers"]}**
- Model baseline barricades: **{resource_plan["model_barricades"]}**
- Final recommended barricades: **{resource_plan["barricades"]}**
- Expected disruption: **{duration["p50"]:.1f} minutes**
- Severe planning window: **{duration["p90"]:.1f} minutes**
- Forecast confidence: **{duration.get("confidence", "UNKNOWN")}**
"""
            )

        # =================================================
        # DOWNLOAD REPORT
        # =================================================
        resource_report = f"""
EVENT CONGESTION RESOURCE DEPLOYMENT PLAN

INCIDENT DETAILS
Cause: {cause_label}
Event Type: {latest["event_type"]}
Location: {latest["lat"]}, {latest["lon"]}
Corridor: {latest["corridor"]}
Day: {day_names_resource[latest["dow"]]}
Time: {latest["hour"]:02d}:00

THREAT ASSESSMENT
Risk Index: {risk_score}/100
Command Level: {resource_plan["command_level"]}
Expected Duration: {duration["p50"]:.1f} minutes
Severe Planning Window: {duration["p90"]:.1f} minutes
Forecast Confidence: {duration.get("confidence", "UNKNOWN")}

FINAL RESOURCE PLAN
Traffic Officers: {resource_plan["officers"]}
Barricades: {resource_plan["barricades"]}
Road Closure: {"YES" if resource_plan["road_closure"] else "NO"}
Diversion: {"YES" if resource_plan["diversion"] else "NO"}

OFFICER DISTRIBUTION
Major Junctions: {resource_plan["junction_officers"]}
Entry and Exit Points: {resource_plan["entry_exit_officers"]}
Mobile Patrol: {resource_plan["mobile_patrol_officers"]}
Control and Reserve: {resource_plan["control_reserve_officers"]}

BARRICADE DISTRIBUTION
Entry Points: {resource_plan["entry_barricades"]}
Exit Points: {resource_plan["exit_barricades"]}
Closure and Buffer Zone: {resource_plan["closure_barricades"]}

OPERATIONAL CONTROL
Required Response: {resource_plan["response_time"]}
Monitoring Interval: {resource_plan["monitoring_interval"]}

AI RATIONALE
{result.get("rationale", "")}
"""

        st.download_button(
            "DOWNLOAD RESOURCE DEPLOYMENT PLAN",
            data=resource_report,
            file_name="resource_deployment_plan.txt",
            mime="text/plain",
            use_container_width=True
        )

# ===================== LIVE MONITOR TAB =====================
with tabs[3]:
    st.markdown("""
    <div class="ops-card">
        <div class="card-title">REAL-TIME TRAFFIC MONITORING CENTER</div>
        <div class="card-sub">
            Enter live traffic conditions from CCTV, sensors or field officers
            to calculate the current congestion threat level.
        </div>
    """, unsafe_allow_html=True)

    input_col, output_col = st.columns([1, 1.35])

    with input_col:
        st.markdown("""
        <div class="card-title">LIVE SENSOR INPUT</div>
        <div class="card-sub">
            Simulate values received from traffic cameras and road sensors.
        </div>
        """, unsafe_allow_html=True)

        with st.form("live_traffic_form"):
            s1, s2 = st.columns(2)

            current_speed = s1.number_input(
                "Current Speed (km/h)",
                min_value=0.0,
                max_value=150.0,
                value=18.0,
                step=1.0
            )

            normal_speed = s2.number_input(
                "Normal Speed (km/h)",
                min_value=1.0,
                max_value=150.0,
                value=50.0,
                step=1.0
            )

            vehicle_density = st.slider(
                "Vehicle Density",
                min_value=0,
                max_value=150,
                value=95,
                help="Estimated vehicles in the monitored road segment."
            )

            queue_length = st.slider(
                "Queue Length",
                min_value=0,
                max_value=2000,
                value=450,
                step=10,
                format="%d m"
            )

            c1, c2 = st.columns(2)

            rainfall = c1.number_input(
                "Rainfall (mm)",
                min_value=0.0,
                max_value=200.0,
                value=5.0,
                step=1.0
            )

            crowd_level = c2.slider(
                "Crowd Level",
                min_value=0,
                max_value=100,
                value=45
            )

            road_blocked = st.slider(
                "Road Capacity Blocked",
                min_value=0,
                max_value=100,
                value=35,
                format="%d%%"
            )

            emergency_required = st.checkbox(
                "Emergency Vehicle Access Required"
            )

            live_submit = st.form_submit_button(
                "ANALYSE LIVE CONDITIONS",
                use_container_width=True
            )

    with output_col:
        if live_submit:
            live_input = LiveTrafficInput(
                current_speed_kmph=current_speed,
                normal_speed_kmph=normal_speed,
                vehicle_density=vehicle_density,
                queue_length_m=queue_length,
                rainfall_mm=rainfall,
                crowd_level=crowd_level,
                road_blocked_percent=road_blocked,
                emergency_vehicle_required=emergency_required
            )

            live_result = live_monitor.calculate_live_risk(
                live_input
            )

            st.session_state["live_result"] = live_result

        live_result = st.session_state.get("live_result")

        if live_result:
            risk_score = live_result["risk_score"]
            risk_level = live_result["risk_level"]
            alert_code = live_result["alert_code"]

            if alert_code == "RED":
                alert_class = "alert-critical"
                badge_class = "badge-red"

            elif alert_code in ["ORANGE", "YELLOW"]:
                alert_class = "alert-medium"
                badge_class = "badge-yellow"

            else:
                alert_class = "alert-low"
                badge_class = "badge-green"

            st.markdown(f"""
            <div class="{alert_class}">
                <div style="
                    display:flex;
                    justify-content:space-between;
                    align-items:center;
                ">
                    <div>
                        <h3 style="margin:0;">
                            {risk_level} LIVE TRAFFIC THREAT
                        </h3>
                        <p style="
                            margin-top:6px;
                            margin-bottom:0;
                            color:#cbd5e1;
                        ">
                            Updated at {live_result["generated_at"]}
                        </p>
                    </div>
                    <span class="{badge_class}">
                        ALERT CODE {alert_code} - INDEX {risk_score}/100
                    </span>
            </div>
            """, unsafe_allow_html=True)

            st.write("")

            gauge = go.Figure(
                go.Indicator(
                    mode="gauge+number",
                    value=risk_score,
                    number={
                        "suffix": "/100",
                        "font": {
                            "size": 34,
                            "color": "#f8fafc"
                        }
                    },
                    title={
                        "text": "LIVE CONGESTION THREAT INDEX",
                        "font": {
                            "size": 14,
                            "color": "#94a3b8"
                        }
                    },
                    gauge={
                        "axis": {
                            "range": [0, 100],
                            "tickcolor": "#94a3b8"
                        },
                        "bar": {
                            "color": "#f59e0b"
                        },
                        "bgcolor": "#0b0e15",
                        "bordercolor": "#252a38",
                        "steps": [
                            {
                                "range": [0, 35],
                                "color": "#123322"
                            },
                            {
                                "range": [35, 60],
                                "color": "#45360d"
                            },
                            {
                                "range": [60, 80],
                                "color": "#4b2910"
                            },
                            {
                                "range": [80, 100],
                                "color": "#45171a"
                            }
                        ],
                        "threshold": {
                            "line": {
                                "color": "#ef4444",
                                "width": 4
                            },
                            "thickness": 0.8,
                            "value": 80
                        }
                    }
                )
            )

            gauge.update_layout(
                height=310,
                margin=dict(
                    l=20,
                    r=20,
                    t=60,
                    b=20
                ),
                paper_bgcolor="#10131c",
                font={
                    "family": "JetBrains Mono",
                    "color": "#e5e7eb"
                }
            )

            st.plotly_chart(
                gauge,
                use_container_width=True
            )

            lm1, lm2, lm3 = st.columns(3)

            lm1.metric(
                "Speed Reduction",
                f"{live_result['speed_reduction_percent']:.1f}%"
            )

            lm2.metric(
                "Estimated Delay",
                f"{live_result['estimated_delay_min']:.0f} min"
            )

            lm3.metric(
                "Next Review",
                f"{live_result['update_interval_min']} min"
            )

        else:
            st.info(
                "Enter live road conditions and click "
                "ANALYSE LIVE CONDITIONS."
            )

    st.markdown("</div>", unsafe_allow_html=True)

    live_result = st.session_state.get("live_result")

    if live_result:
        action_col, alert_col = st.columns(2)

        with action_col:
            st.markdown("""
            <div class="ops-card">
                <div class="card-title">
                    LIVE OPERATIONAL ACTION PLAN
                </div>
                <div class="card-sub">
                    Immediate actions recommended from current conditions.
                </div>
            """, unsafe_allow_html=True)

            for number, action in enumerate(
                live_result["recommended_action"],
                start=1
            ):
                st.markdown(
                    f"""
                    <div class="hot-row">
                        <div class="hot-title">
                            {number:02d} · {action}
                        </div>
                    </div>
                    """,
                    unsafe_allow_html=True
                )

            st.markdown("</div>", unsafe_allow_html=True)

        with alert_col:
            st.markdown("""
            <div class="ops-card">
                <div class="card-title">
                    AUTOMATED ALERT REGISTER
                </div>
                <div class="card-sub">
                    Alerts generated from live traffic indicators.
                </div>
            """, unsafe_allow_html=True)

            for alert in live_result["alerts"]:
                severity = alert["severity"]

                if severity == "CRITICAL":
                    badge = "badge-red"

                elif severity in ["HIGH", "MODERATE"]:
                    badge = "badge-yellow"

                else:
                    badge = "badge-green"

                st.markdown(
                    f"""
                    <div class="hot-row">
                        <div style="
                            display:flex;
                            justify-content:space-between;
                            gap:12px;
                            align-items:flex-start;
                        ">
                            <div class="hot-title">
                                {alert["message"]}
                            </div>

                            <span class="{badge}">
                                {severity}
                            </span>
                        </div>
                    </div>
                    """,
                    unsafe_allow_html=True
                )

            st.markdown("</div>", unsafe_allow_html=True)

        live_plan = f"""
EVENT CONGESTION LIVE MONITORING REPORT

Generated At:
{live_result["generated_at"]}

CURRENT STATUS:
Risk Level: {live_result["risk_level"]}
Risk Score: {live_result["risk_score"]}/100
Alert Code: {live_result["alert_code"]}

LIVE IMPACT:
Speed Reduction:
{live_result["speed_reduction_percent"]}%

Estimated Delay:
{live_result["estimated_delay_min"]} minutes

Recommended Review Interval:
{live_result["update_interval_min"]} minutes

OPERATIONAL ACTIONS:
"""

        for action in live_result["recommended_action"]:
            live_plan += f"\n- {action}"

        live_plan += "\n\nAUTOMATED ALERTS:"

        for alert in live_result["alerts"]:
            live_plan += (
                f"\n- [{alert['severity']}] "
                f"{alert['message']}"
            )

        st.download_button(
            "DOWNLOAD LIVE OPERATIONS REPORT",
            data=live_plan,
            file_name="live_traffic_operations_report.txt",
            mime="text/plain",
            use_container_width=True
        )

# ===================== SCENARIO LAB =====================
with tabs[4]:
    latest = st.session_state.get("latest_result")
    

    if not latest:
        with st.container(border=True):
            st.markdown("### 🧪 Traffic Response Scenario Laboratory")

            st.info(
                "Generate an event forecast in the Incident Planner "
                "before running response simulations."
            )

    else:
        duration = latest["duration"]
        recommendation = latest["result"]

        # Uses the operationally corrected resource recommendation
        resource_plan = build_resource_plan(latest)

        baseline_duration = float(
            duration.get("p50", 0)
        )

        severe_duration = float(
            duration.get("p90", baseline_duration)
        )

        baseline_risk = float(
            latest.get("risk_score", 0)
        )

        recommended_officers = int(
            resource_plan["officers"]
        )

        recommended_barricades = int(
            resource_plan["barricades"]
        )

        recommended_diversion = bool(
            resource_plan["diversion"]
        )

        recommended_closure = bool(
            resource_plan["road_closure"]
        )

        cause_label = (
            latest["cause"]
            .replace("_", " ")
            .title()
        )

        # =============================================
        # HEADER
        # =============================================
        if baseline_risk >= 80:
            status_color = "#ef4444"
            status_background = "rgba(239,68,68,0.14)"
            baseline_level = "CRITICAL"

        elif baseline_risk >= 60:
            status_color = "#f97316"
            status_background = "rgba(249,115,22,0.14)"
            baseline_level = "HIGH"

        elif baseline_risk >= 35:
            status_color = "#f59e0b"
            status_background = "rgba(245,158,11,0.14)"
            baseline_level = "MODERATE"

        else:
            status_color = "#22c55e"
            status_background = "rgba(34,197,94,0.14)"
            baseline_level = "LOW"

        scenario_header = (
            f'<div style="background:{status_background};'
            f'border-left:5px solid {status_color};'
            f'border-radius:12px;padding:18px;margin-bottom:18px;">'
            f'<div style="display:flex;justify-content:space-between;'
            f'align-items:center;gap:16px;">'
            f'<div>'
            f'<div style="color:#f8fafc;font-size:18px;font-weight:800;">'
            f'TRAFFIC RESPONSE SCENARIO LABORATORY'
            f'</div>'
            f'<div style="color:#cbd5e1;font-size:12px;margin-top:6px;">'
            f'{cause_label} &nbsp;•&nbsp; '
            f'{latest["corridor"]} &nbsp;•&nbsp; '
            f'Expected {baseline_duration:.0f} min'
            f'</div>'
            f'</div>'
            f'<div style="color:{status_color};'
            f'border:1px solid {status_color};'
            f'border-radius:8px;padding:7px 12px;'
            f'font-weight:800;white-space:nowrap;">'
            f'{baseline_level} · {baseline_risk:.0f}/100'
            f'</div>'
            f'</div>'
            f'</div>'
        )

        st.markdown(
            scenario_header,
            unsafe_allow_html=True
        )

        # =============================================
        # BASELINE METRICS
        # =============================================
        with st.container(border=True):
            st.markdown("### Baseline Forecast")

            st.caption(
                "Current forecast before applying any additional "
                "traffic-control strategy."
            )

            b1, b2, b3, b4, b5 = st.columns(5)

            b1.metric(
                "Expected Duration",
                f"{baseline_duration:.1f} min"
            )

            b2.metric(
                "Severe Window",
                f"{severe_duration:.1f} min"
            )

            b3.metric(
                "Baseline Risk",
                f"{baseline_risk:.0f}/100"
            )

            b4.metric(
                "Recommended Officers",
                recommended_officers
            )

            b5.metric(
                "Recommended Barricades",
                recommended_barricades
            )

            confidence = duration.get(
                "confidence",
                "UNKNOWN"
            )

            if confidence in ["LOW", "VERY LOW"]:
                st.warning(
                    f"Forecast confidence is {confidence}. "
                    "Scenario comparison should include additional "
                    "standby resources."
                )

        # =============================================
        # STRATEGY CONFIGURATION
        # =============================================
        configuration_col, result_col = st.columns(
            [1, 1.35],
            gap="large"
        )

        with configuration_col:
            with st.container(border=True):
                st.markdown("### Response Strategy Configuration")

                strategy = st.selectbox(
                    "Strategy Preset",
                    [
                        "Recommended Plan",
                        "Lean Deployment",
                        "Balanced Response",
                        "Aggressive Response",
                        "Emergency Control",
                        "Custom Plan",
                    ],
                    key="scenario_strategy"
                )

                if strategy == "Lean Deployment":
                    default_officers = max(
                        4,
                        round(
                            recommended_officers * 0.60
                        )
                    )

                    default_barricades = max(
                        1,
                        round(
                            recommended_barricades * 0.50
                        )
                    )

                    default_diversion = False
                    default_closure = False
                    default_signal = False
                    default_advisory = True

                elif strategy == "Balanced Response":
                    default_officers = max(
                        6,
                        round(
                            recommended_officers * 0.85
                        )
                    )

                    default_barricades = max(
                        2,
                        round(
                            recommended_barricades * 0.80
                        )
                    )

                    default_diversion = recommended_diversion
                    default_closure = False
                    default_signal = baseline_risk >= 60
                    default_advisory = True

                elif strategy == "Aggressive Response":
                    default_officers = max(
                        recommended_officers,
                        round(
                            recommended_officers * 1.30
                        )
                    )

                    default_barricades = max(
                        recommended_barricades,
                        round(
                            recommended_barricades * 1.30
                        )
                    )

                    default_diversion = True
                    default_closure = baseline_risk >= 65
                    default_signal = True
                    default_advisory = True

                elif strategy == "Emergency Control":
                    default_officers = max(
                        recommended_officers + 8,
                        round(
                            recommended_officers * 1.60
                        )
                    )

                    default_barricades = max(
                        recommended_barricades + 6,
                        round(
                            recommended_barricades * 1.60
                        )
                    )

                    default_diversion = True
                    default_closure = True
                    default_signal = True
                    default_advisory = True

                else:
                    default_officers = recommended_officers
                    default_barricades = recommended_barricades
                    default_diversion = recommended_diversion
                    default_closure = recommended_closure
                    default_signal = baseline_risk >= 60
                    default_advisory = True

                with st.form("scenario_configuration_form"):
                    c1, c2 = st.columns(2)

                    officers = c1.number_input(
                        "Traffic Officers",
                        min_value=0,
                        max_value=500,
                        value=int(default_officers),
                        step=1
                    )

                    barricades = c2.number_input(
                        "Barricades",
                        min_value=0,
                        max_value=500,
                        value=int(default_barricades),
                        step=1
                    )

                    diversion = st.checkbox(
                        "Activate Diversion Route",
                        value=default_diversion
                    )

                    road_closure = st.checkbox(
                        "Temporary Road Closure",
                        value=default_closure
                    )

                    signal_override = st.checkbox(
                        "Manual Signal Override",
                        value=default_signal
                    )

                    public_advisory = st.checkbox(
                        "Public Traffic Advisory",
                        value=default_advisory
                    )

                    standby_percentage = st.slider(
                        "Standby Resource Reserve",
                        min_value=0,
                        max_value=50,
                        value=20,
                        step=5,
                        format="%d%%"
                    )

                    simulate_button = st.form_submit_button(
                        "RUN RESPONSE SIMULATION",
                        use_container_width=True,
                        type="primary"
                    )

                reserve_officers = round(
                    officers
                    * standby_percentage
                    / 100
                )

                total_available_officers = (
                    officers + reserve_officers
                )

                st.caption(
                    f"Active officers: {officers} · "
                    f"Reserve officers: {reserve_officers} · "
                    f"Total available: {total_available_officers}"
                )

       # =====================================================
# RUN SIMULATION
# =====================================================
        if simulate_button:
            try:
                # ================= RUN SIMULATION =================
                scenario_result = scenario_simulator.simulate(
                    baseline_duration_min=baseline_duration,
                    baseline_risk_score=baseline_risk,
                    officers=int(officers),
                    barricades=int(barricades),
                    diversion=bool(diversion),
                    road_closure=bool(road_closure),
                    signal_override=bool(signal_override),
                    public_advisory=bool(public_advisory),
                )

                # ================= INPUT PLAN =================
                scenario_input = {
                    "strategy": strategy,
                    "officers": int(officers),
                    "barricades": int(barricades),
                    "diversion": bool(diversion),
                    "road_closure": bool(road_closure),
                    "signal_override": bool(signal_override),
                    "public_advisory": bool(public_advisory),
                    "standby_percentage": int(standby_percentage),
                    "reserve_officers": int(reserve_officers),
                    "total_available_officers": int(
                        total_available_officers
                    ),
                }

                scenario_result["strategy"] = strategy
                scenario_result["standby_percentage"] = int(
                    standby_percentage
                )
                scenario_result["reserve_officers"] = int(
                    reserve_officers
                )
                scenario_result["total_available_officers"] = int(
                    total_available_officers
                )
                scenario_result["scenario_id"] = None

                # ================= SAVE TO MONGODB =================
                forecast_id = latest.get("forecast_id")

                if mongo is None:
                    st.warning(
                        "Scenario generated, but MongoDB is offline."
                    )

                elif not forecast_id:
                    st.warning(
                        "Scenario generated, but forecast ID is missing. "
                        "Generate a new forecast first."
                    )

                else:
                    try:
                        scenario_id = mongo.save_scenario(
                            forecast_id=forecast_id,
                            input_plan=scenario_input,
                            simulation_result=scenario_result,
                        )

                        scenario_result["scenario_id"] = scenario_id

                        st.success(
                            f"Scenario saved to MongoDB: {scenario_id}"
                        )

                    except Exception as mongo_error:
                        st.warning(
                            "Scenario generated, but MongoDB save failed: "
                            f"{mongo_error}"
                        )

                # ================= SAVE TO SESSION =================
                st.session_state["scenario_result"] = scenario_result
                st.session_state["custom_scenario_plan"] = scenario_input

            except Exception as error:
                st.error(
                    f"Scenario simulation failed: {error}"
                )
        # =============================================
        # SIMULATION RESULT
        # =============================================
        with result_col:
            with st.container(border=True):
                st.markdown("### Simulation Output")

                scenario_result = st.session_state.get(
                    "scenario_result"
                )

                if scenario_result:
                    residual_risk = float(
                        scenario_result[
                            "estimated_risk_score"
                        ]
                    )

                    risk_level = scenario_result[
                        "risk_level"
                    ]

                    if residual_risk >= 80:
                        result_color = "#ef4444"
                        result_background = (
                            "rgba(239,68,68,0.14)"
                        )

                    elif residual_risk >= 60:
                        result_color = "#f97316"
                        result_background = (
                            "rgba(249,115,22,0.14)"
                        )

                    elif residual_risk >= 35:
                        result_color = "#f59e0b"
                        result_background = (
                            "rgba(245,158,11,0.14)"
                        )

                    else:
                        result_color = "#22c55e"
                        result_background = (
                            "rgba(34,197,94,0.14)"
                        )

                    result_html = (
                        f'<div style="background:{result_background};'
                        f'border-left:5px solid {result_color};'
                        f'border-radius:12px;padding:17px;'
                        f'margin-bottom:16px;">'
                        f'<div style="display:flex;'
                        f'justify-content:space-between;'
                        f'align-items:center;gap:15px;">'
                        f'<div>'
                        f'<div style="color:#f8fafc;'
                        f'font-size:17px;font-weight:800;">'
                        f'{risk_level} RESIDUAL RISK'
                        f'</div>'
                        f'<div style="color:#cbd5e1;'
                        f'font-size:12px;margin-top:6px;">'
                        f'{scenario_result.get("strategy", "Custom Plan")}'
                        f'</div>'
                        f'</div>'
                        f'<div style="color:{result_color};'
                        f'border:1px solid {result_color};'
                        f'border-radius:8px;padding:7px 12px;'
                        f'font-weight:800;">'
                        f'{residual_risk:.0f}/100'
                        f'</div>'
                        f'</div>'
                        f'</div>'
                    )

                    st.markdown(
                        result_html,
                        unsafe_allow_html=True
                    )

                    o1, o2, o3, o4 = st.columns(4)

                    o1.metric(
                        "New Duration",
                        (
                            f"{scenario_result['estimated_duration_min']:.1f} min"
                        ),
                        (
                            f"-{scenario_result['duration_saved_min']:.1f} min"
                        )
                    )

                    o2.metric(
                        "Residual Risk",
                        (
                            f"{scenario_result['estimated_risk_score']:.0f}/100"
                        ),
                        (
                            f"-{baseline_risk - scenario_result['estimated_risk_score']:.0f}"
                        )
                    )

                    o3.metric(
                        "Efficiency",
                        (
                            f"{scenario_result['efficiency_score']:.0f}/100"
                        )
                    )

                    o4.metric(
                        "Cost Index",
                        (
                            f"{scenario_result['resource_cost_index']:.1f}"
                        )
                    )

                    before_after = pd.DataFrame({
                        "Indicator": [
                            "Duration",
                            "Risk Score",
                        ],
                        "Before": [
                            baseline_duration,
                            baseline_risk,
                        ],
                        "After": [
                            scenario_result[
                                "estimated_duration_min"
                            ],
                            scenario_result[
                                "estimated_risk_score"
                            ],
                        ],
                    })

                    comparison_long = before_after.melt(
                        id_vars="Indicator",
                        value_vars=[
                            "Before",
                            "After",
                        ],
                        var_name="Stage",
                        value_name="Value"
                    )

                    impact_chart = px.bar(
                        comparison_long,
                        x="Indicator",
                        y="Value",
                        color="Stage",
                        barmode="group",
                        text_auto=".0f",
                        template="plotly_dark",
                        color_discrete_map={
                            "Before": "#ef4444",
                            "After": "#22c55e",
                        }
                    )

                    impact_chart.update_layout(
                        height=330,
                        paper_bgcolor="#10131c",
                        plot_bgcolor="#10131c",
                        margin=dict(
                            l=10,
                            r=10,
                            t=20,
                            b=10
                        ),
                        xaxis_title="",
                        yaxis_title="Value"
                    )

                    st.plotly_chart(
                        impact_chart,
                        use_container_width=True
                    )

                    st.markdown(
                        "#### Simulator Recommendations"
                    )

                    for index, item in enumerate(
                        scenario_result[
                            "recommendations"
                        ],
                        start=1
                    ):
                        st.markdown(
                            f"""
                            <div style="
                                background:#0b0e15;
                                border:1px solid #252a38;
                                border-radius:10px;
                                padding:12px;
                                margin-bottom:8px;
                            ">
                                <span style="
                                    color:#f59e0b;
                                    font-weight:800;
                                ">
                                    {index:02d}
                                </span>
                                <span style="
                                    color:#e5e7eb;
                                    margin-left:8px;
                                ">
                                    {item}
                                </span>
                            </div>
                            """,
                            unsafe_allow_html=True
                        )

                else:
                    st.info(
                        "Configure a response strategy and click "
                        "RUN RESPONSE SIMULATION."
                    )

        # =============================================
        # PLAN COMPARISON
        # =============================================
        custom_plan = st.session_state.get(
            "custom_scenario_plan",
            {
                "strategy": "Custom",
                "officers": recommended_officers,
                "barricades": recommended_barricades,
                "diversion": recommended_diversion,
                "road_closure": recommended_closure,
                "signal_override":
                    baseline_risk >= 60,
                "public_advisory": True,
            }
        )

        plans = [
            ScenarioPlan(
                name="Lean",
                officers=max(
                    4,
                    round(
                        recommended_officers * 0.60
                    )
                ),
                barricades=max(
                    1,
                    round(
                        recommended_barricades * 0.50
                    )
                ),
                diversion=False,
                road_closure=False,
                signal_override=False,
                public_advisory=True,
            ),

            ScenarioPlan(
                name="Balanced",
                officers=max(
                    6,
                    round(
                        recommended_officers * 0.85
                    )
                ),
                barricades=max(
                    2,
                    round(
                        recommended_barricades * 0.80
                    )
                ),
                diversion=recommended_diversion,
                road_closure=False,
                signal_override=baseline_risk >= 60,
                public_advisory=True,
            ),

            ScenarioPlan(
                name="Recommended",
                officers=recommended_officers,
                barricades=recommended_barricades,
                diversion=recommended_diversion,
                road_closure=recommended_closure,
                signal_override=baseline_risk >= 60,
                public_advisory=True,
            ),

            ScenarioPlan(
                name="Aggressive",
                officers=max(
                    recommended_officers,
                    round(
                        recommended_officers * 1.30
                    )
                ),
                barricades=max(
                    recommended_barricades,
                    round(
                        recommended_barricades * 1.30
                    )
                ),
                diversion=True,
                road_closure=baseline_risk >= 65,
                signal_override=True,
                public_advisory=True,
            ),

            ScenarioPlan(
                name="Emergency",
                officers=max(
                    recommended_officers + 8,
                    round(
                        recommended_officers * 1.60
                    )
                ),
                barricades=max(
                    recommended_barricades + 6,
                    round(
                        recommended_barricades * 1.60
                    )
                ),
                diversion=True,
                road_closure=True,
                signal_override=True,
                public_advisory=True,
            ),

            ScenarioPlan(
                name="Custom",
                officers=int(
                    custom_plan["officers"]
                ),
                barricades=int(
                    custom_plan["barricades"]
                ),
                diversion=bool(
                    custom_plan["diversion"]
                ),
                road_closure=bool(
                    custom_plan["road_closure"]
                ),
                signal_override=bool(
                    custom_plan[
                        "signal_override"
                    ]
                ),
                public_advisory=bool(
                    custom_plan[
                        "public_advisory"
                    ]
                ),
            ),
        ]

        comparison = scenario_simulator.compare_plans(
            baseline_duration_min=baseline_duration,
            baseline_risk_score=baseline_risk,
            plans=plans
        )

        # Decision score:
        # lower risk, shorter duration and lower cost are preferred
        comparison["Decision Score"] = (
            (
                100
                - comparison["Risk After Plan"]
            ) * 0.50
            +
            (
                100
                - (
                    comparison[
                        "Estimated Duration"
                    ]
                    / max(
                        baseline_duration,
                        1
                    )
                    * 100
                ).clip(
                    lower=0,
                    upper=100
                )
            ) * 0.30
            +
            comparison[
                "Efficiency Score"
            ] * 0.20
        ).round(1)

        with st.container(border=True):
            st.markdown("### Response Plan Comparison")

            st.caption(
                "Comparison of duration, residual risk, cost and "
                "efficiency across alternative strategies."
            )

            chart_data = comparison.melt(
                id_vars="Plan",
                value_vars=[
                    "Estimated Duration",
                    "Risk After Plan",
                    "Efficiency Score",
                ],
                var_name="Indicator",
                value_name="Value"
            )

            comparison_chart = px.bar(
                chart_data,
                x="Plan",
                y="Value",
                color="Indicator",
                barmode="group",
                text_auto=".0f",
                template="plotly_dark"
            )

            comparison_chart.update_layout(
                height=390,
                paper_bgcolor="#10131c",
                plot_bgcolor="#10131c",
                margin=dict(
                    l=10,
                    r=10,
                    t=20,
                    b=10
                ),
                xaxis_title="",
                yaxis_title="Value"
            )

            st.plotly_chart(
                comparison_chart,
                use_container_width=True
            )

            display_columns = [
                "Plan",
                "Officers",
                "Barricades",
                "Diversion",
                "Road Closure",
                "Estimated Duration",
                "Duration Saved",
                "Risk After Plan",
                "Risk Level",
                "Cost Index",
                "Efficiency Score",
                "Decision Score",
            ]

            st.dataframe(
                comparison[display_columns]
                .sort_values(
                    "Decision Score",
                    ascending=False
                ),
                use_container_width=True,
                hide_index=True
            )

            # Prefer plans that reduce risk below 35
            safe_plans = comparison[
                comparison[
                    "Risk After Plan"
                ] < 35
            ]

            if not safe_plans.empty:
                best_plan = safe_plans.sort_values(
                    by=[
                        "Decision Score",
                        "Cost Index",
                    ],
                    ascending=[
                        False,
                        True,
                    ]
                ).iloc[0]

            else:
                best_plan = comparison.sort_values(
                    by=[
                        "Risk After Plan",
                        "Estimated Duration",
                        "Cost Index",
                    ],
                    ascending=[
                        True,
                        True,
                        True,
                    ]
                ).iloc[0]

            st.success(
                f"Recommended strategy: {best_plan['Plan']} — "
                f"estimated duration "
                f"{best_plan['Estimated Duration']:.0f} minutes, "
                f"residual risk "
                f"{best_plan['Risk After Plan']:.0f}/100, "
                f"and efficiency score "
                f"{best_plan['Efficiency Score']:.0f}/100."
            )

            comparison_csv = comparison.to_csv(
                index=False
            )

            st.download_button(
                "DOWNLOAD SCENARIO COMPARISON",
                data=comparison_csv,
                file_name="traffic_response_scenario_comparison.csv",
                mime="text/csv",
                use_container_width=True
            )

        # =============================================
        # OPERATIONAL REPORT
        # =============================================
        scenario_result = st.session_state.get(
            "scenario_result",
            None,
        )

        custom_plan = st.session_state.get(
            "custom_scenario_plan",
            {
                "strategy": "Recommended Plan",
                "officers": recommended_officers,
                "barricades": recommended_barricades,
                "diversion": recommended_diversion,
                "road_closure": recommended_closure,
                "signal_override": baseline_risk >= 60,
                "public_advisory": True,
                "standby_percentage": 20,
                "reserve_officers": round(
                    recommended_officers * 0.20
                ),
                "total_available_officers": (
                    recommended_officers
                    + round(recommended_officers * 0.20)
                ),
            },
        )

        if scenario_result:
            scenario_id = scenario_result.get("scenario_id")

            if scenario_id:
                st.caption(
                    f"Saved scenario record: {scenario_id}"
                )
            else:
                st.caption(
                    "Scenario is available in this session but was not "
                    "saved to MongoDB."
                )

            selected_strategy = scenario_result.get(
                "strategy",
                custom_plan.get("strategy", "Custom Plan"),
            )
            active_officers = int(
                custom_plan.get("officers", recommended_officers)
            )
            active_barricades = int(
                custom_plan.get(
                    "barricades",
                    recommended_barricades,
                )
            )
            diversion_active = bool(
                custom_plan.get("diversion", False)
            )
            road_closure_active = bool(
                custom_plan.get("road_closure", False)
            )
            signal_override_active = bool(
                custom_plan.get("signal_override", False)
            )
            public_advisory_active = bool(
                custom_plan.get("public_advisory", False)
            )

            scenario_report = f"""
EVENT CONGESTION RESPONSE SCENARIO REPORT

INCIDENT DETAILS
Cause: {cause_label}
Event Type: {latest["event_type"]}
Location: {latest["lat"]}, {latest["lon"]}
Corridor: {latest["corridor"]}
Hour: {latest["hour"]:02d}:00

BASELINE FORECAST
Expected Duration: {baseline_duration:.1f} minutes
Severe Planning Window: {severe_duration:.1f} minutes
Baseline Risk: {baseline_risk:.0f}/100
Forecast Confidence: {duration.get("confidence", "UNKNOWN")}

SELECTED STRATEGY
Strategy: {selected_strategy}
Active Officers: {active_officers}
Barricades: {active_barricades}
Diversion: {"YES" if diversion_active else "NO"}
Road Closure: {"YES" if road_closure_active else "NO"}
Signal Override: {"YES" if signal_override_active else "NO"}
Public Advisory: {"YES" if public_advisory_active else "NO"}

SIMULATION OUTPUT
Estimated Duration: {float(scenario_result.get("estimated_duration_min", 0)):.1f} minutes
Duration Saved: {float(scenario_result.get("duration_saved_min", 0)):.1f} minutes
Residual Risk: {float(scenario_result.get("estimated_risk_score", 0)):.1f}/100
Risk Level: {scenario_result.get("risk_level", "UNKNOWN")}
Efficiency Score: {float(scenario_result.get("efficiency_score", 0)):.1f}/100
Resource Cost Index: {float(scenario_result.get("resource_cost_index", 0)):.1f}

SIMULATOR RECOMMENDATIONS
"""

            for item in scenario_result.get(
                "recommendations",
                [],
            ):
                scenario_report += f"\n- {item}"

            st.download_button(
                "DOWNLOAD SELECTED SCENARIO REPORT",
                data=scenario_report,
                file_name=(
                    "selected_traffic_response_scenario.txt"
                ),
                mime="text/plain",
                use_container_width=True,
                key="download_selected_scenario_report",
            )
# ===================== TAB 5 =====================
with tabs[5]:
    a1, a2 = st.columns(2)

    with a1:
        st.markdown("""
        <div class="ops-card">
            <div class="card-title">INCIDENT DISTRIBUTION BY CAUSE</div>
            <div class="card-sub">Historical incident volume grouped by cause.</div>
        """, unsafe_allow_html=True)

        cause_count = events["event_cause"].value_counts().reset_index()
        cause_count.columns = ["Cause", "Count"]

        fig1 = px.bar(
            cause_count.head(12),
            x="Count",
            y="Cause",
            orientation="h",
            template="plotly_dark"
        )

        fig1.update_layout(
            height=410,
            paper_bgcolor="#10131c",
            plot_bgcolor="#10131c",
            margin=dict(l=10, r=10, t=10, b=10)
        )

        st.plotly_chart(fig1, use_container_width=True)
        st.markdown("</div>", unsafe_allow_html=True)

    with a2:
        st.markdown("""
        <div class="ops-card">
            <div class="card-title">DISRUPTION DURATION BY CAUSE</div>
            <div class="card-sub">Spread of event duration across incident classes.</div>
        """, unsafe_allow_html=True)

        fig2 = px.box(
            events,
            x="event_cause",
            y="duration_min",
            template="plotly_dark"
        )

        fig2.update_layout(
            height=410,
            showlegend=False,
            paper_bgcolor="#10131c",
            plot_bgcolor="#10131c",
            margin=dict(l=10, r=10, t=10, b=10),
            yaxis_range=[0, 400]
        )

        fig2.update_xaxes(tickangle=45)

        st.plotly_chart(fig2, use_container_width=True)
        st.markdown("</div>", unsafe_allow_html=True)

    st.markdown("""
    <div class="ops-card">
        <div class="card-title">MODEL EXPLAINABILITY</div>
        <div class="card-sub">Feature importance generated from the trained duration model.</div>
    """, unsafe_allow_html=True)

    if importance is not None:
        fig3 = px.bar(
            importance.head(10),
            x="importance",
            y="feature",
            orientation="h",
            template="plotly_dark"
        )

        fig3.update_layout(
            height=430,
            yaxis=dict(autorange="reversed"),
            paper_bgcolor="#10131c",
            plot_bgcolor="#10131c",
            margin=dict(l=10, r=10, t=10, b=10)
        )

        st.plotly_chart(fig3, use_container_width=True)
    else:
        st.warning("feature_importance.csv not found. Run model training first.")

    st.markdown("""
    <div class="rationale">
        ⚠️ DATA LIMITATION: This system currently uses historical incident data.
        Accuracy can improve further by adding CCTV feeds, GPS speed data,
        weather data, event calendars, and police deployment logs.
    </div>
    """, unsafe_allow_html=True)

    st.markdown("</div>", unsafe_allow_html=True)
    # ===================== TAB 6 =====================
with tabs[6]:
    latest = st.session_state.get("latest_result")

    st.markdown("""
    <div class="ops-card">
        <div class="card-title">POST-EVENT LEARNING CENTER</div>
        <div class="card-sub">
            Compare predicted congestion with the actual result and save
            operational feedback for future model improvement.
        </div>
    """, unsafe_allow_html=True)

    if latest:
        duration = latest["duration"]
        recommendation = latest["result"]

        overview1, overview2, overview3, overview4 = st.columns(4)

        overview1.metric(
            "Predicted Duration",
            f"{duration['p50']:.0f} min"
        )

        overview2.metric(
            "Recommended Officers",
            recommendation["recommend_traffic_constables"]
        )

        overview3.metric(
            "Recommended Barricades",
            recommendation["recommend_barricades"]
        )

        overview4.metric(
            "Predicted Risk",
            f"{latest['risk_score']}/100"
        )

        st.write("")

        left, right = st.columns([1, 1.2])

        with left:
            st.markdown("""
            <div class="card-title">ACTUAL EVENT OUTCOME</div>
            <div class="card-sub">
                Enter the actual values observed after the event.
            </div>
            """, unsafe_allow_html=True)

            with st.form("post_event_review_form"):
                actual_duration = st.number_input(
                    "Actual Congestion Duration (minutes)",
                    min_value=0,
                    max_value=1440,
                    value=int(duration["p50"]),
                    step=5
                )

                c1, c2 = st.columns(2)

                actual_constables = c1.number_input(
                    "Actual Constables Used",
                    min_value=0,
                    max_value=500,
                    value=int(
                        recommendation[
                            "recommend_traffic_constables"
                        ]
                    )
                )

                actual_barricades = c2.number_input(
                    "Actual Barricades Used",
                    min_value=0,
                    max_value=500,
                    value=int(
                        recommendation[
                            "recommend_barricades"
                        ]
                    )
                )

                actual_road_closure = st.checkbox(
                    "Road Was Closed",
                    value=bool(
                        recommendation[
                            "recommend_road_closure"
                        ]
                    )
                )

                actual_diversion = st.checkbox(
                    "Diversion Was Activated",
                    value=bool(
                        recommendation[
                            "recommend_diversion"
                        ]
                    )
                )

                speed_reduction = st.slider(
                    "Average Traffic Speed Reduction",
                    min_value=0,
                    max_value=100,
                    value=40,
                    help="Estimated percentage reduction compared with normal traffic speed."
                )

                complaints = st.number_input(
                    "Public Complaints Received",
                    min_value=0,
                    max_value=10000,
                    value=0
                )

                notes = st.text_area(
                    "Operational Notes",
                    placeholder=(
                        "Example: Diversion was activated at 18:20. "
                        "Traffic recovered faster than expected."
                    )
                )

                review_submitted = st.form_submit_button(
                    "SAVE POST-EVENT REVIEW",
                    use_container_width=True
                )

            if review_submitted:
                saved_record = feedback_logger.save_review(
                    latest_result=latest,
                    actual_duration=actual_duration,
                    actual_constables=actual_constables,
                    actual_barricades=actual_barricades,
                    actual_road_closure=actual_road_closure,
                    actual_diversion=actual_diversion,
                    speed_reduction=speed_reduction,
                    complaints=complaints,
                    notes=notes
                )

                st.session_state["saved_post_event_review"] = saved_record

                st.success(
                    f"Post-event review saved: "
                    f"{saved_record['event_id']}"
                )

        with right:
            st.markdown("""
            <div class="card-title">PREDICTION PERFORMANCE</div>
            <div class="card-sub">
                Evaluation of forecast accuracy and operational response.
            </div>
            """, unsafe_allow_html=True)

            saved_record = st.session_state.get(
                "saved_post_event_review"
            )

            if saved_record:
                p1, p2, p3 = st.columns(3)

                p1.metric(
                    "Duration Error",
                    f"{saved_record['duration_error_min']:.0f} min"
                )

                p2.metric(
                    "Error Percentage",
                    f"{saved_record['duration_error_percent']:.1f}%"
                )

                p3.metric(
                    "Response Score",
                    f"{saved_record['response_success_score']:.0f}/100"
                )

                predicted = saved_record["predicted_p50_min"]
                actual = saved_record["actual_duration_min"]

                comparison = pd.DataFrame({
                    "Type": [
                        "Predicted Duration",
                        "Actual Duration"
                    ],
                    "Minutes": [
                        predicted,
                        actual
                    ]
                })

                comparison_chart = px.bar(
                    comparison,
                    x="Type",
                    y="Minutes",
                    text="Minutes",
                    template="plotly_dark"
                )

                comparison_chart.update_traces(
                    texttemplate="%{text:.0f} min",
                    textposition="outside"
                )

                comparison_chart.update_layout(
                    height=330,
                    showlegend=False,
                    paper_bgcolor="#10131c",
                    plot_bgcolor="#10131c",
                    margin=dict(
                        l=10,
                        r=10,
                        t=20,
                        b=10
                    )
                )

                st.plotly_chart(
                    comparison_chart,
                    use_container_width=True
                )

                score = saved_record["response_success_score"]

                if score >= 80:
                    st.success(
                        "Operational response was highly effective."
                    )
                elif score >= 60:
                    st.warning(
                        "Operational response was acceptable but can improve."
                    )
                else:
                    st.error(
                        "Response performance was below the desired level."
                    )

            else:
                st.info(
                    "Submit the actual event outcome to generate "
                    "performance evaluation."
                )

    else:
        st.warning(
            "Generate a forecast from INCIDENT PLANNER before "
            "submitting a post-event review."
        )

    st.markdown("</div>", unsafe_allow_html=True)

    st.write("")

    reviews = feedback_logger.load_reviews()
    summary = feedback_logger.summary()

    st.markdown("""
    <div class="ops-card">
        <div class="card-title">CONTINUOUS LEARNING REGISTER</div>
        <div class="card-sub">
            Historical comparison between model forecasts and actual outcomes.
        </div>
    """, unsafe_allow_html=True)

    s1, s2, s3, s4 = st.columns(4)

    s1.metric(
        "Reviewed Events",
        summary["total_reviews"]
    )

    s2.metric(
        "Average Error",
        f"{summary['mean_error_percent']:.1f}%"
    )

    s3.metric(
        "Average Success Score",
        f"{summary['mean_success_score']:.1f}/100"
    )

    s4.metric(
        "Average Actual Duration",
        f"{summary['mean_actual_duration']:.0f} min"
    )

    if not reviews.empty:
        st.write("")

        reviews["recorded_at"] = pd.to_datetime(
            reviews["recorded_at"],
            errors="coerce"
        )

        trend = reviews.sort_values("recorded_at")

        trend_chart = px.line(
            trend,
            x="recorded_at",
            y="duration_error_percent",
            markers=True,
            hover_data=[
                "event_id",
                "event_cause",
                "predicted_p50_min",
                "actual_duration_min"
            ],
            labels={
                "recorded_at": "Review Date",
                "duration_error_percent": "Prediction Error (%)"
            },
            template="plotly_dark"
        )

        trend_chart.update_layout(
            height=350,
            paper_bgcolor="#10131c",
            plot_bgcolor="#10131c",
            margin=dict(
                l=10,
                r=10,
                t=20,
                b=10
            )
        )

        st.plotly_chart(
            trend_chart,
            use_container_width=True
        )

        display_columns = [
            "event_id",
            "event_cause",
            "event_type",
            "predicted_p50_min",
            "actual_duration_min",
            "duration_error_percent",
            "response_success_score"
        ]

        st.dataframe(
            reviews[display_columns]
            .sort_index(ascending=False)
            .head(20),
            use_container_width=True,
            hide_index=True
        )

        csv_data = reviews.to_csv(index=False)

        st.download_button(
            "DOWNLOAD LEARNING DATASET",
            data=csv_data,
            file_name="post_event_feedback.csv",
            mime="text/csv",
            use_container_width=True
        )

    else:
        st.info(
            "No post-event reviews have been submitted yet."
        )

    st.markdown("</div>", unsafe_allow_html=True)
