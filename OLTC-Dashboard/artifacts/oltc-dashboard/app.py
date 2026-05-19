import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from datetime import datetime
import time

import api_client as api
from api_client import (
    generate_realtime_sample,
    get_status,
    get_param_status,
    load_data,
    filter_by_period,
    PERIOD_OPTIONS,
)

st.set_page_config(
    page_title="OLTC SCADA — Predictive Maintenance",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ═══════════════════════════════════════════════════════════════════════════════
# DESIGN SYSTEM — Power BI dark SCADA, accent OCP green
# ═══════════════════════════════════════════════════════════════════════════════
ACCENT          = "#008542"   # OCP green
ACCENT_BRIGHT   = "#00B358"
ACCENT_DIM      = "rgba(0,133,66,0.18)"
ACCENT_GLOW     = "rgba(0,133,66,0.35)"

BG              = "#0E1117"
SURFACE         = "#1A1F2E"
SURFACE_HOVER   = "#232A3D"
BORDER          = "#2D3548"
TEXT_PRIMARY    = "#E4E6EB"
TEXT_SECONDARY  = "#9CA3AF"
TEXT_MUTED      = "#6B7280"

C_OK            = "#10B981"
C_WARN          = "#F59E0B"
C_CRIT          = "#EF4444"
C_INFO          = "#3B82F6"

SEVERITY_COLORS = {"Critique": C_CRIT, "Majeur": C_WARN, "Mineur": C_INFO}
STATUS_COLORS   = {"Normal": C_OK, "Surveillance": C_INFO, "Alerte": C_WARN, "Critique": C_CRIT}

# ─── CSS injection ───────────────────────────────────────────────────────────
CSS = f"""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&display=swap');

html, body, [class*="css"] {{
    font-family: 'Segoe UI', 'Inter', system-ui, -apple-system, sans-serif !important;
}}

.stApp {{
    background: {BG};
    color: {TEXT_PRIMARY};
}}

/* ── kill ALL Streamlit transition / stale-fade effects ── */
[data-stale="true"], [data-stale="true"] * {{
    opacity: 1 !important;
    transition: none !important;
}}
[data-testid="stVerticalBlock"],
[data-testid="stHorizontalBlock"],
.element-container,
.stMarkdown {{
    transition: none !important;
    animation: none !important;
}}
div[class*="skeleton"], div[class*="Skeleton"] {{
    display: none !important;
}}

#MainMenu, footer, header[data-testid="stHeader"] {{ visibility: hidden; height: 0; }}
.stDeployButton {{ display: none !important; }}
[data-testid="stStatusWidget"],
[data-testid="stStatusWidget"] * {{ display: none !important; }}
button[kind="header"] {{ display: none !important; }}
[data-testid="stDecoration"] {{ display: none !important; }}
[data-testid="stToolbar"] {{ display: none !important; }}
[data-testid="manage-app-button"] {{ display: none !important; }}
.viewerBadge_container__r5tak {{ display: none !important; }}
.styles_viewerBadge__CvC9N {{ display: none !important; }}
#stDecoration {{ display: none !important; }}

.block-container {{
    padding-top: 1.2rem !important;
    padding-bottom: 2rem !important;
    max-width: 100% !important;
}}

/* ─── Sidebar ─── */
section[data-testid="stSidebar"] > div {{
    background: {SURFACE};
    border-right: 1px solid {BORDER};
}}
section[data-testid="stSidebar"] .stSelectbox label,
section[data-testid="stSidebar"] .stSlider label {{
    color: {TEXT_SECONDARY} !important;
    font-size: 11px !important;
    font-weight: 700;
    text-transform: uppercase;
    letter-spacing: 0.08em;
}}

/* ─── SCADA Header ─── */
.scada-header {{
    background: linear-gradient(135deg, {SURFACE} 0%, {SURFACE_HOVER} 100%);
    border: 1px solid {BORDER};
    border-left: 4px solid {ACCENT};
    padding: 16px 22px;
    border-radius: 4px;
    margin-bottom: 18px;
    display: flex;
    justify-content: space-between;
    align-items: center;
}}
.scada-title {{
    font-size: 20px;
    font-weight: 700;
    color: {TEXT_PRIMARY};
    letter-spacing: -0.01em;
    margin: 0;
}}
.scada-subtitle {{
    font-size: 12px;
    color: {TEXT_SECONDARY};
    margin-top: 3px;
    letter-spacing: 0.02em;
}}
.scada-meta {{
    display: flex;
    gap: 14px;
    align-items: center;
}}
.scada-clock {{
    font-family: 'JetBrains Mono', 'Consolas', monospace;
    font-size: 18px;
    font-weight: 700;
    color: {ACCENT_BRIGHT};
    letter-spacing: 0.05em;
    background: rgba(0,0,0,0.25);
    padding: 4px 10px;
    border-radius: 3px;
    border: 1px solid {BORDER};
}}

/* ─── LIVE indicator ─── */
.live-pill {{
    display: inline-flex;
    align-items: center;
    gap: 6px;
    padding: 4px 10px;
    border-radius: 3px;
    font-size: 10px;
    font-weight: 800;
    letter-spacing: 0.12em;
    text-transform: uppercase;
}}
.live-pill.on {{
    background: rgba(239,68,68,0.18);
    color: {C_CRIT};
    border: 1px solid {C_CRIT};
}}
.live-pill.off {{
    background: {SURFACE_HOVER};
    color: {TEXT_MUTED};
    border: 1px solid {BORDER};
}}
.live-dot {{
    width: 7px;
    height: 7px;
    border-radius: 50%;
    background: {C_CRIT};
    box-shadow: 0 0 8px {C_CRIT};
    animation: live-pulse 1.5s ease-in-out infinite;
}}
@keyframes live-pulse {{
    0%, 100% {{ opacity: 1; transform: scale(1); }}
    50% {{ opacity: 0.4; transform: scale(0.8); }}
}}

/* ─── PBI tile (custom) ─── */
.pbi-tile {{
    background: {SURFACE};
    border: 1px solid {BORDER};
    border-top: 3px solid {ACCENT};
    border-radius: 4px;
    padding: 14px 18px;
    height: 112px;
    display: flex;
    flex-direction: column;
    justify-content: space-between;
    transition: all 0.18s ease;
}}
.pbi-tile:hover {{
    border-color: {ACCENT};
    background: {SURFACE_HOVER};
}}
.pbi-tile.warn   {{ border-top-color: {C_WARN}; }}
.pbi-tile.crit   {{ border-top-color: {C_CRIT}; }}
.pbi-tile.ok     {{ border-top-color: {C_OK};   }}
.pbi-tile.info   {{ border-top-color: {C_INFO}; }}

.pbi-tile-header {{
    display: flex;
    justify-content: space-between;
    align-items: center;
}}
.pbi-tile-label {{
    font-size: 10.5px;
    font-weight: 700;
    color: {TEXT_SECONDARY};
    text-transform: uppercase;
    letter-spacing: 0.1em;
}}
.pbi-tile-icon {{
    font-size: 13px;
    color: {TEXT_MUTED};
}}
.pbi-tile-value {{
    font-size: 30px;
    font-weight: 800;
    color: {TEXT_PRIMARY};
    line-height: 1;
    margin: 2px 0;
    font-variant-numeric: tabular-nums;
}}
.pbi-tile-value.ok   {{ color: {C_OK};   }}
.pbi-tile-value.warn {{ color: {C_WARN}; }}
.pbi-tile-value.crit {{ color: {C_CRIT}; }}
.pbi-tile-delta {{
    font-size: 11.5px;
    font-weight: 600;
    color: {TEXT_MUTED};
}}
.pbi-tile-delta.up    {{ color: {C_OK};   }}
.pbi-tile-delta.down  {{ color: {C_CRIT}; }}

/* ─── Status pill ─── */
.status-pill {{
    display: inline-block;
    padding: 3px 10px;
    border-radius: 11px;
    font-size: 10px;
    font-weight: 800;
    letter-spacing: 0.08em;
    text-transform: uppercase;
}}
.status-pill.ok      {{ background: rgba(16,185,129,0.16); color: {C_OK};   border: 1px solid {C_OK};   }}
.status-pill.warn    {{ background: rgba(245,158,11,0.16); color: {C_WARN}; border: 1px solid {C_WARN}; }}
.status-pill.crit    {{ background: rgba(239,68,68,0.16);  color: {C_CRIT}; border: 1px solid {C_CRIT}; }}
.status-pill.info    {{ background: rgba(59,130,246,0.16); color: {C_INFO}; border: 1px solid {C_INFO}; }}

/* ─── Sensor card ─── */
.sensor-card {{
    background: {SURFACE};
    border: 1px solid {BORDER};
    border-radius: 4px;
    padding: 12px 14px;
    height: 100%;
    transition: all 0.18s ease;
}}
.sensor-card.warn  {{ border-left: 3px solid {C_WARN}; }}
.sensor-card.crit  {{ border-left: 3px solid {C_CRIT}; }}
.sensor-card.ok    {{ border-left: 3px solid {C_OK};   }}
.sensor-head {{
    display: flex; justify-content: space-between; align-items: center;
    margin-bottom: 6px;
}}
.sensor-name {{
    font-size: 11px;
    font-weight: 700;
    color: {TEXT_SECONDARY};
    text-transform: uppercase;
    letter-spacing: 0.06em;
}}
.sensor-value {{
    font-size: 26px;
    font-weight: 800;
    color: {TEXT_PRIMARY};
    line-height: 1;
    font-variant-numeric: tabular-nums;
}}
.sensor-unit {{
    font-size: 12px;
    color: {TEXT_MUTED};
    margin-left: 4px;
    font-weight: 500;
}}
.sensor-thr {{
    font-size: 10px;
    color: {TEXT_MUTED};
    margin-top: 4px;
    font-family: 'Consolas', monospace;
}}

/* ─── Section title ─── */
.section-title {{
    font-size: 11px;
    font-weight: 800;
    color: {TEXT_SECONDARY};
    text-transform: uppercase;
    letter-spacing: 0.14em;
    margin: 18px 0 10px 0;
    padding-bottom: 6px;
    border-bottom: 1px solid {BORDER};
    display: flex;
    align-items: center;
    gap: 8px;
}}
.section-title::before {{
    content: '';
    width: 4px; height: 14px;
    background: {ACCENT};
    border-radius: 2px;
}}

/* ─── Streamlit native overrides ─── */
[data-testid="stMetric"] {{
    background: {SURFACE};
    border: 1px solid {BORDER};
    border-top: 3px solid {ACCENT};
    border-radius: 4px;
    padding: 12px 16px;
}}
[data-testid="stMetricLabel"] {{
    color: {TEXT_SECONDARY} !important;
    font-size: 11px !important;
    font-weight: 700 !important;
    text-transform: uppercase;
    letter-spacing: 0.08em;
}}
[data-testid="stMetricValue"] {{
    color: {TEXT_PRIMARY} !important;
    font-size: 26px !important;
    font-weight: 800 !important;
    font-variant-numeric: tabular-nums;
}}
[data-testid="stMetricDelta"] {{
    font-size: 11px !important;
    font-weight: 600 !important;
}}

.stButton > button {{
    background: {ACCENT};
    color: white;
    border: 0;
    border-radius: 3px;
    font-weight: 700;
    letter-spacing: 0.04em;
    padding: 8px 14px;
    transition: all 0.15s;
}}
.stButton > button:hover {{
    background: {ACCENT_BRIGHT};
    transform: translateY(-1px);
    box-shadow: 0 4px 12px {ACCENT_GLOW};
}}

/* Selectbox + slider styling */
.stSelectbox > div > div, .stMultiSelect > div > div {{
    background: {SURFACE_HOVER};
    border: 1px solid {BORDER};
    color: {TEXT_PRIMARY};
}}
[data-baseweb="select"] > div {{ background: {SURFACE_HOVER} !important; }}

.stSlider [data-baseweb="slider"] [role="slider"] {{ background: {ACCENT}; }}
.stSlider [data-baseweb="slider"] > div > div > div {{ background: {ACCENT} !important; }}

/* Toggle */
[data-testid="stToggle"] label {{ color: {TEXT_PRIMARY}; }}

/* Divider */
hr {{ border-color: {BORDER}; margin: 1rem 0 !important; }}

/* Tabs */
.stTabs [data-baseweb="tab-list"] {{
    background: transparent;
    gap: 0;
    border-bottom: 1px solid {BORDER};
}}
.stTabs [data-baseweb="tab"] {{
    color: {TEXT_SECONDARY};
    background: transparent;
    font-weight: 700;
    font-size: 12px;
    text-transform: uppercase;
    letter-spacing: 0.06em;
    padding: 10px 16px;
}}
.stTabs [aria-selected="true"] {{
    color: {ACCENT_BRIGHT} !important;
    border-bottom: 2px solid {ACCENT} !important;
}}

/* Dataframe */
[data-testid="stDataFrame"] {{
    background: {SURFACE};
    border: 1px solid {BORDER};
    border-radius: 4px;
}}

/* Alert log entries */
.alert-row {{
    padding: 8px 10px;
    border-radius: 3px;
    margin-bottom: 6px;
    font-size: 12px;
    border-left: 3px solid {TEXT_MUTED};
    background: {SURFACE_HOVER};
}}
.alert-row.crit    {{ border-left-color: {C_CRIT}; background: rgba(239,68,68,0.08); }}
.alert-row.warn    {{ border-left-color: {C_WARN}; background: rgba(245,158,11,0.08); }}
.alert-row.info    {{ border-left-color: {C_INFO}; background: rgba(59,130,246,0.06); }}
.alert-time {{
    font-family: 'Consolas', monospace;
    color: {TEXT_MUTED};
    font-size: 10.5px;
    font-weight: 700;
}}
.alert-lvl {{
    font-weight: 800;
    font-size: 10px;
    letter-spacing: 0.08em;
    margin: 0 6px;
}}
.alert-msg {{ color: {TEXT_PRIMARY}; }}

/* Expander */
.streamlit-expanderHeader {{ background: {SURFACE}; border: 1px solid {BORDER}; }}

/* Progress bar */
.stProgress > div > div > div > div {{ background-color: {ACCENT} !important; }}

/* Captions */
.stCaption, [data-testid="stCaptionContainer"] {{ color: {TEXT_MUTED} !important; }}

/* Plotly tooltips */
.modebar {{ display: none !important; }}
</style>
"""

# ─── Plotly dark template ─────────────────────────────────────────────────────
def dark_layout(**overrides):
    base = dict(
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font=dict(family="Segoe UI, Inter, system-ui", color=TEXT_PRIMARY, size=11),
        xaxis=dict(
            gridcolor="rgba(45,53,72,0.4)",
            zerolinecolor=BORDER,
            tickcolor=TEXT_MUTED,
            linecolor=BORDER,
            tickfont=dict(color=TEXT_SECONDARY, size=10),
        ),
        yaxis=dict(
            gridcolor="rgba(45,53,72,0.4)",
            zerolinecolor=BORDER,
            tickcolor=TEXT_MUTED,
            linecolor=BORDER,
            tickfont=dict(color=TEXT_SECONDARY, size=10),
        ),
        margin=dict(l=10, r=20, t=40, b=20),
        hoverlabel=dict(bgcolor=SURFACE, bordercolor=ACCENT, font=dict(color=TEXT_PRIMARY, size=11)),
        legend=dict(font=dict(color=TEXT_SECONDARY, size=10), bgcolor="rgba(0,0,0,0)"),
    )
    base.update(overrides)
    return base


# ═══════════════════════════════════════════════════════════════════════════════
# HELPERS  (load_data / filter_by_period are provided by api_client)
# ═══════════════════════════════════════════════════════════════════════════════
def init_session():
    if "rt_buffer" not in st.session_state:
        st.session_state.rt_buffer = []
    if "alert_log" not in st.session_state:
        st.session_state.alert_log = []
    if "live_mode" not in st.session_state:
        st.session_state.live_mode = True


def add_alert(level: str, message: str):
    ts = datetime.now().strftime("%H:%M:%S")
    st.session_state.alert_log.insert(0, {"time": ts, "level": level, "message": message})
    if len(st.session_state.alert_log) > 50:
        st.session_state.alert_log = st.session_state.alert_log[:50]


def compute_fault_stats(df_full: pd.DataFrame) -> list:
    rows = []
    for fk, fv in api.FAULT_SCENARIOS.items():
        fd = pd.Timestamp(fv["date"])
        row_at_fault = df_full[df_full["timestamp"] <= fd]
        hi_at_fault = float(row_at_fault["health_index"].iloc[-1]) if not row_at_fault.empty else 0.0
        detection_days = fv["drift_weeks"] * 7
        rows.append({
            "Panne": fk,
            "Date": fv["date"],
            "Sévérité": fv["severity"],
            "MWh": f"{fv['mwh_lost']:.1f}",
            "HI panne": f"{hi_at_fault:.0f}%",
            "Avance détection": f"{detection_days}j",
            "Description": fv["description"],
        })
    return sorted(rows, key=lambda x: x["Date"], reverse=True)


def compute_hi_from_vals(vals: dict) -> float:
    weights = {"dga_h2_ppm": 0.35, "temp_huile_c": 0.20, "vib_ms2": 0.20, "t_comm_ms": 0.15, "vcc_v": 0.10}
    total = 0.0
    for param, w in weights.items():
        nom = api.NOMINAL[param]["mean"]
        warn = api.THRESHOLDS[param]["warn"]
        crit = api.THRESHOLDS[param]["critical"]
        val = vals[param]
        if param == "vcc_v":
            score = 100.0 if val >= warn else (50.0 if val >= crit else 0.0)
        else:
            if val <= nom * 1.1:
                score = 100.0
            elif val <= warn:
                score = 100 - 50 * (val - nom * 1.1) / (warn - nom * 1.1)
            elif val <= crit:
                score = 50 - 50 * (val - warn) / (crit - warn)
            else:
                score = 0.0
        total += score * w
    return max(0.0, min(100.0, total))


# ═══════════════════════════════════════════════════════════════════════════════
# HTML COMPONENTS
# ═══════════════════════════════════════════════════════════════════════════════
def tile_html(label: str, value: str, delta: str = "", tone: str = "", value_tone: str = "",
              delta_dir: str = "", icon: str = "") -> str:
    tone_cls = f" {tone}" if tone else ""
    vcls = f" {value_tone}" if value_tone else ""
    dcls = f" {delta_dir}" if delta_dir else ""
    return f"""
    <div class="pbi-tile{tone_cls}">
        <div class="pbi-tile-header">
            <span class="pbi-tile-label">{label}</span>
            <span class="pbi-tile-icon">{icon}</span>
        </div>
        <div class="pbi-tile-value{vcls}">{value}</div>
        <div class="pbi-tile-delta{dcls}">{delta}</div>
    </div>
    """


def status_pill_html(label: str, tone: str) -> str:
    return f'<span class="status-pill {tone}">{label}</span>'


def status_tone(status: str) -> str:
    return {"Normal": "ok", "Surveillance": "info", "Alerte": "warn", "Critique": "crit"}.get(status, "info")


# ═══════════════════════════════════════════════════════════════════════════════
# CHART BUILDERS — Dark SCADA Power BI
# ═══════════════════════════════════════════════════════════════════════════════
def build_hi_timeseries(df: pd.DataFrame) -> go.Figure:
    fault_list = sorted(api.FAULT_SCENARIOS.items(), key=lambda x: x[1]["date"])
    fig = go.Figure()


    # Glow effect: wider transparent line underneath
    fig.add_trace(go.Scatter(
        x=df["timestamp"], y=df["health_index"],
        mode="lines", line=dict(color=ACCENT, width=8),
        opacity=0.18, showlegend=False, hoverinfo="skip",
    ))
    fig.add_trace(go.Scatter(
        x=df["timestamp"], y=df["health_index"],
        mode="lines", name="Health Index",
        line=dict(color=ACCENT_BRIGHT, width=2),
        fill="tozeroy", fillcolor="rgba(0,133,66,0.06)",
        hovertemplate="<b>%{x|%d %b %Y · %H:%M}</b><br>HI = %{y:.1f}%<extra></extra>",
    ))

    fig.add_hline(y=60, line_dash="dash", line_color=C_WARN, line_width=1,
                  annotation_text="60% Alerte", annotation_position="right",
                  annotation_font_color=C_WARN, annotation_font_size=9)
    fig.add_hline(y=40, line_dash="dash", line_color=C_CRIT, line_width=1,
                  annotation_text="40% Critique", annotation_position="right",
                  annotation_font_color=C_CRIT, annotation_font_size=9)

    for fk, fv in fault_list:
        color = SEVERITY_COLORS.get(fv["severity"], C_CRIT)
        fig.add_vline(
            x=pd.Timestamp(fv["date"]).timestamp() * 1000,
            line_color=color, line_dash="dot", line_width=1.5,
            annotation_text=f"<b>{fk}</b>", annotation_position="top",
            annotation_font_size=10, annotation_font_color=color,
        )

    fig.update_layout(
        **dark_layout(
            title=dict(text="<b>HEALTH INDEX — Évolution</b>",
                       font=dict(size=12, color=TEXT_SECONDARY), x=0.01),
            yaxis=dict(title=dict(text="HI (%)", font=dict(color=TEXT_MUTED, size=10)),
                       range=[0, 105], gridcolor="rgba(45,53,72,0.4)",
                       tickcolor=TEXT_MUTED, linecolor=BORDER,
                       tickfont=dict(color=TEXT_SECONDARY, size=10)),
            height=340,
            hovermode="x unified",
            showlegend=False,
        )
    )
    return fig


def build_gauge(hi: float) -> go.Figure:
    status, _ = get_status(hi)
    color = STATUS_COLORS.get(status, ACCENT)
    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=hi,
        number={"suffix": "%", "font": {"size": 44, "color": color, "family": "Segoe UI"}},
        domain={"x": [0, 1], "y": [0.05, 0.95]},
        title={"text": f"<span style='color:{TEXT_SECONDARY};font-size:11px;letter-spacing:0.1em'>HEALTH INDEX</span><br>"
                       f"<span style='color:{color};font-size:13px;font-weight:700'>{status.upper()}</span>",
               "font": {"size": 13}},
        gauge={
            "axis": {"range": [0, 100], "tickwidth": 1, "tickcolor": TEXT_MUTED,
                     "tickfont": {"color": TEXT_SECONDARY, "size": 9}},
            "bar": {"color": color, "thickness": 0.28, "line": {"color": color, "width": 1}},
            "bgcolor": SURFACE_HOVER,
            "borderwidth": 1,
            "bordercolor": BORDER,
            "steps": [
                {"range": [0,  40], "color": "rgba(239,68,68,0.18)"},
                {"range": [40, 60], "color": "rgba(245,158,11,0.18)"},
                {"range": [60, 80], "color": "rgba(59,130,246,0.14)"},
                {"range": [80, 100], "color": "rgba(16,185,129,0.16)"},
            ],
            "threshold": {"line": {"color": C_CRIT, "width": 3}, "thickness": 0.78, "value": 40},
        },
    ))
    dist = max(0, hi - 40)
    fig.update_layout(
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        height=300, margin=dict(l=20, r=20, t=70, b=10),
        font=dict(family="Segoe UI"),
    )
    return fig, dist


def build_sparkline(values: list, tone: str = "ok") -> go.Figure:
    color = {"ok": C_OK, "warn": C_WARN, "crit": C_CRIT}.get(tone, ACCENT_BRIGHT)
    fig = go.Figure()
    # glow
    fig.add_trace(go.Scatter(y=values, mode="lines",
                             line=dict(color=color, width=5), opacity=0.25, hoverinfo="skip"))
    # main
    fig.add_trace(go.Scatter(y=values, mode="lines",
                             line=dict(color=color, width=1.6),
                             fill="tozeroy", fillcolor=f"rgba({int(color[1:3],16)},{int(color[3:5],16)},{int(color[5:7],16)},0.08)",
                             hoverinfo="skip"))
    fig.update_layout(
        height=42, margin=dict(l=0, r=0, t=2, b=0),
        xaxis=dict(visible=False), yaxis=dict(visible=False),
        paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
        showlegend=False,
    )
    return fig


def build_anomaly_score_chart(df: pd.DataFrame) -> go.Figure:
    normal  = df[~df["is_anomaly"]]
    anomaly = df[df["is_anomaly"]]
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=normal["timestamp"], y=normal["anomaly_score"],
        mode="markers", name="Normal",
        marker=dict(color=ACCENT, size=3, opacity=0.4),
        hovertemplate="%{x|%d %b %Y}<br>Score %{y:.3f}<extra></extra>",
    ))
    fig.add_trace(go.Scatter(
        x=anomaly["timestamp"], y=anomaly["anomaly_score"],
        mode="markers", name="Anomalie",
        marker=dict(color=C_CRIT, size=7, symbol="diamond",
                    line=dict(color="white", width=0.5)),
        hovertemplate="%{x|%d %b %Y}<br><b>Score %{y:.3f} ⚠</b><extra></extra>",
    ))
    fig.add_hline(y=0.47, line_dash="dash", line_color=C_WARN, line_width=1,
                  annotation_text="0.47 alerte", annotation_position="right",
                  annotation_font_color=C_WARN, annotation_font_size=9)
    fig.add_hline(y=0.60, line_dash="dash", line_color=C_CRIT, line_width=1,
                  annotation_text="0.60 critique", annotation_position="right",
                  annotation_font_color=C_CRIT, annotation_font_size=9)
    fig.update_layout(
        **dark_layout(
            title=dict(text="<b>ISOLATION FOREST — Score d'anomalie</b>",
                       font=dict(size=12, color=TEXT_SECONDARY), x=0.01),
            yaxis=dict(title=dict(text="Score", font=dict(color=TEXT_MUTED, size=10)),
                       gridcolor="rgba(45,53,72,0.4)",
                       tickcolor=TEXT_MUTED, linecolor=BORDER,
                       tickfont=dict(color=TEXT_SECONDARY, size=10)),
            height=300, hovermode="x unified",
            legend=dict(orientation="h", y=1.1, font=dict(color=TEXT_SECONDARY, size=10)),
        )
    )
    return fig


def build_normalized_multicapteur(df: pd.DataFrame, motor_data=None) -> go.Figure:
    fig = go.Figure()
    palette = [ACCENT_BRIGHT, C_CRIT, C_INFO, C_WARN]
    sample = df.iloc[:: max(1, len(df) // 500)]

    # 4 IoT sensors — skip vcc_v (replaced by motor current)
    sensor_params = [p for p in api.NOMINAL.keys() if p != "vcc_v"]
    for i, param in enumerate(sensor_params):
        nom = api.NOMINAL[param]["mean"]
        std = api.NOMINAL[param]["std"]
        normalized = (sample[param] - nom) / std
        fig.add_trace(go.Scatter(
            x=sample["timestamp"], y=normalized,
            mode="lines", name=f"{api.PARAM_ICONS[param]} {api.PARAM_LABELS[param]}",
            line=dict(color=palette[i % len(palette)], width=1.4),
            hovertemplate=f"{api.PARAM_LABELS[param]}: %{{y:.2f}}σ<extra></extra>",
        ))

    # Motor current z-score (replaces Tension 127 Vcc)
    if motor_data:
        history = motor_data.get("history", [])
        if history and not df.empty:
            df_m = pd.DataFrame({
                "timestamp": pd.to_datetime([h["timestamp"] for h in history]),
                "z_score":   [h["z_score"] for h in history],
            })
            t_min, t_max = df["timestamp"].min(), df["timestamp"].max()
            df_m = df_m[(df_m["timestamp"] >= t_min) & (df_m["timestamp"] <= t_max)]
            if not df_m.empty:
                fig.add_trace(go.Scatter(
                    x=df_m["timestamp"], y=df_m["z_score"],
                    mode="lines", name="⚡ Courant Moteur",
                    line=dict(color="#A78BFA", width=1.4),
                    hovertemplate="Courant Moteur: %{y:.2f}σ<extra></extra>",
                ))

    fig.add_hline(y=0, line_dash="dot", line_color=TEXT_MUTED, line_width=1)
    fig.add_hline(y=2, line_dash="dash", line_color=C_WARN, line_width=1)
    fig.add_hline(y=-2, line_dash="dash", line_color=C_WARN, line_width=1)
    fig.update_layout(
        **dark_layout(
            title=dict(text="<b>MULTI-CAPTEURS — Z-score</b>",
                       font=dict(size=12, color=TEXT_SECONDARY), x=0.01),
            yaxis=dict(title=dict(text="σ", font=dict(color=TEXT_MUTED, size=10)),
                       gridcolor="rgba(45,53,72,0.4)",
                       tickcolor=TEXT_MUTED, linecolor=BORDER,
                       tickfont=dict(color=TEXT_SECONDARY, size=10)),
            height=300, hovermode="x unified",
            legend=dict(orientation="h", y=1.12, font=dict(color=TEXT_SECONDARY, size=10)),
        )
    )
    return fig


def build_hi_distribution(df: pd.DataFrame) -> go.Figure:
    bins = [(0, 40, "Critique", C_CRIT), (40, 60, "Alerte", C_WARN),
            (60, 80, "Surveillance", C_INFO), (80, 101, "Normal", C_OK)]
    fig = go.Figure()
    for lo, hi_val, label, color in bins:
        mask = (df["health_index"] >= lo) & (df["health_index"] < hi_val)
        chunk = df.loc[mask, "health_index"]
        if not chunk.empty:
            fig.add_trace(go.Histogram(
                x=chunk, name=label, nbinsx=20,
                marker=dict(color=color, line=dict(color=BG, width=0.5)),
                opacity=0.85,
                hovertemplate=f"{label}<br>HI %{{x:.1f}}%<br>n=%{{y}}<extra></extra>",
            ))
    fig.update_layout(
        **dark_layout(
            title=dict(text="<b>DISTRIBUTION DU HEALTH INDEX</b>",
                       font=dict(size=12, color=TEXT_SECONDARY), x=0.01),
            xaxis=dict(title=dict(text="HI (%)", font=dict(color=TEXT_MUTED, size=10)),
                       gridcolor="rgba(45,53,72,0.4)",
                       tickcolor=TEXT_MUTED, linecolor=BORDER,
                       tickfont=dict(color=TEXT_SECONDARY, size=10)),
            yaxis=dict(title=dict(text="Fréquence", font=dict(color=TEXT_MUTED, size=10)),
                       gridcolor="rgba(45,53,72,0.4)",
                       tickcolor=TEXT_MUTED, linecolor=BORDER,
                       tickfont=dict(color=TEXT_SECONDARY, size=10)),
            barmode="stack", height=300,
            legend=dict(orientation="h", y=1.1, font=dict(color=TEXT_SECONDARY, size=10)),
        )
    )
    return fig


# ═══════════════════════════════════════════════════════════════════════════════
# LIVE FRAGMENT — auto-refreshes only this block
# ═══════════════════════════════════════════════════════════════════════════════
def render_kpi_row(hi_now, status_label, if_score_now, last_fault_key, last_fault_date, mwh_avoided, hi_delta):
    k1, k2, k3, k4, k5, k6 = st.columns(6)
    hi_tone = "ok" if hi_now >= 80 else ("info" if hi_now >= 60 else ("warn" if hi_now >= 40 else "crit"))
    hi_val_tone = "ok" if hi_now >= 80 else ("warn" if hi_now >= 40 else "crit")
    delta_dir = "up" if hi_delta >= 0 else "down"
    delta_arrow = "▲" if hi_delta >= 0 else "▼"

    # Availability: (8760 - total_outage_hours) / 8760 × 100
    # total_outage_hours estimated from END ÷ rated power (25 MW, 5-yr period)
    total_mwh = sum(v["mwh_lost"] for v in api.FAULT_SCENARIOS.values())
    total_outage_hours = total_mwh / 25.0
    dispo = (8760 - total_outage_hours) / 8760 * 100

    with k1:
        st.markdown(tile_html(
            "Health Index", f"{hi_now:.1f}%",
            delta=f"{delta_arrow} {abs(hi_delta):.1f}% vs moy.",
            tone=hi_tone, value_tone=hi_val_tone, delta_dir=delta_dir,
            icon="⚕"
        ), unsafe_allow_html=True)
    with k2:
        st.markdown(tile_html(
            "Pannes Totales", "6",
            delta="P1–P6 · 2017–2022",
            tone="warn", value_tone="warn", icon="⚠"
        ), unsafe_allow_html=True)
    with k3:
        st.markdown(tile_html(
            "Confirmées OLTC", "2",
            delta="P5, P6 · défaut positionnement",
            tone="crit", value_tone="crit", icon="🔴"
        ), unsafe_allow_html=True)
    with k4:
        st.markdown(tile_html(
            "Probables OLTC", "4",
            delta="P1–P4 · thermique / 87T",
            tone="warn", value_tone="warn", icon="🟠"
        ), unsafe_allow_html=True)
    with k5:
        st.markdown(tile_html(
            "END Totale (MWh)", f"{total_mwh:.2f}",
            delta="Énergie non distribuée",
            tone="crit", value_tone="crit", icon="⚡"
        ), unsafe_allow_html=True)
    with k6:
        st.markdown(tile_html(
            "Disponibilité", f"{dispo:.2f}%",
            delta="(8760 − h_arrêt) / 8760",
            tone="ok", value_tone="ok", icon="✓"
        ), unsafe_allow_html=True)


def render_sensor_grid(current_vals, spark_data, motor_data=None):
    cols = st.columns(6)
    # Display IoT sensors (skip vcc_v — replaced by motor current in last column)
    display_params = [p for p in api.NOMINAL.keys() if p != "vcc_v"]

    for i, param in enumerate(display_params):
        val = current_vals.get(param, api.NOMINAL[param]["mean"])
        unit = api.PARAM_UNITS[param]
        icon = api.PARAM_ICONS[param]
        label = api.PARAM_LABELS[param]
        ps = get_param_status(param, val)
        tone = "ok" if ps == "normal" else ("warn" if ps == "warning" else "crit")

        nom_thr   = api.THRESHOLDS[param]["nominal"]
        alarm_thr = api.THRESHOLDS[param]["alarm"]
        trip_thr  = api.THRESHOLDS[param]["trip"]
        thr_text  = f"Nom ≤{nom_thr} · Al ≤{alarm_thr} · Trip ≤{trip_thr}"
        pill_label = {"ok": "Normal", "warn": "Attention", "crit": "Danger"}[tone]

        with cols[i]:
            st.markdown(f"""
            <div class="sensor-card {tone}">
                <div class="sensor-head">
                    <span class="sensor-name">{icon} {label}</span>
                    {status_pill_html(pill_label, tone)}
                </div>
                <div>
                    <span class="sensor-value">{val:.2f}</span><span class="sensor-unit">{unit}</span>
                </div>
                <div class="sensor-thr">{thr_text}</div>
            </div>
            """, unsafe_allow_html=True)
            spark_vals = spark_data.get(param, [val])
            st.plotly_chart(build_sparkline(spark_vals, tone),
                            use_container_width=True, config={"displayModeBar": False})

    # 6th column — Courant Moteur (replaces Tension 127 Vcc)
    with cols[5]:
        if motor_data:
            history = motor_data.get("history", [])
            last_val = float(history[-1]["current_A"]) if history else float(motor_data["baseline_current_A"])
            baseline = float(motor_data["baseline_current_A"])
            alert_level = motor_data.get("alert_level", "normal")
            tone = {"normal": "ok", "warning": "warn", "critical": "crit"}.get(alert_level, "ok")
            pill_label = {"ok": "Normal", "warn": "Attention", "crit": "Danger"}[tone]
            thr_warn = baseline * 1.20
            thr_crit = baseline * 1.30
            thr_text = f"Nom <{thr_warn:.0f} · Al <{thr_crit:.0f} · Trip ≥{thr_crit:.0f}"
            spark_vals = [h["current_A"] for h in history[-50:]] if history else [last_val]
            st.markdown(f"""
            <div class="sensor-card {tone}">
                <div class="sensor-head">
                    <span class="sensor-name">⚡ Courant Moteur</span>
                    {status_pill_html(pill_label, tone)}
                </div>
                <div>
                    <span class="sensor-value">{last_val:.1f}</span><span class="sensor-unit">A</span>
                </div>
                <div class="sensor-thr">{thr_text}</div>
            </div>
            """, unsafe_allow_html=True)
            st.plotly_chart(build_sparkline(spark_vals, tone),
                            use_container_width=True, config={"displayModeBar": False})
        else:
            # Fallback: show vcc_v if motor API unavailable
            param = "vcc_v"
            val = current_vals.get(param, api.NOMINAL[param]["mean"])
            ps = get_param_status(param, val)
            tone = "ok" if ps == "normal" else ("warn" if ps == "warning" else "crit")
            pill_label = {"ok": "Normal", "warn": "Attention", "crit": "Danger"}[tone]
            nom_thr = api.THRESHOLDS[param]["nominal"]
            alarm_thr = api.THRESHOLDS[param]["alarm"]
            trip_thr = api.THRESHOLDS[param]["trip"]
            st.markdown(f"""
            <div class="sensor-card {tone}">
                <div class="sensor-head">
                    <span class="sensor-name">{api.PARAM_ICONS[param]} {api.PARAM_LABELS[param]}</span>
                    {status_pill_html(pill_label, tone)}
                </div>
                <div>
                    <span class="sensor-value">{val:.2f}</span>
                    <span class="sensor-unit">{api.PARAM_UNITS[param]}</span>
                </div>
                <div class="sensor-thr">Nom ≥{nom_thr} · Al ≥{alarm_thr} · Trip ≥{trip_thr}</div>
            </div>
            """, unsafe_allow_html=True)
            st.plotly_chart(build_sparkline(spark_data.get(param, [val]), tone),
                            use_container_width=True, config={"displayModeBar": False})


def render_alert_log():
    if not st.session_state.alert_log:
        st.markdown(f'<div style="color:{TEXT_MUTED};font-size:12px;padding:8px">Aucune alerte. Système en surveillance normale.</div>',
                    unsafe_allow_html=True)
        return
    html_parts = []
    for entry in st.session_state.alert_log[:20]:
        lvl = entry["level"]
        cls = "crit" if lvl == "CRITICAL" else ("warn" if lvl == "WARNING" else "info")
        color = {"crit": C_CRIT, "warn": C_WARN, "info": C_INFO}[cls]
        html_parts.append(
            f'<div class="alert-row {cls}">'
            f'<span class="alert-time">{entry["time"]}</span>'
            f'<span class="alert-lvl" style="color:{color}">{lvl}</span>'
            f'<span class="alert-msg">{entry["message"]}</span>'
            f'</div>'
        )
    st.markdown("".join(html_parts), unsafe_allow_html=True)


# ═══════════════════════════════════════════════════════════════════════════════
# CHART BUILDERS — Motor current & forecast
# ═══════════════════════════════════════════════════════════════════════════════

def build_motor_chart(df_m: pd.DataFrame, baseline: float, equipment: str) -> go.Figure:
    """Main motor current chart with thresholds, rolling mean and fault markers."""
    fig = go.Figure()

    # Glow effect on raw trace
    fig.add_trace(go.Scatter(
        x=df_m["timestamp"], y=df_m["current_A"],
        mode="lines", line=dict(color=C_INFO, width=7),
        opacity=0.15, showlegend=False, hoverinfo="skip",
    ))
    # Raw I_motor
    fig.add_trace(go.Scatter(
        x=df_m["timestamp"], y=df_m["current_A"],
        mode="lines", name="I_moteur (raw)",
        line=dict(color=C_INFO, width=1.6),
        hovertemplate="<b>%{x|%d %b %Y}</b><br>I = %{y:.1f} A<extra></extra>",
    ))
    # Rolling 30-day mean
    roll_mean = df_m["current_A"].rolling(30, min_periods=1).mean()
    fig.add_trace(go.Scatter(
        x=df_m["timestamp"], y=roll_mean,
        mode="lines", name="Moy. glissante 30j",
        line=dict(color=C_WARN, width=2, dash="dash"),
        hovertemplate="Moy 30j = %{y:.1f} A<extra></extra>",
    ))

    # Threshold lines
    fig.add_hline(y=baseline * 1.30, line_dash="dash", line_color=C_CRIT, line_width=1.5,
                  annotation_text=f"Critique ×1.30 ({baseline*1.30:.0f} A)",
                  annotation_position="right", annotation_font_color=C_CRIT, annotation_font_size=9)
    fig.add_hline(y=baseline * 1.20, line_dash="dash", line_color=C_WARN, line_width=1.5,
                  annotation_text=f"Alerte ×1.20 ({baseline*1.20:.0f} A)",
                  annotation_position="right", annotation_font_color=C_WARN, annotation_font_size=9)
    fig.add_hline(y=baseline, line_dash="dot", line_color=C_OK, line_width=1,
                  annotation_text=f"Baseline {baseline:.0f} A",
                  annotation_position="right", annotation_font_color=C_OK, annotation_font_size=9)

    # Fault date vertical lines
    fault_list = sorted(api.FAULT_SCENARIOS.items(), key=lambda x: x[1]["date"])
    for fk, fv in fault_list:
        color = SEVERITY_COLORS.get(fv["severity"], C_CRIT)
        fig.add_vline(
            x=pd.Timestamp(fv["date"]).timestamp() * 1000,
            line_color=color, line_dash="dot", line_width=1.2,
            annotation_text=f"<b>{fk}</b>", annotation_position="top",
            annotation_font_size=9, annotation_font_color=color,
        )

    fig.update_layout(
        **dark_layout(
            title=dict(text=f"<b>COURANT MOTEUR — {equipment} · Évolution & Seuils IEC</b>",
                       font=dict(size=12, color=TEXT_SECONDARY), x=0.01),
            yaxis=dict(title=dict(text="Courant (A)", font=dict(color=TEXT_MUTED, size=10)),
                       gridcolor="rgba(45,53,72,0.4)", tickcolor=TEXT_MUTED,
                       linecolor=BORDER, tickfont=dict(color=TEXT_SECONDARY, size=10)),
            height=380, hovermode="x unified",
            legend=dict(orientation="h", y=1.1, font=dict(color=TEXT_SECONDARY, size=10)),
        )
    )
    return fig


def build_zscore_chart(df_m: pd.DataFrame) -> go.Figure:
    """Z-score bar chart with color coding: green/orange/red bands."""
    # Sample to ~600 points to keep chart responsive
    step = max(1, len(df_m) // 600)
    df_s = df_m.iloc[::step].copy()

    colors = []
    for z in df_s["z_score"]:
        az = abs(z)
        if az >= 2.5:
            colors.append(C_CRIT)
        elif az >= 2.0:
            colors.append(C_WARN)
        else:
            colors.append(C_OK)

    fig = go.Figure()
    fig.add_trace(go.Bar(
        x=df_s["timestamp"], y=df_s["z_score"],
        marker_color=colors,
        name="Z-score",
        hovertemplate="<b>%{x|%d %b %Y}</b><br>z = %{y:.2f}<extra></extra>",
    ))
    fig.add_hline(y=2.5, line_dash="dash", line_color=C_CRIT, line_width=1,
                  annotation_text="2.5 anomalie", annotation_position="right",
                  annotation_font_color=C_CRIT, annotation_font_size=9)
    fig.add_hline(y=-2.5, line_dash="dash", line_color=C_CRIT, line_width=1)
    fig.add_hline(y=2.0, line_dash="dot", line_color=C_WARN, line_width=1)
    fig.add_hline(y=-2.0, line_dash="dot", line_color=C_WARN, line_width=1)

    fig.update_layout(
        **dark_layout(
            title=dict(text="<b>Z-SCORE PAR COMMUTATION</b>",
                       font=dict(size=12, color=TEXT_SECONDARY), x=0.01),
            yaxis=dict(title=dict(text="z", font=dict(color=TEXT_MUTED, size=10)),
                       gridcolor="rgba(45,53,72,0.4)", tickcolor=TEXT_MUTED,
                       linecolor=BORDER, tickfont=dict(color=TEXT_SECONDARY, size=10)),
            height=280, showlegend=False,
        )
    )
    return fig


def build_forecast_chart(fc: dict) -> go.Figure:
    """Prophet forecast chart with HI historical, forecast band, and threshold crossings."""
    hist = pd.DataFrame(fc["historical"])
    hist["ds"] = pd.to_datetime(hist["date"])

    fcast = pd.DataFrame(fc["forecast"])
    fcast["ds"] = pd.to_datetime(fcast["date"])
    future = fcast[fcast["is_future"]]
    past_fc = fcast[~fcast["is_future"]]

    fig = go.Figure()

    # Historical HI — glow + line
    fig.add_trace(go.Scatter(
        x=hist["ds"], y=hist["hi"],
        mode="lines", line=dict(color=ACCENT, width=7), opacity=0.15,
        showlegend=False, hoverinfo="skip",
    ))
    fig.add_trace(go.Scatter(
        x=hist["ds"], y=hist["hi"],
        mode="lines", name="HI historique",
        line=dict(color=ACCENT_BRIGHT, width=2),
        hovertemplate="<b>%{x|%d %b %Y}</b><br>HI = %{y:.1f}%<extra></extra>",
    ))

    # Prophet fitted (past)
    if not past_fc.empty:
        fig.add_trace(go.Scatter(
            x=past_fc["ds"], y=past_fc["yhat"],
            mode="lines", name="Modèle Prophet (ajusté)",
            line=dict(color=C_INFO, width=1.2, dash="dot"),
            opacity=0.6, hoverinfo="skip",
        ))

    # Confidence band for future
    if not future.empty:
        fig.add_trace(go.Scatter(
            x=pd.concat([future["ds"], future["ds"].iloc[::-1]]),
            y=pd.concat([future["yhat_upper"], future["yhat_lower"].iloc[::-1]]),
            fill="toself", fillcolor="rgba(59,130,246,0.10)",
            line=dict(color="rgba(0,0,0,0)"),
            name="IC 80%", showlegend=True, hoverinfo="skip",
        ))
        fig.add_trace(go.Scatter(
            x=future["ds"], y=future["yhat"],
            mode="lines", name="Prévision 60j",
            line=dict(color=C_INFO, width=2, dash="dash"),
            hovertemplate="<b>%{x|%d %b %Y}</b><br>HI prévu = %{y:.1f}%<extra></extra>",
        ))

    # Threshold lines
    fig.add_hline(y=60, line_dash="dash", line_color=C_WARN, line_width=1.5,
                  annotation_text="60% Alerte", annotation_position="right",
                  annotation_font_color=C_WARN, annotation_font_size=9)
    fig.add_hline(y=40, line_dash="dash", line_color=C_CRIT, line_width=1.5,
                  annotation_text="40% Critique", annotation_position="right",
                  annotation_font_color=C_CRIT, annotation_font_size=9)

    # Threshold crossing markers
    alert_date = fc.get("alert_date")
    crit_date = fc.get("critical_date")
    if alert_date:
        fig.add_vline(x=pd.Timestamp(alert_date).timestamp() * 1000,
                      line_color=C_WARN, line_dash="dash", line_width=2,
                      annotation_text=f"⚠ Alerte {alert_date}",
                      annotation_position="top left",
                      annotation_font_color=C_WARN, annotation_font_size=9)
    if crit_date:
        fig.add_vline(x=pd.Timestamp(crit_date).timestamp() * 1000,
                      line_color=C_CRIT, line_dash="dash", line_width=2,
                      annotation_text=f"🔴 Critique {crit_date}",
                      annotation_position="top left",
                      annotation_font_color=C_CRIT, annotation_font_size=9)

    fig.update_layout(
        **dark_layout(
            title=dict(text="<b>HEALTH INDEX — Historique + Prévision Prophet 60 jours</b>",
                       font=dict(size=12, color=TEXT_SECONDARY), x=0.01),
            yaxis=dict(title=dict(text="HI (%)", font=dict(color=TEXT_MUTED, size=10)),
                       range=[0, 110], gridcolor="rgba(45,53,72,0.4)",
                       tickcolor=TEXT_MUTED, linecolor=BORDER,
                       tickfont=dict(color=TEXT_SECONDARY, size=10)),
            height=400, hovermode="x unified",
            legend=dict(orientation="h", y=1.1, font=dict(color=TEXT_SECONDARY, size=10)),
        )
    )
    return fig


# ═══════════════════════════════════════════════════════════════════════════════
# TAB RENDERERS
# ═══════════════════════════════════════════════════════════════════════════════

def render_tab_parc_oltc():
    """🔧 Parc OLTC — Fleet inventory, bubble chart, arc energy formula."""
    st.markdown('<div class="section-title">Inventaire Parc OLTC — Jorf Lasfar</div>',
                unsafe_allow_html=True)

    fleet = [
        {"Poste": "PJ1 (CGE)", "Constructeur": "ALSTHOM", "Type OLTC": "JANSEN V III Y",
         "Courant (A)": 350, "Nb prises": 21, "Plage HT (kV)": "72.45 → 53.55",
         "Mise en service": "Inconnue", "Ancienneté (ans)": ">25", "Criticité": "🔴 Critique",
         "_mva": 25, "_age": 27, "_crit": "Critique"},
        {"Poste": "PJ1 (Alcatel)", "Constructeur": "JANSEN", "Type OLTC": "V III",
         "Courant (A)": 350, "Nb prises": 21, "Plage HT (kV)": "72.45 → 53.55",
         "Mise en service": "1998", "Ancienneté (ans)": "26", "Criticité": "🔴 Critique",
         "_mva": 25, "_age": 26, "_crit": "Critique"},
        {"Poste": "PJ2 / PJ10", "Constructeur": "ALSTHOM", "Type OLTC": "JANSEN V III Y",
         "Courant (A)": 200, "Nb prises": 21, "Plage HT (kV)": "72.45 → 53.55",
         "Mise en service": "Inconnue", "Ancienneté (ans)": ">20", "Criticité": "⚠️ Élevée",
         "_mva": 12.5, "_age": 22, "_crit": "Élevée"},
        {"Poste": "PJ3", "Constructeur": "Non documenté", "Type OLTC": "—",
         "Courant (A)": "200*", "Nb prises": 21, "Plage HT (kV)": "72.45 → 53.55",
         "Mise en service": "Inconnue", "Ancienneté (ans)": ">20", "Criticité": "⚠️ Élevée",
         "_mva": 18.75, "_age": 22, "_crit": "Élevée"},
        {"Poste": "PJ11", "Constructeur": "MR", "Type OLTC": "V III 350-Y-76-10",
         "Courant (A)": 268, "Nb prises": 17, "Plage HT (kV)": "66 → 54",
         "Mise en service": "2007", "Ancienneté (ans)": "17", "Criticité": "⚠️ Élevée",
         "_mva": 25, "_age": 17, "_crit": "Élevée"},
        {"Poste": "PJ10 (FT)", "Constructeur": "MR", "Type OLTC": "V III 200Y-76-16-19",
         "Courant (A)": 134, "Nb prises": 17, "Plage HT (kV)": "66 → 54",
         "Mise en service": "2007", "Ancienneté (ans)": "17", "Criticité": "🟡 Modérée",
         "_mva": 12.5, "_age": 17, "_crit": "Modérée"},
    ]

    display_cols = ["Poste", "Constructeur", "Type OLTC", "Courant (A)", "Nb prises",
                    "Plage HT (kV)", "Mise en service", "Ancienneté (ans)", "Criticité"]
    df_fleet = pd.DataFrame([{k: v for k, v in r.items() if not k.startswith("_")}
                              for r in fleet])
    st.dataframe(df_fleet[display_cols], use_container_width=True, hide_index=True)

    with st.expander("ℹ️ Pourquoi ce tableau ?"):
        st.markdown(
            "Le parc OLTC de Jorf Lasfar est hétérogène : des appareils JANSEN à 350 A côtoient "
            "des MR à 134 A. L'**énergie d'arc** par commutation croît comme **I²×t**, ce qui fait "
            "des unités 350 A une priorité absolue de surveillance."
        )

    # Bubble chart
    crit_colors = {"Critique": C_CRIT, "Élevée": C_WARN, "Modérée": "#EAB308"}
    fig = go.Figure()
    for row in fleet:
        color = crit_colors.get(row["_crit"], C_INFO)
        c_num = row["Courant (A)"] if isinstance(row["Courant (A)"], int) else 200
        arc_ratio = round((c_num / 134) ** 2, 1)
        hover = (
            f"<b>{row['Poste']}</b><br>"
            f"Constructeur : {row['Constructeur']}<br>"
            f"Type : {row['Type OLTC']}<br>"
            f"Courant : {row['Courant (A)']} A<br>"
            f"Nb prises : {row['Nb prises']}<br>"
            f"Plage HT : {row['Plage HT (kV)']} kV<br>"
            f"Mise en service : {row['Mise en service']}<br>"
            f"Ancienneté : {row['Ancienneté (ans)']} ans<br>"
            f"Puissance : {row['_mva']} MVA<br>"
            f"Énergie arc vs PJ10 : <b>{arc_ratio}×</b><extra></extra>"
        )
        fig.add_trace(go.Scatter(
            x=[row["_age"]], y=[c_num],
            mode="markers+text",
            name=row["Poste"],
            text=[row["Poste"]],
            textposition="top center",
            textfont=dict(size=9, color=TEXT_SECONDARY),
            marker=dict(size=row["_mva"] * 2.8, color=color, opacity=0.80,
                        line=dict(color="white", width=1.2)),
            hovertemplate=hover,
        ))

    fig.update_layout(
        **dark_layout(
            title=dict(
                text="<b>PARC OLTC — Ancienneté vs Courant nominal (taille ∝ MVA)</b>",
                font=dict(size=12, color=TEXT_SECONDARY), x=0.01),
            xaxis=dict(title=dict(text="Ancienneté (ans)", font=dict(color=TEXT_MUTED, size=10)),
                       range=[12, 32], gridcolor="rgba(45,53,72,0.4)",
                       tickcolor=TEXT_MUTED, linecolor=BORDER,
                       tickfont=dict(color=TEXT_SECONDARY, size=10)),
            yaxis=dict(title=dict(text="Courant nominal (A)", font=dict(color=TEXT_MUTED, size=10)),
                       range=[50, 430], gridcolor="rgba(45,53,72,0.4)",
                       tickcolor=TEXT_MUTED, linecolor=BORDER,
                       tickfont=dict(color=TEXT_SECONDARY, size=10)),
            height=430, showlegend=False,
        )
    )
    st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False})

    # Arc energy formula + callout
    st.latex(r"""
        E_{arc} \propto I^2 \times t_{arc}
        \quad \Rightarrow \quad
        \frac{E_{arc}(350\text{A})}{E_{arc}(134\text{A})}
        = \left(\frac{350}{134}\right)^2 \approx 6.8\times
    """)
    st.markdown(f"""
    <div style="background:rgba(239,68,68,0.10);border:1px solid {C_CRIT};
                border-left:4px solid {C_CRIT};border-radius:4px;
                padding:14px 18px;margin-top:10px">
        <span style="color:{TEXT_PRIMARY}">
        Un OLTC à 350 A accumule <b>6,8× plus d'énergie d'arc</b> par commutation
        qu'un OLTC à 134 A — ce qui justifie sa priorité de surveillance absolue.
        </span>
    </div>
    """, unsafe_allow_html=True)


def render_tab_moteur_oltc(equipment: str):
    """⚡ Moteur OLTC — KPI tiles, motor current chart, z-score chart."""
    try:
        motor = api.get_motor_data(equipment)
    except Exception as e:
        st.error(f"Impossible de charger les données moteur : {e}")
        return

    baseline = motor["baseline_current_A"]
    mean_curr = motor["mean_per_commutation"]
    trend = motor["trend_slope_A_per_month"]
    n_anom = motor["anomaly_count_last_90d"]
    alert_level = motor["alert_level"]

    # ── Row 1 — 3 KPI tiles ──────────────────────────────────────────────────
    r1a, r1b, r1c = st.columns(3)

    curr_tone = "ok" if alert_level == "normal" else ("warn" if alert_level == "warning" else "crit")
    delta_curr = mean_curr - baseline
    delta_sign = "▲" if delta_curr >= 0 else "▼"
    trend_tone = "ok" if abs(trend) < 0.5 else ("warn" if abs(trend) < 1.0 else "crit")
    trend_arrow = "↑" if trend >= 0 else "↓"
    anom_tone = "ok" if n_anom == 0 else ("warn" if n_anom < 5 else "crit")

    with r1a:
        st.markdown(tile_html(
            "Courant Moyen / Comm.", f"{mean_curr:.1f} A",
            delta=f"{delta_sign} {abs(delta_curr):.1f} A vs baseline {baseline:.0f} A",
            tone=curr_tone, value_tone=curr_tone, icon="⚡"
        ), unsafe_allow_html=True)
    with r1b:
        st.markdown(tile_html(
            "Dérive Mensuelle", f"{trend:+.2f} A/mois",
            delta=f"{trend_arrow} Tendance sur 6 mois",
            tone=trend_tone, value_tone=trend_tone, icon="📈"
        ), unsafe_allow_html=True)
    with r1c:
        st.markdown(tile_html(
            "Anomalies 90 jours", str(n_anom),
            delta="|z| > 2.5 sur fenêtre 30j",
            tone=anom_tone, value_tone=anom_tone, icon="◈"
        ), unsafe_allow_html=True)

    with st.expander("ℹ️ Pourquoi surveiller le courant moteur ?"):
        st.markdown(
            "Le courant moteur reflète l'état mécanique de l'OLTC. "
            "Une dérive de **+20–30 %** précède en général la défaillance de **4 à 8 semaines** "
            "— c'est le précurseur manquant des pannes P1 et P4."
        )

    # ── Row 2 — Main chart ───────────────────────────────────────────────────
    history = motor["history"]
    df_m = pd.DataFrame(history)
    df_m["timestamp"] = pd.to_datetime(df_m["timestamp"])
    df_m.sort_values("timestamp", inplace=True, ignore_index=True)

    st.plotly_chart(
        build_motor_chart(df_m, baseline, equipment),
        use_container_width=True, config={"displayModeBar": False}
    )

    # ── Row 3 — Z-score + explanation ────────────────────────────────────────
    col_z, col_exp = st.columns([3, 2])
    with col_z:
        st.plotly_chart(
            build_zscore_chart(df_m),
            use_container_width=True, config={"displayModeBar": False}
        )
    with col_exp:
        st.markdown(f"""
        <div style="background:{SURFACE};border:1px solid {BORDER};border-radius:4px;
                    padding:16px 18px;margin-top:4px">
            <div style="font-size:12px;font-weight:700;color:{ACCENT_BRIGHT};
                        text-transform:uppercase;letter-spacing:0.06em;margin-bottom:10px">
                Pourquoi surveiller le courant moteur ?
            </div>
            <div style="font-size:12px;color:{TEXT_PRIMARY};line-height:1.65;margin-bottom:10px">
                Le moteur de commande déplace le commutateur via un réducteur mécanique
                et un ressort accumulateur.
            </div>
            <table style="width:100%;font-size:11px;border-collapse:collapse">
              <tr style="color:{TEXT_SECONDARY};font-weight:700;
                         border-bottom:1px solid {BORDER}">
                <td style="padding:4px 8px">Cause</td>
                <td style="padding:4px 8px">Effet sur I_moteur</td>
              </tr>
              <tr><td style="padding:4px 8px;border-bottom:1px solid {BORDER};color:{TEXT_PRIMARY}">Contacts usés</td>
                  <td style="padding:4px 8px;border-bottom:1px solid {BORDER};color:{TEXT_PRIMARY}">+friction → ↑ I</td></tr>
              <tr><td style="padding:4px 8px;border-bottom:1px solid {BORDER};color:{TEXT_PRIMARY}">Huile visqueuse</td>
                  <td style="padding:4px 8px;border-bottom:1px solid {BORDER};color:{TEXT_PRIMARY}">+couple → ↑ I</td></tr>
              <tr><td style="padding:4px 8px;border-bottom:1px solid {BORDER};color:{TEXT_PRIMARY}">Ressort fatigué</td>
                  <td style="padding:4px 8px;border-bottom:1px solid {BORDER};color:{TEXT_PRIMARY}">plus lent → ↑ I×t</td></tr>
              <tr><td style="padding:4px 8px;color:{TEXT_PRIMARY}">Engrenage dégradé</td>
                  <td style="padding:4px 8px;color:{TEXT_PRIMARY}">pertes → ↑ I</td></tr>
            </table>
            <div style="margin-top:12px;padding:8px 10px;
                        background:rgba(245,158,11,0.10);border:1px solid {C_WARN};
                        border-radius:3px;font-size:11px;color:{C_WARN}">
                ⚠️ Une dérive de +20–30% précède la défaillance de
                <b>4 à 8 semaines</b> — précurseur manquant des pannes P1 et P4.
            </div>
        </div>
        """, unsafe_allow_html=True)


def render_tab_prevision_prophet(equipment: str):
    """🔮 Prévision Prophet — forecast chart, SAP PM work order, threshold table."""
    if not api.PROPHET_AVAILABLE:
        st.warning(
            "Prophet n'est pas installé sur ce serveur. "
            "Exécutez `pip install prophet` pour activer les prévisions."
        )
        return

    try:
        fc = api.get_forecast(equipment)
    except Exception as e:
        st.error(f"Erreur prévision : {e}")
        return

    # 1. Forecast chart
    st.plotly_chart(
        build_forecast_chart(fc),
        use_container_width=True, config={"displayModeBar": False}
    )

    with st.expander("ℹ️ Pourquoi ce KPI ?"):
        st.markdown(
            "Prophet modélise la tendance long-terme du Health Index avec saisonnalité "
            "hebdomadaire et mensuelle. La prévision à 60 jours permet de planifier les "
            "interventions préventives avant que le HI franchisse les seuils 60 % (alerte) "
            "et 40 % (critique)."
        )

    # 2. SAP PM work order
    wo = fc.get("work_order")
    if wo:
        wo_type = wo.get("type", "preventive")
        wo_color = C_CRIT if wo_type == "urgent" else C_WARN
        wo_bg = "rgba(239,68,68,0.10)" if wo_type == "urgent" else "rgba(245,158,11,0.10)"
        sap_order = wo.get("sap_pm_order", "—")
        st.markdown(f"""
        <div style="background:{wo_bg};border:1px solid {wo_color};border-left:4px solid {wo_color};
                    border-radius:4px;padding:16px 20px;margin:14px 0">
            <div style="font-size:11px;font-weight:800;color:{wo_color};
                        text-transform:uppercase;letter-spacing:0.1em;margin-bottom:10px">
                SAP PM — Ordre de travail préventif
            </div>
            <div style="display:grid;grid-template-columns:1fr 1fr 1fr;gap:12px;font-size:12px">
                <div><span style="color:{TEXT_MUTED};font-size:10px">N° Ordre SAP</span><br>
                     <span style="color:{TEXT_PRIMARY};font-weight:700;font-family:Consolas">{sap_order}</span></div>
                <div><span style="color:{TEXT_MUTED};font-size:10px">Équipement</span><br>
                     <span style="color:{TEXT_PRIMARY};font-weight:700">{wo.get('equipment','—')}</span></div>
                <div><span style="color:{TEXT_MUTED};font-size:10px">Priorité</span><br>
                     <span style="color:{wo_color};font-weight:700">{wo.get('priority','—')}</span></div>
                <div><span style="color:{TEXT_MUTED};font-size:10px">Date alerte prévue</span><br>
                     <span style="color:{TEXT_PRIMARY};font-weight:700">{wo.get('alert_date','—')}</span></div>
                <div><span style="color:{TEXT_MUTED};font-size:10px">Jours restants</span><br>
                     <span style="color:{wo_color};font-weight:700">{wo.get('days_until_alert','—')} j</span></div>
                <div><span style="color:{TEXT_MUTED};font-size:10px">Action recommandée</span><br>
                     <span style="color:{TEXT_PRIMARY}">{wo.get('action','—')}</span></div>
            </div>
        </div>
        """, unsafe_allow_html=True)
    else:
        st.markdown(f"""
        <div style="background:rgba(16,185,129,0.10);border:1px solid {C_OK};
                    border-left:4px solid {C_OK};border-radius:4px;padding:14px 18px">
            <span style="color:{C_OK};font-weight:700">✓ Aucune intervention requise</span>
            <span style="color:{TEXT_SECONDARY};font-size:12px;margin-left:10px">
                Le Health Index reste au-dessus du seuil d'alerte sur l'horizon de prévision.
            </span>
        </div>
        """, unsafe_allow_html=True)

    # 3. Threshold reference table
    st.markdown(f'<div class="section-title" style="margin-top:20px">Seuils de surveillance OLTC</div>',
                unsafe_allow_html=True)
    thresholds = [
        ("T° huile OLTC",    "< 70°C",        "70–90°C",         "> 90°C",           "PT100 immergé"),
        ("ΔT OLTC vs cuve",  "< 10°C",        "10–20°C",         "> 20°C",           "PT100 différentiel"),
        ("C₂H₂ dissous",     "< 5 ppm",       "> 30 ppm",        "> 100 ppm",        "DGA en ligne"),
        ("H₂ dissous",       "< 50 ppm",      "> 150 ppm",       "> 300 ppm",        "DGA en ligne"),
        ("t_commutation",    "< 100 ms",      "100–150 ms",      "> 150 ms",         "Encodeur"),
        ("Courant moteur",   "< I_base×1.2",  "I_base×1.2–1.3",  "> I_base×1.3",     "TC effet Hall"),
        ("Health Index",     "> 80%",         "60–80%",          "< 60%",            "Composite"),
    ]
    header = ("Paramètre", "Normal", "Alerte ⚠️", "Critique 🔴", "Capteur")
    rows_html = "".join(
        f"""<tr>
            <td style="padding:7px 10px;color:{TEXT_PRIMARY};border-bottom:1px solid {BORDER}">{r[0]}</td>
            <td style="padding:7px 10px;color:{C_OK};border-bottom:1px solid {BORDER}">{r[1]}</td>
            <td style="padding:7px 10px;color:{C_WARN};border-bottom:1px solid {BORDER}">{r[2]}</td>
            <td style="padding:7px 10px;color:{C_CRIT};border-bottom:1px solid {BORDER}">{r[3]}</td>
            <td style="padding:7px 10px;color:{TEXT_MUTED};border-bottom:1px solid {BORDER};
                       font-family:Consolas,monospace;font-size:10px">{r[4]}</td>
        </tr>"""
        for r in thresholds
    )
    st.markdown(f"""
    <table style="width:100%;border-collapse:collapse;font-size:12px;
                  background:{SURFACE};border:1px solid {BORDER};border-radius:4px">
      <thead>
        <tr style="background:{SURFACE_HOVER}">
          {"".join(f'<th style="padding:8px 10px;color:{TEXT_SECONDARY};font-weight:700;'
                   f'text-transform:uppercase;letter-spacing:0.06em;font-size:10px;'
                   f'text-align:left;border-bottom:2px solid {BORDER}">{h}</th>' for h in header)}
        </tr>
      </thead>
      <tbody>{rows_html}</tbody>
    </table>
    """, unsafe_allow_html=True)


# ═══════════════════════════════════════════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════════════════════════════════════════
def main():
    init_session()
    st.markdown(CSS, unsafe_allow_html=True)

    # ── API CHECK ─────────────────────────────────────────────────────────────
    if not api.API_OK:
        api.bootstrap()  # retry once in case the API just came up
    if not api.API_OK:
        st.markdown(f"""
        <div style="background:rgba(239,68,68,0.10);border:1px solid {C_CRIT};
                    border-left:4px solid {C_CRIT};border-radius:4px;
                    padding:18px 22px;margin:20px 0">
            <div style="font-size:14px;font-weight:800;color:{C_CRIT};
                        text-transform:uppercase;letter-spacing:0.06em;margin-bottom:8px">
                ⚠ API Back-end injoignable
            </div>
            <div style="color:{TEXT_PRIMARY};font-size:13px;line-height:1.6">
                Le dashboard ne peut pas charger les données — le serveur FastAPI
                est introuvable à <code style="color:{ACCENT_BRIGHT}">{api.API_BASE}</code>.<br>
                <span style="color:{TEXT_SECONDARY}">Erreur : {api.API_ERROR or 'inconnue'}</span>
            </div>
            <div style="margin-top:14px;padding-top:12px;border-top:1px solid {BORDER};
                        font-size:12px;color:{TEXT_SECONDARY};font-family:Consolas,monospace">
                <div style="color:{TEXT_MUTED};margin-bottom:4px">Démarrer le back-end :</div>
                <div>cd back-end && python main.py</div>
                <div style="margin-top:8px;color:{TEXT_MUTED}">Personnaliser l'URL :</div>
                <div>set OLTC_API_URL=http://votre-host:8000</div>
            </div>
        </div>
        """, unsafe_allow_html=True)
        st.stop()

    # ── SIDEBAR ──────────────────────────────────────────────────────────────
    with st.sidebar:
        st.markdown(f"""
        <div style="text-align:center;padding:10px 0 18px 0;border-bottom:1px solid {BORDER};margin-bottom:14px">
            <div style="font-size:24px;font-weight:800;color:{ACCENT_BRIGHT};letter-spacing:-0.02em">⚡ OLTC</div>
            <div style="font-size:10px;color:{TEXT_MUTED};text-transform:uppercase;letter-spacing:0.15em;margin-top:2px">
                SCADA Monitor
            </div>
        </div>
        """, unsafe_allow_html=True)

        equipment = st.selectbox("Équipement", api.EQUIPMENT_LIST, index=0)
        period    = st.selectbox("Période", list(PERIOD_OPTIONS.keys()), index=5)

        st.markdown(f'<hr style="border-color:{BORDER};margin:14px 0">', unsafe_allow_html=True)

        live_mode = st.toggle("Mode Temps Réel", value=st.session_state.live_mode)
        st.session_state.live_mode = live_mode
        refresh_rate = st.slider("Rafraîchissement (s)", 2, 30, 5,
                                 disabled=not live_mode)

        if st.button("Recharger les données", use_container_width=True):
            st.session_state.rt_buffer = []
            st.cache_data.clear()
            add_alert("INFO", f"Données rechargées · {equipment}")
            st.rerun()

        st.markdown(f'<hr style="border-color:{BORDER};margin:14px 0">', unsafe_allow_html=True)

        api_dot_color = C_OK if api.API_OK else C_CRIT
        prophet_dot_color = C_OK if api.PROPHET_AVAILABLE else TEXT_MUTED
        st.markdown(f"""
        <div style="font-size:10px;color:{TEXT_MUTED};line-height:1.7;letter-spacing:0.02em">
            <div style="color:{TEXT_SECONDARY};text-transform:uppercase;font-weight:700;
                        letter-spacing:0.1em;font-size:9px;margin-bottom:6px">Back-end API</div>
            <div><span style="color:{api_dot_color}">●</span> Connecté · v{api.API_VERSION or '?'}</div>
            <div style="color:{TEXT_MUTED};font-family:Consolas,monospace;font-size:9.5px">{api.API_BASE}</div>
            <div><span style="color:{prophet_dot_color}">●</span> Prophet {'OK' if api.PROPHET_AVAILABLE else 'indisponible'}</div>
            <div style="margin-top:8px;color:{TEXT_SECONDARY};text-transform:uppercase;font-weight:700;
                        letter-spacing:0.1em;font-size:9px">Moteurs IA</div>
            <div>● Isolation Forest 200 est.</div>
            <div>● Health Index pondéré</div>
            <div style="margin-top:8px;color:{TEXT_SECONDARY};text-transform:uppercase;font-weight:700;
                        letter-spacing:0.1em;font-size:9px">Session</div>
            <div>Équipement · <span style="color:{TEXT_PRIMARY}">{equipment}</span></div>
            <div>Build {datetime.now().strftime('%d/%m/%Y')}</div>
            <div style="margin-top:8px;color:{TEXT_SECONDARY};text-transform:uppercase;font-weight:700;
                        letter-spacing:0.1em;font-size:9px">Site</div>
            <div>OCP · Terminal Slurry Pipeline</div>
        </div>
        """, unsafe_allow_html=True)

    # ── LOAD DATA ─────────────────────────────────────────────────────────────
    df_full = load_data(equipment)
    df = filter_by_period(df_full, period)

    # ── STATIC COMPUTATIONS (done once per main() run, not per tick) ────────
    last_fault_key  = sorted(api.FAULT_SCENARIOS.keys(),
                             key=lambda k: api.FAULT_SCENARIOS[k]["date"],
                             reverse=True)[0]
    last_fault_date = pd.Timestamp(api.FAULT_SCENARIOS[last_fault_key]["date"]).strftime("%d/%m/%Y")
    mwh_avoided     = sum(v["mwh_lost"] for v in api.FAULT_SCENARIOS.values())
    if_score_now    = float(df_full["anomaly_score"].iloc[-1]) if not df_full.empty else 0.0

    fault_stats = compute_fault_stats(df_full)
    fault_df = pd.DataFrame(fault_stats)[[
        "Panne", "Date", "Sévérité", "MWh", "HI panne", "Avance détection", "Description"
    ]]
    period_range = ""
    if not df.empty:
        period_range = f"{df['timestamp'].min().strftime('%d %b %Y')} → {df['timestamp'].max().strftime('%d %b %Y')}"
    hi_timeseries_fig = build_hi_timeseries(df)

    # ── LIVE BLOCK (only this section re-runs on each tick, no full reload) ─
    @st.fragment(run_every=refresh_rate if live_mode else None)
    def _live_section():
        # Sampling
        if live_mode:
            new_pt = generate_realtime_sample(equipment, seed=int(time.time() * 1000) % 100000)
            st.session_state.rt_buffer.append(new_pt)
            if len(st.session_state.rt_buffer) > 200:
                st.session_state.rt_buffer = st.session_state.rt_buffer[-200:]
            current_vals = {p: new_pt[p] for p in api.NOMINAL}

            for param in api.NOMINAL:
                ps = get_param_status(param, current_vals[param])
                if ps == "critical":
                    add_alert("CRITICAL",
                              f"{api.PARAM_LABELS[param]} = {current_vals[param]:.2f} {api.PARAM_UNITS[param]} — seuil trip dépassé")

            if len(st.session_state.rt_buffer) >= 10:
                try:
                    from sklearn.preprocessing import StandardScaler
                    from sklearn.ensemble import IsolationForest
                    buf_df = pd.DataFrame(st.session_state.rt_buffer)
                    X = StandardScaler().fit_transform(buf_df[[p for p in api.NOMINAL]].values)
                    clf = IsolationForest(n_estimators=50, contamination=0.05, random_state=42)
                    clf.fit(X)
                    sc = float(-clf.decision_function(X[-1:])[-1])
                    if sc > 0.60:
                        add_alert("CRITICAL", f"Score IF live = {sc:.2f} — Anomalie critique détectée")
                    elif sc > 0.47:
                        add_alert("WARNING", f"Score IF live = {sc:.2f} — Surveillance renforcée")
                except ImportError:
                    pass
        else:
            if not df_full.empty:
                current_vals = {p: float(df_full.iloc[-1][p]) for p in api.NOMINAL}
            else:
                current_vals = {p: api.NOMINAL[p]["mean"] for p in api.NOMINAL}

        if live_mode and st.session_state.rt_buffer:
            hi_now = compute_hi_from_vals(current_vals)
        else:
            hi_now = float(df_full["health_index"].iloc[-1]) if not df_full.empty else 80.0

        status_label, _ = get_status(hi_now)
        hi_delta = hi_now - float(df_full["health_index"].mean()) if not df_full.empty else 0

        # HEADER
        live_class = "on" if live_mode else "off"
        live_inner = '<span class="live-dot"></span><span>● LIVE</span>' if live_mode else '<span>○ HISTORIQUE</span>'
        st.markdown(f"""
        <div class="scada-header">
            <div>
                <div class="scada-title">⚡ OLTC · Maintenance Prédictive</div>
                <div class="scada-subtitle">OCP · Terminal Slurry Pipeline · Équipement <b style="color:{ACCENT_BRIGHT}">{equipment}</b> · {period_range}</div>
            </div>
            <div class="scada-meta">
                <span class="live-pill {live_class}">{live_inner}</span>
                <span class="scada-clock">{datetime.now().strftime('%H:%M:%S')}</span>
            </div>
        </div>
        """, unsafe_allow_html=True)

        # KPI ROW
        render_kpi_row(hi_now, status_label, if_score_now, last_fault_key,
                       last_fault_date, mwh_avoided, hi_delta)

        # MAIN CHARTS (HI timeseries cached + dynamic gauge)
        st.markdown('<div class="section-title">Vue d\'ensemble</div>', unsafe_allow_html=True)
        col_main, col_gauge = st.columns([3, 2])
        with col_main:
            st.plotly_chart(hi_timeseries_fig, use_container_width=True,
                            config={"displayModeBar": False})
        with col_gauge:
            gauge_fig, dist_to_crit = build_gauge(hi_now)
            st.plotly_chart(gauge_fig, use_container_width=True,
                            config={"displayModeBar": False})
            progress_pct = int(max(0, min(100, dist_to_crit / 60 * 100)))
            st.markdown(f"""
            <div style="padding:0 6px">
                <div style="font-size:10px;color:{TEXT_SECONDARY};text-transform:uppercase;
                            letter-spacing:0.1em;font-weight:700;margin-bottom:4px">
                    Marge au seuil critique
                </div>
                <div style="background:{SURFACE_HOVER};border:1px solid {BORDER};
                            border-radius:3px;height:14px;overflow:hidden">
                    <div style="width:{progress_pct}%;height:100%;
                                background:linear-gradient(90deg,{ACCENT} 0%,{ACCENT_BRIGHT} 100%);
                                box-shadow:0 0 8px {ACCENT_GLOW}"></div>
                </div>
                <div style="font-size:11px;color:{TEXT_MUTED};margin-top:4px">
                    {dist_to_crit:.1f} pts au-dessus du seuil 40% · marge {progress_pct}%
                </div>
            </div>
            """, unsafe_allow_html=True)

        # SENSORS
        st.markdown('<div class="section-title">Capteurs · Lecture instantanée</div>',
                    unsafe_allow_html=True)
        spark_data = {}
        if live_mode and len(st.session_state.rt_buffer) >= 5:
            for param in api.NOMINAL:
                spark_data[param] = [pt[param] for pt in st.session_state.rt_buffer[-50:]]
        else:
            tail = df_full.tail(50)
            for param in api.NOMINAL:
                spark_data[param] = tail[param].tolist()
        try:
            motor_data = api.get_motor_data(equipment)
        except Exception:
            motor_data = None
        render_sensor_grid(current_vals, spark_data, motor_data)

        # FAULT TABLE + ALERT LOG
        st.markdown('<div class="section-title">Historique pannes & Journal d\'alertes</div>',
                    unsafe_allow_html=True)
        col_faults, col_alerts = st.columns([3, 2])
        with col_faults:
            st.markdown(f'<div style="font-size:11px;color:{TEXT_SECONDARY};text-transform:uppercase;'
                        f'letter-spacing:0.08em;font-weight:700;margin-bottom:8px">Pannes historiques P1–P6</div>',
                        unsafe_allow_html=True)
            st.dataframe(fault_df, use_container_width=True, hide_index=True, height=300)
        with col_alerts:
            st.markdown(f'<div style="font-size:11px;color:{TEXT_SECONDARY};text-transform:uppercase;'
                        f'letter-spacing:0.08em;font-weight:700;margin-bottom:8px">'
                        f'Journal temps réel <span style="color:{ACCENT_BRIGHT}">●</span></div>',
                        unsafe_allow_html=True)
            if not st.session_state.alert_log:
                add_alert("INFO", f"Système initialisé · {equipment} en surveillance")
            with st.container(height=300):
                render_alert_log()

    _live_section()

    # ── ANALYSIS SECTION ─────────────────────────────────────────────────────
    st.markdown('<div class="section-title">Analyse approfondie</div>', unsafe_allow_html=True)
    tab_if, tab_multi, tab_dist, tab_parc, tab_moteur, tab_prophet = st.tabs([
        "Isolation Forest",
        "Multi-capteurs (z-score)",
        "Distribution HI",
        "🔧 Parc OLTC",
        "⚡ Moteur OLTC",
        "🔮 Prévision Prophet",
    ])

    with tab_if:
        st.plotly_chart(build_anomaly_score_chart(df), use_container_width=True,
                        config={"displayModeBar": False})
        n_anom = int(df["is_anomaly"].sum())
        pct = n_anom / len(df) * 100 if len(df) > 0 else 0
        ca, cb, cc = st.columns(3)
        with ca: st.metric("Mesures analysées", f"{len(df):,}")
        with cb: st.metric("Anomalies détectées", f"{n_anom:,}")
        with cc: st.metric("Taux d'anomalies", f"{pct:.2f}%")

    with tab_multi:
        try:
            _motor_data_multi = api.get_motor_data(equipment)
        except Exception:
            _motor_data_multi = None
        st.plotly_chart(build_normalized_multicapteur(df, _motor_data_multi),
                        use_container_width=True, config={"displayModeBar": False})

    with tab_dist:
        col_d1, col_d2 = st.columns([3, 2])
        with col_d1:
            st.plotly_chart(build_hi_distribution(df), use_container_width=True,
                            config={"displayModeBar": False})
        with col_d2:
            st.markdown(f'<div style="font-size:11px;color:{TEXT_SECONDARY};text-transform:uppercase;'
                        f'letter-spacing:0.08em;font-weight:700;margin-bottom:8px">'
                        f'Répartition par état</div>', unsafe_allow_html=True)
            counts = {
                "Normal (>80%)":         (int((df["health_index"] >= 80).sum()), C_OK),
                "Surveillance (60-80%)": (int(((df["health_index"] >= 60) & (df["health_index"] < 80)).sum()), C_INFO),
                "Alerte (40-60%)":       (int(((df["health_index"] >= 40) & (df["health_index"] < 60)).sum()), C_WARN),
                "Critique (<40%)":       (int((df["health_index"] < 40).sum()), C_CRIT),
            }
            for label, (cnt, color) in counts.items():
                pct_s = cnt / len(df) * 100 if len(df) > 0 else 0
                st.markdown(f"""
                <div style="background:{SURFACE};border:1px solid {BORDER};
                            border-left:3px solid {color};border-radius:3px;
                            padding:10px 12px;margin-bottom:6px">
                    <div style="font-size:10px;color:{TEXT_SECONDARY};
                                text-transform:uppercase;letter-spacing:0.06em;font-weight:600">{label}</div>
                    <div style="font-size:22px;font-weight:800;color:{TEXT_PRIMARY};line-height:1.1">{cnt:,}
                        <span style="font-size:11px;color:{color};margin-left:6px">{pct_s:.1f}%</span>
                    </div>
                </div>
                """, unsafe_allow_html=True)

    # ── TAB: PARC OLTC ────────────────────────────────────────────────────────
    with tab_parc:
        render_tab_parc_oltc()

    # ── TAB: MOTEUR OLTC ─────────────────────────────────────────────────────
    with tab_moteur:
        render_tab_moteur_oltc(equipment)

    # ── TAB: PRÉVISION PROPHET ───────────────────────────────────────────────
    with tab_prophet:
        render_tab_prevision_prophet(equipment)

    # Footer
    st.markdown(f"""
    <div style="margin-top:30px;padding-top:14px;border-top:1px solid {BORDER};
                text-align:center;color:{TEXT_MUTED};font-size:10px;letter-spacing:0.05em">
        OLTC SCADA · Predictive Maintenance v2.1 · Powered by Health Index + Isolation Forest
    </div>
    """, unsafe_allow_html=True)

    # No global auto-refresh: the live region uses @st.fragment(run_every=…)
    # so only that region re-renders on each tick — no full-page reload, no flash.


if __name__ == "__main__":
    main()
