"""
HTTP client to the OLTC FastAPI back-end.

Drop-in replacement for the legacy `data_generator` module: exposes the same
constants (NOMINAL, FAULT_SCENARIOS, …) and helper functions so the Streamlit
dashboard can swap its data source without touching the UI code.

Configure the API base URL with the `OLTC_API_URL` env var
(default: http://localhost:8000).
"""
from __future__ import annotations

import os
from datetime import datetime, timedelta
from typing import Optional

import httpx
import pandas as pd
import streamlit as st


API_BASE = os.environ.get("OLTC_API_URL", "http://localhost:8000")
TIMEOUT = httpx.Timeout(15.0, connect=3.0)


# ─── Low-level HTTP ──────────────────────────────────────────────────────────
def _get(path: str, params: Optional[dict] = None) -> dict | list:
    with httpx.Client(timeout=TIMEOUT) as client:
        r = client.get(f"{API_BASE}{path}", params=params)
        r.raise_for_status()
        return r.json()


# ─── Metadata bootstrap (cached for the session) ─────────────────────────────
@st.cache_data(ttl=3600, show_spinner=False)
def _fetch_metadata() -> dict:
    sensors  = _get("/api/sensors")
    faults   = _get("/api/faults")
    equip    = _get("/api/equipment")
    health   = _get("/api/health")
    return {
        "sensors": sensors, "faults": faults,
        "equipment": equip, "health": health,
    }


def _build_constants(meta: dict) -> dict:
    sensors = meta["sensors"]
    faults  = meta["faults"]
    equip   = meta["equipment"]
    return {
        "NOMINAL": {s["id"]: {"mean": s["nominal_mean"], "std": s["nominal_std"]} for s in sensors},
        "THRESHOLDS": {s["id"]: s["thresholds"] for s in sensors},
        "PARAM_LABELS": {s["id"]: s["label"] for s in sensors},
        "PARAM_ICONS":  {s["id"]: s["icon"]  for s in sensors},
        "PARAM_UNITS":  {s["id"]: s["unit"]  for s in sensors},
        "FAULT_SCENARIOS": {
            f["id"]: {
                "date": f["date"], "drift_weeks": f["drift_weeks"],
                "params": f["params"], "description": f["description"],
                "severity": f["severity"], "mwh_lost": f["mwh_lost"],
            }
            for f in faults
        },
        "EQUIPMENT_LIST": [e["id"] for e in equip],
        "EQUIPMENT_SEEDS": {e["id"]: e["seed"] for e in equip},
    }


# Module-level constants — populated at import time. If the API is unreachable,
# these stay as empty defaults and `API_OK` flips to False so the UI can warn.
NOMINAL: dict = {}
THRESHOLDS: dict = {}
PARAM_LABELS: dict = {}
PARAM_ICONS: dict = {}
PARAM_UNITS: dict = {}
FAULT_SCENARIOS: dict = {}
EQUIPMENT_LIST: list = []
EQUIPMENT_SEEDS: dict = {}

API_OK: bool = False
API_ERROR: Optional[str] = None
API_VERSION: Optional[str] = None
PROPHET_AVAILABLE: bool = False


def bootstrap() -> bool:
    """Fetch metadata once and populate module-level constants.
    Returns True on success, False on any error (use API_ERROR to inspect)."""
    global NOMINAL, THRESHOLDS, PARAM_LABELS, PARAM_ICONS, PARAM_UNITS
    global FAULT_SCENARIOS, EQUIPMENT_LIST, EQUIPMENT_SEEDS
    global API_OK, API_ERROR, API_VERSION, PROPHET_AVAILABLE

    try:
        meta = _fetch_metadata()
        c = _build_constants(meta)
        NOMINAL          = c["NOMINAL"]
        THRESHOLDS       = c["THRESHOLDS"]
        PARAM_LABELS     = c["PARAM_LABELS"]
        PARAM_ICONS      = c["PARAM_ICONS"]
        PARAM_UNITS      = c["PARAM_UNITS"]
        FAULT_SCENARIOS  = c["FAULT_SCENARIOS"]
        EQUIPMENT_LIST   = c["EQUIPMENT_LIST"]
        EQUIPMENT_SEEDS  = c["EQUIPMENT_SEEDS"]
        API_VERSION      = meta["health"]["version"]
        PROPHET_AVAILABLE = meta["health"]["prophet_available"]
        API_OK = True
        API_ERROR = None
        return True
    except Exception as e:
        API_OK = False
        API_ERROR = f"{type(e).__name__}: {e}"
        return False


# ─── Period mapping (UI labels → API codes) ──────────────────────────────────
PERIOD_OPTIONS = {
    "Dernière heure": timedelta(hours=1),
    "24h":  timedelta(hours=24),
    "7j":   timedelta(days=7),
    "30j":  timedelta(days=30),
    "1 an": timedelta(days=365),
    "Tout": None,
}


# ─── Data fetching ───────────────────────────────────────────────────────────
@st.cache_data(ttl=3600, show_spinner="Chargement des données…")
def load_data(equipment: str, freq_hours: int = 6) -> pd.DataFrame:
    """Fetch the full scored historical series from the API."""
    payload = _get(f"/api/historical/{equipment}",
                   {"freq_hours": freq_hours, "period": "all"})
    df = pd.DataFrame(payload["points"])
    if not df.empty:
        df["timestamp"] = pd.to_datetime(df["timestamp"])
    return df


def filter_by_period(df: pd.DataFrame, period_label: str) -> pd.DataFrame:
    delta = PERIOD_OPTIONS.get(period_label)
    if delta is None or df.empty:
        return df
    cutoff = df["timestamp"].max() - delta
    return df[df["timestamp"] >= cutoff].copy()


def generate_realtime_sample(equipment: str, seed: Optional[int] = None) -> dict:
    """Get the next realtime sample from the API."""
    params: dict = {"freq_hours": 6}
    if seed is not None:
        params["seed"] = seed
    payload = _get(f"/api/realtime/{equipment}/sample", params)
    payload["timestamp"] = pd.to_datetime(payload["timestamp"])
    return payload


def get_anomaly_report(equipment: str, freq_hours: int = 6) -> dict:
    return _get(f"/api/anomalies/{equipment}/report", {"freq_hours": freq_hours})


def get_forecast(equipment: str, days: int = 60, freq_hours: int = 6) -> dict:
    return _get(f"/api/forecast/{equipment}",
                {"days": days, "freq_hours": freq_hours})


@st.cache_data(ttl=300, show_spinner="Chargement courant moteur…")
def get_motor_data(equipment_id: str) -> dict:
    """Fetch motor current surveillance data for the given OLTC unit (cached 5 min)."""
    return _get(f"/api/motor/{equipment_id}")


# ─── Pure helpers (replicated locally to avoid an HTTP round-trip per tick) ──
def get_status(hi: float) -> tuple[str, str]:
    if hi >= 80: return "Normal", "🟢"
    if hi >= 60: return "Surveillance", "🟡"
    if hi >= 40: return "Alerte", "🟠"
    return "Critique", "🔴"


def get_param_status(param: str, value: float) -> str:
    if param not in THRESHOLDS:
        return "normal"
    warn = THRESHOLDS[param]["warn"]
    crit = THRESHOLDS[param]["critical"]
    if param == "vcc_v":
        if value >= warn: return "normal"
        if value >= crit: return "warning"
        return "critical"
    if value <= warn: return "normal"
    if value <= crit: return "warning"
    return "critical"


# Auto-bootstrap on import. If the API is down at first import, the module
# still loads but with empty constants — main() should call bootstrap() again
# and surface API_ERROR in the UI.
bootstrap()
