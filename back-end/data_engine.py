"""
OLTC data engine — single source of truth for the simulated dataset.

Generates historical IoT series with injected fault drifts, computes the
weighted Health Index, and produces realtime samples. Kept dependency-free
from FastAPI so it can be reused by Streamlit, notebooks, or other clients.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from functools import lru_cache


# ─── Reference parameters ─────────────────────────────────────────────────────
NOMINAL = {
    "dga_h2_ppm":    {"mean": 30,  "std": 8},
    "dga_c2h2_ppm":  {"mean": 2.0, "std": 0.8},
    "temp_huile_c":  {"mean": 65,  "std": 3},
    "vib_ms2":       {"mean": 1.2, "std": 0.3},
    "t_comm_ms":     {"mean": 68,  "std": 5},
    "vcc_v":         {"mean": 125, "std": 1.5},
}

THRESHOLDS = {
    "dga_h2_ppm":   {"nominal": 50,  "alarm": 100, "trip": 300, "warn": 50,  "critical": 100},
    "dga_c2h2_ppm": {"nominal": 5,   "alarm": 30,  "trip": 100, "warn": 30,  "critical": 100},
    "temp_huile_c": {"nominal": 70,  "alarm": 80,  "trip": 95,  "warn": 75,  "critical": 85},
    "vib_ms2":      {"nominal": 2.0, "alarm": 4.0, "trip": 7.0, "warn": 2.0, "critical": 3.0},
    "t_comm_ms":    {"nominal": 80,  "alarm": 100, "trip": 150, "warn": 80,  "critical": 95},
    "vcc_v":        {"nominal": 115, "alarm": 110, "trip": 95,  "warn": 122, "critical": 119},
}

PARAM_LABELS = {
    "dga_h2_ppm":   "DGA H₂",
    "dga_c2h2_ppm": "DGA C₂H₂",
    "temp_huile_c": "Temp. Huile",
    "vib_ms2":      "Vibrations",
    "t_comm_ms":    "Temps Comm.",
    "vcc_v":        "Tension 127 Vcc",
}

PARAM_ICONS = {
    "dga_h2_ppm":   "🔥",
    "dga_c2h2_ppm": "⚗️",
    "temp_huile_c": "🌡️",
    "vib_ms2":      "📳",
    "t_comm_ms":    "⏱️",
    "vcc_v":        "⚡",
}

PARAM_UNITS = {
    "dga_h2_ppm":   "ppm",
    "dga_c2h2_ppm": "ppm",
    "temp_huile_c": "°C",
    "vib_ms2":      "m/s²",
    "t_comm_ms":    "ms",
    "vcc_v":        "V",
}

HI_WEIGHTS = {
    "dga_h2_ppm":   0.35,
    "dga_c2h2_ppm": 0.00,   # monitoring only — kept out of HI to preserve existing score
    "temp_huile_c": 0.20,
    "vib_ms2":      0.20,
    "t_comm_ms":    0.15,
    "vcc_v":        0.10,
}

FAULT_SCENARIOS = {
    "P1": {
        "date": "2022-07-27", "drift_weeks": 8,
        "params": {"dga_h2_ppm": 3.5, "dga_c2h2_ppm": 5.0, "temp_huile_c": 0.8, "vib_ms2": 0.6, "t_comm_ms": 1.8},
        "description": "Décomposition thermique de l'huile — arc électrique",
        "severity": "Critique", "mwh_lost": 24.5,
    },
    "P2": {
        "date": "2022-02-20", "drift_weeks": 6,
        "params": {"dga_h2_ppm": 2.0, "t_comm_ms": 2.2},
        "description": "Usure mécanique des contacts — commutation lente",
        "severity": "Majeur", "mwh_lost": 8.2,
    },
    "P3": {
        "date": "2021-12-14", "drift_weeks": 4,
        "params": {"vib_ms2": 0.8, "t_comm_ms": 2.5},
        "description": "Desserrage mécanique — vibrations anormales",
        "severity": "Majeur", "mwh_lost": 11.3,
    },
    "P4": {
        "date": "2019-09-30", "drift_weeks": 4,
        "params": {"t_comm_ms": 3.0},
        "description": "Dégradation ressort de commutation",
        "severity": "Mineur", "mwh_lost": 3.1,
    },
    "P5": {
        "date": "2017-07-01", "drift_weeks": 10,
        "params": {"dga_h2_ppm": 4.0, "dga_c2h2_ppm": 3.0, "temp_huile_c": 1.5, "vib_ms2": 0.9},
        "description": "Surchauffe chronique — contamination huile",
        "severity": "Critique", "mwh_lost": 16.0,
    },
    "P6": {
        "date": "2017-08-09", "drift_weeks": 2,
        "params": {"vcc_v": -3.5},
        "description": "Défaut alimentation commande 127 Vcc",
        "severity": "Mineur", "mwh_lost": 2.8,
    },
}

EQUIPMENT_LIST = ["PJ1", "PJ2", "PJ3", "PJ11"]
EQUIPMENT_SEEDS = {"PJ1": 42, "PJ2": 43, "PJ3": 44, "PJ11": 45}

# ─── Motor current parameters ─────────────────────────────────────────────────
MOTOR_BASELINES = {  # Amperes — nameplate data per unit
    "PJ1": 350,   # JANSEN V III Y
    "PJ2": 350,   # JANSEN V III Y
    "PJ3": 200,   # manufacturer unknown → estimated
    "PJ11": 268,  # MR V III 350-Y-76-10
    "PJ10": 134,  # MR V III 200Y
}

MOTOR_DRIFT_RATES = {  # A/month — slow linear degradation by unit age
    "PJ1": 1.2,   # 1998 vintage, ~26 yrs → fastest contact wear
    "PJ2": 0.8,
    "PJ3": 0.8,
    "PJ11": 0.5,  # 2007 vintage, 17 yrs
    "PJ10": 0.3,
}

PERIOD_DELTAS = {
    "1h": timedelta(hours=1),
    "24h": timedelta(hours=24),
    "7d": timedelta(days=7),
    "30d": timedelta(days=30),
    "1y": timedelta(days=365),
    "all": None,
}


# ─── Status helpers ───────────────────────────────────────────────────────────
def get_status(hi: float) -> str:
    if hi >= 80:
        return "Normal"
    if hi >= 60:
        return "Surveillance"
    if hi >= 40:
        return "Alerte"
    return "Critique"


def get_param_status(param: str, value: float) -> str:
    warn = THRESHOLDS[param]["warn"]
    crit = THRESHOLDS[param]["critical"]
    if param == "vcc_v":
        if value >= warn:
            return "normal"
        if value >= crit:
            return "warning"
        return "critical"
    if value <= warn:
        return "normal"
    if value <= crit:
        return "warning"
    return "critical"


# ─── Fault drift application ──────────────────────────────────────────────────
def _apply_fault_drift(df: pd.DataFrame, fault_key: str) -> pd.DataFrame:
    fault = FAULT_SCENARIOS[fault_key]
    fault_date = pd.Timestamp(fault["date"])
    drift_days = fault["drift_weeks"] * 7

    mask_drift = (df["timestamp"] >= fault_date - timedelta(days=drift_days)) & (
        df["timestamp"] < fault_date
    )
    for param, multiplier in fault["params"].items():
        if param not in df.columns:
            continue
        drift_indices = df.index[mask_drift]
        n = len(drift_indices)
        if n == 0:
            continue
        progress = np.linspace(0, 1, n) ** 2
        delta = multiplier * NOMINAL[param]["std"]
        df.loc[drift_indices, param] += delta * progress

    mask_post = df["timestamp"] >= fault_date
    days_post = (df.loc[mask_post, "timestamp"] - fault_date).dt.days
    recovery = np.exp(-days_post / 30.0).values
    for param, multiplier in fault["params"].items():
        if param not in df.columns:
            continue
        delta = multiplier * NOMINAL[param]["std"]
        df.loc[mask_post, param] += delta * recovery

    return df


def _compute_health_index(df: pd.DataFrame) -> pd.DataFrame:
    scores = pd.DataFrame(index=df.index)
    for param, w in HI_WEIGHTS.items():
        nom = NOMINAL[param]["mean"]
        warn = THRESHOLDS[param]["warn"]
        crit = THRESHOLDS[param]["critical"]
        val = df[param]
        if param == "vcc_v":
            score = np.where(val >= warn, 100, np.where(val >= crit, 50, 0)).astype(float)
        else:
            score = np.where(
                val <= nom * 1.1,
                100,
                np.where(
                    val <= warn,
                    100 - 50 * (val - nom * 1.1) / (warn - nom * 1.1),
                    np.where(val <= crit, 50 - 50 * (val - warn) / (crit - warn), 0),
                ),
            ).astype(float)
        scores[param] = score * w
    df["health_index"] = scores.sum(axis=1).clip(0, 100)
    return df


def compute_hi_from_vals(vals: dict) -> float:
    total = 0.0
    for param, w in HI_WEIGHTS.items():
        nom = NOMINAL[param]["mean"]
        warn = THRESHOLDS[param]["warn"]
        crit = THRESHOLDS[param]["critical"]
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


# ─── Generators ───────────────────────────────────────────────────────────────
def generate_historical(
    equipment: str,
    start_date: str = "2017-01-01",
    end_date: str | None = None,
    freq_hours: int = 6,
) -> pd.DataFrame:
    seed = EQUIPMENT_SEEDS.get(equipment, 42)
    np.random.seed(seed)
    if end_date is None:
        end_date = datetime.now().strftime("%Y-%m-%d")

    timestamps = pd.date_range(start=start_date, end=end_date, freq=f"{freq_hours}h")
    n = len(timestamps)
    data: dict = {"timestamp": timestamps}
    for param, stats in NOMINAL.items():
        noise = np.random.normal(0, 1, n)
        seasonal = stats["std"] * 0.3 * np.sin(2 * np.pi * np.arange(n) / (365 * 24 / freq_hours))
        data[param] = stats["mean"] + stats["std"] * noise + seasonal

    df = pd.DataFrame(data)
    df["dga_h2_ppm"]   = df["dga_h2_ppm"].clip(lower=0)
    df["dga_c2h2_ppm"] = df["dga_c2h2_ppm"].clip(lower=0)
    df["vib_ms2"] = df["vib_ms2"].clip(lower=0.1)
    df["vcc_v"] = df["vcc_v"].clip(lower=115)
    df["t_comm_ms"] = df["t_comm_ms"].clip(lower=55)

    for fault_key in FAULT_SCENARIOS:
        df = _apply_fault_drift(df, fault_key)

    df = _compute_health_index(df)
    return df.reset_index(drop=True)


@lru_cache(maxsize=8)
def get_dataset(equipment: str, freq_hours: int = 6) -> pd.DataFrame:
    """Cached historical dataset with HI but without anomaly scores."""
    return generate_historical(equipment, freq_hours=freq_hours)


def generate_realtime_sample(last_row: dict | None = None, seed: int | None = None) -> dict:
    if seed is not None:
        np.random.seed(seed)
    sample = {}
    for param, stats in NOMINAL.items():
        if last_row is not None and param in last_row:
            prev = float(last_row[param])
            delta = np.random.normal(0, stats["std"] * 0.08)
            val = prev + delta
        else:
            val = np.random.normal(stats["mean"], stats["std"])
        sample[param] = float(val)
    sample["dga_h2_ppm"] = max(0, sample["dga_h2_ppm"])
    sample["vib_ms2"] = max(0.1, sample["vib_ms2"])
    sample["timestamp"] = datetime.now()
    sample["health_index"] = compute_hi_from_vals(sample)
    sample["status"] = get_status(sample["health_index"])
    return sample


def filter_by_period(df: pd.DataFrame, period: str) -> pd.DataFrame:
    delta = PERIOD_DELTAS.get(period)
    if delta is None:
        return df
    cutoff = df["timestamp"].max() - delta
    return df[df["timestamp"] >= cutoff].copy()


@lru_cache(maxsize=8)
def generate_motor_current(equipment: str) -> dict:
    """
    Generate motor current history (one sample per day) for an OLTC unit.

    Motor current increases when: contacts are worn (higher resistance → more torque
    needed), oil is too viscous (cold weather or degraded oil), mechanical components
    are seizing, or the drive spring is weakening. A +30% current drift detected 4–8
    weeks before failure was the missing precursor for faults P1 and P4
    (PJ15, thermal trips 2017 & 2020).

    Alert thresholds (IEC / MR recommendations):
      Normal   : I_motor < baseline × 1.20
      Warning  : baseline × 1.20 ≤ I_motor < baseline × 1.30
      Critical : I_motor ≥ baseline × 1.30

    Z-score anomaly: flag if |z| > 2.5 over a 30-day rolling window.
    """
    seed = EQUIPMENT_SEEDS.get(equipment, 42)
    np.random.seed(seed)

    baseline = float(MOTOR_BASELINES.get(equipment, 200))
    drift_rate = MOTOR_DRIFT_RATES.get(equipment, 0.5)  # A/month

    # Daily timeline from 2017-01-01 to today for full fault coverage
    start_dt = pd.Timestamp("2017-01-01")
    end_dt = pd.Timestamp(datetime.now().date())
    ts_series = pd.date_range(start=start_dt, end=end_dt, freq="D")
    n = len(ts_series)

    # Slow linear drift: A/day = A/month ÷ 30
    drift = np.arange(n, dtype=float) * (drift_rate / 30.0)

    # Gaussian noise (~3% baseline) + winter viscosity seasonal
    noise = np.random.normal(0.0, baseline * 0.03, n)
    seasonal = baseline * 0.02 * np.sin(2.0 * np.pi * np.arange(n) / 365.25)

    currents = baseline + drift + noise + seasonal
    currents = np.maximum(currents, baseline * 0.5)

    # Vectorised rolling z-score (30-day window)
    s = pd.Series(currents)
    roll_mean = s.rolling(window=30, min_periods=2).mean().bfill()
    roll_std = s.rolling(window=30, min_periods=2).std().bfill().fillna(1e-9)
    z_scores = ((s - roll_mean) / (roll_std + 1e-9)).values
    is_anomaly = np.abs(z_scores) > 2.5

    # Trend slope from last 180 days
    tail = min(180, n)
    slope_per_day = float(np.polyfit(np.arange(tail, dtype=float), currents[n - tail:], 1)[0])
    trend_slope_A_per_month = slope_per_day * 30.0

    last_val = float(currents[-1])
    if last_val >= baseline * 1.30:
        alert_level = "critical"
    elif last_val >= baseline * 1.20:
        alert_level = "warning"
    else:
        alert_level = "normal"

    # Anomaly count for the last 90 days
    anomaly_count_90d = int(is_anomaly[max(0, n - 90):].sum())

    history = [
        {
            "timestamp": str(ts.date()),
            "current_A": round(float(c), 3),
            "commutation_index": int(i + 1),
            "z_score": round(float(z), 4),
            "is_anomaly": bool(a),
        }
        for i, (ts, c, z, a) in enumerate(zip(ts_series, currents, z_scores, is_anomaly))
    ]

    return {
        "equipment_id": equipment,
        "baseline_current_A": baseline,
        "nominal_current_A": baseline,
        "history": history,
        "mean_per_commutation": round(float(currents.mean()), 2),
        "trend_slope_A_per_month": round(trend_slope_A_per_month, 3),
        "alert_level": alert_level,
        "anomaly_count_last_90d": anomaly_count_90d,
    }
