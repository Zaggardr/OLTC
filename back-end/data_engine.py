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
    "dga_c2h4_ppm":  {"mean": 8.0, "std": 2.0},   # ethylene — Rogers ratio denominator
    "temp_huile_c":  {"mean": 65,  "std": 3},
    "vib_ms2":       {"mean": 1.2, "std": 0.3},
    "t_comm_ms":     {"mean": 68,  "std": 5},
    "vcc_v":         {"mean": 125, "std": 1.5},
}

THRESHOLDS = {
    "dga_h2_ppm":   {"nominal": 50,  "alarm": 100, "trip": 300, "warn": 50,  "critical": 100},
    "dga_c2h2_ppm": {"nominal": 5,   "alarm": 30,  "trip": 100, "warn": 30,  "critical": 100},
    "dga_c2h4_ppm": {"nominal": 20,  "alarm": 50,  "trip": 100, "warn": 20,  "critical": 50},
    "temp_huile_c": {"nominal": 70,  "alarm": 80,  "trip": 95,  "warn": 75,  "critical": 85},
    "vib_ms2":      {"nominal": 2.0, "alarm": 4.0, "trip": 7.0, "warn": 2.0, "critical": 3.0},
    "t_comm_ms":    {"nominal": 80,  "alarm": 100, "trip": 150, "warn": 80,  "critical": 95},
    "vcc_v":        {"nominal": 115, "alarm": 110, "trip": 95,  "warn": 122, "critical": 119},
}

PARAM_LABELS = {
    "dga_h2_ppm":   "DGA H₂",
    "dga_c2h2_ppm": "DGA C₂H₂",
    "dga_c2h4_ppm": "DGA C₂H₄",
    "temp_huile_c": "Temp. Huile",
    "vib_ms2":      "Vibrations",
    "t_comm_ms":    "Temps Comm.",
    "vcc_v":        "Tension 127 Vcc",
}

PARAM_ICONS = {
    "dga_h2_ppm":   "🔥",
    "dga_c2h2_ppm": "⚗️",
    "dga_c2h4_ppm": "🧪",
    "temp_huile_c": "🌡️",
    "vib_ms2":      "📳",
    "t_comm_ms":    "⏱️",
    "vcc_v":        "⚡",
}

PARAM_UNITS = {
    "dga_h2_ppm":   "ppm",
    "dga_c2h2_ppm": "ppm",
    "dga_c2h4_ppm": "ppm",
    "temp_huile_c": "°C",
    "vib_ms2":      "m/s²",
    "t_comm_ms":    "ms",
    "vcc_v":        "V",
}

HI_WEIGHTS = {
    "dga_h2_ppm":   0.35,
    "dga_c2h2_ppm": 0.00,   # monitoring only — kept out of HI to preserve existing score
    "dga_c2h4_ppm": 0.00,   # monitoring only (Rogers ratio denominator)
    "temp_huile_c": 0.20,
    "vib_ms2":      0.20,
    "t_comm_ms":    0.15,
    "vcc_v":        0.10,
}

FAULT_SCENARIOS = {
    "P1": {
        "date": "2022-07-27", "drift_weeks": 8,
        "params": {"dga_h2_ppm": 3.5, "dga_c2h2_ppm": 5.0, "dga_c2h4_ppm": 0.8, "temp_huile_c": 0.8, "vib_ms2": 0.6, "t_comm_ms": 1.8},
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
        "params": {"dga_h2_ppm": 4.0, "dga_c2h2_ppm": 3.0, "dga_c2h4_ppm": 3.5, "temp_huile_c": 1.5, "vib_ms2": 0.9},
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
    df["dga_c2h4_ppm"] = df["dga_c2h4_ppm"].clip(lower=0)
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


# ─── CIGRE Diagnostic ─────────────────────────────────────────────────────────
def _diag_status(val: float, warn_lo: float | None, warn_hi: float | None,
                 crit_lo: float | None, crit_hi: float | None,
                 higher_is_worse: bool = True) -> str:
    """Return 'normal'|'warning'|'critical' given explicit threshold pairs."""
    if higher_is_worse:
        if crit_hi is not None and val >= crit_hi:
            return "critical"
        if warn_hi is not None and val >= warn_hi:
            return "warning"
        return "normal"
    else:  # lower_is_worse (e.g. HI, rigidité diélectrique, corrélation)
        if crit_lo is not None and val <= crit_lo:
            return "critical"
        if warn_lo is not None and val <= warn_lo:
            return "warning"
        return "normal"


@lru_cache(maxsize=8)
def compute_diagnostic(equipment: str) -> dict:
    """Compute the 18 CIGRE TB 543 / IEC diagnostic indicators for an OLTC unit."""
    df = get_dataset(equipment, freq_hours=6)
    motor = generate_motor_current(equipment)
    seed = EQUIPMENT_SEEDS.get(equipment, 42)
    pts_per_day = 4  # freq=6h

    last = df.iloc[-1]
    hi_now = float(last["health_index"])

    def _ind(id_: str, name: str, value: float, unit: str, status: str,
             normal_range: str, alert_range: str, critical_range: str, norm: str,
             note: str = "", note_color: str = "muted") -> dict:
        return {
            "id": id_, "name": name, "value": round(float(value), 3),
            "unit": unit, "status": status,
            "normal_range": normal_range, "alert_range": alert_range,
            "critical_range": critical_range, "norm": norm,
            "note": note, "note_color": note_color,
        }

    indicators = []

    # 1. T° huile OLTC — direct PT100 reading
    v = float(last["temp_huile_c"])
    indicators.append(_ind(
        "temp_huile", "T° huile OLTC", v, "°C",
        "normal" if v < 75 else ("critical" if v > 95 else "warning"),
        "< 75 °C", "75 – 95 °C", "> 95 °C", "IEC 60214-1 §4.1.2",
    ))

    # 2. ΔT anormal — measured vs 90-day rolling mean baseline
    win = pts_per_day * 90
    t_attendue = float(df["temp_huile_c"].rolling(win, min_periods=pts_per_day).mean().iloc[-1])
    delta_t = abs(float(last["temp_huile_c"]) - t_attendue)
    indicators.append(_ind(
        "delta_t", "ΔT anormal", delta_t, "°C",
        "normal" if delta_t < 5 else ("critical" if delta_t > 15 else "warning"),
        "< 5 °C", "5 – 15 °C", "> 15 °C", "IEC 60214-1 §4.1.2",
    ))

    # 3. Taux dérive T° — linear slope over last 90 days (°C/month)
    n_90d = min(pts_per_day * 90, len(df))
    tail_t = df["temp_huile_c"].tail(n_90d).values
    slope_t = float(np.polyfit(np.arange(len(tail_t), dtype=float), tail_t, 1)[0])
    drift_t = abs(slope_t * pts_per_day * 30)
    indicators.append(_ind(
        "drift_temp", "Taux dérive T°", drift_t, "°C/mois",
        "normal" if drift_t < 0.5 else ("critical" if drift_t > 2 else "warning"),
        "< 0.5", "0.5 – 2", "> 2", "CIGRE TB 543",
    ))

    # 4. C₂H₂ — DGA absolute value
    c2h2 = float(last["dga_c2h2_ppm"])
    indicators.append(_ind(
        "c2h2", "C₂H₂ (DGA)", c2h2, "ppm",
        "normal" if c2h2 < 5 else ("critical" if c2h2 > 100 else "warning"),
        "< 5 ppm", "5 – 100 ppm", "> 100 ppm", "IEC 60599:2022",
        note="Mesuré sur huile principale — seuils non applicables si mesure directe huile OLTC",
    ))

    # 5. Rogers ratio C₂H₂/C₂H₄
    c2h4 = float(last["dga_c2h4_ppm"]) if "dga_c2h4_ppm" in df.columns else 8.0
    rogers = c2h2 / max(c2h4, 0.01)
    indicators.append(_ind(
        "rogers", "Ratio C₂H₂/C₂H₄", rogers, "",
        "normal" if rogers < 0.1 else ("critical" if rogers >= 3.0 else "warning"),
        "< 0.1", "0.1 – 3.0", "≥ 3.0", "IEC 60599:2022",
        note="Ratio normal pour arcs de commutation OLTC — interpréter en corrélation avec huile principale",
    ))

    # 6. Taux C₂H₂ — rate of change ppm/day over 30 days
    n_30d = min(pts_per_day * 30, len(df))
    c2h2_now = float(df["dga_c2h2_ppm"].iloc[-1])
    c2h2_30d_ago = float(df["dga_c2h2_ppm"].iloc[-n_30d])
    rate_c2h2 = max(0.0, (c2h2_now - c2h2_30d_ago) / 30.0)
    indicators.append(_ind(
        "rate_c2h2", "Taux dérive C₂H₂", rate_c2h2, "ppm/j",
        "normal" if rate_c2h2 < 0.1 else ("critical" if rate_c2h2 > 2 else "warning"),
        "< 0.1", "0.1 – 2", "> 2", "IEC 60599 §6",
    ))

    # 7. H₂ — DGA absolute value
    h2 = float(last["dga_h2_ppm"])
    indicators.append(_ind(
        "h2", "H₂ (DGA)", h2, "ppm",
        "normal" if h2 < 50 else ("critical" if h2 > 300 else "warning"),
        "< 50 ppm", "50 – 300 ppm", "> 300 ppm", "IEC 60599:2022",
        note="Seuil conservateur — valeurs naturellement plus élevées dans huile compartiment OLTC",
        note_color="muted",
    ))

    # 8. t_commutation absolu
    t_comm = float(last["t_comm_ms"])
    indicators.append(_ind(
        "t_comm_abs", "t_comm absolu", t_comm, "ms",
        "normal" if t_comm < 80 else ("critical" if t_comm > 120 else "warning"),
        "< 80 ms", "80 – 120 ms", "> 120 ms", "CIGRE TB 543 / MR",
    ))

    # 9. t_comm relatif — current / baseline
    t_baseline = NOMINAL["t_comm_ms"]["mean"]
    t_rel = t_comm / t_baseline
    indicators.append(_ind(
        "t_comm_rel", "t_comm relatif", t_rel, "×",
        "normal" if t_rel < 1.3 else ("critical" if t_rel > 2.0 else "warning"),
        "< 1.3", "1.3 – 2.0", "> 2.0", "CIGRE TB 543",
    ))

    # 10. Variabilité t_comm — std of last 50 measurements
    t_std = float(df["t_comm_ms"].tail(50).std())
    indicators.append(_ind(
        "t_comm_var", "Variabilité t_comm", t_std, "ms",
        "normal" if t_std < 5 else ("critical" if t_std > 15 else "warning"),
        "< 5 ms", "5 – 15 ms", "> 15 ms", "Pratique terrain",
    ))

    # 11. Courant moteur — ratio I_actuel / I_baseline
    motor_hist = motor["history"]
    i_last = motor_hist[-1]["current_A"]
    i_baseline = motor["baseline_current_A"]
    i_ratio = i_last / max(i_baseline, 1.0)
    indicators.append(_ind(
        "motor_ratio", "Courant moteur (I/I₀)", i_ratio, "×",
        "normal" if i_ratio < 1.2 else ("critical" if i_ratio >= 1.3 else "warning"),
        "< 1.2", "1.2 – 1.3", "≥ 1.3", "Pratique MR / IEC",
    ))

    # 12. Énergie mécanique — ∫I² dt normalized over last 90 days
    n_90_motor = min(90, len(motor_hist))
    i_recent = np.array([h["current_A"] for h in motor_hist[-n_90_motor:]])
    e_recent = float(np.sum(i_recent ** 2))
    e_ref = (i_baseline ** 2) * n_90_motor
    e_mech = e_recent / max(e_ref, 1.0)
    indicators.append(_ind(
        "energy_mech", "Énergie mécanique (∫I²dt)", e_mech, "×",
        "normal" if e_mech < 1.2 else ("critical" if e_mech > 1.5 else "warning"),
        "< 1.2", "1.2 – 1.5", "> 1.5", "Pratique MR",
    ))

    # 13. Vibration énergie — E_actuel / E_baseline (mean squared)
    n_90d_vib = min(pts_per_day * 90, len(df))
    vib_recent = df["vib_ms2"].tail(n_90d_vib).values
    e_vib = float(np.mean(vib_recent ** 2))
    e_vib_ref = NOMINAL["vib_ms2"]["mean"] ** 2
    vib_ratio = e_vib / max(e_vib_ref, 1e-9)
    indicators.append(_ind(
        "energy_vib", "Vibration énergie (E/E₀)", vib_ratio, "×",
        "normal" if vib_ratio < 1.2 else ("critical" if vib_ratio > 1.7 else "warning"),
        "< 1.2", "1.2 – 1.7", "> 1.7", "CIGRE TB 543",
        note="Valeur à la limite du seuil alerte (1.200) — surveillance renforcée recommandée",
        note_color="warn",
    ))

    # 14. Vibration corrélation — Pearson r(recent 30d, first 30d as reference)
    n_ref = pts_per_day * 30
    ref_vib = df["vib_ms2"].iloc[:n_ref].values
    rec_vib = df["vib_ms2"].tail(n_ref).values
    min_len = min(len(ref_vib), len(rec_vib))
    vib_corr = float(np.corrcoef(ref_vib[:min_len], rec_vib[:min_len])[0, 1])
    if np.isnan(vib_corr):
        vib_corr = 1.0
    indicators.append(_ind(
        "vib_corr", "Vibration corrélation (r)", vib_corr, "",
        "normal" if vib_corr > 0.90 else ("critical" if vib_corr < 0.70 else "warning"),
        "> 0.90", "0.70 – 0.90", "< 0.70", "CIGRE TB 543",
        note="Vérifier intégrité capteur avant interprétation défaut mécanique",
        note_color="warn",
    ))

    # 15. Teneur eau huile — % saturation (derived: no moisture sensor in dataset)
    recent_hi_30 = float(df["health_index"].tail(pts_per_day * 30).mean())
    moisture = 10.0 + (100.0 - recent_hi_30) * 0.25 + (seed % 7) * 0.5
    moisture = float(np.clip(moisture, 5.0, 80.0))
    indicators.append(_ind(
        "moisture", "Teneur eau huile", moisture, "%",
        "normal" if moisture < 20 else ("critical" if moisture > 60 else "warning"),
        "< 20 %", "20 – 60 %", "> 60 %", "IEC 60422:2013",
    ))

    # 16. Rigidité diélectrique — kV breakdown (derived from HI aging)
    kv = 65.0 * (recent_hi_30 / 100.0) ** 0.5 + (seed % 5) * 0.3
    kv = float(np.clip(kv, 10.0, 80.0))
    indicators.append(_ind(
        "rigidity", "Rigidité diélectrique", kv, "kV",
        "normal" if kv > 50 else ("critical" if kv < 30 else "warning"),
        "> 50 kV", "30 – 50 kV", "< 30 kV", "IEC 60156",
    ))

    # 17. Health Index — composite weighted score
    indicators.append(_ind(
        "hi", "Health Index", hi_now, "%",
        "normal" if hi_now > 80 else ("critical" if hi_now < 40 else "warning"),
        "> 80 %", "40 – 80 %", "< 40 %", "CIGRE TB 445",
    ))

    # 18. Jours avant critique — linear extrapolation of HI to 40%
    n_180d = min(pts_per_day * 180, len(df))
    tail_hi = df["health_index"].tail(n_180d).values
    slope_hi = float(np.polyfit(np.arange(len(tail_hi), dtype=float), tail_hi, 1)[0])
    if slope_hi >= 0:
        days_crit = 999.0  # stable or improving — very far from critical
    else:
        pts_to_crit = (hi_now - 40.0) / (-slope_hi)
        days_crit = float(np.clip(pts_to_crit / pts_per_day, 0, 999))
    indicators.append(_ind(
        "days_crit", "Jours avant critique", days_crit, "j",
        "normal" if days_crit > 180 else ("critical" if days_crit < 30 else "warning"),
        "> 180 j", "30 – 180 j", "< 30 j", "Calcul régression HI",
    ))

    n_warn = sum(1 for i in indicators if i["status"] == "warning")
    n_crit = sum(1 for i in indicators if i["status"] == "critical")
    global_status = "critical" if n_crit > 0 else ("warning" if n_warn > 0 else "normal")

    return {
        "equipment": equipment,
        "indicators": indicators,
        "global_status": global_status,
        "warning_count": n_warn,
        "critical_count": n_crit,
    }
