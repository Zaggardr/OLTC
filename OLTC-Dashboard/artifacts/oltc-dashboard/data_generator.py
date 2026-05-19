import numpy as np
import pandas as pd
from datetime import datetime, timedelta

NOMINAL = {
    "dga_h2_ppm": {"mean": 30, "std": 8},
    "temp_huile_c": {"mean": 65, "std": 3},
    "vib_ms2": {"mean": 1.2, "std": 0.3},
    "t_comm_ms": {"mean": 68, "std": 5},
    "vcc_v": {"mean": 125, "std": 1.5},
}

FAULT_SCENARIOS = {
    "P1": {
        "date": "2022-07-27",
        "drift_weeks": 8,
        "params": {"dga_h2_ppm": 3.5, "temp_huile_c": 0.8, "vib_ms2": 0.6, "t_comm_ms": 1.8},
        "description": "Décomposition thermique de l'huile — arc électrique",
        "severity": "Critique",
        "mwh_lost": 24.5,
    },
    "P2": {
        "date": "2022-02-20",
        "drift_weeks": 6,
        "params": {"dga_h2_ppm": 2.0, "t_comm_ms": 2.2},
        "description": "Usure mécanique des contacts — commutation lente",
        "severity": "Majeur",
        "mwh_lost": 8.2,
    },
    "P3": {
        "date": "2021-12-14",
        "drift_weeks": 4,
        "params": {"vib_ms2": 0.8, "t_comm_ms": 2.5},
        "description": "Desserrage mécanique — vibrations anormales",
        "severity": "Majeur",
        "mwh_lost": 11.3,
    },
    "P4": {
        "date": "2019-09-30",
        "drift_weeks": 4,
        "params": {"t_comm_ms": 3.0},
        "description": "Dégradation ressort de commutation",
        "severity": "Mineur",
        "mwh_lost": 3.1,
    },
    "P5": {
        "date": "2017-07-01",
        "drift_weeks": 10,
        "params": {"dga_h2_ppm": 4.0, "temp_huile_c": 1.5, "vib_ms2": 0.9},
        "description": "Surchauffe chronique — contamination huile",
        "severity": "Critique",
        "mwh_lost": 16.0,
    },
    "P6": {
        "date": "2017-08-09",
        "drift_weeks": 2,
        "params": {"vcc_v": -3.5},
        "description": "Défaut alimentation commande 127 Vcc",
        "severity": "Mineur",
        "mwh_lost": 2.8,
    },
}

THRESHOLDS = {
    "dga_h2_ppm":  {"nominal": 50,  "alarm": 100, "trip": 300,  "warn": 50,  "critical": 100},
    "temp_huile_c": {"nominal": 70,  "alarm": 80,  "trip": 95,   "warn": 75,  "critical": 85},
    "vib_ms2":     {"nominal": 2.0, "alarm": 4.0, "trip": 7.0,  "warn": 2.0, "critical": 3.0},
    "t_comm_ms":   {"nominal": 80,  "alarm": 100, "trip": 150,  "warn": 80,  "critical": 95},
    "vcc_v":       {"nominal": 115, "alarm": 110, "trip": 95,   "warn": 122, "critical": 119},
}

PARAM_LABELS = {
    "dga_h2_ppm":  "DGA H₂",
    "temp_huile_c": "Temp. Huile",
    "vib_ms2":     "Vibrations",
    "t_comm_ms":   "Temps Comm.",
    "vcc_v":       "Tension 127 Vcc",
}

PARAM_ICONS = {
    "dga_h2_ppm":  "🔥",
    "temp_huile_c": "🌡️",
    "vib_ms2":     "📳",
    "t_comm_ms":   "⏱️",
    "vcc_v":       "⚡",
}

PARAM_UNITS = {
    "dga_h2_ppm":  "ppm",
    "temp_huile_c": "°C",
    "vib_ms2":     "m/s²",
    "t_comm_ms":   "ms",
    "vcc_v":       "V",
}

EQUIPMENT_LIST = ["PJ1", "PJ2", "PJ3", "PJ11"]
EQUIPMENT_SEEDS = {"PJ1": 42, "PJ2": 43, "PJ3": 44, "PJ11": 45}

PERIOD_OPTIONS = {
    "Dernière heure": timedelta(hours=1),
    "24h": timedelta(hours=24),
    "7j": timedelta(days=7),
    "30j": timedelta(days=30),
    "1 an": timedelta(days=365),
    "Tout": None,
}


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
    weights = {
        "dga_h2_ppm": 0.35,
        "temp_huile_c": 0.20,
        "vib_ms2": 0.20,
        "t_comm_ms": 0.15,
        "vcc_v": 0.10,
    }
    scores = pd.DataFrame(index=df.index)
    for param, w in weights.items():
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


def generate_historical_data(
    start_date: str = "2017-01-01",
    end_date: str = None,
    freq_hours: int = 6,
    seed: int = 42,
) -> pd.DataFrame:
    np.random.seed(seed)
    if end_date is None:
        end_date = datetime.now().strftime("%Y-%m-%d")

    timestamps = pd.date_range(start=start_date, end=end_date, freq=f"{freq_hours}h")
    n = len(timestamps)
    data = {"timestamp": timestamps}
    for param, stats in NOMINAL.items():
        noise = np.random.normal(0, 1, n)
        seasonal = stats["std"] * 0.3 * np.sin(2 * np.pi * np.arange(n) / (365 * 24 / freq_hours))
        data[param] = stats["mean"] + stats["std"] * noise + seasonal

    df = pd.DataFrame(data)
    df["dga_h2_ppm"] = df["dga_h2_ppm"].clip(lower=0)
    df["vib_ms2"] = df["vib_ms2"].clip(lower=0.1)
    df["vcc_v"] = df["vcc_v"].clip(lower=115)
    df["t_comm_ms"] = df["t_comm_ms"].clip(lower=55)

    for fault_key in FAULT_SCENARIOS:
        df = _apply_fault_drift(df, fault_key)

    df = _compute_health_index(df)
    return df.reset_index(drop=True)


def compute_anomaly_scores(df: pd.DataFrame) -> pd.DataFrame:
    from sklearn.ensemble import IsolationForest
    from sklearn.preprocessing import StandardScaler

    features = list(NOMINAL.keys())
    X = df[features].values
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    clf = IsolationForest(n_estimators=200, contamination=0.05, random_state=42, n_jobs=-1)
    clf.fit(X_scaled)
    scores = clf.decision_function(X_scaled)
    labels = clf.predict(X_scaled)
    df = df.copy()
    df["anomaly_score"] = -scores
    df["is_anomaly"] = labels == -1
    return df


def get_status(hi: float) -> tuple:
    if hi >= 80:
        return "Normal", "🟢"
    elif hi >= 60:
        return "Surveillance", "🟡"
    elif hi >= 40:
        return "Alerte", "🟠"
    else:
        return "Critique", "🔴"


def get_param_status(param: str, value: float) -> str:
    warn = THRESHOLDS[param]["warn"]
    crit = THRESHOLDS[param]["critical"]
    if param == "vcc_v":
        if value >= warn:
            return "normal"
        elif value >= crit:
            return "warning"
        else:
            return "critical"
    else:
        if value <= warn:
            return "normal"
        elif value <= crit:
            return "warning"
        else:
            return "critical"


def generate_realtime_sample(last_row: pd.Series = None, seed: int = None) -> dict:
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
    return sample
