"""
Isolation Forest service — anomaly detection on OLTC sensor data.
Caches a trained model per (equipment, freq_hours, contamination) combination.
"""
from __future__ import annotations

from functools import lru_cache
from typing import Any

import pandas as pd
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler

from data_engine import (
    NOMINAL,
    FAULT_SCENARIOS,
    get_dataset,
)


FEATURES = list(NOMINAL.keys())
THRESHOLD_ALERT = 0.47
THRESHOLD_CRITICAL = 0.60


def _train_model(df: pd.DataFrame, contamination: float = 0.05, baseline_days: int = 90):
    scaler = StandardScaler()
    X = scaler.fit_transform(df[FEATURES])

    rows_per_day = max(1, len(df) // max(1, (df["timestamp"].max() - df["timestamp"].min()).days or 1))
    n_baseline = min(len(X), baseline_days * rows_per_day)
    X_baseline = X[:n_baseline]

    model = IsolationForest(
        n_estimators=200,
        contamination=contamination,
        max_samples="auto",
        random_state=42,
        n_jobs=-1,
    )
    model.fit(X_baseline)
    return model, scaler, X


@lru_cache(maxsize=8)
def compute_scored_dataset(equipment: str, freq_hours: int = 6,
                           contamination: float = 0.05) -> pd.DataFrame:
    """Returns the cached dataset with anomaly_score + is_anomaly columns."""
    df = get_dataset(equipment, freq_hours=freq_hours).copy()
    model, scaler, X = _train_model(df, contamination=contamination)
    df["anomaly_score"] = -model.decision_function(X)
    df["is_anomaly"] = model.predict(X) == -1
    return df


def detection_report(equipment: str, freq_hours: int = 6) -> list[dict[str, Any]]:
    """For each historical fault, when did IF first flag an anomaly within
    the pre-fault drift window? Uses sklearn's binary `is_anomaly` flag —
    the score threshold (0.60) is kept only for the score telemetry."""
    df = compute_scored_dataset(equipment, freq_hours=freq_hours).set_index("timestamp")
    rows = []
    for fk, fv in FAULT_SCENARIOS.items():
        fault_date = pd.Timestamp(fv["date"])
        # Look only inside the pre-fault drift window so we don't pick up
        # noise from years earlier.
        drift_start = fault_date - pd.Timedelta(days=fv["drift_weeks"] * 7)
        window = df.loc[drift_start:fault_date]
        anomalies = window[window["is_anomaly"]]
        if not anomalies.empty:
            first_alert = anomalies.index[0]
            days_advance = int((fault_date - first_alert).days)
            first_alert_str = first_alert.date().isoformat()
        else:
            first_alert_str = None
            days_advance = None
        rows.append({
            "fault_id": fk,
            "fault_date": fv["date"],
            "severity": fv["severity"],
            "first_alert": first_alert_str,
            "days_advance": days_advance,
            "max_score": float(window["anomaly_score"].max()) if not window.empty else 0.0,
            "description": fv["description"],
        })
    return sorted(rows, key=lambda r: r["fault_date"], reverse=True)
