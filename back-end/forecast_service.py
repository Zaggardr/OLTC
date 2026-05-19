"""
Prophet forecast service — projects the Health Index `forecast_days` into the future,
detects threshold crossings, and emits a simulated SAP PM work order.
"""
from __future__ import annotations

import warnings
from functools import lru_cache
from typing import Any

import pandas as pd

warnings.filterwarnings("ignore")

try:
    from prophet import Prophet  # type: ignore[import-not-found]
    PROPHET_AVAILABLE = True
except ImportError:
    Prophet = None  # type: ignore[assignment, misc]
    PROPHET_AVAILABLE = False

from data_engine import get_dataset


ALERT_THRESHOLD = 60.0     # HI % — schedule preventive maintenance
CRITICAL_THRESHOLD = 40.0  # HI % — urgent planned outage


def _build_model() -> "Prophet":
    if not PROPHET_AVAILABLE:
        raise RuntimeError("Prophet is not installed.")
    model = Prophet(
        interval_width=0.80,
        changepoint_prior_scale=0.05,
        seasonality_mode="additive",
        daily_seasonality=False,
        weekly_seasonality=True,
        yearly_seasonality=False,
    )
    model.add_seasonality(name="monthly", period=30.5, fourier_order=5)
    return model


@lru_cache(maxsize=8)
def run_forecast(equipment: str, freq_hours: int = 6, forecast_days: int = 60) -> dict[str, Any]:
    if not PROPHET_AVAILABLE:
        raise RuntimeError("Prophet is not installed.")

    df = get_dataset(equipment, freq_hours=freq_hours)

    daily_hi = (
        df.set_index("timestamp")["health_index"]
        .resample("D").mean()
        .dropna()
        .reset_index()
    )
    daily_hi.columns = ["ds", "y"]

    model = _build_model()
    model.fit(daily_hi)

    future = model.make_future_dataframe(periods=forecast_days)
    forecast = model.predict(future)

    last_obs = daily_hi["ds"].max()
    future_only = forecast[forecast["ds"] > last_obs].copy()

    alert_rows = future_only[future_only["yhat"] < ALERT_THRESHOLD]
    crit_rows = future_only[future_only["yhat"] < CRITICAL_THRESHOLD]
    alert_date = alert_rows["ds"].min() if not alert_rows.empty else None
    crit_date = crit_rows["ds"].min() if not crit_rows.empty else None

    # Serialise the forecast columns we expose
    historical_payload = [
        {"date": row["ds"].date().isoformat(), "hi": float(row["y"])}
        for _, row in daily_hi.iterrows()
    ]
    forecast_payload = [
        {
            "date": row["ds"].date().isoformat(),
            "yhat": float(row["yhat"]),
            "yhat_lower": float(row["yhat_lower"]),
            "yhat_upper": float(row["yhat_upper"]),
            "is_future": bool(row["ds"] > last_obs),
        }
        for _, row in forecast.iterrows()
    ]

    work_order = _build_work_order(equipment, alert_date, crit_date)

    return {
        "equipment": equipment,
        "forecast_days": forecast_days,
        "last_observation": last_obs.date().isoformat(),
        "alert_threshold": ALERT_THRESHOLD,
        "critical_threshold": CRITICAL_THRESHOLD,
        "alert_date": alert_date.date().isoformat() if alert_date is not None else None,
        "critical_date": crit_date.date().isoformat() if crit_date is not None else None,
        "historical": historical_payload,
        "forecast": forecast_payload,
        "work_order": work_order,
    }


def _build_work_order(equipment: str, alert_date, crit_date) -> dict[str, Any] | None:
    if alert_date is None:
        return None
    days_to_alert = int((alert_date - pd.Timestamp.now()).days)
    urgent = days_to_alert <= 14
    return {
        "equipment": f"OLTC-{equipment}",
        "type": "urgent" if urgent else "preventive",
        "priority": "P1-URGENCE" if urgent else "P2-PLANIFIE",
        "alert_date": alert_date.date().isoformat(),
        "days_until_alert": days_to_alert,
        "critical_date": crit_date.date().isoformat() if crit_date is not None else None,
        "action": "Inspection OLTC + analyse DGA + mesure t_comm",
    }
