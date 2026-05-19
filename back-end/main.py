"""
OLTC Predictive Maintenance API — FastAPI back-end.

Exposes the data engine, Isolation Forest anomaly scoring, and Prophet forecasting
as HTTP endpoints for the Streamlit dashboard (or any other client).

Run:
    uvicorn main:app --host 0.0.0.0 --port 8000 --reload
"""
from __future__ import annotations

from typing import Optional

from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware

import data_engine as eng
import anomaly_service as anomaly
import forecast_service as forecast
import schemas as S


API_VERSION = "1.0.0"


app = FastAPI(
    title="OLTC Predictive Maintenance API",
    version=API_VERSION,
    description="HT/THT transformer OLTC monitoring — Health Index, Isolation Forest, Prophet forecasting.",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ─────────────────────────────────────────────────────────────────────────────
# Health
# ─────────────────────────────────────────────────────────────────────────────
@app.get("/api/health", response_model=S.HealthResponse, tags=["meta"])
def health():
    return {
        "status": "ok",
        "version": API_VERSION,
        "prophet_available": forecast.PROPHET_AVAILABLE,
    }


# ─────────────────────────────────────────────────────────────────────────────
# Metadata
# ─────────────────────────────────────────────────────────────────────────────
@app.get("/api/equipment", response_model=list[S.EquipmentInfo], tags=["metadata"])
def list_equipment():
    return [{"id": eq, "seed": eng.EQUIPMENT_SEEDS[eq]} for eq in eng.EQUIPMENT_LIST]


@app.get("/api/sensors", response_model=list[S.SensorInfo], tags=["metadata"])
def list_sensors():
    return [
        {
            "id": pid,
            "label": eng.PARAM_LABELS[pid],
            "unit": eng.PARAM_UNITS[pid],
            "icon": eng.PARAM_ICONS[pid],
            "nominal_mean": eng.NOMINAL[pid]["mean"],
            "nominal_std": eng.NOMINAL[pid]["std"],
            "weight": eng.HI_WEIGHTS[pid],
            "thresholds": eng.THRESHOLDS[pid],
        }
        for pid in eng.NOMINAL
    ]


@app.get("/api/faults", response_model=list[S.FaultInfo], tags=["metadata"])
def list_faults():
    return [
        {
            "id": fid,
            "date": fv["date"],
            "severity": fv["severity"],
            "mwh_lost": fv["mwh_lost"],
            "drift_weeks": fv["drift_weeks"],
            "description": fv["description"],
            "params": fv["params"],
        }
        for fid, fv in eng.FAULT_SCENARIOS.items()
    ]


def _validate_equipment(equipment: str):
    if equipment not in eng.EQUIPMENT_LIST:
        raise HTTPException(
            status_code=404,
            detail=f"Unknown equipment '{equipment}'. Allowed: {eng.EQUIPMENT_LIST}",
        )


# ─────────────────────────────────────────────────────────────────────────────
# Historical data
# ─────────────────────────────────────────────────────────────────────────────
@app.get("/api/historical/{equipment}", response_model=S.HistoricalResponse, tags=["data"])
def historical(
    equipment: str,
    freq_hours: int = Query(6, ge=1, le=24),
    period: str = Query("all", regex="^(1h|24h|7d|30d|1y|all)$"),
):
    _validate_equipment(equipment)
    df = anomaly.compute_scored_dataset(equipment, freq_hours=freq_hours)
    df = eng.filter_by_period(df, period)
    points = df.to_dict(orient="records")
    return {
        "equipment": equipment,
        "freq_hours": freq_hours,
        "period": period,
        "count": len(points),
        "points": points,
    }


@app.get("/api/latest/{equipment}", response_model=S.LatestPoint, tags=["data"])
def latest(equipment: str, freq_hours: int = Query(6, ge=1, le=24)):
    _validate_equipment(equipment)
    df = eng.get_dataset(equipment, freq_hours=freq_hours)
    if df.empty:
        raise HTTPException(status_code=404, detail="No data available")
    row = df.iloc[-1].to_dict()
    row["status"] = eng.get_status(row["health_index"])
    return row


@app.get("/api/realtime/{equipment}/sample", response_model=S.RealtimeSample, tags=["data"])
def realtime_sample(
    equipment: str,
    seed: Optional[int] = Query(None),
    freq_hours: int = Query(6, ge=1, le=24),
):
    _validate_equipment(equipment)
    df = eng.get_dataset(equipment, freq_hours=freq_hours)
    last_row = df.iloc[-1].to_dict() if not df.empty else None
    sample = eng.generate_realtime_sample(last_row=last_row, seed=seed)
    return sample


# ─────────────────────────────────────────────────────────────────────────────
# Anomaly detection
# ─────────────────────────────────────────────────────────────────────────────
@app.get("/api/anomalies/{equipment}/report", response_model=S.DetectionReport, tags=["ml"])
def anomaly_detection_report(equipment: str, freq_hours: int = Query(6, ge=1, le=24)):
    _validate_equipment(equipment)
    rows = anomaly.detection_report(equipment, freq_hours=freq_hours)
    return {
        "equipment": equipment,
        "threshold_critical": anomaly.THRESHOLD_CRITICAL,
        "rows": rows,
    }


# ─────────────────────────────────────────────────────────────────────────────
# Forecast
# ─────────────────────────────────────────────────────────────────────────────
@app.get("/api/forecast/{equipment}", response_model=S.ForecastResponse, tags=["ml"])
def hi_forecast(
    equipment: str,
    days: int = Query(60, ge=7, le=365),
    freq_hours: int = Query(6, ge=1, le=24),
):
    _validate_equipment(equipment)
    if not forecast.PROPHET_AVAILABLE:
        raise HTTPException(
            status_code=503,
            detail="Prophet is not installed on this server. Run `pip install prophet`.",
        )
    return forecast.run_forecast(equipment, freq_hours=freq_hours, forecast_days=days)


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)
