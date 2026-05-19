# OLTC Predictive Maintenance — Back-end (FastAPI)

REST API serving the OLTC predictive-maintenance stack: simulated IoT data,
weighted Health Index, Isolation Forest anomaly detection, and Prophet
forecasting.

## Run

```bash
pip install -r requirements.txt
python main.py
# or
uvicorn main:app --host 0.0.0.0 --port 8000 --reload
```

Interactive docs: <http://localhost:8000/docs>
OpenAPI schema: <http://localhost:8000/openapi.json>

## Endpoints

| Method | Path | Description |
|--------|------|-------------|
| GET | `/api/health` | Service health + Prophet availability |
| GET | `/api/equipment` | List equipment IDs |
| GET | `/api/sensors` | Sensor metadata (labels, units, thresholds, weights) |
| GET | `/api/faults` | Historical fault scenarios P1–P6 |
| GET | `/api/historical/{eq}?freq_hours=6&period=all` | Full series with HI + IF |
| GET | `/api/latest/{eq}` | Most recent measurement + status |
| GET | `/api/realtime/{eq}/sample?seed=…` | Synthesised next realtime sample |
| GET | `/api/anomalies/{eq}/report` | IF detection lead-time per fault |
| GET | `/api/forecast/{eq}?days=60` | Prophet forecast + SAP PM work order |

`{eq}` ∈ `PJ1`, `PJ2`, `PJ3`, `PJ11`
`period` ∈ `1h`, `24h`, `7d`, `30d`, `1y`, `all`

## Caching

`get_dataset`, `compute_scored_dataset`, and `run_forecast` are cached with
`functools.lru_cache`, so subsequent calls for the same `(equipment, freq_hours)`
combination are instantaneous. Restart the server (or call from a fresh process)
to refresh.

## Files

- `data_engine.py` — single source of truth for sensors, thresholds, faults, HI
- `anomaly_service.py` — trained Isolation Forest + detection report
- `forecast_service.py` — Prophet model + SAP PM work-order generation
- `schemas.py` — Pydantic response models
- `main.py` — FastAPI app
- `isolation_forest.py`, `prophet_forecast.py` — legacy standalone scripts (matplotlib)
