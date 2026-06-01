"""Pydantic response models for the OLTC API."""
from __future__ import annotations

from datetime import datetime
from typing import Literal, Optional

from pydantic import BaseModel, Field


class HealthResponse(BaseModel):
    status: Literal["ok"] = "ok"
    version: str
    prophet_available: bool


class EquipmentInfo(BaseModel):
    id: str
    seed: int


class SensorThresholds(BaseModel):
    nominal: float
    alarm: float
    trip: float
    warn: float
    critical: float


class SensorInfo(BaseModel):
    id: str
    label: str
    unit: str
    icon: str
    nominal_mean: float
    nominal_std: float
    weight: float
    thresholds: SensorThresholds


class FaultInfo(BaseModel):
    id: str
    date: str
    severity: Literal["Critique", "Majeur", "Mineur"]
    mwh_lost: float
    drift_weeks: int
    description: str
    params: dict[str, float]


class MeasurementPoint(BaseModel):
    timestamp: datetime
    dga_h2_ppm: float
    dga_c2h2_ppm: float
    dga_c2h4_ppm: float
    temp_huile_c: float
    vib_ms2: float
    t_comm_ms: float
    vcc_v: float
    health_index: float


class ScoredMeasurement(MeasurementPoint):
    anomaly_score: float
    is_anomaly: bool


class HistoricalResponse(BaseModel):
    equipment: str
    freq_hours: int
    period: str
    count: int
    points: list[ScoredMeasurement]


class LatestPoint(BaseModel):
    timestamp: datetime
    dga_h2_ppm: float
    dga_c2h2_ppm: float
    dga_c2h4_ppm: float
    temp_huile_c: float
    vib_ms2: float
    t_comm_ms: float
    vcc_v: float
    health_index: float
    status: Literal["Normal", "Surveillance", "Alerte", "Critique"]


class RealtimeSample(LatestPoint):
    anomaly_score: Optional[float] = None


class DetectionRow(BaseModel):
    fault_id: str
    fault_date: str
    severity: str
    first_alert: Optional[str]
    days_advance: Optional[int]
    max_score: float
    description: str


class DetectionReport(BaseModel):
    equipment: str
    threshold_critical: float
    rows: list[DetectionRow]


class ForecastPoint(BaseModel):
    date: str
    yhat: float
    yhat_lower: float
    yhat_upper: float
    is_future: bool


class WorkOrder(BaseModel):
    equipment: str
    type: Literal["urgent", "preventive"]
    priority: str
    alert_date: str
    days_until_alert: int
    critical_date: Optional[str]
    action: str
    sap_pm_order: Optional[str] = None


class ForecastResponse(BaseModel):
    equipment: str
    forecast_days: int
    last_observation: str
    alert_threshold: float
    critical_threshold: float
    alert_date: Optional[str]
    critical_date: Optional[str]
    historical: list[dict]
    forecast: list[ForecastPoint]
    work_order: Optional[WorkOrder]


class MotorCurrentPoint(BaseModel):
    timestamp: str
    current_A: float
    commutation_index: int
    z_score: float
    is_anomaly: bool


class MotorCurrentResponse(BaseModel):
    equipment_id: str
    baseline_current_A: float
    nominal_current_A: float
    history: list[MotorCurrentPoint]
    mean_per_commutation: float
    trend_slope_A_per_month: float
    alert_level: Literal["normal", "warning", "critical"]
    anomaly_count_last_90d: int


class DiagnosticIndicator(BaseModel):
    id: str
    name: str
    value: float
    unit: str
    status: Literal["normal", "warning", "critical"]
    normal_range: str
    alert_range: str
    critical_range: str
    norm: str
    note: Optional[str] = ""
    note_color: Optional[str] = "muted"


class DiagnosticReport(BaseModel):
    equipment: str
    indicators: list[DiagnosticIndicator]
    global_status: Literal["normal", "warning", "critical"]
    warning_count: int
    critical_count: int
