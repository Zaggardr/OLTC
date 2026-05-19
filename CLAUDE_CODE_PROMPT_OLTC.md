# CLAUDE CODE — OLTC Predictive Maintenance Dashboard
# Paste this entire file as your first message in a Claude Code session (VS Code)

---

## CONTEXT — READ BEFORE WRITING ANY CODE

You are working on an existing PFE (final-year engineering project) for OCP · Jorf Lasfar.
The project monitors On-Load Tap Changers (OLTC) on HV/EHV power transformers.

**Existing stack (already built, do NOT rebuild from scratch):**
- Back-end : FastAPI on port 8000
  - `back-end/main.py`          — 9 REST routes
  - `back-end/data_engine.py`   — sensor simulation, Health Index, fault data
  - `back-end/anomaly_service.py` — Isolation Forest (200 trees, 5% contamination)
  - `back-end/forecast_service.py` — Prophet 60-day forecast
- Front-end : Streamlit on port 5000
  - `OLTC-Dashboard/artifacts/oltc-dashboard/app.py`      — main UI (~700 lines)
  - `OLTC-Dashboard/artifacts/oltc-dashboard/api_client.py` — HTTP client

**Design system (enforce strictly):**
- Background `#0E1117`, surfaces `#1A1F2E`, borders `#2D3548`
- Primary accent OCP green `#008542` / light `#00B358`
- Semantic: green OK / orange warning `#FFA500` / red critical `#FF4444` / blue info `#2196F3`
- All Plotly charts: transparent bg, glow-effect traces, subtle gridlines
- KPI tiles: Power BI style — colored top border, bold 30px tabular-nums value

**Real data available (hard-coded in `data_engine.py`):**
```
6 historical faults P1–P6 (2017–2022), total 62.46 MWh lost
4 OLTC units: PJ1 (1998, 350A, critical), PJ2, PJ3, PJ11
5 simulated sensors: DGA H₂, Temperature, Vibrations, t_commutation, Motor current
Isolation Forest detection advance: avg 36 days (max 62 days on PJ1)
```

---

## YOUR TASK

Extend the existing Streamlit dashboard by implementing the changes listed below.
Work file by file. Before editing any file, read it with the Read tool.
After each change, confirm what was modified and what remains.

---

### TASK 1 — New API endpoint: Motor Current detail
**File:** `back-end/data_engine.py` and `back-end/main.py`

Add a new endpoint `GET /api/motor/{equipment_id}` that returns:
```json
{
  "equipment_id": "PJ1",
  "baseline_current_A": 134,
  "nominal_current_A": 134,
  "history": [
    { "timestamp": "...", "current_A": 138.2, "commutation_index": 1, "z_score": 0.4, "is_anomaly": false }
  ],
  "mean_per_commutation": 137.5,
  "trend_slope_A_per_month": 0.8,
  "alert_level": "normal"  // "normal" | "warning" | "critical"
}
```

**Engineering logic for motor current (implement exactly this):**
- Baseline per unit from real specs:
  - PJ1 / PJ2: 350 A (JANSEN V III Y)
  - PJ3: unknown → use 200 A estimate
  - PJ11: 268 A (MR V III 350-Y-76-10)
  - PJ10: 134 A (MR V III 200Y)
- Alert thresholds (IEC / MR recommendations):
  - Normal   : I_motor < baseline × 1.20
  - Warning  : baseline × 1.20 ≤ I_motor < baseline × 1.30
  - Critical : I_motor ≥ baseline × 1.30
- Z-score anomaly: flag if |z| > 2.5 over a 30-commutation rolling window
- Simulate degradation trend: slow linear drift +0.3–1.2 A/month depending on unit age
  - PJ1 (1998, 26 yrs): +1.2 A/month
  - PJ2/PJ3 (unknown): +0.8 A/month
  - PJ11 (2007, 17 yrs): +0.5 A/month

**Why this KPI matters (include as a docstring in the endpoint):**
Motor current increases when: contacts are worn (higher resistance → more torque needed),
oil is too viscous (cold weather or degraded oil), mechanical components are seizing,
or the drive spring is weakening. A +30% current drift detected 4–8 weeks before failure
was the missing precursor for faults P1 and P4 (PJ15, thermal trips 2017 & 2020).

---

### TASK 2 — OLTC Fleet inventory page
**File:** `OLTC-Dashboard/artifacts/oltc-dashboard/app.py`

Add a new Streamlit tab labelled **"🔧 Parc OLTC"**.

Display a styled dataframe with these exact columns and values:

| Poste | Constructeur | Type OLTC | Courant (A) | Nb prises | Plage HT (kV) | Mise en service | Ancienneté (ans) | Criticité |
|---|---|---|---|---|---|---|---|---|
| PJ1 (CGE) | ALSTHOM | JANSEN V III Y | 350 | 21 | 72.45 → 53.55 | Inconnue | >25 | 🔴 Critique |
| PJ1 (Alcatel) | JANSEN | V III | 350 | 21 | 72.45 → 53.55 | 1998 | 26 | 🔴 Critique |
| PJ2 / PJ10 | ALSTHOM | JANSEN V III Y | 200 | 21 | 72.45 → 53.55 | Inconnue | >20 | ⚠️ Élevée |
| PJ3 | Non documenté | — | 200* | 21 | 72.45 → 53.55 | Inconnue | >20 | ⚠️ Élevée |
| PJ11 | MR | V III 350-Y-76-10 | 268 | 17 | 66 → 54 | 2007 | 17 | ⚠️ Élevée |
| PJ10 (FT) | MR | V III 200Y-76-16-19 | 134 | 17 | 66 → 54 | 2007 | 17 | 🟡 Modérée |

Below the table, add a Plotly bubble chart:
- X-axis: Ancienneté (years)
- Y-axis: Courant nominal (A)
- Bubble size: Puissance transformateur (MVA) — use 25, 25, 12.5, 18.75, 25, 12.5
- Color: criticité (red=Critique, orange=Élevée, yellow=Modérée)
- Hover: show all technical fields + arc energy ratio vs PJ10 baseline

Below the chart, display this arc energy formula with st.latex:
```
E_{arc} \propto I^2 \times t_{arc}
\quad \Rightarrow \quad
\frac{E_{arc}(350A)}{E_{arc}(134A)} = \left(\frac{350}{134}\right)^2 \approx 6.8\times
```
Then a callout box (styled div): "Un OLTC à 350 A accumule **6,8× plus d'énergie d'arc** par
commutation qu'un OLTC à 134 A — ce qui justifie sa priorité de surveillance absolue."

---

### TASK 3 — Motor current surveillance section
**File:** `OLTC-Dashboard/artifacts/oltc-dashboard/app.py`
**Requires:** Task 1 endpoint to be working first.

Add a new Streamlit tab **"⚡ Moteur OLTC"**.

Layout (use st.columns):
**Row 1 — 3 KPI tiles** (OCP green style):
- Courant moyen / commutation (A) — value from API + delta vs baseline
- Dérive mensuelle (A/mois) — trend slope from API + arrow indicator
- Anomalies détectées — count of is_anomaly=True in last 90 days

**Row 2 — Main chart (full width):**
Plotly line chart of motor current over time with:
- Blue trace: I_motor per commutation (raw)
- Orange dashed line: rolling 30-commutation mean
- Red horizontal line: critical threshold (baseline × 1.30)
- Orange horizontal line: warning threshold (baseline × 1.20)
- Green horizontal line: baseline
- Vertical red bands: known fault dates (P1–P6) with annotation
- Glow effect on the main trace (same style as existing dashboard)

**Row 3 — 2 columns:**
Left (60%): Z-score chart per commutation. Color bars: green |z|<2, orange 2≤|z|<2.5, red |z|≥2.5
Right (40%): Explanation card (styled st.markdown):
```
### Pourquoi surveiller le courant moteur ?

Le moteur de commande déplace le commutateur de charge via un réducteur
mécanique et un ressort accumulateur. Sa consommation en courant reflète
directement l'état mécanique de l'OLTC :

| Cause | Effet sur I_moteur |
|---|---|
| Contacts usés | +résistance friction → ↑ I |
| Huile visqueuse | +couple résistant → ↑ I |
| Ressort fatigué | Commutation plus lente → ↑ I × t |
| Engrenage dégradé | Pertes mécaniques → ↑ I |

⚠️ Une dérive de +20–30% du courant moteur précède en général la défaillance
de **4 à 8 semaines** — c'est le précurseur manquant des pannes P1 et P4.
```

---

### TASK 4 — Predictive maintenance thresholds page
**File:** `OLTC-Dashboard/artifacts/oltc-dashboard/app.py`

Add a new tab **"🔮 Prévision Prophet"** that calls `GET /api/forecast/{equipment_id}`.

Display:
1. Plotly chart: historical HI (solid line) + Prophet 60-day forecast (dashed) + 80% confidence band (filled area, low opacity). Mark the predicted date when HI crosses 60% (warning) and 40% (critical) with vertical dashed lines + annotation.
2. Below chart: SAP PM work order recommendation box — show the `sap_pm_order` field from the API response in a styled alert box.
3. Threshold reference table:

| Paramètre | Normal | Alerte ⚠️ | Critique 🔴 | Capteur |
|---|---|---|---|---|
| T° huile OLTC | < 70°C | 70–90°C | > 90°C | PT100 immergé |
| ΔT OLTC vs cuve | < 10°C | 10–20°C | > 20°C | PT100 différentiel |
| C₂H₂ dissous | < 5 ppm | > 30 ppm | > 100 ppm | DGA en ligne |
| H₂ dissous | < 50 ppm | > 150 ppm | > 300 ppm | DGA en ligne |
| t_commutation | < 100 ms | 100–150 ms | > 150 ms | Encodeur |
| Courant moteur | < I_base×1.2 | I_base×1.2–1.3 | > I_base×1.3 | TC effet Hall |
| Health Index | > 80% | 60–80% | < 60% | Composite |

---

### TASK 5 — Updated KPI bar (existing section, not new tab)
**File:** `OLTC-Dashboard/artifacts/oltc-dashboard/app.py`

In the existing top KPI row, ensure these 6 tiles are present (add missing ones):
1. **Health Index** — existing ✓
2. **Pannes totales** — count all P1–P6 (= 6)
3. **Pannes confirmées OLTC** — 2 (P5, P6 — défaut positionnement)
4. **Pannes probables OLTC** — 4 (P1, P2, P3, P4 — thermique / 87T)
5. **END totale** — 62.46 MWh (red tile)
6. **Disponibilité** — calculate as `(8760 - total_outage_hours) / 8760 * 100` for the monitored period

For tiles 2–4: add a small footnote below the value:
- Confirmée = défaut positionnement encodeur (courant homopolaire)
- Probable = déclenchement thermique / 87T lié à usure contacts

---

### TASK 6 — Code quality and integration checks

After completing Tasks 1–5:

1. In `back-end/main.py`: add the new `/api/motor/{equipment_id}` route to the OpenAPI tags, add a 404 guard if equipment_id not in ["PJ1","PJ2","PJ3","PJ11"].
2. In `api_client.py`: add `get_motor_data(equipment_id)` function with `st.cache_data(ttl=300)`.
3. Run a quick smoke test — print the output of:
   ```
   curl http://localhost:8000/api/motor/PJ1
   curl http://localhost:8000/api/motor/PJ99   # should return 404
   ```
4. Verify Streamlit still starts without error on port 5000 after changes.

---

## CONSTRAINTS (enforce throughout)

- Never invent data that contradicts the real fault record (P1–P6, dates, END values)
- All simulated values must be seeded (NumPy seed = hash of equipment_id) for reproducibility
- No new pip dependencies unless strictly necessary — existing stack: fastapi, uvicorn, streamlit, plotly, pandas, numpy, scikit-learn, prophet, httpx
- Keep every new Plotly chart consistent with the dark SCADA theme: `paper_bgcolor="rgba(0,0,0,0)"`, `plot_bgcolor="rgba(0,0,0,0)"`, white axis text
- Every new section must have a collapsible `st.expander("ℹ️ Pourquoi ce KPI ?")` with the engineering justification in plain French

## STARTUP REMINDER

```powershell
# Terminal 1
cd c:\Users\User\Desktop\Aicha\back-end
python main.py

# Terminal 2
cd c:\Users\User\Desktop\Aicha\OLTC-Dashboard\artifacts\oltc-dashboard
python -m streamlit run app.py --server.port 5000
```

Start with Task 1. Read `back-end/data_engine.py` first before writing any code.
