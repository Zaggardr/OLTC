# OLTC Surveillance Prédictive

Application Streamlit de dashboard de surveillance temps réel pour la maintenance prédictive d'un OLTC (Changeur de Prises en Charge) de transformateur HT/THT.

## Run & Operate

Le dashboard a maintenant **deux processus** : le back-end FastAPI (port 8000) et le front-end Streamlit (port 5000). Les deux doivent tourner.

- `cd ../../../back-end && python main.py` — démarrer le back-end (port 8000)
- `cd artifacts/oltc-dashboard && streamlit run app.py --server.port 5000` — démarrer le dashboard
- Workflow configuré : **OLTC Dashboard** (port 5000, webview)
- Si le back-end tourne ailleurs : `OLTC_API_URL=http://host:8000 streamlit run app.py`

## Stack

- Python 3.11 + Streamlit 1.57 (front) + FastAPI + uvicorn (back)
- httpx — client HTTP du dashboard vers l'API
- Pandas, NumPy — traitement local des séries renvoyées par l'API
- Scikit-learn — Isolation Forest (côté back-end + ré-entraînement live côté front)
- Prophet — prévision Health Index 60j (côté back-end)
- Plotly — visualisations interactives sombres (séries, jauges)
- Statsmodels — régressions OLS pour scatter plots

## Where things live

- `back-end/main.py` — FastAPI app (9 routes)
- `back-end/data_engine.py` — source de vérité simulation + Health Index
- `back-end/anomaly_service.py` — Isolation Forest + rapport de détection
- `back-end/forecast_service.py` — Prophet + ordre SAP PM
- `artifacts/oltc-dashboard/app.py` — Dashboard Streamlit (Power BI dark SCADA)
- `artifacts/oltc-dashboard/api_client.py` — Client HTTP vers FastAPI (drop-in pour l'ancien data_generator)
- `artifacts/oltc-dashboard/data_generator.py` — Module legacy conservé pour référence, plus utilisé par le dashboard
- `artifacts/oltc-dashboard/.streamlit/config.toml` — Configuration serveur Streamlit

## Architecture decisions

- Architecture **2 tiers** : FastAPI back-end (calculs + simulation) + Streamlit front (UI). Le front fait tout via HTTP.
- Données 100% simulées côté back-end — génération reproductible via seed NumPy par équipement
- Dérive de pannes : progression quadratique pré-panne, récupération exponentielle post-panne (τ=30j)
- Health Index pondéré (H₂: 35%, temp: 20%, vibrations: 20%, t_comm: 15%, Vcc: 10%)
- Isolation Forest avec 5% contamination et 200 estimateurs — entraîné côté back-end et mis en cache par équipement
- Rapport de détection IF utilise la classification binaire `is_anomaly` (pas un seuil de score fragile)
- Cache : `lru_cache` côté back-end pour les datasets, `st.cache_data(ttl=3600)` côté front pour les chargements
- API URL configurable via env var `OLTC_API_URL` (default `http://localhost:8000`)

## Product

- **Tableau de Bord** : jauge HI globale, KPI capteurs temps réel, évolution HI, profil radar normalisé
- **Analyse Temporelle** : séries temporelles par capteur avec seuils d'alerte et marqueurs pannes
- **Détection d'Anomalies** : scatter Isolation Forest, matrice de corrélation, histogramme des scores
- **Historique Pannes** : timeline P1-P6, fiches détaillées, statistiques de sévérité
- **Comparaison Paramètres** : analyse par période de panne, scatter plots de corrélation avec régression

## User preferences

- Application en français
- Données simulées intégrées au code (pas de fichiers CSV externes)
- Architecture IoT + IA pour maintenance prédictive industrielle

## Gotchas

- Le back-end doit tourner AVANT le dashboard, sinon le dashboard affiche un banner d'erreur API et `st.stop()`
- Si le back-end tourne sur un autre host, exporter `OLTC_API_URL` avant `streamlit run`
- `statsmodels` est requis pour `trendline="ols"` dans Plotly (installé séparément)
- Mode temps réel : `st.rerun()` uniquement (pas `experimental_rerun`)
- Prophet : ~2 min à installer la 1re fois (cmdstanpy + dépendances Stan)
- Le bootstrap api_client tente automatiquement de re-fetch les métadonnées si la 1re connexion a échoué

## Pointers

- FAULT_SCENARIOS dans `back-end/data_engine.py` — source de vérité pour les 6 pannes historiques
- NOMINAL et THRESHOLDS dans `back-end/data_engine.py` — paramètres de référence des capteurs
- Docs API interactives : <http://localhost:8000/docs> (Swagger) ou <http://localhost:8000/redoc>
- Voir le skill `pnpm-workspace` pour la structure workspace Node.js (API server séparé)
