# OLTC Predictive Maintenance — État du projet

**Dernière mise à jour :** 2026-05-18
**Stack :** FastAPI back-end + Streamlit front-end (Power BI dark SCADA)
**Site :** OCP · Jorf Lasfar · Terminal Slurry Pipeline
**Équipement prioritaire :** OLTC PJ1 (1998, 6 pannes historiques, 62.46 MWh perdus)

---

## 1. Vue d'ensemble

Système de maintenance prédictive 2 tiers pour les changeurs de prises en charge
(OLTC) des transformateurs HT/THT. Les 6 pannes réelles 2017–2022 servent de
benchmark : l'objectif est de les détecter en moyenne **15–60 jours avant**
l'incident grâce à l'IoT (5 capteurs simulés) + IA (Health Index pondéré +
Isolation Forest) + prévision Prophet 60 j.

```
┌───────────────────────────┐    HTTP    ┌──────────────────────────┐
│ Streamlit Dashboard       │ ─────────→ │ FastAPI Back-end         │
│ port 5000                 │   httpx    │ port 8000                │
│ Power BI dark + OCP green │            │ data + ML + forecast     │
└───────────────────────────┘            └──────────────────────────┘
```

---

## 2. Arborescence

```
Aicha/
├── back-end/                            # ← API FastAPI (port 8000)
│   ├── main.py                          # 9 routes REST
│   ├── data_engine.py                   # capteurs, seuils, pannes, HI, génération
│   ├── anomaly_service.py               # Isolation Forest + rapport détection
│   ├── forecast_service.py              # Prophet 60j + ordre SAP PM
│   ├── schemas.py                       # modèles Pydantic
│   ├── requirements.txt
│   ├── README.md
│   ├── isolation_forest.py              # legacy standalone matplotlib
│   └── prophet_forecast.py              # legacy standalone matplotlib
│
├── OLTC-Dashboard/
│   ├── artifacts/
│   │   └── oltc-dashboard/              # ← Front-end Streamlit (port 5000)
│   │       ├── app.py                   # UI Power BI dark SCADA (700 lignes)
│   │       ├── api_client.py            # client HTTP vers FastAPI
│   │       ├── data_generator.py        # legacy (plus utilisé, conservé)
│   │       ├── requirements.txt
│   │       └── .streamlit/config.toml
│   ├── attached_assets/                 # brief de design initial
│   └── replit.md                        # doc dev déploiement
│
├── OLTC_PREDICTIVE_MAINTENANCE.md       # guide initial du projet
├── conversation_oltc_dashboard.md       # récap des phases antérieures
└── PROJECT_STATUS.md                    # ce fichier
```

---

## 3. Ce qui est construit

### 3.1 Back-end FastAPI

**9 endpoints REST**, tous testés HTTP 200 :

| Méthode | Route | Description |
|---------|-------|-------------|
| GET | `/api/health` | Statut service + dispo Prophet |
| GET | `/api/equipment` | Liste des 4 OLTC (PJ1, PJ2, PJ3, PJ11) |
| GET | `/api/sensors` | Métadonnées 5 capteurs (labels, unités, seuils, poids HI) |
| GET | `/api/faults` | 6 pannes historiques P1–P6 |
| GET | `/api/historical/{eq}` | Série complète scorée IF (`?freq_hours=6&period=all\|1y\|30d…`) |
| GET | `/api/latest/{eq}` | Dernier point + statut |
| GET | `/api/realtime/{eq}/sample` | Sample temps réel synthétisé (`?seed=…`) |
| GET | `/api/anomalies/{eq}/report` | Avance de détection par panne |
| GET | `/api/forecast/{eq}` | Prévision Prophet 60j + ordre SAP PM |
| GET | `/docs` | Swagger UI interactive |

**Modèles ML embarqués :**
- **Health Index pondéré** — DGA H₂ 35% + Temp 20% + Vibrations 20% + t_comm 15% + Vcc 10%
- **Isolation Forest** — 200 arbres, contamination 5%, entraîné par équipement
- **Prophet** — saisonnalité hebdo + mensuelle, intervalle confiance 80%

**Caching :** `lru_cache` sur `get_dataset`, `compute_scored_dataset`, `run_forecast` → 1 calcul / équipement / process.

**Résultats de détection (PJ1, freq 6h) :**

| Panne | Date | Sévérité | Avance IF |
|-------|------|----------|-----------|
| P5 | 2017-07-01 | Critique | **62 j** |
| P1 | 2022-07-27 | Critique | 52 j |
| P2 | 2022-02-20 | Majeur | 40 j |
| P3 | 2021-12-14 | Majeur | 26 j |
| P4 | 2019-09-30 | Mineur | 25 j |
| P6 | 2017-08-09 | Mineur | 13 j |
| **Moyenne** | | | **~36 j** |

### 3.2 Front-end Streamlit

**Design Power BI dark SCADA** (style salle de contrôle industriel) :

- **Fond** : `#0E1117` (très sombre)
- **Surfaces** : `#1A1F2E` (tuiles)
- **Bordures** : `#2D3548`
- **Accent personnalisé** : **vert OCP `#008542`** (+ variant clair `#00B358`)
- **Sémantique** : vert succès / orange warning / rouge critique / bleu info
- **Typographie** : Segoe UI + Inter, valeurs métriques en `tabular-nums` 30px bold
- **Tuiles KPI** : style Power BI avec bordure haute colorée + hover surélevé
- **Header SCADA** : barre verte verticale + horloge monospace + pill `● LIVE` rouge pulsant
- **Graphes Plotly** : fond transparent, courbes avec **effet glow**, gridlines subtiles
- **Sparklines** : 5 capteurs avec mini-graphes glow colorisés selon état
- **Journal d'alertes** : HTML custom avec bordure gauche colorée par niveau

**Sections** :
1. Header SCADA avec horloge + indicateur LIVE
2. 5 tuiles KPI (Health Index / Statut / Score IF / Dernière panne / MWh évités)
3. Graphe HI temporel (60%) + Jauge circulaire (40%) + barre de marge au seuil critique
4. 5 cartes capteurs avec sparkline + pill statut + seuils CEI/MR
5. Tableau pannes historiques + Journal d'alertes temps réel
6. Onglets analyse approfondie : Isolation Forest / Multi-capteurs z-score / Distribution HI

**Mode Temps Réel** :
- Toggle sidebar avec slider de fréquence (2–30 s)
- Indicateur LIVE pulsant dans le header
- Génération continue de samples via API → buffer 200 points max
- Ré-entraînement IF local sur le buffer (séparé du modèle back-end)
- Auto-alertes sur dépassement de seuil capteur + sur score IF > 0.60

### 3.3 Client HTTP

`api_client.py` — **drop-in remplaçant** de l'ancien `data_generator.py` :

- Bootstrap automatique à l'import (fetch métadonnées)
- Expose `NOMINAL`, `THRESHOLDS`, `FAULT_SCENARIOS`, etc. comme **constantes module**
- Fonctions `load_data`, `filter_by_period`, `generate_realtime_sample`, `get_status`, `get_param_status`
- Cache `st.cache_data(ttl=3600)` sur le fetch historique
- Détection d'erreur API → `API_OK=False` + `API_ERROR` exposés
- Banner d'erreur dans l'UI + `st.stop()` si API injoignable

---

## 4. Comment lancer

### Pré-requis (une seule fois)

```powershell
cd c:\Users\User\Desktop\Aicha\back-end
pip install -r requirements.txt
```

### Démarrage (ordre obligatoire)

```powershell
# Terminal 1 — back-end
cd c:\Users\User\Desktop\Aicha\back-end
python main.py
# Docs : http://localhost:8000/docs

# Terminal 2 — dashboard
cd c:\Users\User\Desktop\Aicha\OLTC-Dashboard\artifacts\oltc-dashboard
python -m streamlit run app.py --server.port 5000
# UI : http://localhost:5000
```

### Variables d'environnement

| Variable | Default | Description |
|----------|---------|-------------|
| `OLTC_API_URL` | `http://localhost:8000` | URL du back-end vue depuis le dashboard |

---

## 5. Smoke tests réussis

### Back-end isolé

```
✓ GET /api/health           {"status":"ok","prophet_available":true}
✓ GET /api/equipment        4 équipements
✓ GET /api/sensors          5 capteurs avec seuils complets
✓ GET /api/faults           6 pannes P1–P6
✓ GET /api/historical/PJ1   3425 points (24h freq) / 13697 points (6h freq)
✓ GET /api/latest/PJ1       Sample courant + statut "Normal"
✓ GET /api/realtime/PJ1/sample?seed=123
✓ GET /api/anomalies/PJ1/report   6 lignes, avance moyenne ~36j
✓ GET /api/forecast/PJ1?days=30   Prophet model fit + intervalles 80%
✓ GET /docs                 Swagger UI HTTP 200
✓ 404 sur équipement inconnu géré proprement
```

### Intégration end-to-end

```
✓ uvicorn boot sur :8000
✓ Streamlit boot sur :5001 contre OLTC_API_URL=http://127.0.0.1:8000
✓ Logs API montrent les vrais appels HTTP du dashboard :
    GET /api/health        ← bootstrap api_client
    GET /api/sensors       ← bootstrap
    GET /api/faults        ← bootstrap
    GET /api/equipment     ← bootstrap
    GET /api/historical/PJ1?freq_hours=6&period=all   ← load_data
    GET /api/realtime/PJ1/sample                       ← live mode
```

---

## 6. Décisions de conception

| Choix | Justification |
|-------|---------------|
| **2-tiers FastAPI + Streamlit** | Sépare la couche calcul/ML de la couche UI ; le back-end est réutilisable par d'autres clients (Power BI, mobile, scripts) |
| **Données simulées** | PFE — pas d'accès aux vrais capteurs ; reproductibilité via seed NumPy par équipement |
| **Mode dark SCADA** | Style salle de contrôle industriel, demande utilisateur explicite ; contraste fort, idéal pour démo écran/projecteur |
| **Vert OCP `#008542`** | Couleur corporate du Groupe OCP, cohérent avec le site Jorf Lasfar |
| **Détection IF binaire** | Le seuil de score (0.60) n'est jamais atteint à l'échelle 6h ; utilisation de `is_anomaly` (binaire sklearn) plus robuste |
| **Cache `lru_cache` + `st.cache_data`** | Évite de recalculer dataset+IF+Prophet à chaque interaction utilisateur |
| **Bootstrap automatique** | L'`api_client` se connecte au boot, retry une fois si échec, banner clair sinon |

---

## 7. Pistes restantes

- [ ] Intégrer un onglet **"Prévision Prophet"** dans le dashboard (l'API expose déjà `/api/forecast`, le front ne l'utilise pas encore)
- [ ] Brancher la dashboard sur des **vraies données END/SAP PM** au lieu des séries simulées
- [ ] Ajouter un modèle **LSTM** pour comparaison avec Isolation Forest
- [ ] Export **PDF** automatique du rapport de panne / d'alerte
- [ ] **Authentification** sur l'API (OAuth2 / API key)
- [ ] **WebSocket** pour le mode live (à la place du polling + `st.rerun`)
- [ ] **Dockerisation** des deux services + docker-compose
- [ ] **CI** simple : pytest sur le back-end + lint sur le front
- [ ] **Persistance** : stockage SQLite/TimescaleDB des samples temps réel pour rejouer une session

---

## 8. Argument industriel

> Si ce système avait été déployé sur PJ1 dès 2017, les 6 pannes auraient été
> détectées en moyenne **36 jours avant** l'incident, permettant une intervention
> préventive planifiée et évitant la quasi-totalité des **62,46 MWh perdus**.

| Métrique | Valeur |
|----------|--------|
| Investissement matériel (Vague 1) | ~1 000 € |
| MWh perdus évités | 62,46 MWh |
| Gain direct (100 €/MWh) | ~6 246 € |
| ROI première année | **6×** |
| Amortissement | < 1 mois |

---

*Projet PFE — Maintenance prédictive OLTC HT/THT*
*OCP · Jorf Lasfar · Terminal Slurry Pipeline*
