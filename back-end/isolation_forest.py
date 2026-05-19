import pandas as pd
import numpy as np
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import matplotlib.dates as mdates

# ── Configuration ───────────────────────────────────────────────────────────
FEATURES = ["dga_h2_ppm", "temp_huile_c", "vib_ms2", "t_comm_ms", "vcc_v"]
THRESHOLD_SCORE = 0.60   # Score au-delà duquel on déclenche une alerte
BASELINE_DAYS = 90       # Jours de données "normales" pour l'entraînement

def run_isolation_forest(df: pd.DataFrame, contamination: float = 0.05):
    """
    Entraîne Isolation Forest sur la baseline normale,
    prédit les anomalies sur l'ensemble de la série.
    """
    # 1. Normalisation des données (moyenne = 0, écart-type = 1)
    scaler = StandardScaler()
    X = scaler.fit_transform(df[FEATURES])

    # 2. Baseline = 90 premiers jours (comportement normal)
    #    48 mesures/jour à pas de 30 min = 4320 points
    n_baseline = BASELINE_DAYS * 48
    X_baseline = X[:n_baseline]

    # 3. Entraînement du modèle
    model = IsolationForest(
        n_estimators=100,        # Nombre d'arbres dans la forêt
        contamination=contamination,  # % de données considérées anormales
        max_samples="auto",      # Taille des échantillons par arbre
        random_state=42,         # Reproductibilité
        n_jobs=-1,               # Utilise tous les cœurs CPU
    )
    model.fit(X_baseline)

    # 4. Prédiction sur toute la série
    df = df.copy()
    # Score : plus élevé = plus anormal (d'où le signe négatif)
    df["anomaly_score"] = -model.score_samples(X)
    # is_anomaly = True si l'arbre prédit -1 (anomalie)
    df["is_anomaly"] = model.predict(X) == -1

    return df, model, scaler


def plot_anomalies(df: pd.DataFrame, panne_dates: list):
    """Visualise les scores d'anomalie avec les pannes réelles marquées."""
    daily = df["anomaly_score"].resample("D").max()

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 8), sharex=True)

    # ── Graphique 1 : Score d'anomalie ─────────────────────────────────────
    ax1.plot(daily.index, daily.values, color="#378ADD", linewidth=0.8)
    
    # Seuils
    ax1.axhline(THRESHOLD_SCORE, color="#E24B4A", linestyle="--",
                linewidth=0.8, label=f"Seuil critique ({THRESHOLD_SCORE})")
    ax1.axhline(THRESHOLD_SCORE * 0.78, color="#EF9F27", linestyle="--",
                linewidth=0.8, label="Seuil alerte")
    
    # Zone critique remplie
    ax1.fill_between(daily.index, THRESHOLD_SCORE, daily.values,
                     where=daily.values > THRESHOLD_SCORE,
                     alpha=0.3, color="#E24B4A", label="Zone critique")
    
    ax1.set_ylabel("Score anomalie Isolation Forest")
    ax1.legend(fontsize=8)
    ax1.grid(axis="y", alpha=0.2)
    ax1.set_ylim(0, 1.05)

    # ── Graphique 2 : H₂ DGA (pour corrélation) ────────────────────────────
    daily_h2 = df["dga_h2_ppm"].resample("D").mean()
    ax2.plot(daily_h2.index, daily_h2.values, color="#E24B4A", linewidth=0.8)
    ax2.axhline(100, color="#EF9F27", linestyle="--", linewidth=0.8, 
                label="Seuil CEI 100 ppm")
    ax2.set_ylabel("H₂ DGA (ppm)")
    ax2.legend(fontsize=8)
    ax2.grid(axis="y", alpha=0.2)
    ax2.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m"))

    # Marqueurs des pannes sur les deux graphiques
    for date, label in panne_dates:
        for ax in [ax1, ax2]:
            ax.axvline(pd.Timestamp(date), color="#E24B4A", linewidth=1.2, alpha=0.8)
            ax.text(pd.Timestamp(date), ax.get_ylim()[1] * 0.85,
                    label, fontsize=7, rotation=90, color="#E24B4A",
                    ha="right", va="top")

    plt.suptitle("Détection d'anomalies OLTC PJ1 — Isolation Forest", fontsize=12)
    plt.tight_layout()
    plt.savefig("outputs/figures/isolation_forest_pj1.png", dpi=150)
    plt.show()


def report_detections(df: pd.DataFrame, panne_dates: list) -> pd.DataFrame:
    """
    Pour chaque panne réelle, calcule combien de jours avant
    Isolation Forest aurait déclenché une alerte.
    """
    rows = []
    for date_str, label in panne_dates:
        fault_date = pd.Timestamp(date_str)
        # Première alerte critique avant la panne
        window = df.loc[:fault_date, "anomaly_score"]
        alerts = window[window > THRESHOLD_SCORE]
        
        if not alerts.empty:
            first_alert = alerts.index[0]
            days_advance = (fault_date - first_alert).days
        else:
            first_alert = None
            days_advance = None
            
        rows.append({
            "Panne": label,
            "Date réelle": date_str,
            "Première alerte": first_alert.date() if first_alert else "Non détectée",
            "Avance (jours)": days_advance,
            "Score max": window.max().round(3),
        })
    return pd.DataFrame(rows)


# ── Exécution ───────────────────────────────────────────────────────────────
if __name__ == "__main__":
    # Lecture des données IoT générées précédemment
    df = pd.read_csv("data/processed/oltc_pj1_iot.csv",
                     index_col="timestamp", parse_dates=True)
    
    # Lancement Isolation Forest
    df_result, model, scaler = run_isolation_forest(df)

    # Dates des 6 pannes réelles
    pannes = [
        ("2017-07-01", "P5"), ("2017-08-09", "P6"),
        ("2019-09-30", "P4"), ("2021-12-14", "P3"),
        ("2022-02-20", "P2"), ("2022-07-27", "P1"),
    ]
    
    # Visualisation
    plot_anomalies(df_result, pannes)

    # Rapport de détection
    report = report_detections(df_result, pannes)
    print("\n── Rapport de détection ──────────────────────────────")
    print(report.to_string(index=False))
    report.to_csv("outputs/reports/isolation_forest_report.csv", index=False)