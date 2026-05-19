import pandas as pd
import numpy as np
from prophet import Prophet
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings("ignore")

# ── Seuils d'action ─────────────────────────────────────────────────────────
ALERT_THRESHOLD = 60      # HI % — déclenche planification maintenance
CRITICAL_THRESHOLD = 40   # HI % — arrêt programmé urgent


def run_prophet_forecast(df_hi: pd.DataFrame,
                         forecast_days: int = 60,
                         interval_width: float = 0.80) -> tuple:
    """
    Ajuste Prophet sur l'historique Health Index,
    prédit les 'forecast_days' jours suivants.
    
    Retourne : (modèle, dataframe prévision, date_alerte, date_critique)
    """
    # Prophet attend obligatoirement les colonnes 'ds' (date) et 'y' (valeur)
    daily_hi = df_hi["health_index"].resample("D").mean().reset_index()
    daily_hi.columns = ["ds", "y"]
    daily_hi = daily_hi.dropna()

    # Configuration du modèle
    model = Prophet(
        interval_width=interval_width,           # Largeur intervalle confiance
        changepoint_prior_scale=0.05,            # Sensibilité aux ruptures
        seasonality_mode="additive",             # Saisonnalités additives
        daily_seasonality=False,                 # Pas de cycle journalier
        weekly_seasonality=True,                 # Cycles de charge hebdo
        yearly_seasonality=False,                # Pas de cycle annuel
    )
    # Ajout d'une saisonnalité mensuelle personnalisée
    model.add_seasonality(name="monthly", period=30.5, fourier_order=5)
    
    # Entraînement
    model.fit(daily_hi)

    # Génération des dates futures
    future = model.make_future_dataframe(periods=forecast_days)
    forecast = model.predict(future)

    # ── Détection des franchissements de seuils ────────────────────────────
    # On ne regarde que la partie future de la prévision
    future_only = forecast[forecast["ds"] > daily_hi["ds"].max()]

    alert_rows = future_only[future_only["yhat"] < ALERT_THRESHOLD]
    crit_rows = future_only[future_only["yhat"] < CRITICAL_THRESHOLD]

    alert_date = alert_rows["ds"].min() if not alert_rows.empty else None
    crit_date = crit_rows["ds"].min() if not crit_rows.empty else None

    return model, forecast, alert_date, crit_date


def plot_prophet(model, forecast, df_hi, alert_date, crit_date, title="PJ1"):
    """Visualisation soignée de la prévision Prophet."""
    daily_hi = df_hi["health_index"].resample("D").mean()
    last_obs = daily_hi.index.max()

    fig, ax = plt.subplots(figsize=(14, 5))

    # ── Données historiques ────────────────────────────────────────────────
    ax.plot(daily_hi.index, daily_hi.values,
            color="#378ADD", linewidth=1.2, label="HI observé")

    # ── Prévision Prophet ──────────────────────────────────────────────────
    future_mask = forecast["ds"] > last_obs
    f = forecast[future_mask]
    
    ax.plot(f["ds"], f["yhat"], color="#7F77DD", linewidth=1.5,
            linestyle="--", label="Prévision Prophet")
    ax.fill_between(f["ds"], f["yhat_lower"], f["yhat_upper"],
                    alpha=0.15, color="#7F77DD", 
                    label="Intervalle confiance 80%")

    # ── Seuils horizontaux ─────────────────────────────────────────────────
    ax.axhline(ALERT_THRESHOLD, color="#EF9F27", linestyle=":", linewidth=1, 
               label="Seuil alerte 60%")
    ax.axhline(CRITICAL_THRESHOLD, color="#E24B4A", linestyle=":", linewidth=1, 
               label="Seuil critique 40%")

    # ── Marqueurs de franchissement ────────────────────────────────────────
    if alert_date:
        ax.axvline(alert_date, color="#EF9F27", linewidth=1.2, alpha=0.8)
        ax.text(alert_date, 65, f"Alerte\n{alert_date.strftime('%d/%m/%y')}",
                fontsize=8, color="#EF9F27", ha="left")
                
    if crit_date:
        ax.axvline(crit_date, color="#E24B4A", linewidth=1.2, alpha=0.8)
        ax.text(crit_date, 45, f"Critique\n{crit_date.strftime('%d/%m/%y')}",
                fontsize=8, color="#E24B4A", ha="left")

    # ── Mise en forme ──────────────────────────────────────────────────────
    ax.set_ylim(0, 105)
    ax.set_ylabel("Health Index (%)")
    ax.set_title(f"Prévision Prophet — Health Index OLTC {title}")
    ax.legend(fontsize=8, loc="upper right")
    ax.grid(axis="y", alpha=0.2)
    # Zone grisée pour la prévision
    ax.axvspan(last_obs, forecast["ds"].max(), alpha=0.04, color="#7F77DD")
    
    plt.tight_layout()
    plt.savefig(f"outputs/figures/prophet_forecast_{title.lower()}.png", dpi=150)
    plt.show()


def generate_maintenance_order(alert_date, crit_date, equipment="OLTC-PJ1"):
    """Simulation d'un ordre de travail SAP PM."""
    if alert_date is None:
        print("Aucune intervention prévue dans la fenêtre de prévision.")
        return
        
    days_to_alert = (alert_date - pd.Timestamp.now()).days
    urgency = "urgent" if days_to_alert <= 14 else "préventif"
    
    print(f"\n── Ordre de travail SAP PM généré ────────────────────")
    print(f"  Équipement  : {equipment}")
    print(f"  Type        : {urgency.upper()}")
    print(f"  Date alerte : {alert_date.strftime('%d/%m/%Y')} (J+{days_to_alert})")
    print(f"  Date limite : {crit_date.strftime('%d/%m/%Y') if crit_date else 'N/A'}")
    print(f"  Action      : Inspection OLTC + analyse DGA + mesure t_comm")
    print(f"  Priorité    : {'P1 - URGENCE' if urgency == 'urgent' else 'P2 - PLANIFIÉ'}")


# ── Exécution ───────────────────────────────────────────────────────────────
if __name__ == "__main__":
    # Lecture du Health Index calculé précédemment
    df_hi = pd.read_csv("data/processed/health_index_pj1.csv",
                        index_col="timestamp", parse_dates=True)
    
    # Lancement Prophet (prévision sur 60 jours)
    model, forecast, alert_date, crit_date = run_prophet_forecast(
        df_hi, forecast_days=60
    )
    
    # Visualisation
    plot_prophet(model, forecast, df_hi, alert_date, crit_date)
    
    # Génération ordre de maintenance
    generate_maintenance_order(alert_date, crit_date)