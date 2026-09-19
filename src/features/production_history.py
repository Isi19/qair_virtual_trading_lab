"""Shared production lags and rolling statistics for wind and solar."""

from __future__ import annotations

import pandas as pd


INTERVAL = pd.Timedelta(minutes=15)


def production_history_at_delivery(
    delivery_timestamps: pd.DatetimeIndex,
    historical_production: pd.DataFrame,
    *,
    require_complete: bool = True,
) -> pd.DataFrame:
    """Calcule les lags et rollings pour les dates demandées.

    Avec require_complete=False, le training conserve les NaN initiaux.
    Avec True, le forecast exige toutes les observations nécessaires.
    Les fenêtres rolling se terminent 48 heures avant chaque livraison.
    """
    deliveries = pd.DatetimeIndex(pd.to_datetime(delivery_timestamps, utc=True))
    history = historical_production[["timestamp", "production_mw"]].copy()
    history["timestamp"] = pd.to_datetime(history["timestamp"], utc=True)
    production = (
        history.drop_duplicates("timestamp", keep="last")
        .set_index("timestamp")["production_mw"]
        .sort_index()
        .astype(float)
    )
    if production.empty or production.isna().any():
        raise ValueError("Historical production is empty or contains missing values.")

    end = max(production.index.max(), deliveries.max())

    # Prolonger la grille permet de calculer les lags des livraisons futures.
    # La production future reste manquante : aucune valeur n'est inventée.
    grid = pd.date_range(production.index.min(), end, freq=INTERVAL)
    production = production.reindex(grid)
    available_production = production.shift(48 * 4)
    rolling_4h = available_production.rolling(4 * 4)
    rolling_24h = available_production.rolling(24 * 4)

    history_features = pd.DataFrame({
        "production_lag_48h_mw": available_production,
        "production_lag_7d_mw": production.shift(7 * 24 * 4),
        "production_rolling_mean_4h_lag48h_mw": rolling_4h.mean(),
        "production_rolling_std_4h_lag48h_mw": rolling_4h.std(ddof=0),
        "production_rolling_max_4h_lag48h_mw": rolling_4h.max(),
        "production_rolling_mean_24h_lag48h_mw": rolling_24h.mean(),
        "production_rolling_std_24h_lag48h_mw": rolling_24h.std(ddof=0),
        # MW × 0,25 h par quart d'heure donne une énergie en MWh.
        "production_rolling_energy_24h_lag48h_mwh": rolling_24h.sum() * 0.25,
    })
    # Garder les dates demandées dans le même ordre que les features météo.
    features = history_features.reindex(deliveries).reset_index(drop=True)
    if require_complete and features.isna().any().any():
        raise ValueError("Not enough observed production history for forecast predictors.")
    return features
