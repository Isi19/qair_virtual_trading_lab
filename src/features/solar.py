"""Build leakage-safe predictors for Perleberg solar forecasting.

Production-derived features stop at t-48h so that they are known for every
quarter-hour of the following delivery day.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pvlib

from src.features.production_history import production_history_at_delivery


INTERVAL = pd.Timedelta(minutes=15)
QUARTERS_PER_HOUR = 4

# Circular encodings avoid false discontinuities at the end of a cycle.
DERIVED_FEATURE_COLUMNS = [
    "wind_direction_10m_sin",
    "wind_direction_10m_cos",
    "local_time_sin",
    "local_time_cos",
    "day_of_year_sin",
    "day_of_year_cos",
    "solar_elevation_deg",
    "solar_azimuth_sin",
    "solar_azimuth_cos",
    "is_day",
]

# A 24-hour lag would not be known for all periods of tomorrow.
PRODUCTION_LAG_COLUMNS = [
    "production_lag_48h_mw",
    "production_lag_7d_mw",
]

# Each rolling window ends at t-48h.
PRODUCTION_ROLLING_COLUMNS = [
    "production_rolling_mean_4h_lag48h_mw",
    "production_rolling_std_4h_lag48h_mw",
    "production_rolling_max_4h_lag48h_mw",
    "production_rolling_mean_24h_lag48h_mw",
    "production_rolling_std_24h_lag48h_mw",
    "production_rolling_energy_24h_lag48h_mwh",
]

PRODUCTION_HISTORY_FEATURE_COLUMNS = [
    *PRODUCTION_LAG_COLUMNS,
    *PRODUCTION_ROLLING_COLUMNS,
]


def _feature_weather_columns(asset_config: dict) -> list[str]:
    """Return weather variables shared by ERA5 and the J-1 forecast."""
    settings = asset_config.get("historical_reanalysis", {})
    columns = settings.get("feature_variables")
    if not columns:
        raise ValueError("Solar feature configuration is missing feature_variables.")
    return list(dict.fromkeys(columns))


def _weather_feature_columns(asset_config: dict) -> list[str]:
    """Keep weather values and replace the wind direction with sin/cos."""
    return [
        column
        for column in _feature_weather_columns(asset_config)
        if column != "wind_direction_10m"
    ]


def solar_feature_columns(asset_config: dict) -> list[str]:
    """Return the exact ordered predictor contract consumed by every model."""
    return [
        *_weather_feature_columns(asset_config),
        *DERIVED_FEATURE_COLUMNS,
        *PRODUCTION_HISTORY_FEATURE_COLUMNS,
    ]


def _validated_timestamps(
    frame: pd.DataFrame, *, require_continuity: bool = True
) -> pd.DatetimeIndex:
    """Require a unique, ordered and complete UTC-compatible 15-minute grid."""
    if "timestamp" not in frame:
        raise ValueError("Solar data must contain a timestamp column.")
    timestamps = pd.DatetimeIndex(pd.to_datetime(frame["timestamp"]))
    if timestamps.tz is None:
        raise ValueError("Solar timestamps must be timezone-aware.")
    timestamps = timestamps.tz_convert("UTC")
    if timestamps.has_duplicates or not timestamps.is_monotonic_increasing:
        raise ValueError("Solar timestamps must be unique and sorted.")
    if require_continuity and (
        len(timestamps) > 1
        and not timestamps.to_series().diff().dropna().eq(INTERVAL).all()
    ):
        raise ValueError("Solar data must be continuous at 15-minute resolution.")
    return timestamps


def _validated_input_columns(frame: pd.DataFrame, asset_config: dict) -> list[str]:
    """Validate the site contract and return its configured weather inputs."""
    if asset_config.get("technology") != "solar":
        raise ValueError("Solar features require an asset configured as solar.")
    required_config = {
        "latitude",
        "longitude",
        "market_timezone",
    }
    missing_config = sorted(
        field for field in required_config if asset_config.get(field) is None
    )
    if missing_config:
        raise ValueError(f"Solar feature configuration is missing: {missing_config}")

    weather_columns = _feature_weather_columns(asset_config)
    required_columns = list(
        dict.fromkeys([*weather_columns, "wind_direction_10m"])
    )
    missing_columns = sorted(set(required_columns).difference(frame.columns))
    if missing_columns:
        raise ValueError(f"Solar weather data is missing columns: {missing_columns}")
    columns_with_nulls = frame[required_columns].columns[
        frame[required_columns].isna().any()
    ].tolist()
    if columns_with_nulls:
        raise ValueError(
            f"Solar weather data contains missing values: {columns_with_nulls}"
        )
    return weather_columns


def _add_wind_direction_features(features: pd.DataFrame) -> None:
    """Replace angular degrees with continuous sine and cosine components."""
    wind_angle_radians = np.deg2rad(
        features["wind_direction_10m"].to_numpy() % 360
    )
    features["wind_direction_10m_sin"] = np.sin(wind_angle_radians)
    features["wind_direction_10m_cos"] = np.cos(wind_angle_radians)


def _add_local_calendar_features(
    features: pd.DataFrame,
    timestamps: pd.DatetimeIndex,
    market_timezone: str,
) -> None:
    """Encode local delivery time and season without adding a horizon feature."""
    local_timestamps = timestamps.tz_convert(market_timezone)
    quarter_of_day = local_timestamps.hour * 4 + local_timestamps.minute // 15
    local_time_angle = 2 * np.pi * quarter_of_day / 96
    features["local_time_sin"] = np.sin(local_time_angle)
    features["local_time_cos"] = np.cos(local_time_angle)

    day_of_year_angle = 2 * np.pi * (local_timestamps.dayofyear - 1) / 365.2425
    features["day_of_year_sin"] = np.sin(day_of_year_angle)
    features["day_of_year_cos"] = np.cos(day_of_year_angle)


def _add_solar_position_features(
    features: pd.DataFrame,
    timestamps: pd.DatetimeIndex,
    asset_config: dict,
) -> None:
    """Calculate deterministic solar geometry from time and site coordinates."""
    solar_position = pvlib.solarposition.get_solarposition(
        timestamps,
        latitude=float(asset_config["latitude"]),
        longitude=float(asset_config["longitude"]),
    )
    solar_elevation = solar_position["apparent_elevation"].to_numpy()
    solar_azimuth_radians = np.deg2rad(solar_position["azimuth"].to_numpy())
    features["solar_elevation_deg"] = solar_elevation
    features["solar_azimuth_sin"] = np.sin(solar_azimuth_radians)
    features["solar_azimuth_cos"] = np.cos(solar_azimuth_radians)
    features["is_day"] = (solar_elevation > 0).astype("int8")


def _build_solar_features(
    weather_data: pd.DataFrame,
    asset_config: dict,
    *,
    require_continuity: bool = True,
) -> pd.DataFrame:
    """Prépare la météo, la direction du vent, le calendrier et le soleil."""
    timestamps = _validated_timestamps(
        weather_data, require_continuity=require_continuity
    )
    weather_columns = _validated_input_columns(weather_data, asset_config)
    features = weather_data[weather_columns].astype(float).reset_index(drop=True)
    features.insert(0, "timestamp", timestamps)

    _add_wind_direction_features(features)
    _add_local_calendar_features(
        features, timestamps, str(asset_config["market_timezone"])
    )
    _add_solar_position_features(features, timestamps, asset_config)
    return features


def build_solar_training_frame(
    historical_data: pd.DataFrame, asset_config: dict
) -> pd.DataFrame:
    """Prépare les features historiques et la cible pour le modeling.

    Les premières lignes gardent leurs lags manquants. Le découpage en train,
    validation et test se fait ensuite dans le notebook.
    """
    features = _build_solar_features(historical_data, asset_config)
    history_features = production_history_at_delivery(
        features["timestamp"], historical_data, require_complete=False
    )
    features[PRODUCTION_HISTORY_FEATURE_COLUMNS] = history_features
    features["production_mw"] = historical_data["production_mw"].to_numpy(dtype=float)
    return features[
        ["timestamp", *solar_feature_columns(asset_config), "production_mw"]
    ]


def build_solar_forecast_frame(
    forecast_weather: pd.DataFrame,
    historical_production: pd.DataFrame,
    asset_config: dict,
) -> pd.DataFrame:
    """Prépare les mêmes features avec la météo J-1 et la production passée."""
    features = _build_solar_features(
        forecast_weather, asset_config, require_continuity=False
    )
    history_features = production_history_at_delivery(
        features["timestamp"], historical_production
    )
    features[PRODUCTION_HISTORY_FEATURE_COLUMNS] = history_features
    return features[["timestamp", *solar_feature_columns(asset_config)]]
