"""Build leakage-safe predictors for day-ahead wind generation forecasting.

Langer Wald is forecast one complete day at a time. Production lags and rolling
statistics therefore stop at t-48h, so every feature is already known when the
forecast for tomorrow is issued.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from src.features.production_history import production_history_at_delivery


INTERVAL = pd.Timedelta(minutes=15)
QUARTERS_PER_HOUR = 4

DRY_AIR_GAS_CONSTANT = 287.05

CALENDAR_FEATURE_COLUMNS = [
    "local_time_sin",
    "local_time_cos",
    "day_of_year_sin",
    "day_of_year_cos",
]

WIND_PHYSICS_FEATURE_COLUMNS = [
    "wind_speed_hub_m",
    "wind_speed_hub_cubed",
    "wind_gust_excess_10m",
    "air_density_kg_m3",
]

PRODUCTION_LAG_COLUMNS = [
    "production_lag_48h_mw",
    "production_lag_7d_mw",
]

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
        raise ValueError("Wind feature configuration is missing feature_variables.")
    return list(dict.fromkeys(columns))


def _direction_columns(asset_config: dict) -> list[str]:
    """Return configured wind directions in their original order."""
    return [
        column
        for column in _feature_weather_columns(asset_config)
        if column.startswith("wind_direction_")
    ]


def _weather_feature_columns(asset_config: dict) -> list[str]:
    """Keep forecastable weather values, replacing directions with sin/cos."""
    return [
        column
        for column in _feature_weather_columns(asset_config)
        if not column.startswith("wind_direction_")
    ]


def _direction_feature_columns(asset_config: dict) -> list[str]:
    direction_features = []
    for column in _direction_columns(asset_config):
        direction_features.extend([f"{column}_sin", f"{column}_cos"])
    return direction_features


def wind_feature_columns(asset_config: dict) -> list[str]:
    """Return the ordered predictor contract shared by training and replay."""
    return [
        *_weather_feature_columns(asset_config),
        *_direction_feature_columns(asset_config),
        *CALENDAR_FEATURE_COLUMNS,
        *WIND_PHYSICS_FEATURE_COLUMNS,
        *PRODUCTION_HISTORY_FEATURE_COLUMNS,
    ]


def _validated_timestamps(
    frame: pd.DataFrame, *, require_continuity: bool = True
) -> pd.DatetimeIndex:
    if "timestamp" not in frame:
        raise ValueError("Wind data must contain a timestamp column.")
    timestamps = pd.DatetimeIndex(pd.to_datetime(frame["timestamp"]))
    if timestamps.tz is None:
        raise ValueError("Wind timestamps must be timezone-aware.")
    timestamps = timestamps.tz_convert("UTC")
    if timestamps.has_duplicates or not timestamps.is_monotonic_increasing:
        raise ValueError("Wind timestamps must be unique and sorted.")
    if require_continuity and (
        len(timestamps) > 1
        and not timestamps.to_series().diff().dropna().eq(INTERVAL).all()
    ):
        raise ValueError("Wind data must be continuous at 15-minute resolution.")
    return timestamps


def _validated_weather_columns(frame: pd.DataFrame, asset_config: dict) -> list[str]:
    if asset_config.get("technology") != "wind":
        raise ValueError("Wind features require an asset configured as wind.")
    for field in ["market_timezone", "hub_height_m"]:
        if asset_config.get(field) is None:
            raise ValueError(f"Wind feature configuration is missing: {field}")

    weather_columns = _feature_weather_columns(asset_config)
    missing_columns = sorted(set(weather_columns) - set(frame.columns))
    if missing_columns:
        raise ValueError(f"Wind weather data is missing columns: {missing_columns}")
    columns_with_nulls = frame[weather_columns].columns[
        frame[weather_columns].isna().any()
    ].tolist()
    if columns_with_nulls:
        raise ValueError(
            f"Wind weather data contains missing values: {columns_with_nulls}"
        )
    return weather_columns


def _add_direction_features(features: pd.DataFrame, direction_columns: list[str]) -> None:
    """Encode circular degrees without a false break between 359° and 0°."""
    for column in direction_columns:
        angle_radians = np.deg2rad(features[column].to_numpy() % 360)
        features[f"{column}_sin"] = np.sin(angle_radians)
        features[f"{column}_cos"] = np.cos(angle_radians)


def _add_calendar_features(
    features: pd.DataFrame,
    timestamps: pd.DatetimeIndex,
    market_timezone: str,
) -> None:
    """Encode local time and season as continuous cycles."""
    local_timestamps = timestamps.tz_convert(market_timezone)
    quarter_of_day = local_timestamps.hour * 4 + local_timestamps.minute // 15
    local_time_angle = 2 * np.pi * quarter_of_day / 96
    features["local_time_sin"] = np.sin(local_time_angle)
    features["local_time_cos"] = np.cos(local_time_angle)
    # 365.2425 tient compte des années bissextiles dans le cycle saisonnier.
    day_angle = 2 * np.pi * (local_timestamps.dayofyear - 1) / 365.2425
    features["day_of_year_sin"] = np.sin(day_angle)
    features["day_of_year_cos"] = np.cos(day_angle)


def _add_wind_physics_features(features: pd.DataFrame, asset_config: dict) -> None:
    """Create simple physical summaries from forecastable weather variables."""
    subtype = asset_config.get("subtype", "onshore")
    shear_exponent = 0.1 if subtype == "offshore_floating" else 0.2
    hub_height_m = float(asset_config["hub_height_m"])
    # ERA5 and ICON-D2 both provide the wind at 100 m.
    wind_speed_hub = features["wind_speed_100m"].clip(lower=0) * (
        hub_height_m / 100.0
    ) ** shear_exponent
    features["wind_speed_hub_m"] = wind_speed_hub
    features["wind_speed_hub_cubed"] = wind_speed_hub**3

    features["wind_gust_excess_10m"] = (
        features["wind_gusts_10m"] - features["wind_speed_10m"]
    ).clip(lower=0)

    temperature_kelvin = features["temperature_2m"] + 273.15
    pressure_pascal = features["surface_pressure"] * 100
    features["air_density_kg_m3"] = pressure_pascal / (
        DRY_AIR_GAS_CONSTANT * temperature_kelvin
    )




def _build_wind_features(
    weather_data: pd.DataFrame,
    asset_config: dict,
    *,
    require_continuity: bool = True,
) -> pd.DataFrame:
    """Apply the weather, direction, calendar and physics transformations."""
    timestamps = _validated_timestamps(
        weather_data, require_continuity=require_continuity
    )
    weather_columns = _validated_weather_columns(weather_data, asset_config)
    direction_columns = _direction_columns(asset_config)

    features = weather_data[weather_columns].astype(float).reset_index(drop=True)
    features.insert(0, "timestamp", timestamps)
    _add_direction_features(features, direction_columns)
    _add_calendar_features(features, timestamps, str(asset_config["market_timezone"]))
    _add_wind_physics_features(features, asset_config)
    return features


def build_wind_training_frame(
    historical_data: pd.DataFrame, asset_config: dict
) -> pd.DataFrame:
    """Return one historical target and its reproducible day-ahead predictors."""
    if "production_mw" not in historical_data:
        raise ValueError("Wind training data must contain production_mw.")
    if historical_data["production_mw"].isna().any():
        raise ValueError("Wind training production contains missing values.")

    features = _build_wind_features(historical_data, asset_config)

    history_features = production_history_at_delivery(
        features["timestamp"], historical_data, require_complete=False
    )
    features[PRODUCTION_HISTORY_FEATURE_COLUMNS] = history_features
    features["production_mw"] = historical_data["production_mw"].to_numpy(dtype=float)

    return features[
        ["timestamp", *wind_feature_columns(asset_config), "production_mw"]
    ]


def build_wind_forecast_frame(
    forecast_weather: pd.DataFrame,
    historical_production: pd.DataFrame,
    asset_config: dict,
) -> pd.DataFrame:
    """Build wind predictors from archived forecast weather and known output."""
    features = _build_wind_features(
        forecast_weather, asset_config, require_continuity=False
    )
    timestamps = pd.DatetimeIndex(features["timestamp"])
    history_features = production_history_at_delivery(timestamps, historical_production)
    for column in PRODUCTION_HISTORY_FEATURE_COLUMNS:
        features[column] = history_features[column].to_numpy()
    return features[["timestamp", *wind_feature_columns(asset_config)]]
