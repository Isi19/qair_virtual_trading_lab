"""Power-curve based virtual wind generation model."""

from __future__ import annotations

import numpy as np
import pandas as pd
from windpowerlib import WindTurbine


DEFAULT_HUB_HEIGHT_M = 135.0


def extrapolate_wind_speed(
    wind_speed_100m: pd.Series, hub_height_m: float, *, alpha: float
) -> pd.Series:
    """Extrapolate 100 m wind speed to hub height with the Hellmann power law."""
    return wind_speed_100m.clip(lower=0) * (hub_height_m / 100.0) ** alpha


def _load_power_curve(asset_config: dict, hub_height_m: float) -> pd.DataFrame:
    """Load the explicitly declared windpowerlib turbine curve."""
    turbine_type = asset_config.get("windpowerlib_turbine_type")
    if not turbine_type:
        raise ValueError("Wind asset must define windpowerlib_turbine_type.")
    try:
        turbine = WindTurbine(turbine_type=turbine_type, hub_height=hub_height_m)
    except Exception as exc:
        raise ValueError(
            f"Unknown windpowerlib turbine type '{turbine_type}'. Set "
            "windpowerlib_turbine_type to an installed standard turbine."
        ) from exc
    return turbine.power_curve.sort_values("wind_speed")


def transform(weather_df: pd.DataFrame, asset_config: dict) -> pd.DataFrame:
    """Convert site weather into capacity-scaled, bounded wind production."""

    required_columns = {"wind_speed_100m", "temperature_2m", "surface_pressure"}
    
    missing = required_columns.difference(weather_df.columns)
    if missing:
        raise ValueError(f"Wind weather is missing columns: {sorted(missing)}")
    capacity_mw = asset_config.get("capacity_mw")
    if capacity_mw is None or float(capacity_mw) <= 0:
        raise ValueError("Wind asset must define a positive capacity_mw.")

    subtype = asset_config.get("subtype", "onshore")
    alpha = 0.1 if subtype == "offshore_floating" else 0.2
    hub_height_m = float(asset_config.get("hub_height_m", DEFAULT_HUB_HEIGHT_M))
    wind_at_hub = extrapolate_wind_speed(
        weather_df["wind_speed_100m"], hub_height_m, alpha=alpha
    )
    curve = _load_power_curve(asset_config, hub_height_m)
    turbine_power_w = np.interp(
        wind_at_hub.to_numpy(),
        curve["wind_speed"].to_numpy(),
        curve["value"].to_numpy(),
        left=0,
        right=0,
    )
    rated_power_w = float(curve["value"].max())
    normalized_power = np.clip(turbine_power_w / rated_power_w, 0, 1)
    turbine_count = asset_config.get("number_of_turbines")
    turbine_rated_power_mw = asset_config.get("turbine_rated_power_mw")
    if turbine_count is not None and turbine_rated_power_mw is not None:
        declared_capacity_mw = int(turbine_count) * float(turbine_rated_power_mw)
        if not np.isclose(float(capacity_mw), declared_capacity_mw):
            raise ValueError(
                "capacity_mw must equal number_of_turbines * turbine_rated_power_mw."
            )
    result = weather_df.copy()
    result["wind_speed_hub_m"] = wind_at_hub
    result["production_mw"] = normalized_power * float(capacity_mw)
    result["capacity_factor"] = normalized_power
    return result
