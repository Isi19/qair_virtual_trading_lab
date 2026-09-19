"""PVWatts-based virtual solar generation model."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pvlib


REQUIRED_WEATHER_COLUMNS = {
    "temperature_2m",
    "wind_speed_10m",
    "shortwave_radiation",
    "diffuse_radiation",
    "direct_normal_irradiance",
}


def transform(weather_df: pd.DataFrame, asset_config: dict) -> pd.DataFrame:
    """Convert site weather to bounded MW production with a PVWatts model."""
    missing = REQUIRED_WEATHER_COLUMNS.difference(weather_df.columns)
    if missing:
        raise ValueError(f"Solar weather is missing columns: {sorted(missing)}")
    required_asset_fields = {"latitude", "longitude", "capacity_mw", "tilt", "azimuth"}
    missing_asset_fields = [
        field for field in required_asset_fields if asset_config.get(field) is None
    ]
    if missing_asset_fields:
        raise ValueError(f"Solar asset is missing fields: {missing_asset_fields}")

    weather = weather_df.copy()
    if weather.index.tz is None:
        raise ValueError("Solar weather timestamps must be timezone-aware.")
    # Calculate solar position for the given timestamps and asset location
    solar_position = pvlib.solarposition.get_solarposition(
        weather.index, asset_config["latitude"], asset_config["longitude"]
    )
    # Calculate total irradiance on the tilted surface using the solar position and weather data
    # what does tilt and azimuth mean: tilt is the angle of the PV panel from the horizontal, azimuth is the compass direction the panel faces              
    irradiance = pvlib.irradiance.get_total_irradiance(
        surface_tilt=asset_config["tilt"],
        surface_azimuth=asset_config["azimuth"],
        solar_zenith=solar_position["apparent_zenith"],
        solar_azimuth=solar_position["azimuth"],
        dni=weather["direct_normal_irradiance"].clip(lower=0),
        ghi=weather["shortwave_radiation"].clip(lower=0),
        dhi=weather["diffuse_radiation"].clip(lower=0),
    )

    # Calculate cell temperature using the Faiman model
    # Faiman model estimates the PV cell temperature based on the plane of array irradiance, ambient temperature, and wind speed.
    # why cell temperature is needed: PV module performance depends on the temperature of the solar cells, which affects the DC (direct current) power output. DC power decreases as cell temperature increases.
    cell_temperature = pvlib.temperature.faiman(
        poa_global=irradiance["poa_global"].clip(lower=0),
        temp_air=weather["temperature_2m"],
        wind_speed=weather["wind_speed_10m"].clip(lower=0),
    )

    # Convert the asset capacity from MW to W for DC power calculation
    capacity_w = float(asset_config["capacity_mw"]) * 1000000
    # Calculate the DC power output of the PV system using the PVWatts model
    dc_power_w = pvlib.pvsystem.pvwatts_dc(
        irradiance["poa_global"].clip(lower=0),
        cell_temperature,
        capacity_w,
        gamma_pdc=-0.003,
    )

    # Estimate system losses using the PVWatts loss model
    loss_fraction = pvlib.pvsystem.pvwatts_losses() / 100

    # Calculate the AC (alternating current) power output by applying system losses to the DC power
    production_mw = (dc_power_w * (1 - loss_fraction) / 1_000_000).clip(
        lower=0, upper=float(asset_config["capacity_mw"])
    )
    # Set production to zero when the sun is below the horizon
    production_mw = production_mw.where(solar_position["apparent_zenith"] < 90, 0.0)
    result = weather.copy()
    result["production_mw"] = production_mw
    result["capacity_factor"] = result["production_mw"] / float(asset_config["capacity_mw"])
    return result
