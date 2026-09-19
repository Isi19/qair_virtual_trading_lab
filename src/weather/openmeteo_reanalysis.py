"""Download historical ERA5 weather for virtual production targets."""

from __future__ import annotations

import time
from collections.abc import Iterable

import numpy as np
import pandas as pd
import requests


HISTORICAL_WEATHER_URL = "https://archive-api.open-meteo.com/v1/archive"
RADIATION_COLUMNS = {
    "shortwave_radiation",
    "direct_radiation",
    "diffuse_radiation",
    "direct_normal_irradiance",
}


class ReanalysisDownloadError(RuntimeError):
    """Raised when Open-Meteo does not provide a usable ERA5 response."""


def hourly_to_15_minutes(
    hourly_weather: pd.DataFrame,
    expected_index: pd.DatetimeIndex,
) -> pd.DataFrame:
    """Place hourly Open-Meteo weather on a specified 15-minute UTC grid.

    Temperatures, wind speeds and cloud cover are interpolated. Radiation is
    held constant across the four quarters of its reporting hour, preserving
    its hourly mean. Precipitation is divided across the four quarters because
    it is an hourly accumulation.
    """
    if hourly_weather.index.tz is None or expected_index.tz is None:
        raise ValueError("Weather and target timestamps must be timezone-aware.")

    source = hourly_weather.sort_index()
    expanded_index = pd.date_range(
        source.index.min(), source.index.max(), freq="15min", tz="UTC"
    )
    result = source.reindex(expanded_index)
    directions = [name for name in source if name.startswith("wind_direction_")]
    radiation = list(RADIATION_COLUMNS.intersection(source.columns))
    precipitation = [name for name in source if name == "precipitation"]
    continuous = [
        name
        for name in source
        if name not in directions + radiation + precipitation
    ]

    result[continuous] = result[continuous].interpolate(
        method="time", limit_area="inside"
    )
    for name in directions:
        angles = np.deg2rad(source[name])
        sine = pd.Series(np.sin(angles), index=source.index).reindex(expanded_index)
        cosine = pd.Series(np.cos(angles), index=source.index).reindex(expanded_index)
        result[name] = np.rad2deg(
            np.arctan2(
                sine.interpolate(method="time", limit_area="inside"),
                cosine.interpolate(method="time", limit_area="inside"),
            )
        ) % 360

    for name in radiation + precipitation:
        values = source[name].copy()
        values.index = values.index - pd.Timedelta(hours=1)
        values = values.reindex(expanded_index).ffill().clip(lower=0)
        result[name] = values / 4 if name in precipitation else values

    result = result.reindex(expected_index)
    missing = result.columns[result.isna().any()].tolist()
    if missing:
        raise ValueError(f"Hourly weather conversion contains missing values: {missing}")
    return result


def _request_hourly_weather(params: dict[str, str]) -> pd.DataFrame:
    """Request ERA5 data and return it on a UTC timestamp index."""
    response: requests.Response | None = None
    for attempt in range(3):
        try:
            response = requests.get(
                HISTORICAL_WEATHER_URL,
                params=params,
                timeout=60,
            )
            response.raise_for_status()
            break
        except requests.RequestException as exc:
            if attempt == 2:
                raise ReanalysisDownloadError("Open-Meteo ERA5 request failed.") from exc
            time.sleep(2**attempt)

    if response is None:
        raise ReanalysisDownloadError("Open-Meteo ERA5 request did not return a response.")

    hourly = response.json().get("hourly")
    if not isinstance(hourly, dict) or "time" not in hourly:
        raise ReanalysisDownloadError("Open-Meteo did not return hourly ERA5 data.")

    weather = pd.DataFrame(hourly)
    weather["timestamp"] = pd.to_datetime(weather.pop("time"), utc=True)
    weather = weather.set_index("timestamp").sort_index()
    if weather.empty or weather.index.has_duplicates:
        raise ReanalysisDownloadError("Open-Meteo returned an invalid ERA5 time index.")
    return weather


def get_historical_reanalysis_weather(
    latitude: float,
    longitude: float,
    start_date: str | pd.Timestamp,
    end_date: str | pd.Timestamp,
    variables: Iterable[str],
    *,
    model: str = "era5",
) -> pd.DataFrame:
    """Download hourly ERA5 weather for one site and one historical period.

    This function only downloads and structures weather data. Interpolation to
    15 minutes and virtual production calculations belong to the pipeline.
    """
    requested_variables = list(dict.fromkeys(variables))
    if not requested_variables:
        raise ValueError("At least one reanalysis weather variable is required.")

    params = {
        "latitude": str(latitude),
        "longitude": str(longitude),
        "start_date": pd.Timestamp(start_date).date().isoformat(),
        "end_date": pd.Timestamp(end_date).date().isoformat(),
        "hourly": ",".join(requested_variables),
        "models": model,
        "timezone": "GMT",
        "wind_speed_unit": "ms",
    }
    weather = _request_hourly_weather(params)
    missing = [name for name in requested_variables if name not in weather]
    if missing:
        raise ReanalysisDownloadError(
            f"Historical reanalysis is missing variables: {missing}"
        )

    weather.attrs["weather_model"] = model
    weather.attrs["source_resolution_minutes"] = 60
    weather.attrs["is_native_resolution"] = True
    return weather
