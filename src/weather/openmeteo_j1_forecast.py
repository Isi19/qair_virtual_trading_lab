"""Retrieve one archived weather-model run available before a J-1 cutoff."""

from __future__ import annotations

import time
from collections.abc import Iterable

import pandas as pd
import requests


SINGLE_RUNS_URL = "https://single-runs-api.open-meteo.com/v1/forecast"


class J1ForecastDownloadError(RuntimeError):
    """Raised when an archived weather-model run cannot support the request."""


def _as_utc(timestamp: str | pd.Timestamp) -> pd.Timestamp:
    """Return a timezone-aware timestamp converted to UTC."""
    value = pd.Timestamp(timestamp)
    if value.tzinfo is None:
        raise ValueError("The cutoff timestamp must include a timezone.")
    return value.tz_convert("UTC")


def select_latest_available_run(
    cutoff_timestamp: str | pd.Timestamp,
    *,
    run_frequency_hours: int = 3,
    publication_delay_hours: int = 3,
) -> pd.Timestamp:
    """Select the latest model initialisation safely available before cutoff.

    The delay represents the maximum time allowed for the model computation and
    distribution. For ICON D2, a three-hour delay is deliberately conservative.
    """
    latest_available_initialisation = _as_utc(cutoff_timestamp) - pd.Timedelta(
        hours=publication_delay_hours
    )
    return latest_available_initialisation.floor(f"{run_frequency_hours}h")


def _delivery_index(
    delivery_date: str | pd.Timestamp,
    market_timezone: str,
) -> pd.DatetimeIndex:
    """Build all 15-minute delivery intervals for one local market day."""
    local_start = pd.Timestamp(delivery_date).tz_localize(market_timezone)
    local_end = local_start + pd.DateOffset(days=1)
    return pd.date_range(
        local_start.tz_convert("UTC"),
        local_end.tz_convert("UTC") - pd.Timedelta(minutes=15),
        freq="15min",
    )


def _request_minutely_forecast(params: dict[str, str]) -> pd.DataFrame:
    """Request a single archived run and return its 15-minute forecast grid."""
    response: requests.Response | None = None
    for attempt in range(3):
        try:
            response = requests.get(SINGLE_RUNS_URL, params=params, timeout=60)
            response.raise_for_status()
            break
        except requests.RequestException as exc:
            if attempt == 2:
                raise J1ForecastDownloadError("Open-Meteo single-run request failed.") from exc
            time.sleep(2**attempt)

    if response is None:
        raise J1ForecastDownloadError("Open-Meteo single-run request did not return a response.")

    forecast = response.json().get("minutely_15")
    # Ensure that the forecast contains the expected time series data
    if not isinstance(forecast, dict) or "time" not in forecast:
        raise J1ForecastDownloadError("Open-Meteo did not return 15-minute forecast data.")

    weather = pd.DataFrame(forecast)
    # Create a timestamp column from the "time" array in the forecast data
    weather["timestamp"] = pd.to_datetime(weather.pop("time"), utc=True)
    # Convert the time column to a timezone-aware datetime index in UTC
    weather = weather.set_index("timestamp").sort_index()
    if weather.empty or weather.index.has_duplicates:
        raise J1ForecastDownloadError("Open-Meteo returned an invalid forecast time index.")
    return weather


def get_j1_forecast_weather(
    latitude: float,
    longitude: float,
    delivery_date: str | pd.Timestamp,
    cutoff_timestamp: str | pd.Timestamp,
    variables: Iterable[str],
    *,
    model: str = "icon_d2",
    market_timezone: str = "Europe/Berlin",
    run_frequency_hours: int = 3,
    publication_delay_hours: int = 3,
    forecast_horizon_hours: int = 48,
) -> pd.DataFrame:
    """Return one D-1 run's weather forecast for the full delivery day.

    The output index is UTC. Metadata columns are retained so they persist when
    the pipeline writes the result to Parquet.
    """
    requested_variables = list(dict.fromkeys(variables))
    if not requested_variables:
        raise ValueError("At least one forecast weather variable is required.")

    latest_run = select_latest_available_run(
        cutoff_timestamp,
        run_frequency_hours=run_frequency_hours,
        publication_delay_hours=publication_delay_hours,
    )
    delivery_index = _delivery_index(delivery_date, market_timezone)

    # An archived run can exceptionally be incomplete. Use the most recent
    # earlier run that contains the full delivery day instead of filling data.
    for attempt in range(4):
        run_timestamp = latest_run - pd.Timedelta(hours=attempt * run_frequency_hours)
        if delivery_index[-1] >= run_timestamp + pd.Timedelta(hours=forecast_horizon_hours):
            continue
        params = {
            "latitude": str(latitude),
            "longitude": str(longitude),
            "run": run_timestamp.strftime("%Y-%m-%dT%H:%M"),
            "minutely_15": ",".join(requested_variables),
            "models": model,
            "timezone": "GMT",
            "wind_speed_unit": "ms",
        }
        try:
            weather = _request_minutely_forecast(params)
        except J1ForecastDownloadError:
            continue
        missing = [name for name in requested_variables if name not in weather]
        if missing:
            continue
        weather = weather.reindex(delivery_index)
        if weather.isna().any().any():
            continue

        weather["forecast_run_timestamp_utc"] = run_timestamp
        weather["forecast_cutoff_timestamp_utc"] = _as_utc(cutoff_timestamp)
        weather["forecast_model"] = model
        weather.index.name = "timestamp"
        return weather

    raise J1ForecastDownloadError("No run before the cutoff covers the full delivery day.")
