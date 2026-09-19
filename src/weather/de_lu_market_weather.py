"""Collect wide regional weather tables for the DE-LU price study."""

from __future__ import annotations

from collections.abc import Mapping

import pandas as pd

from src.weather.openmeteo_j1_forecast import get_j1_forecast_weather
from src.weather.openmeteo_reanalysis import (
    get_historical_reanalysis_weather,
    hourly_to_15_minutes,
)


def _delivery_index(start_date: str, end_date: str, timezone: str) -> pd.DatetimeIndex:
    """Return every 15-minute interval of inclusive local delivery dates."""
    start = pd.Timestamp(start_date).tz_localize(timezone).tz_convert("UTC")
    end = (
        (pd.Timestamp(end_date) + pd.Timedelta(days=1))
        .tz_localize(timezone)
        .tz_convert("UTC")
    )
    return pd.date_range(start, end, freq="15min", inclusive="left")


def _wide_regional_frame(frames: Mapping[str, pd.DataFrame]) -> pd.DataFrame:
    """Prefix weather columns with their region and join them on UTC time."""
    return pd.concat(
        [frame.add_prefix(f"{region}_") for region, frame in frames.items()], axis=1
    )


def get_de_lu_reanalysis_weather(
    start_date: str,
    end_date: str,
    points: Mapping[str, Mapping[str, float | str]],
    variables: list[str],
    *,
    timezone: str = "Europe/Berlin",
    model: str = "era5",
) -> pd.DataFrame:
    """Return 15-minute ERA5 weather for all configured DE-LU regions."""
    expected_index = _delivery_index(start_date, end_date, timezone)
    frames = {}
    for region, point in points.items():
        hourly = get_historical_reanalysis_weather(
            float(point["latitude"]),
            float(point["longitude"]),
            expected_index[0].date(),
            expected_index[-1].date() + pd.Timedelta(days=1),
            variables,
            model=model,
        )
        frames[region] = hourly_to_15_minutes(hourly, expected_index)

    result = _wide_regional_frame(frames)
    result.index.name = "delivery_start_utc"
    return result


def get_de_lu_j1_weather(
    delivery_date: str,
    cutoff_timestamp: pd.Timestamp,
    points: Mapping[str, Mapping[str, float | str]],
    variables: list[str],
    *,
    timezone: str = "Europe/Berlin",
    model: str = "icon_d2",
    publication_delay_hours: int = 3,
) -> pd.DataFrame:
    """Return one cutoff-safe forecast snapshot for every DE-LU region."""
    frames = {}
    metadata: pd.DataFrame | None = None
    for region, point in points.items():
        forecast = get_j1_forecast_weather(
            float(point["latitude"]),
            float(point["longitude"]),
            delivery_date,
            cutoff_timestamp,
            variables,
            model=model,
            market_timezone=timezone,
            publication_delay_hours=publication_delay_hours,
        )
        if metadata is None:
            metadata = forecast[
                [
                    "forecast_run_timestamp_utc",
                    "forecast_cutoff_timestamp_utc",
                    "forecast_model",
                ]
            ]
        elif not forecast[
            [
                "forecast_run_timestamp_utc",
                "forecast_cutoff_timestamp_utc",
                "forecast_model",
            ]
        ].equals(metadata):
            raise ValueError("DE-LU regions must use the same forecast snapshot.")
        frames[region] = forecast[variables]

    result = _wide_regional_frame(frames)
    if metadata is None:
        raise ValueError("At least one DE-LU weather region is required.")
    result = result.join(metadata)
    result.index.name = "delivery_start_utc"
    return result
