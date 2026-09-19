"""Dataset quality checks shared by the historical pipeline and tests."""

from __future__ import annotations

import pandas as pd


def validate_hourly_dataset(dataset: pd.DataFrame, capacity_mw: float) -> dict[str, int]:
    """Validate timestamp continuity and physical generation bounds."""
    return validate_interval_dataset(dataset, capacity_mw, pd.Timedelta(hours=1))


def validate_interval_dataset(
    dataset: pd.DataFrame,
    capacity_mw: float,
    expected_interval: pd.Timedelta,
    *,
    require_continuity: bool = True,
) -> dict[str, int]:
    """Validate timestamp continuity, physical bounds, and an expected interval."""
    if "timestamp" not in dataset.columns:
        raise ValueError("Dataset must include timestamp.")
    timestamps = pd.to_datetime(dataset["timestamp"], utc=True)
    if timestamps.duplicated().any():
        raise ValueError("Dataset contains duplicate timestamps.")
    if not timestamps.is_monotonic_increasing:
        raise ValueError("Dataset timestamps are not sorted.")
    if dataset["production_mw"].isna().any() or dataset["capacity_factor"].isna().any():
        raise ValueError("Dataset contains missing generation values.")
    if (dataset["production_mw"] < 0).any() or (dataset["production_mw"] > capacity_mw).any():
        raise ValueError("Production is outside [0, capacity_mw].")
    if (dataset["capacity_factor"] < 0).any() or (dataset["capacity_factor"] > 1).any():
        raise ValueError("Capacity factor is outside [0, 1].")

    deltas = timestamps.diff().dropna()
    irregular_intervals = int(deltas.ne(expected_interval).sum())
    if require_continuity and irregular_intervals:
        raise ValueError(
            f"Dataset contains {irregular_intervals} intervals different from "
            f"{expected_interval}."
        )
    missing_intervals = int(deltas.gt(expected_interval).sum())
    weather_columns = [
        column
        for column in dataset.columns
        if column
        not in {
            "timestamp", "site_id", "country", "latitude", "longitude", "technology",
            "capacity_mw", "production_mw", "capacity_factor",
        }
    ]
    missing_weather_values = int(dataset[weather_columns].isna().sum().sum())
    if missing_weather_values:
        raise ValueError(
            f"Dataset contains {missing_weather_values} missing weather or provenance values."
        )
    return {
        "missing_intervals": missing_intervals,
        "missing_hours": missing_intervals,
        "missing_weather_values": missing_weather_values,
    }
