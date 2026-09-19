"""Safe incremental Parquet storage for time-series pipelines."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Iterable

import pandas as pd


TIMESTAMP_COLUMN = "timestamp"


def normalize_timestamp_frame(frame: pd.DataFrame) -> pd.DataFrame:
    """Return a UTC, de-duplicated dataframe sorted by timestamp."""
    if TIMESTAMP_COLUMN not in frame.columns:
        raise ValueError("Dataset must include timestamp.")
    normalized = frame.copy()
    normalized[TIMESTAMP_COLUMN] = pd.to_datetime(
        normalized[TIMESTAMP_COLUMN], utc=True
    )
    return (
        normalized.drop_duplicates(subset=[TIMESTAMP_COLUMN], keep="last")
        .sort_values(TIMESTAMP_COLUMN)
        .reset_index(drop=True)
    )


def merge_timestamp_frames(
    existing: pd.DataFrame | None, new: pd.DataFrame
) -> pd.DataFrame:
    
    """Merge two timestamp datasets deterministically, keeping new values."""

    frames = [new] if existing is None or existing.empty else [existing, new]
    return normalize_timestamp_frame(pd.concat(frames, ignore_index=True))


def atomic_write_parquet(frame: pd.DataFrame, path: str | Path) -> None:
    """Write a Parquet file atomically so an interrupted run keeps the old file."""
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = output_path.with_name(f".{output_path.name}.tmp")
    frame.to_parquet(temporary_path, index=False)
    os.replace(temporary_path, output_path)


def read_parquet_if_exists(path: str | Path) -> pd.DataFrame | None:
    """Read a Parquet file if it exists, otherwise return None."""
    input_path = Path(path)
    if not input_path.exists():
        return None
    return normalize_timestamp_frame(pd.read_parquet(input_path))


def _next_incomplete_date(
    path: Path, configured_start: pd.Timestamp, interval: pd.Timedelta
) -> pd.Timestamp:
    """Find the first date that is missing or incomplete in one stored dataset."""
    if not path.exists():
        return configured_start
    stored = pd.read_parquet(path, columns=[TIMESTAMP_COLUMN])
    if stored.empty:
        return configured_start
    timestamps = (
        pd.to_datetime(stored[TIMESTAMP_COLUMN], utc=True)
        .drop_duplicates()
        .sort_values()
        .reset_index(drop=True)
    )
    relevant = timestamps[timestamps >= configured_start]
    if relevant.empty or relevant.iloc[0] > configured_start:
        return configured_start
    expected_through_last = pd.date_range(
        configured_start, relevant.iloc[-1], freq=interval, tz="UTC"
    )
    missing = expected_through_last.difference(pd.DatetimeIndex(relevant))
    if len(missing):
        return missing[0].normalize()
    last_day = relevant.iloc[-1].normalize()
    day_values = relevant[relevant.dt.normalize() == last_day]
    expected = pd.date_range(
        last_day,
        last_day + pd.Timedelta(days=1) - interval,
        freq=interval,
        tz="UTC",
    )
    day_is_complete = len(day_values) == len(expected) and set(day_values) == set(expected)
    candidate = last_day + pd.Timedelta(days=1) if day_is_complete else last_day
    return max(configured_start, candidate)


def incremental_start_date(
    paths: Iterable[str | Path],
    configured_start: str | pd.Timestamp,
    interval: pd.Timedelta,
) -> str:
    """Return the earliest resume date required to keep all outputs aligned."""
    start = pd.Timestamp(configured_start, tz="UTC").normalize()
    candidates = [_next_incomplete_date(Path(path), start, interval) for path in paths]
    return min(candidates, default=start).date().isoformat()
