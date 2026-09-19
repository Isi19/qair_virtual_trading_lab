"""Build canonical DE-LU system tables for price EDA and forecasting."""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd
import yaml

from src.market_data.de_lu_system import (
    get_de_lu_actual_system_features,
    get_de_lu_forecast_system_features,
)
from src.pipelines.storage import atomic_write_parquet


PROJECT_ROOT = Path(__file__).resolve().parents[2]
CONFIG_PATH = PROJECT_ROOT / "config" / "day_ahead_price_features.yaml"
CONTRACT_PATH = PROJECT_ROOT / "config" / "operational_contract.yaml"
ACTUAL_PATH = PROJECT_ROOT / "data" / "market_15m" / "de_lu_system_actual_15m.parquet"
J1_PATH = PROJECT_ROOT / "data" / "market_15m" / "de_lu_system_j1_load_forecast_15m.parquet"


def _download_ranges(
    existing: pd.DataFrame | None,
    start_date: str,
    end_date: str,
    timezone: str,
) -> list[tuple[str, str]]:
    """Backfill missing history or refresh the latest stored local day."""
    if existing is None or existing.empty:
        return [(start_date, end_date)]

    dates = (
        pd.to_datetime(existing["delivery_start_utc"], utc=True)
        .dt.tz_convert(timezone)
        .dt.tz_localize(None)
        .dt.normalize()
    )
    stored_start, stored_end = dates.min(), dates.max()
    requested_start, requested_end = pd.Timestamp(start_date), pd.Timestamp(end_date)
    ranges = []
    if requested_start < stored_start:
        ranges.append((requested_start.date().isoformat(), (stored_start - pd.Timedelta(days=1)).date().isoformat()))
    if requested_end >= stored_end:
        ranges.append((max(requested_start, stored_end).date().isoformat(), requested_end.date().isoformat()))
    return ranges


def _merge(existing: pd.DataFrame | None, batches: list[pd.DataFrame]) -> pd.DataFrame:
    """Merge refreshed delivery periods, keeping the newly collected values."""
    frames = batches if existing is None else [existing, *batches]
    return (
        pd.concat(frames, ignore_index=True)
        .drop_duplicates("delivery_start_utc", keep="last")
        .sort_values("delivery_start_utc")
        .reset_index(drop=True)
    )


def _forecast_cutoffs(timestamps: pd.Series, timezone: str, cutoff_time: str) -> pd.Series:
    """Attach the D-1 operational cutoff associated with each delivery row."""
    local_dates = pd.to_datetime(timestamps, utc=True).dt.tz_convert(timezone).dt.date
    return pd.to_datetime(
        [f"{pd.Timestamp(date) - pd.Timedelta(days=1):%Y-%m-%d} {cutoff_time}" for date in local_dates]
    ).tz_localize(timezone).tz_convert("UTC")


def build_de_lu_system_dataset(
    start_date: str,
    end_date: str,
    *,
    kind: str = "both",
    config_path: Path = CONFIG_PATH,
    contract_path: Path = CONTRACT_PATH,
) -> dict[str, int]:
    """Collect actual system data and the eligible load forecast in separate files."""
    settings = yaml.safe_load(config_path.read_text(encoding="utf-8"))["de_lu_sources"]
    cutoff_time = yaml.safe_load(contract_path.read_text(encoding="utf-8"))["day_ahead"]["observation_cutoff_local"]
    smard = settings["smard"]
    timezone = settings["timezone"]
    result: dict[str, int] = {}

    if kind in {"actual", "both"}:
        existing = pd.read_parquet(ACTUAL_PATH) if ACTUAL_PATH.exists() else None
        batches = []
        for range_start, range_end in _download_ranges(existing, start_date, end_date, timezone):
            batch = get_de_lu_actual_system_features(
                range_start, range_end, smard["actual_series"],
                region=smard["region"], timezone=timezone,
            ).reset_index()
            batch["source"] = "smard"
            batch["value_type"] = "actual"
            batches.append(batch)
        if batches:
            combined = _merge(existing, batches)
            atomic_write_parquet(combined, ACTUAL_PATH)
            result["actual_downloaded_rows"] = sum(len(batch) for batch in batches)
            result["actual_total_rows"] = len(combined)

    if kind in {"j1", "both"}:
        existing = pd.read_parquet(J1_PATH) if J1_PATH.exists() else None
        batches = []
        for range_start, range_end in _download_ranges(existing, start_date, end_date, timezone):
            batch = get_de_lu_forecast_system_features(
                range_start, range_end, smard["forecast_series"],
                region=smard["region"], timezone=timezone,
            ).reset_index()
            batch["forecast_cutoff_timestamp_utc"] = _forecast_cutoffs(
                batch["delivery_start_utc"], timezone, cutoff_time
            )
            batch["source"] = "smard"
            batch["value_type"] = "day_ahead_forecast"
            batch["availability_note"] = "SMARD load forecast published by 10:00 local D-1"
            batches.append(batch)
        if batches:
            combined = _merge(existing, batches)
            atomic_write_parquet(combined, J1_PATH)
            result["j1_downloaded_rows"] = sum(len(batch) for batch in batches)
            result["j1_total_rows"] = len(combined)
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--start-date", required=True)
    parser.add_argument("--end-date", required=True)
    parser.add_argument("--kind", choices=["actual", "j1", "both"], default="both")
    args = parser.parse_args()
    print(build_de_lu_system_dataset(args.start_date, args.end_date, kind=args.kind))


if __name__ == "__main__":
    main()
