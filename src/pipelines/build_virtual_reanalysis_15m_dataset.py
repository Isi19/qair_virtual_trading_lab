"""Build one 15-minute historical ERA5 dataset with a virtual production target."""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from src.config import load_assets
from src.generation import solar, wind
from src.pipelines.quality import validate_interval_dataset
from src.pipelines.storage import (
    atomic_write_parquet,
    incremental_start_date,
    merge_timestamp_frames,
    read_parquet_if_exists,
)
from src.weather.openmeteo_reanalysis import (
    get_historical_reanalysis_weather,
    hourly_to_15_minutes as convert_hourly_to_15_minutes,
)


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_DIRECTORY = PROJECT_ROOT / "data" / "asset_data"
INTERVAL = pd.Timedelta(minutes=15)


def hourly_to_15_minutes(
    hourly_weather: pd.DataFrame,
    start_date: str,
    end_date: str,
) -> pd.DataFrame:
    """Convert ERA5 hourly weather to a 15-minute grid for virtual generation.

    Continuous variables are linearly interpolated. Wind direction is treated
    on a circle. ERA5 radiation and precipitation describe the preceding hour,
    so they are spread across its four quarter-hours to preserve hourly energy
    and precipitation totals.
    """
    if hourly_weather.index.tz is None:
        raise ValueError("ERA5 timestamps must be timezone-aware.")

    expected_index = pd.date_range(
        pd.Timestamp(start_date, tz="UTC"),
        pd.Timestamp(end_date, tz="UTC")
        + pd.Timedelta(days=1)
        - INTERVAL,
        freq="15min",
    )
    weather_15m = convert_hourly_to_15_minutes(hourly_weather, expected_index)
    weather_15m.index.name = "timestamp"
    return weather_15m


def build_asset_reanalysis_dataset(
    asset_id: str,
    asset_config: dict,
    *,
    start_date: str | None,
    end_date: str,
    incremental: bool = True,
) -> dict[str, int | str]:
    """Create or extend the single historical ERA5 table for one virtual asset."""
    settings = asset_config.get("historical_reanalysis")
    if not isinstance(settings, dict):
        raise ValueError(f"Asset '{asset_id}' has no historical_reanalysis settings.")
    target_variables = settings.get("target_variables", [])
    feature_variables = settings.get("feature_variables", [])
    if not settings.get("model") or not target_variables or not feature_variables:
        raise ValueError(
            f"Asset '{asset_id}' must define a model, target_variables and feature_variables."
        )
    weather_variables = list(dict.fromkeys(target_variables + feature_variables))
    configured_start = pd.Timestamp(asset_config["historical_start_date"])
    requested_start = configured_start if start_date is None else max(
        configured_start, pd.Timestamp(start_date)
    )
    output_path = DATA_DIRECTORY / f"{asset_id}_historical_reanalysis_15m.parquet"
    resume_start = (
        incremental_start_date([output_path], requested_start, INTERVAL)
        if incremental
        else requested_start.date().isoformat()
    )
    existing = read_parquet_if_exists(output_path)

    downloaded_rows = 0
    if pd.Timestamp(resume_start) <= pd.Timestamp(end_date):
        source_end = (pd.Timestamp(end_date) + pd.Timedelta(days=1)).date()
        hourly_weather = get_historical_reanalysis_weather(
            float(asset_config["latitude"]),
            float(asset_config["longitude"]),
            resume_start,
            source_end,
            weather_variables,
            model=settings["model"],
        )
        weather_15m = hourly_to_15_minutes(hourly_weather, resume_start, end_date)
        target_weather = weather_15m[target_variables]
        production = (
            wind.transform(target_weather, asset_config)
            if asset_config["technology"] == "wind"
            else solar.transform(target_weather, asset_config)
        )
        dataset_batch = weather_15m.reset_index()
        dataset_batch.insert(1, "site_id", asset_id)
        dataset_batch.insert(2, "technology", asset_config["technology"])
        dataset_batch.insert(3, "capacity_mw", float(asset_config["capacity_mw"]))
        dataset_batch["production_mw"] = production["production_mw"].to_numpy()
        dataset_batch["capacity_factor"] = production["capacity_factor"].to_numpy()
        dataset_batch["weather_model"] = settings["model"]
        dataset_batch["weather_source_resolution_minutes"] = 60
        dataset_batch["weather_resolution_minutes"] = 15
        dataset_batch["weather_conversion"] = "hourly_to_15m"
        dataset_batch["weather_retrieved_at_utc"] = pd.Timestamp.now(tz="UTC").isoformat()
        existing = merge_timestamp_frames(existing, dataset_batch)
        validate_interval_dataset(
            existing,
            float(asset_config["capacity_mw"]),
            INTERVAL,
            require_continuity=True,
        )
        atomic_write_parquet(existing, output_path)
        downloaded_rows = len(dataset_batch)
        print(f"{asset_id}: {resume_start} -> {end_date} ({len(dataset_batch)} rows)")

    return {
        "downloaded_rows": downloaded_rows,
        "total_rows": 0 if existing is None else len(existing),
        "resume_start": resume_start,
    }


def rebuild_asset_production(asset_id: str, asset_config: dict) -> int:
    """Recalculate virtual production from the weather already stored on disk."""
    settings = asset_config.get("historical_reanalysis", {})
    target_variables = settings.get("target_variables", [])
    output_path = DATA_DIRECTORY / f"{asset_id}_historical_reanalysis_15m.parquet"
    dataset = read_parquet_if_exists(output_path)
    if dataset is None or dataset.empty:
        raise ValueError(f"No historical reanalysis dataset found for '{asset_id}'.")

    missing = set(target_variables).difference(dataset.columns)
    if missing:
        raise ValueError(f"Stored dataset is missing target weather variables: {sorted(missing)}")

    production = (
        wind.transform(dataset[target_variables], asset_config)
        if asset_config["technology"] == "wind"
        else solar.transform(dataset[target_variables], asset_config)
    )
    dataset["production_mw"] = production["production_mw"].to_numpy()
    dataset["capacity_factor"] = production["capacity_factor"].to_numpy()
    atomic_write_parquet(dataset, output_path)
    return len(dataset)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--asset", help="Build only one configured asset.")
    parser.add_argument("--start-date")
    parser.add_argument(
        "--end-date",
        default=(pd.Timestamp.now(tz="UTC") - pd.Timedelta(days=7)).date().isoformat(),
    )
    parser.add_argument("--force-from-start", action="store_true")
    parser.add_argument(
        "--rebuild-production",
        action="store_true",
        help="Recalculate production from the weather already stored for one asset.",
    )
    parser.add_argument(
        "--config", type=Path, default=PROJECT_ROOT / "config" / "assets.yaml"
    )
    args = parser.parse_args()
    if args.rebuild_production and not args.asset:
        parser.error("--rebuild-production requires --asset.")

    assets = load_assets(args.config)
    if args.asset:
        assets = {args.asset: assets[args.asset]}
    else:
        assets = {
            asset_id: config
            for asset_id, config in assets.items()
            if "historical_reanalysis" in config
        }
    for asset_id, asset_config in assets.items():
        if args.rebuild_production:
            rows = rebuild_asset_production(asset_id, asset_config)
            print(f"{asset_id}: {rows} rows rebuilt")
            continue
        summary = build_asset_reanalysis_dataset(
            asset_id,
            asset_config,
            start_date=args.start_date,
            end_date=args.end_date,
            incremental=not args.force_from_start,
        )
        print(
            f"{asset_id}: {summary['downloaded_rows']} downloaded, "
            f"{summary['total_rows']} total rows, resume={summary['resume_start']}"
        )


if __name__ == "__main__":
    main()
