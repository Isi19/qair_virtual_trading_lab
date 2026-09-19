"""Build archived J-1 weather forecasts for a delivery-period backtest."""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd
import yaml

from src.config import load_assets
from src.pipelines.storage import (
    atomic_write_parquet,
    incremental_start_date,
    merge_timestamp_frames,
    read_parquet_if_exists,
)
from src.weather.openmeteo_j1_forecast import J1ForecastDownloadError, get_j1_forecast_weather


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_DIRECTORY = PROJECT_ROOT / "data" / "asset_data"
INTERVAL = pd.Timedelta(minutes=15)


def load_j1_settings(contract_path: Path) -> dict:
    """Load the market cutoff and weather-run settings for the J-1 snapshot."""
    with contract_path.open(encoding="utf-8") as stream:
        day_ahead = (yaml.safe_load(stream) or {}).get("day_ahead", {})
    weather_j1 = day_ahead.get("weather_j1", {})
    required = ("observation_cutoff_local", "model", "publication_delay_hours")
    missing = [name for name in required if name not in day_ahead | weather_j1]
    if missing:
        raise ValueError(f"The operational contract is missing: {missing}")
    return {
        "cutoff_time": day_ahead["observation_cutoff_local"],
        "model": weather_j1["model"],
        "publication_delay_hours": int(weather_j1["publication_delay_hours"]),
    }


def cutoff_for_delivery(
    delivery_date: pd.Timestamp,
    market_timezone: str,
    cutoff_time: str,
) -> pd.Timestamp:
    """Return the local J-1 decision cutoff for one delivery date."""
    cutoff_date = delivery_date - pd.Timedelta(days=1)
    return pd.Timestamp(f"{cutoff_date.date()} {cutoff_time}").tz_localize(
        market_timezone
    )


def build_asset_j1_forecast_dataset(
    asset_id: str,
    asset_config: dict,
    j1_settings: dict,
    *,
    start_date: str,
    end_date: str,
    incremental: bool = True,
) -> dict[str, int | str]:
    """Create one J-1 forecast table for a site's delivery-period backtest."""
    reanalysis = asset_config["historical_reanalysis"]
    variables = reanalysis["feature_variables"]
    output_path = DATA_DIRECTORY / f"{asset_id}_j1_forecast_weather_15m.parquet"
    resume_start = (
        incremental_start_date([output_path], start_date, INTERVAL)
        if incremental
        else start_date
    )
    existing = read_parquet_if_exists(output_path)
    downloaded_rows = 0
    runs_downloaded = 0
    unavailable_days = 0

    for delivery_date in pd.date_range(resume_start, end_date, freq="D"):
        cutoff_timestamp = cutoff_for_delivery(
            delivery_date,
            asset_config["market_timezone"],
            j1_settings["cutoff_time"],
        )
        try:
            forecast = get_j1_forecast_weather(
                float(asset_config["latitude"]),
                float(asset_config["longitude"]),
                delivery_date,
                cutoff_timestamp,
                variables,
                model=j1_settings["model"],
                market_timezone=asset_config["market_timezone"],
                publication_delay_hours=j1_settings["publication_delay_hours"],
            )
        except J1ForecastDownloadError as error:
            unavailable_days += 1
            print(f"{asset_id}: delivery={delivery_date.date()}, unavailable ({error})")
            continue
        batch = forecast.reset_index()
        batch.insert(1, "site_id", asset_id)
        batch.insert(2, "technology", asset_config["technology"])
        batch.insert(3, "capacity_mw", float(asset_config["capacity_mw"]))
        existing = merge_timestamp_frames(existing, batch)
        atomic_write_parquet(existing, output_path)
        downloaded_rows += len(batch)
        runs_downloaded += 1
        print(
            f"{asset_id}: delivery={delivery_date.date()}, "
            f"run={forecast['forecast_run_timestamp_utc'].iloc[0]}"
        )

    return {
        "downloaded_rows": downloaded_rows,
        "runs_downloaded": runs_downloaded,
        "unavailable_days": unavailable_days,
        "total_rows": 0 if existing is None else len(existing),
        "resume_start": resume_start,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--asset", help="Build only one configured asset.")
    parser.add_argument("--start-date", required=True)
    parser.add_argument("--end-date", required=True)
    parser.add_argument("--force-from-start", action="store_true")
    parser.add_argument(
        "--config", type=Path, default=PROJECT_ROOT / "config" / "assets.yaml"
    )
    parser.add_argument(
        "--contract",
        type=Path,
        default=PROJECT_ROOT / "config" / "operational_contract.yaml",
    )
    args = parser.parse_args()

    assets = load_assets(args.config)
    if args.asset:
        assets = {args.asset: assets[args.asset]}
    else:
        assets = {
            asset_id: config
            for asset_id, config in assets.items()
            if "historical_reanalysis" in config
        }
    j1_settings = load_j1_settings(args.contract)

    for asset_id, asset_config in assets.items():
        summary = build_asset_j1_forecast_dataset(
            asset_id,
            asset_config,
            j1_settings,
            start_date=args.start_date,
            end_date=args.end_date,
            incremental=not args.force_from_start,
        )
        print(
            f"{asset_id}: {summary['runs_downloaded']} runs, "
            f"{summary['unavailable_days']} unavailable days, "
            f"{summary['downloaded_rows']} rows, "
            f"{summary['total_rows']} total rows"
        )


if __name__ == "__main__":
    main()
