"""Build incremental regional weather tables for DE-LU Day-Ahead prices."""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd
import yaml

from src.pipelines.storage import atomic_write_parquet
from src.weather.de_lu_market_weather import (
    get_de_lu_j1_weather,
    get_de_lu_reanalysis_weather,
)
from src.weather.openmeteo_j1_forecast import J1ForecastDownloadError


PROJECT_ROOT = Path(__file__).resolve().parents[2]
CONFIG_PATH = PROJECT_ROOT / "config" / "day_ahead_price_features.yaml"
CONTRACT_PATH = PROJECT_ROOT / "config" / "operational_contract.yaml"
REANALYSIS_PATH = PROJECT_ROOT / "data" / "market_15m" / "de_lu_regional_reanalysis_weather_15m.parquet"
J1_PATH = PROJECT_ROOT / "data" / "market_15m" / "de_lu_regional_j1_forecast_weather_15m.parquet"


def _requested_ranges(
    existing: pd.DataFrame | None,
    start_date: str,
    end_date: str,
    timezone: str,
) -> list[tuple[str, str]]:
    """Return missing local delivery-date ranges and refresh the latest day."""
    requested = pd.date_range(start_date, end_date, freq="D")
    if existing is None or existing.empty:
        dates = requested
    else:
        local_dates = (
            pd.to_datetime(existing["delivery_start_utc"], utc=True)
            .dt.tz_convert(timezone)
            .dt.tz_localize(None)
            .dt.normalize()
        )
        complete_dates = set()
        for date, count in local_dates.value_counts().items():
            local_start = pd.Timestamp(date).tz_localize(timezone)
            local_end = local_start + pd.DateOffset(days=1)
            expected = len(
                pd.date_range(
                    local_start.tz_convert("UTC"),
                    local_end.tz_convert("UTC"),
                    freq="15min",
                    inclusive="left",
                )
            )
            if count == expected:
                complete_dates.add(date)
        missing = [date for date in requested if date not in complete_dates]
        latest = local_dates.max()
        dates = pd.DatetimeIndex(sorted(set(missing + [latest])))
        dates = dates[(dates >= requested.min()) & (dates <= requested.max())]

    if dates.empty:
        return []
    ranges = []
    range_start = previous = dates[0]
    for date in dates[1:]:
        if date - previous > pd.Timedelta(days=1):
            ranges.append((range_start.date().isoformat(), previous.date().isoformat()))
            range_start = date
        previous = date
    ranges.append((range_start.date().isoformat(), previous.date().isoformat()))
    return ranges


def _merge(existing: pd.DataFrame | None, batches: list[pd.DataFrame]) -> pd.DataFrame:
    """Keep the freshly collected value when a delivery interval is refreshed."""
    frames = batches if existing is None else [existing, *batches]
    return (
        pd.concat(frames, ignore_index=True)
        .drop_duplicates("delivery_start_utc", keep="last")
        .sort_values("delivery_start_utc")
        .reset_index(drop=True)
    )


def build_de_lu_market_weather(
    start_date: str,
    end_date: str,
    *,
    kind: str = "both",
    config_path: Path = CONFIG_PATH,
    contract_path: Path = CONTRACT_PATH,
) -> dict[str, int]:
    """Collect requested DE-LU weather tables without rebuilding model features."""
    source = yaml.safe_load(config_path.read_text(encoding="utf-8"))["de_lu_sources"]
    contract = yaml.safe_load(contract_path.read_text(encoding="utf-8"))["day_ahead"]
    weather = source["weather"]
    timezone = source["timezone"]
    result: dict[str, int] = {}

    if kind in {"reanalysis", "both"}:
        existing = pd.read_parquet(REANALYSIS_PATH) if REANALYSIS_PATH.exists() else None
        batches = [
            get_de_lu_reanalysis_weather(
                range_start, range_end, weather["points"], weather["variables"],
                timezone=timezone, model=weather["historical_model"],
            )
            .reset_index()
            .assign(
                weather_source="open_meteo_archive",
                weather_model=weather["historical_model"],
                weather_source_resolution_minutes=60,
                weather_resolution_minutes=15,
                weather_conversion="hourly_to_15m",
            )
            for range_start, range_end in _requested_ranges(existing, start_date, end_date, timezone)
        ]
        if batches:
            combined = _merge(existing, batches)
            atomic_write_parquet(combined, REANALYSIS_PATH)
            result["reanalysis_downloaded_rows"] = sum(len(batch) for batch in batches)
            result["reanalysis_total_rows"] = len(combined)

    if kind in {"j1", "both"}:
        existing = pd.read_parquet(J1_PATH) if J1_PATH.exists() else None
        batches = []
        unavailable_days = []
        for range_start, range_end in _requested_ranges(existing, start_date, end_date, timezone):
            for delivery_date in pd.date_range(range_start, range_end, freq="D"):
                cutoff = pd.Timestamp(
                    f"{(delivery_date - pd.Timedelta(days=1)).date()} "
                    f"{contract['observation_cutoff_local']}"
                ).tz_localize(timezone)
                try:
                    batch = get_de_lu_j1_weather(
                        delivery_date.date().isoformat(),
                        cutoff,
                        weather["points"],
                        weather["variables"],
                        timezone=timezone,
                        model=contract["weather_j1"]["model"],
                        publication_delay_hours=contract["weather_j1"]["publication_delay_hours"],
                    ).reset_index()
                except J1ForecastDownloadError:
                    unavailable_days.append(delivery_date.date().isoformat())
                    continue
                batch["weather_source"] = "open_meteo_single_runs"
                batch["weather_resolution_minutes"] = 15
                batches.append(batch)
        if batches:
            combined = _merge(existing, batches)
            atomic_write_parquet(combined, J1_PATH)
            result["j1_downloaded_rows"] = sum(len(batch) for batch in batches)
            result["j1_total_rows"] = len(combined)
        result["j1_unavailable_days"] = len(unavailable_days)
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--start-date", required=True)
    parser.add_argument("--end-date", required=True)
    parser.add_argument("--kind", choices=["reanalysis", "j1", "both"], default="both")
    args = parser.parse_args()
    print(build_de_lu_market_weather(args.start_date, args.end_date, kind=args.kind))


if __name__ == "__main__":
    main()
