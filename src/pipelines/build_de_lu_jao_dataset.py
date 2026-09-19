"""Store DE-LU JAO features separately from the price-model dataset."""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from src.market_data.jao_core import get_de_lu_jao_features
from src.pipelines.storage import atomic_write_parquet


PROJECT_ROOT = Path(__file__).resolve().parents[2]
OUTPUT_PATH = PROJECT_ROOT / "data" / "market_15m" / "de_lu_jao_historical_features_15m.parquet"
TRAINING_PATHS = [
    PROJECT_ROOT / "data" / "modeling" / "de_lu_day_ahead_price_training_full.parquet",
]
TIMEZONE = "Europe/Berlin"
# JAO's public d2CF endpoint has no data for this local delivery day. The gap
# stays visible; the price dataset drops this day instead of inventing values.
KNOWN_UNAVAILABLE_DATES = {pd.Timestamp("2024-06-25")}


def _stored_jao_data(paths: list[Path]) -> pd.DataFrame | None:
    """Reuse JAO columns already present in the former training datasets."""
    batches = []
    for path in paths:
        if not path.exists():
            continue
        frame = pd.read_parquet(path)
        columns = [name for name in frame if name.startswith("jao_")]
        if not columns:
            continue
        batch = frame[["delivery_start_utc", *columns]].copy()
        batch["source"] = "jao_core_publication_tool"
        batch["value_type"] = "historical_pre_clearing_publication"
        batches.append(batch)
    return _merge(None, batches) if batches else None


def _download_ranges(
    existing: pd.DataFrame | None,
    start_date: str,
    end_date: str,
) -> list[tuple[str, str]]:
    """Return all local days missing from the stored JAO table."""
    if existing is None or existing.empty:
        return [(start_date, end_date)]
    local_dates = (
        pd.to_datetime(existing["delivery_start_utc"], utc=True)
        .dt.tz_convert(TIMEZONE)
        .dt.tz_localize(None)
        .dt.normalize()
    )
    requested = pd.date_range(start_date, end_date, freq="D")
    counts = local_dates.value_counts()
    missing = []
    for date in requested:
        if date in KNOWN_UNAVAILABLE_DATES:
            continue
        local_start = date.tz_localize(TIMEZONE)
        expected = len(
            pd.date_range(
                local_start.tz_convert("UTC"),
                (local_start + pd.DateOffset(days=1)).tz_convert("UTC"),
                freq="15min",
                inclusive="left",
            )
        )
        if counts.get(date, 0) != expected:
            missing.append(date)

    if not missing:
        return []
    ranges = []
    first = previous = missing[0]
    for date in missing[1:]:
        if date - previous > pd.Timedelta(days=1):
            ranges.append((first.date().isoformat(), previous.date().isoformat()))
            first = date
        previous = date
    ranges.append((first.date().isoformat(), previous.date().isoformat()))
    return ranges


def _merge(existing: pd.DataFrame | None, batches: list[pd.DataFrame]) -> pd.DataFrame:
    """Keep the newly retrieved publication when a period is refreshed."""
    frames = batches if existing is None else [existing, *batches]
    return (
        pd.concat(frames, ignore_index=True)
        .drop_duplicates("delivery_start_utc", keep="last")
        .sort_values("delivery_start_utc")
        .reset_index(drop=True)
    )


def build_de_lu_jao_dataset(
    start_date: str,
    end_date: str,
    *,
    output_path: Path = OUTPUT_PATH,
    training_paths: list[Path] | None = None,
) -> dict[str, int]:
    """Create or extend the JAO table without rebuilding model features."""
    stored = _stored_jao_data(training_paths or TRAINING_PATHS)
    output = pd.read_parquet(output_path) if output_path.exists() else None
    existing = _merge(stored, [output]) if output is not None else stored
    batches = []
    for range_start, range_end in _download_ranges(existing, start_date, end_date):
        batch = get_de_lu_jao_features(range_start, range_end).reset_index()
        batch["source"] = "jao_core_publication_tool"
        batch["value_type"] = "historical_pre_clearing_publication"
        batches.append(batch)

    if existing is None and not batches:
        raise ValueError("No JAO data is available for the requested period.")
    combined = _merge(existing, batches) if batches else existing
    atomic_write_parquet(combined, output_path)
    return {
        "downloaded_rows": sum(len(batch) for batch in batches),
        "total_rows": len(combined),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--start-date", required=True)
    parser.add_argument("--end-date", required=True)
    args = parser.parse_args()
    print(build_de_lu_jao_dataset(args.start_date, args.end_date))


if __name__ == "__main__":
    main()
