"""Build one wide DE-LU neighbouring-price file for market EDA."""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd
import yaml

from src.market_data.de_lu_neighbor_prices import get_de_lu_neighbor_prices
from src.pipelines.storage import atomic_write_parquet


DEFAULT_CONFIG_PATH = Path("config/day_ahead_price_features.yaml")
DEFAULT_OUTPUT_PATH = Path(
    "data/market_15m/de_lu_neighbor_day_ahead_prices.parquet"
)


def build_de_lu_neighbor_price_dataset(
    start_date: str,
    end_date: str,
    *,
    config_path: str | Path = DEFAULT_CONFIG_PATH,
    output_path: str | Path = DEFAULT_OUTPUT_PATH,
) -> dict:
    """Backfill or extend the six Core-neighbour prices in one Parquet file."""
    with Path(config_path).open(encoding="utf-8") as config_file:
        source_config = yaml.safe_load(config_file)["de_lu_sources"]

    smard = source_config["smard"]
    output = Path(output_path)
    existing = pd.read_parquet(output) if output.exists() else None

    requested_start = pd.Timestamp(start_date)
    requested_end = pd.Timestamp(end_date)
    download_ranges: list[tuple[str, str]] = []

    if existing is None or existing.empty:
        download_ranges.append((start_date, end_date))
    else:
        stored_local_dates = pd.to_datetime(
            existing["delivery_start_utc"], utc=True
        ).dt.tz_convert(source_config["timezone"]).dt.tz_localize(None).dt.normalize()
        stored_start = stored_local_dates.min()
        stored_end = stored_local_dates.max()

        # Download only the missing history before the first stored day.
        if requested_start < stored_start:
            backfill_end = min(requested_end, stored_start - pd.Timedelta(days=1))
            download_ranges.append(
                (requested_start.date().isoformat(), backfill_end.date().isoformat())
            )

        # Refresh the latest stored day, then append any newer days.
        if requested_end >= stored_end:
            extension_start = max(requested_start, stored_end)
            download_ranges.append(
                (extension_start.date().isoformat(), requested_end.date().isoformat())
            )

    downloaded_frames: list[pd.DataFrame] = []
    for range_start, range_end in download_ranges:
        downloaded_frames.append(
            get_de_lu_neighbor_prices(
                range_start,
                range_end,
                smard["neighbor_price_series"],
                timezone=source_config["timezone"],
                region=smard["region"],
            ).reset_index()
        )

    frames = downloaded_frames if existing is None else [existing, *downloaded_frames]
    prices = (
        pd.concat(frames, ignore_index=True)
        .drop_duplicates("delivery_start_utc", keep="last")
        .sort_values("delivery_start_utc")
        .reset_index(drop=True)
    )
    atomic_write_parquet(prices, output)
    return {
        "rows": len(prices),
        "downloaded_rows": sum(len(frame) for frame in downloaded_frames),
        "columns": len(prices.columns) - 1,
        "output_path": str(output),
    }


def main() -> None:
    """Parse command-line dates and build the neighbouring-price dataset."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--start-date", required=True)
    parser.add_argument("--end-date", required=True)
    parser.add_argument("--config", default=str(DEFAULT_CONFIG_PATH))
    parser.add_argument("--output-path", default=str(DEFAULT_OUTPUT_PATH))
    args = parser.parse_args()

    result = build_de_lu_neighbor_price_dataset(
        args.start_date,
        args.end_date,
        config_path=args.config,
        output_path=args.output_path,
    )
    print(
        f"Prix voisins DE-LU : {result['downloaded_rows']} lignes téléchargées, "
        f"{result['rows']} lignes stockées, "
        f"{result['columns']} zones -> {result['output_path']}"
    )


if __name__ == "__main__":
    main()
