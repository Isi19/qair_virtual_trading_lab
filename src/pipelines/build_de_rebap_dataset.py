"""Backfill or refresh final German reBAP settlement prices."""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from src.market_data.netztransparenz import fetch_de_rebap
from src.pipelines.storage import atomic_write_parquet


DEFAULT_OUTPUT_PATH = Path("data/market_15m/de_lu_imbalance_prices.parquet")


def build_de_rebap_dataset(
    start_date: str,
    end_date: str,
    *,
    env_path: str | Path = "refab.env",
    output_path: str | Path = DEFAULT_OUTPUT_PATH,
) -> pd.DataFrame:
    """Store one canonical, de-duplicated reBAP series.

    The requested window is deliberately re-downloaded.  This makes the
    pipeline incremental while allowing a later official re-publication to
    replace older values for the same delivery quarter-hour.
    """
    new_values = fetch_de_rebap(start_date, end_date, env_path=env_path)
    output = Path(output_path)
    existing = pd.read_parquet(output) if output.exists() else pd.DataFrame()
    combined = (
        pd.concat([existing, new_values], ignore_index=True)
        .drop_duplicates("delivery_start_utc", keep="last")
        .sort_values("delivery_start_utc")
        .reset_index(drop=True)
    )
    atomic_write_parquet(combined, output)
    return combined


def main() -> None:
    """Parse a delivery-date range and print a compact collection summary."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--start-date", required=True)
    parser.add_argument("--end-date", required=True)
    parser.add_argument("--env-path", default="refab.env")
    parser.add_argument("--output-path", default=str(DEFAULT_OUTPUT_PATH))
    args = parser.parse_args()

    result = build_de_rebap_dataset(
        args.start_date,
        args.end_date,
        env_path=args.env_path,
        output_path=args.output_path,
    )
    print(f"reBAP DE : {len(result)} lignes stockées -> {args.output_path}")


if __name__ == "__main__":
    main()
