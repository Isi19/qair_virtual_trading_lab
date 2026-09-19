"""Calculate portfolio forecast exposure and diversification from the joined table."""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from src.analysis.portfolio import (
    add_portfolio_exposure_metrics,
    portfolio_exposure_summary,
)
from src.pipelines.storage import atomic_write_parquet


PROJECT_ROOT = Path(__file__).resolve().parents[2]
INPUT_PATH = (
    PROJECT_ROOT
    / "data"
    / "backtesting"
    / "portfolio"
    / "de_lu_portfolio_backtest_20260701_20260831.parquet"
)
DETAIL_OUTPUT_PATH = (
    PROJECT_ROOT
    / "data"
    / "backtesting"
    / "portfolio"
    / "de_lu_portfolio_exposure_20260701_20260831.parquet"
)
SUMMARY_OUTPUT_PATH = (
    PROJECT_ROOT
    / "data"
    / "backtesting"
    / "portfolio"
    / "de_lu_portfolio_exposure_summary_20260701_20260831.parquet"
)


def build_portfolio_exposure(
    input_path: Path = INPUT_PATH,
    detail_output_path: Path = DETAIL_OUTPUT_PATH,
    summary_output_path: Path = SUMMARY_OUTPUT_PATH,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Write one detailed table and one one-row summary for the portfolio risk."""
    portfolio = pd.read_parquet(input_path)
    detailed = add_portfolio_exposure_metrics(portfolio)
    summary = portfolio_exposure_summary(detailed)
    atomic_write_parquet(detailed, detail_output_path)
    atomic_write_parquet(summary, summary_output_path)
    return detailed, summary


def main() -> None:
    """Build portfolio exposure files and display the main metrics."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-path", type=Path, default=INPUT_PATH)
    parser.add_argument("--detail-output-path", type=Path, default=DETAIL_OUTPUT_PATH)
    parser.add_argument("--summary-output-path", type=Path, default=SUMMARY_OUTPUT_PATH)
    args = parser.parse_args()

    _, summary = build_portfolio_exposure(
        input_path=args.input_path,
        detail_output_path=args.detail_output_path,
        summary_output_path=args.summary_output_path,
    )
    print(summary.round(3).to_string(index=False))


if __name__ == "__main__":
    main()
