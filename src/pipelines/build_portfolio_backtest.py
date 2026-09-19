"""Assemble the active DE-LU wind and solar forecasts into one portfolio table."""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from src.pipelines.storage import atomic_write_parquet


PROJECT_ROOT = Path(__file__).resolve().parents[2]
WIND_PATH = (
    PROJECT_ROOT
    / "data"
    / "backtesting"
    / "production_forecasts"
    / "langer_wald_j1_forecasts_20260701_20260831.parquet"
)
SOLAR_PATH = (
    PROJECT_ROOT
    / "data"
    / "backtesting"
    / "production_forecasts"
    / "perleberg_j1_forecasts_20260701_20260831.parquet"
)
PRICE_PATH = (
    PROJECT_ROOT
    / "data"
    / "backtesting"
    / "price_forecasts"
    / "de_lu_price_forecasts_20260701_20260831.parquet"
)
OUTPUT_PATH = (
    PROJECT_ROOT
    / "data"
    / "backtesting"
    / "portfolio"
    / "de_lu_portfolio_backtest_20260701_20260831.parquet"
)


def load_asset_forecast(path: Path, asset_name: str) -> pd.DataFrame:
    """Load one asset and prefix its actual and forecast production columns."""
    columns = [
        "delivery_start_utc",
        "actual_generation_mw",
        "generation_p10_mw",
        "generation_p50_mw",
        "generation_p90_mw",
    ]
    asset = pd.read_parquet(path, columns=columns)
    return asset.rename(
        columns={
            "actual_generation_mw": f"{asset_name}_actual_mw",
            "generation_p10_mw": f"{asset_name}_p10_mw",
            "generation_p50_mw": f"{asset_name}_p50_mw",
            "generation_p90_mw": f"{asset_name}_p90_mw",
        }
    )


def build_portfolio_backtest(
    wind_path: Path = WIND_PATH,
    solar_path: Path = SOLAR_PATH,
    price_path: Path = PRICE_PATH,
    output_path: Path = OUTPUT_PATH,
) -> pd.DataFrame:
    """Join both active assets with prices and calculate portfolio MW and MWh."""
    wind = load_asset_forecast(wind_path, "wind")
    solar = load_asset_forecast(solar_path, "solar")
    price_columns = [
        "delivery_start_utc",
        "actual_day_ahead_price_eur_mwh",
        "price_p10_eur_mwh",
        "price_p50_eur_mwh",
        "price_p90_eur_mwh",
    ]
    price = pd.read_parquet(price_path, columns=price_columns)

    portfolio = wind.merge(solar, on="delivery_start_utc").merge(
        price, on="delivery_start_utc"
    )
    for scenario in ("actual", "p10", "p50", "p90"):
        portfolio[f"portfolio_{scenario}_mw"] = (
            portfolio[f"wind_{scenario}_mw"] + portfolio[f"solar_{scenario}_mw"]
        )
        # A row covers a 15-minute delivery period.
        portfolio[f"portfolio_{scenario}_mwh"] = (
            portfolio[f"portfolio_{scenario}_mw"] * 0.25
        )

    portfolio = portfolio.sort_values("delivery_start_utc").reset_index(drop=True)
    atomic_write_parquet(portfolio, output_path)
    return portfolio


def main() -> None:
    """Build the portfolio table and print a short sample."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--wind-path", type=Path, default=WIND_PATH)
    parser.add_argument("--solar-path", type=Path, default=SOLAR_PATH)
    parser.add_argument("--price-path", type=Path, default=PRICE_PATH)
    parser.add_argument("--output-path", type=Path, default=OUTPUT_PATH)
    args = parser.parse_args()

    portfolio = build_portfolio_backtest(
        wind_path=args.wind_path,
        solar_path=args.solar_path,
        price_path=args.price_path,
        output_path=args.output_path,
    )
    print(f"{len(portfolio)} lignes écrites dans {args.output_path}")
    print(portfolio.head(4).to_string(index=False))


if __name__ == "__main__":
    main()
