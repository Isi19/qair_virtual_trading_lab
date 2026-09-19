"""Build the information set available before the Day-Ahead nomination."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from src.pipelines.add_nomination_market_risk import attach_market_spreads
from src.pipelines.storage import atomic_write_parquet


DEFAULT_PORTFOLIO_PATH = Path(
    "data/backtesting/portfolio/de_lu_portfolio_backtest_20260701_20260831.parquet"
)
DEFAULT_OUTPUT_PATH = Path(
    "data/backtesting/nomination/de_lu_nomination_ex_ante_20260701_20260831.parquet"
)
DEFAULT_REBAP_PATH = Path("data/market_15m/de_lu_imbalance_prices.parquet")
DEFAULT_DA_PRICE_PATH = Path("data/market_15m/de_lu_day_ahead_prices.parquet")


def build_nomination_ex_ante(
    *,
    portfolio_path: str | Path = DEFAULT_PORTFOLIO_PATH,
    rebap_path: str | Path = DEFAULT_REBAP_PATH,
    day_ahead_price_path: str | Path = DEFAULT_DA_PRICE_PATH,
    output_path: str | Path = DEFAULT_OUTPUT_PATH,
) -> pd.DataFrame:
    """Build the complete ex-ante scenario table in one pass."""
    columns = [
        "delivery_start_utc",
        "portfolio_p10_mwh",
        "portfolio_p50_mwh",
        "portfolio_p90_mwh",
        "price_p10_eur_mwh",
        "price_p50_eur_mwh",
        "price_p90_eur_mwh",
    ]
    information_set = pd.read_parquet(portfolio_path, columns=columns)
    scenario_weights = {"p10": 0.25, "p50": 0.50, "p90": 0.25}

    candidates = []
    for strategy_name, quantile in (
        ("p10_prudente", "p10"),
        ("p50_centrale", "p50"),
        ("p90_agressive", "p90"),
    ):
        for production_quantile, production_weight in scenario_weights.items():
            for price_quantile, price_weight in scenario_weights.items():
                candidate = information_set.copy()
                candidate["strategy_name"] = strategy_name
                candidate["nomination_mwh"] = candidate[f"portfolio_{quantile}_mwh"]
                candidate["production_scenario"] = production_quantile
                candidate["price_scenario"] = price_quantile
                candidate["production_scenario_mwh"] = candidate[
                    f"portfolio_{production_quantile}_mwh"
                ]
                candidate["price_scenario_eur_mwh"] = candidate[
                    f"price_{price_quantile}_eur_mwh"
                ]
                candidate["scenario_weight"] = production_weight * price_weight
                candidates.append(candidate)

    scenarios = pd.concat(candidates, ignore_index=True).sort_values(
        ["delivery_start_utc", "strategy_name", "production_scenario", "price_scenario"]
    ).reset_index(drop=True)
    rebap = pd.read_parquet(rebap_path)
    day_ahead_price = pd.read_parquet(
        day_ahead_price_path,
        columns=["delivery_start_utc", "day_ahead_price"],
    ).rename(columns={"day_ahead_price": "actual_day_ahead_price_eur_mwh"})
    result = attach_market_spreads(scenarios, rebap, day_ahead_price)

    # A negative net imbalance means that production is below the nomination.
    # Keep the two directions explicit because they have different economics.
    imbalance = result["production_scenario_mwh"] - result["nomination_mwh"]
    result["scenario_shortfall_mwh"] = (
        result["nomination_mwh"] - result["production_scenario_mwh"]
    ).clip(lower=0)
    result["scenario_surplus_mwh"] = (
        result["production_scenario_mwh"] - result["nomination_mwh"]
    ).clip(lower=0)
    result["expected_spread_eur_mwh"] = np.where(
        imbalance < 0,
        result["expected_spread_undercovered_eur_mwh"],
        result["expected_spread_overcovered_eur_mwh"],
    )
    result["expected_rebap_eur_mwh"] = (
        result["price_scenario_eur_mwh"] + result["expected_spread_eur_mwh"]
    )
    result["scenario_day_ahead_revenue_eur"] = (
        result["nomination_mwh"] * result["price_scenario_eur_mwh"]
    )
    result["scenario_imbalance_settlement_eur"] = (
        imbalance * result["expected_rebap_eur_mwh"]
    )
    result["scenario_cash_flow_eur"] = (
        result["scenario_day_ahead_revenue_eur"]
        + result["scenario_imbalance_settlement_eur"]
    )
    result["weighted_cash_flow_eur"] = (
        result["scenario_weight"] * result["scenario_cash_flow_eur"]
    )
    result["weighted_da_revenue_eur"] = (
        result["scenario_weight"] * result["scenario_day_ahead_revenue_eur"]
    )
    result["weighted_shortfall_mwh"] = (
        result["scenario_weight"] * result["scenario_shortfall_mwh"]
    )
    result["weighted_surplus_mwh"] = (
        result["scenario_weight"] * result["scenario_surplus_mwh"]
    )
    result["weighted_imbalance_settlement_eur"] = (
        result["scenario_weight"] * result["scenario_imbalance_settlement_eur"]
    )
    atomic_write_parquet(result, output_path)
    return result


def main() -> None:
    """Build the ex-ante scenario table and print a compact summary."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--portfolio-path", default=str(DEFAULT_PORTFOLIO_PATH))
    parser.add_argument("--rebap-path", default=str(DEFAULT_REBAP_PATH))
    parser.add_argument("--day-ahead-price-path", default=str(DEFAULT_DA_PRICE_PATH))
    parser.add_argument("--output-path", default=str(DEFAULT_OUTPUT_PATH))
    args = parser.parse_args()
    result = build_nomination_ex_ante(**vars(args))
    print(
        f"Candidatures ex ante : {len(result)} lignes, "
        f"{result['delivery_start_utc'].nunique()} quarts -> {args.output_path}"
    )


if __name__ == "__main__":
    main()
