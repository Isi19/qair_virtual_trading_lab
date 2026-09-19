"""Settle fixed and dynamic nomination policies against realised values."""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from src.analysis.settlement import settle_nomination
from src.pipelines.storage import atomic_write_parquet


DEFAULT_SCORES_PATH = Path(
    "data/backtesting/nomination/de_lu_nomination_scores_20260701_20260831.parquet"
)
DEFAULT_DECISIONS_PATH = Path(
    "data/backtesting/nomination/de_lu_nomination_decisions_20260701_20260831.parquet"
)
DEFAULT_PORTFOLIO_PATH = Path(
    "data/backtesting/portfolio/de_lu_portfolio_backtest_20260701_20260831.parquet"
)
DEFAULT_REBAP_PATH = Path("data/market_15m/de_lu_imbalance_prices.parquet")
DEFAULT_OUTPUT_PATH = Path(
    "data/backtesting/nomination/de_lu_nomination_backtest_20260701_20260831.parquet"
)


def build_nomination_backtest(
    *,
    scores_path: str | Path = DEFAULT_SCORES_PATH,
    decisions_path: str | Path = DEFAULT_DECISIONS_PATH,
    portfolio_path: str | Path = DEFAULT_PORTFOLIO_PATH,
    rebap_path: str | Path = DEFAULT_REBAP_PATH,
    output_path: str | Path = DEFAULT_OUTPUT_PATH,
) -> pd.DataFrame:
    """Apply every frozen nomination decision to production and prices realised later."""
    scores = pd.read_parquet(scores_path)
    decisions = pd.read_parquet(decisions_path)
    portfolio = pd.read_parquet(portfolio_path)
    rebap = pd.read_parquet(rebap_path)

    actual_columns = [
        "delivery_start_utc",
        "portfolio_actual_mwh",
        "actual_day_ahead_price_eur_mwh",
        "rebap_undercovered_eur_mwh",
        "rebap_overcovered_eur_mwh",
    ]
    actuals = portfolio[actual_columns[:3]].merge(
        rebap[actual_columns[:1] + actual_columns[3:]],
        on="delivery_start_utc",
        how="inner",
        validate="one_to_one",
    )

    policies = []
    for strategy_name, frame in scores.groupby("strategy_name"):
        fixed = frame[["delivery_start_utc", "nomination_mwh"]].copy()
        fixed["policy_name"] = f"always_{strategy_name}"
        policies.append(fixed)

    dynamic = decisions[["delivery_start_utc", "nomination_mwh"]].copy()
    dynamic["policy_name"] = "dynamic_expected_pnl"
    policies.append(dynamic)

    nominated = pd.concat(policies, ignore_index=True).merge(
        actuals, on="delivery_start_utc", how="inner", validate="many_to_one"
    )
    if len(nominated) != 4 * len(portfolio):
        raise ValueError("Some delivery periods are missing from the settlement inputs.")
    settled = settle_nomination(nominated, nomination_column="nomination_mwh")
    settled = settled.sort_values(
        ["policy_name", "delivery_start_utc"]
    ).reset_index(drop=True)
    atomic_write_parquet(settled, output_path)
    return settled


def main() -> None:
    """Build the realised policy comparison and print aggregate PnL."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scores-path", default=str(DEFAULT_SCORES_PATH))
    parser.add_argument("--decisions-path", default=str(DEFAULT_DECISIONS_PATH))
    parser.add_argument("--portfolio-path", default=str(DEFAULT_PORTFOLIO_PATH))
    parser.add_argument("--rebap-path", default=str(DEFAULT_REBAP_PATH))
    parser.add_argument("--output-path", default=str(DEFAULT_OUTPUT_PATH))
    args = parser.parse_args()
    result = build_nomination_backtest(**vars(args))
    print(result.groupby("policy_name")["gross_pnl_eur"].sum().round(2).to_string())


if __name__ == "__main__":
    main()
