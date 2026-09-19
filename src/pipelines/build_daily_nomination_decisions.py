"""Choose a P10, P50 or P90 nomination for each delivery quarter."""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from src.pipelines.storage import atomic_write_parquet


INPUT_PATH = Path(
    "data/backtesting/nomination/de_lu_nomination_ex_ante_20260701_20260831.parquet"
)
SCORES_PATH = Path(
    "data/backtesting/nomination/de_lu_nomination_scores_20260701_20260831.parquet"
)
DECISIONS_PATH = Path(
    "data/backtesting/nomination/de_lu_nomination_decisions_20260701_20260831.parquet"
)


def build_daily_nomination_decisions(
    *,
    input_path: str | Path = INPUT_PATH,
    scores_path: str | Path = SCORES_PATH,
    decisions_path: str | Path = DECISIONS_PATH,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Select the nomination with the highest expected PnL per quarter.

    The ex-ante scenario table is read only. Its cash flows were calculated
    when it was built, before the decision is made.
    """
    candidates = pd.read_parquet(input_path)

    scores = (
        candidates.groupby(
            ["delivery_start_utc", "delivery_date_local", "strategy_name"],
            as_index=False,
        )
        .agg(
            nomination_mwh=("nomination_mwh", "first"),
            expected_da_revenue_eur=("weighted_da_revenue_eur", "sum"),
            expected_shortfall_mwh=("weighted_shortfall_mwh", "sum"),
            expected_surplus_mwh=("weighted_surplus_mwh", "sum"),
            expected_imbalance_settlement_eur=(
                "weighted_imbalance_settlement_eur",
                "sum",
            ),
            expected_pnl_eur=("weighted_cash_flow_eur", "sum"),
        )
        .sort_values(["delivery_start_utc", "strategy_name"])
        .reset_index(drop=True)
    )
    decisions = scores.loc[
        scores.groupby("delivery_start_utc")["expected_pnl_eur"].idxmax()
    ].sort_values("delivery_start_utc").reset_index(drop=True)

    atomic_write_parquet(scores, scores_path)
    atomic_write_parquet(decisions, decisions_path)
    return scores, decisions


def main() -> None:
    """Build nomination scores and print the selected-profile counts."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-path", default=str(INPUT_PATH))
    parser.add_argument("--scores-path", default=str(SCORES_PATH))
    parser.add_argument("--decisions-path", default=str(DECISIONS_PATH))
    args = parser.parse_args()
    _, decisions = build_daily_nomination_decisions(**vars(args))
    print("Décisions journalières ex ante :")
    print(decisions.groupby("strategy_name").size().to_string())


if __name__ == "__main__":
    main()
