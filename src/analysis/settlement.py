"""Simple settlement calculations for a nominated renewable portfolio."""

from __future__ import annotations

import numpy as np
import pandas as pd


def settle_nomination(
    frame: pd.DataFrame,
    *,
    nomination_column: str,
) -> pd.DataFrame:
    """Settle one nominated volume against realised DA and reBAP prices.

    A positive imbalance means the portfolio generated more than nominated and
    is therefore overcovered.  A negative imbalance means it is undercovered.
    All quantities are MWh and all prices EUR/MWh.
    """
    required = {
        nomination_column,
        "portfolio_actual_mwh",
        "actual_day_ahead_price_eur_mwh",
        "rebap_undercovered_eur_mwh",
        "rebap_overcovered_eur_mwh",
    }
    missing = sorted(required - set(frame.columns))
    if missing:
        raise ValueError(f"Settlement input is missing: {', '.join(missing)}")

    settled = frame.copy()
    nominated = settled[nomination_column]
    settled["imbalance_volume_mwh"] = settled["portfolio_actual_mwh"] - nominated
    settled["applied_rebap_eur_mwh"] = np.where(
        settled["imbalance_volume_mwh"] < 0,
        settled["rebap_undercovered_eur_mwh"],
        settled["rebap_overcovered_eur_mwh"],
    )
    settled["day_ahead_revenue_eur"] = (
        nominated * settled["actual_day_ahead_price_eur_mwh"]
    )
    settled["imbalance_settlement_eur"] = (
        settled["imbalance_volume_mwh"] * settled["applied_rebap_eur_mwh"]
    )
    settled["gross_pnl_eur"] = (
        settled["day_ahead_revenue_eur"] + settled["imbalance_settlement_eur"]
    )
    return settled
