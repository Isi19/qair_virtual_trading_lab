"""Add point-in-time reBAP and spread expectted values to nomination candidates."""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from src.pipelines.storage import atomic_write_parquet


DEFAULT_CANDIDATES_PATH = Path(
    "data/backtesting/nomination/de_lu_nomination_ex_ante_20260701_20260831.parquet"
)
DEFAULT_REBAP_PATH = Path("data/market_15m/de_lu_imbalance_prices.parquet")
DEFAULT_DA_PRICE_PATH = Path("data/market_15m/de_lu_day_ahead_prices.parquet")
DEFAULT_OUTPUT_PATH = Path(
    "data/backtesting/nomination/de_lu_nomination_ex_ante_20260701_20260831.parquet"
)


def _add_time_groups(frame: pd.DataFrame) -> pd.DataFrame:
    """Add German local month and four trader-friendly hour blocks."""
    result = frame.copy()
    local = pd.to_datetime(result["delivery_start_utc"], utc=True).dt.tz_convert(
        "Europe/Berlin"
    )
    result["delivery_month"] = local.dt.month
    result["hour_block"] = pd.cut(
        local.dt.hour,
        bins=[-1, 5, 11, 17, 23],
        labels=["00_05", "06_11", "12_17", "18_23"],
    ).astype(str)
    result["delivery_date_local"] = local.dt.tz_localize(None).dt.normalize()
    return result


def _market_statistics(history: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Return signed spread means and standard deviations by month and block."""
    values = history.copy()
    values["spread_undercovered_eur_mwh"] = (
        values["rebap_undercovered_eur_mwh"] - values["actual_day_ahead_price_eur_mwh"]
    )
    values["spread_overcovered_eur_mwh"] = (
        values["rebap_overcovered_eur_mwh"] - values["actual_day_ahead_price_eur_mwh"]
    )
    metrics = ["spread_undercovered_eur_mwh", "spread_overcovered_eur_mwh"]
    grouped = values.groupby(["delivery_month", "hour_block"])[metrics].agg(["mean", "std"])
    grouped.columns = [
        f"expected_{metric}"
        if statistic == "mean"
        else f"historical_std_{metric}"
        for metric, statistic in grouped.columns
    ]
    fallback_values = {}
    for metric in metrics:
        fallback_values[f"expected_{metric}"] = values[metric].mean()
        fallback_values[f"historical_std_{metric}"] = values[metric].std()
    fallback = pd.DataFrame([fallback_values])
    return grouped.reset_index(), fallback


def attach_market_spreads(
    candidates: pd.DataFrame,
    rebap: pd.DataFrame,
    day_ahead_price: pd.DataFrame,
) -> pd.DataFrame:
    """Attach point-in-time signed spread statistics to scenario rows."""
    candidates = _add_time_groups(candidates)
    history = _add_time_groups(
        rebap.merge(day_ahead_price, on="delivery_start_utc", how="inner", validate="one_to_one")
    )

    enriched_parts = []
    for delivery_date, day_candidates in candidates.groupby("delivery_date_local"):
        known_history = history[history["delivery_date_local"] < delivery_date]
        grouped, fallback = _market_statistics(known_history)
        part = day_candidates.merge(
            grouped, on=["delivery_month", "hour_block"], how="left", validate="many_to_one"
        )
        for column in fallback.columns:
            part[column] = part[column].fillna(fallback.iloc[0][column])
        enriched_parts.append(part)

    return (
        pd.concat(enriched_parts, ignore_index=True)
        .sort_values(["delivery_start_utc", "strategy_name"])
        .drop(columns=["delivery_month", "hour_block"])
        .reset_index(drop=True)
    )


def add_nomination_market_risk(
    *,
    candidates_path: str | Path = DEFAULT_CANDIDATES_PATH,
    rebap_path: str | Path = DEFAULT_REBAP_PATH,
    day_ahead_price_path: str | Path = DEFAULT_DA_PRICE_PATH,
    output_path: str | Path = DEFAULT_OUTPUT_PATH,
) -> pd.DataFrame:
    """Attach signed spread statistics using the history before each delivery."""
    candidates = pd.read_parquet(candidates_path)
    rebap = pd.read_parquet(rebap_path)
    da_price = pd.read_parquet(
        day_ahead_price_path,
        columns=["delivery_start_utc", "day_ahead_price"],
    ).rename(columns={"day_ahead_price": "actual_day_ahead_price_eur_mwh"})
    result = attach_market_spreads(candidates, rebap, da_price)
    atomic_write_parquet(result, output_path)
    return result


def main() -> None:
    """Add ex-ante market-risk signals and print the stored row count."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--candidates-path", default=str(DEFAULT_CANDIDATES_PATH))
    parser.add_argument("--rebap-path", default=str(DEFAULT_REBAP_PATH))
    parser.add_argument("--day-ahead-price-path", default=str(DEFAULT_DA_PRICE_PATH))
    parser.add_argument("--output-path", default=str(DEFAULT_OUTPUT_PATH))
    args = parser.parse_args()
    result = add_nomination_market_risk(**vars(args))
    print(f"Signaux financiers ex ante : {len(result)} lignes -> {args.output_path}")


if __name__ == "__main__":
    main()
