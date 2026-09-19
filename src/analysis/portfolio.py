"""Metrics that describe the production forecast risk of a two-asset portfolio."""

from __future__ import annotations

import numpy as np
import pandas as pd


def add_portfolio_exposure_metrics(portfolio: pd.DataFrame) -> pd.DataFrame:
    """Add forecast errors, interval diagnostics and diversification gain.

    The diversification gain is measured in MWh for each 15-minute period:
    ``|wind error| + |solar error| - |portfolio error|``.

    It is positive when one asset's error partly offsets the other asset's
    error. The calculation uses P50 production forecasts only.
    """
    result = portfolio.copy()
    for asset in ("wind", "solar"):
        result[f"{asset}_error_mwh"] = (
            result[f"{asset}_actual_mw"] - result[f"{asset}_p50_mw"]
        ) * 0.25
        result[f"{asset}_absolute_error_mwh"] = result[f"{asset}_error_mwh"].abs()

    result["portfolio_error_mwh"] = (
        result["portfolio_actual_mwh"] - result["portfolio_p50_mwh"]
    )
    result["portfolio_absolute_error_mwh"] = result["portfolio_error_mwh"].abs()
    result["standalone_absolute_error_mwh"] = (
        result["wind_absolute_error_mwh"] + result["solar_absolute_error_mwh"]
    )
    result["diversification_gain_mwh"] = (
        result["standalone_absolute_error_mwh"]
        - result["portfolio_absolute_error_mwh"]
    )

    # Adding marginal P10/P90 forecasts gives conservative joint scenarios.
    # They are not exact portfolio quantiles without an error-dependence model.
    result["portfolio_p10_p90_width_mwh"] = (
        result["portfolio_p90_mwh"] - result["portfolio_p10_mwh"]
    )
    result["portfolio_inside_p10_p90"] = (
        result["portfolio_actual_mwh"].ge(result["portfolio_p10_mwh"]) # ge stands for greater than or equal to
        & result["portfolio_actual_mwh"].le(result["portfolio_p90_mwh"]) # le stands for less than or equal to
    )
    return result


def portfolio_exposure_summary(portfolio: pd.DataFrame) -> pd.DataFrame:
    """Return one readable row of portfolio forecast and diversification metrics."""
    required_columns = {
        "wind_absolute_error_mwh",
        "solar_absolute_error_mwh",
        "portfolio_absolute_error_mwh",
        "standalone_absolute_error_mwh",
        "diversification_gain_mwh",
        "portfolio_p10_p90_width_mwh",
        "portfolio_inside_p10_p90",
    }
    missing = required_columns.difference(portfolio.columns)
    if missing:
        raise ValueError(f"Portfolio metrics are missing: {sorted(missing)}")

    standalone_error = portfolio["standalone_absolute_error_mwh"].sum()
    diversification_gain = portfolio["diversification_gain_mwh"].sum()
    diversification_gain_pct = (
        100 * diversification_gain / standalone_error if standalone_error else np.nan
    )
    return pd.DataFrame(
        [
            {
                "wind_mae_mwh": portfolio["wind_absolute_error_mwh"].mean(),
                "solar_mae_mwh": portfolio["solar_absolute_error_mwh"].mean(),
                "portfolio_mae_mwh": portfolio["portfolio_absolute_error_mwh"].mean(),
                "standalone_absolute_error_mwh": standalone_error,
                "portfolio_absolute_error_mwh": portfolio[
                    "portfolio_absolute_error_mwh"
                ].sum(),
                "diversification_gain_mwh": diversification_gain,
                "diversification_gain_pct": diversification_gain_pct,
                "p10_p90_scenario_coverage_pct": 100
                * portfolio["portfolio_inside_p10_p90"].mean(),
                "mean_p10_p90_scenario_width_mwh": portfolio[
                    "portfolio_p10_p90_width_mwh"
                ].mean(),
            }
        ]
    )
