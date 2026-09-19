"""Read existing backtest tables without modifying them."""

from pathlib import Path

import pandas as pd
import streamlit as st

PROJECT_ROOT = Path(__file__).resolve().parents[2]
BACKTEST_ROOT = PROJECT_ROOT / "data" / "backtesting"
DEFAULT_MARKET_ZONE = "DE-LU"
MARKET_ZONES = {
    "DE-LU": {
        "display_name": "Germany–Luxembourg",
        "portfolio_name": "Virtual Qair Portfolio",
        "timezone": "Europe/Berlin",
        "resolution_minutes": 15,
        "assets": [
            {
                "id": "langer_wald",
                "name": "Langer Wald",
                "technology": "Wind",
                "capacity_mw": 31.0,
                "portfolio_actual_column": "wind_actual_mw",
                "color": "#4F8FCB",
                "stack_order": 1,
            },
            {
                "id": "perleberg",
                "name": "Perleberg",
                "technology": "Solar",
                "capacity_mw": 22.1,
                "portfolio_actual_column": "solar_actual_mw",
                "color": "#D9B36C",
                "stack_order": 0,
            },
        ],
        "tables": {
            "portfolio": (
                "portfolio/de_lu_portfolio_backtest_20260701_20260831.parquet"
            ),
            "asset_forecasts": {
                "langer_wald": (
                    "production_forecasts/langer_wald_j1_forecasts_20260701_20260831.parquet"
                ),
                "perleberg": (
                    "production_forecasts/perleberg_j1_forecasts_20260701_20260831.parquet"
                ),
            },
            "price": (
                "price_forecasts/de_lu_price_forecasts_20260701_20260831.parquet"
            ),
            "scores": "nomination/de_lu_nomination_scores_20260701_20260831.parquet",
            "decisions": (
                "nomination/de_lu_nomination_decisions_20260701_20260831.parquet"
            ),
            "settlement": (
                "nomination/de_lu_nomination_backtest_20260701_20260831.parquet"
            ),
        },
    }
}


@st.cache_data(show_spinner=False)
def _read_parquet(path: Path, modified_ns: int, timezone: str) -> pd.DataFrame:
    """Cache a file until its modification time changes."""
    frame = pd.read_parquet(path)
    frame["delivery_start_local"] = pd.to_datetime(
        frame["delivery_start_utc"], utc=True
    ).dt.tz_convert(timezone)
    return frame


def selected_market_zone_id() -> str:
    """Return the zone selected in the shared sidebar."""
    return st.session_state.get("market_zone", DEFAULT_MARKET_ZONE)


def market_zone_config(zone_id: str | None = None) -> dict:
    """Return the configuration for the selected market zone."""
    return MARKET_ZONES[zone_id or selected_market_zone_id()]


def _load_path(relative_path: str, zone_id: str | None = None) -> pd.DataFrame:
    zone = market_zone_config(zone_id)
    path = BACKTEST_ROOT / relative_path
    return _read_parquet(path, path.stat().st_mtime_ns, zone["timezone"])


def load_table(name: str, zone_id: str | None = None) -> pd.DataFrame:
    """Load a market-zone table with its local delivery timestamps."""
    return _load_path(market_zone_config(zone_id)["tables"][name], zone_id)


def load_asset_forecast(asset_id: str, zone_id: str | None = None) -> pd.DataFrame:
    """Load the selected zone's forecast for one of its virtual assets."""
    relative_path = market_zone_config(zone_id)["tables"]["asset_forecasts"][asset_id]
    return _load_path(relative_path, zone_id)


def available_period(zone_id: str | None = None):
    """Return the date range shared by every table for a market zone."""
    tables = market_zone_config(zone_id)["tables"]
    relative_paths = [
        path
        for name, path in tables.items()
        if name != "asset_forecasts"
    ]
    relative_paths.extend(tables["asset_forecasts"].values())
    dates = [
        _load_path(path, zone_id)["delivery_start_local"].dt.date
        for path in relative_paths
    ]
    first_day = max(series.min() for series in dates)
    last_day = min(series.max() for series in dates)
    if first_day > last_day:
        raise ValueError("The input tables have no common delivery period.")
    return first_day, last_day


def filter_period(frame: pd.DataFrame, start, end) -> pd.DataFrame:
    """Include both boundary dates in German local time."""
    dates = frame["delivery_start_local"].dt.date
    return frame.loc[dates.between(start, end)].copy()
