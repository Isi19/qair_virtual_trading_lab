"""Collect Day-Ahead prices for the Core neighbours of DE-LU."""

from __future__ import annotations

from collections.abc import Mapping

import pandas as pd
import requests

from src.market_data.day_ahead_prices import (
    _local_delivery_bounds,
    fetch_smard_day_ahead_prices,
)


def get_de_lu_neighbor_prices(
    start_date: str,
    end_date: str,
    series_ids: Mapping[str, int],
    *,
    timezone: str = "Europe/Berlin",
    region: str = "DE",
    native_15m_start_date: str = "2025-10-01",
    request_get=requests.get,
) -> pd.DataFrame:
    """Return neighbouring auction prices in EUR/MWh on the DE-LU time grid.

    SMARD exposes these prices through the same API as the DE-LU price. A wide
    table is easier to inspect in the EDA and avoids creating one file per
    neighbouring market.
    """
    columns: dict[str, pd.Series] = {}
    for zone, filter_id in series_ids.items():
        zone_prices = fetch_smard_day_ahead_prices(
            start_date,
            end_date,
            {
                "filter_id": filter_id,
                "region": region,
                "timezone": timezone,
                "currency": "EUR",
                "native_15m_start_date": native_15m_start_date,
            },
            request_get=request_get,
        )
        columns[f"neighbor_price_{zone}_eur_per_mwh"] = zone_prices.set_index(
            "delivery_start_utc"
        )["day_ahead_price"]

    result = pd.DataFrame(columns).sort_index()
    utc_start, utc_end = _local_delivery_bounds(start_date, end_date, timezone)
    expected_index = pd.date_range(
        utc_start,
        utc_end,
        freq="15min",
        inclusive="left",
        name="delivery_start_utc",
    )
    result = result.reindex(expected_index)
    missing = result.isna().sum()
    missing = missing[missing.gt(0)]
    if not missing.empty:
        raise ValueError(f"Missing neighbouring prices: {missing.to_dict()}")

    result.attrs["source"] = "smard"
    result.attrs["unit"] = "EUR/MWh"
    result.attrs["value_type"] = "day_ahead_clearing_price"
    return result
