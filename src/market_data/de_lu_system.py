"""Collect German system variables used by the DE-LU price study."""

from __future__ import annotations

from collections.abc import Mapping

import pandas as pd
import requests

from src.market_data.day_ahead_prices import SMARD_BASE_URL, _request_with_retry


INTERVAL = pd.Timedelta(minutes=15)


def _market_day_bounds(
    start_date: str, end_date: str, timezone: str
) -> tuple[pd.Timestamp, pd.Timestamp]:
    """Return UTC bounds for inclusive local market dates."""
    local_start = pd.Timestamp(start_date).tz_localize(timezone)
    local_end = (pd.Timestamp(end_date) + pd.Timedelta(days=1)).tz_localize(timezone)
    return local_start.tz_convert("UTC"), local_end.tz_convert("UTC")


def _fetch_smard_series(
    filter_id: int,
    start_date: str,
    end_date: str,
    *,
    region: str,
    timezone: str,
    request_get=requests.get,
) -> pd.Series:
    """Download one quarter-hourly SMARD series and convert MWh to average MW."""
    index_url = (
        f"{SMARD_BASE_URL}/{filter_id}/{region}/index_quarterhour.json"
    )
    response = _request_with_retry(request_get, index_url, timeout=30)
    block_starts = sorted(response.json()["timestamps"])
    utc_start, utc_end = _market_day_bounds(start_date, end_date, timezone)

    selected_blocks: list[int] = []
    for position, block_start_ms in enumerate(block_starts):
        block_start = pd.to_datetime(block_start_ms, unit="ms", utc=True)
        next_block_start = (
            pd.to_datetime(block_starts[position + 1], unit="ms", utc=True)
            if position + 1 < len(block_starts)
            else block_start + pd.Timedelta(days=7)
        )
        if block_start < utc_end and next_block_start > utc_start:
            selected_blocks.append(block_start_ms)

    rows: list[list[float | int | None]] = []
    for block_start_ms in selected_blocks:
        filename = (
            f"{filter_id}_{region}_quarterhour_{block_start_ms}.json"
        )
        block_url = f"{SMARD_BASE_URL}/{filter_id}/{region}/{filename}"
        block_response = _request_with_retry(
            request_get, block_url, timeout=30
        )
        rows.extend(block_response.json()["series"])

    frame = pd.DataFrame(rows, columns=["delivery_start_ms", "value_mwh"])
    if frame.empty:
        raise ValueError(f"SMARD returned no values for series {filter_id}.")

    frame["delivery_start_utc"] = pd.to_datetime(
        frame.pop("delivery_start_ms"), unit="ms", utc=True
    )
    frame["value_mwh"] = pd.to_numeric(frame["value_mwh"], errors="coerce")
    frame = frame[
        frame["delivery_start_utc"].ge(utc_start)
        & frame["delivery_start_utc"].lt(utc_end)
    ].drop_duplicates("delivery_start_utc", keep="last")

    # SMARD reports energy within each quarter-hour. Multiplying by four gives
    # the average power in MW, which matches the JAO system variables.
    values_mw = frame.set_index("delivery_start_utc")["value_mwh"] * 4
    expected_index = pd.date_range(
        utc_start, utc_end, freq=INTERVAL, inclusive="left"
    )
    values_mw = values_mw.reindex(expected_index)
    if values_mw.isna().any():
        missing = int(values_mw.isna().sum())
        raise ValueError(
            f"SMARD series {filter_id} is missing {missing} quarter-hours."
        )
    values_mw.index.name = "delivery_start_utc"
    return values_mw


def get_de_lu_system_features(
    start_date: str,
    end_date: str,
    series_ids: Mapping[str, int],
    *,
    region: str = "DE",
    timezone: str = "Europe/Berlin",
    request_get=requests.get,
) -> pd.DataFrame:
    """Return named quarter-hourly SMARD series in MW."""
    columns = {
        name: _fetch_smard_series(
            filter_id,
            start_date,
            end_date,
            region=region,
            timezone=timezone,
            request_get=request_get,
        )
        for name, filter_id in series_ids.items()
    }
    result = pd.DataFrame(columns)
    return pd.DataFrame(columns)


def get_de_lu_actual_system_features(
    start_date: str,
    end_date: str,
    series_ids: Mapping[str, int],
    *,
    region: str = "DE",
    timezone: str = "Europe/Berlin",
    request_get=requests.get,
) -> pd.DataFrame:
    """Return observed system variables for training and EDA."""
    result = get_de_lu_system_features(
        start_date,
        end_date,
        series_ids,
        region=region,
        timezone=timezone,
        request_get=request_get,
    )
    result.attrs.update(source="smard", value_type="actual", unit="MW")
    return result


def get_de_lu_forecast_system_features(
    start_date: str,
    end_date: str,
    series_ids: Mapping[str, int],
    *,
    region: str = "DE",
    timezone: str = "Europe/Berlin",
    request_get=requests.get,
) -> pd.DataFrame:
    """Return SMARD Day-Ahead system forecasts for the delivery period."""
    result = get_de_lu_system_features(
        start_date,
        end_date,
        series_ids,
        region=region,
        timezone=timezone,
        request_get=request_get,
    )
    result.attrs.update(source="smard", value_type="day_ahead_forecast", unit="MW")
    return result


def get_de_lu_load_forecast(
    target_date: str,
    *,
    filter_id: int = 411,
    region: str = "DE",
    timezone: str = "Europe/Berlin",
    request_get=requests.get,
) -> pd.DataFrame:
    """Return the complete national load forecast for one local delivery day."""
    result = get_de_lu_forecast_system_features(
        target_date,
        target_date,
        {"national_load_mw": filter_id},
        region=region,
        timezone=timezone,
        request_get=request_get,
    )
    result.attrs["target_date"] = target_date
    return result
