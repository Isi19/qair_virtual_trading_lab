"""Collect pre-clearing Core capacity data published by JAO."""

from __future__ import annotations

from collections.abc import Mapping
import time

import pandas as pd
import requests


JAO_CORE_API_URL = "https://publicationtool.jao.eu/core/api/data"
CORE_NEIGHBOURS = ("AT", "BE", "CZ", "FR", "NL", "PL")


def _market_day_bounds(
    start_date: str, end_date: str, timezone: str
) -> tuple[pd.Timestamp, pd.Timestamp]:
    """Convert inclusive local delivery dates to a half-open UTC interval."""
    local_start = pd.Timestamp(start_date).tz_localize(timezone)
    local_end = (pd.Timestamp(end_date) + pd.Timedelta(days=1)).tz_localize(timezone)
    return local_start.tz_convert("UTC"), local_end.tz_convert("UTC")


def _fetch_jao_table(
    endpoint: str,
    start_date: str,
    end_date: str,
    *,
    timezone: str,
    request_get,
    allow_missing_hours: bool = False,
) -> pd.DataFrame:
    """Download one hourly JAO table and check its delivery-time coverage."""
    utc_start, utc_end = _market_day_bounds(start_date, end_date, timezone)
    url = f"{JAO_CORE_API_URL}/{endpoint}"
    params = {"FromUtc": utc_start.isoformat(), "ToUtc": utc_end.isoformat()}

    # JAO rate-limits long historical collections. Retry only temporary 429
    # and server errors; a permanent 4xx is raised immediately.
    for attempt in range(5):
        response = request_get(url, params=params, timeout=60)
        status_code = getattr(response, "status_code", 200)
        if status_code != 429 and status_code < 500:
            response.raise_for_status()
            break
        if attempt == 4:
            response.raise_for_status()
        time.sleep(5 * (attempt + 1))

    payload = response.json()

    if payload.get("rejected"):
        raise ValueError(f"JAO rejected the {endpoint} query: {payload.get('messages')}")

    frame = pd.DataFrame(payload.get("data", []))
    if frame.empty or "dateTimeUtc" not in frame:
        raise ValueError(f"JAO returned no {endpoint} data for the requested dates.")

    frame["delivery_start_utc"] = pd.to_datetime(frame.pop("dateTimeUtc"), utc=True)
    frame = (
        frame[frame["delivery_start_utc"].ge(utc_start)]
        .loc[lambda rows: rows["delivery_start_utc"].lt(utc_end)]
        .drop_duplicates("delivery_start_utc", keep="last")
        .set_index("delivery_start_utc")
        .sort_index()
    )

    # A local delivery day contains 23, 24 or 25 hourly MTUs. Building the
    # expectation from localized midnights handles both clock changes.
    expected_hours = pd.date_range(
        utc_start, utc_end, freq="h", inclusive="left", name="delivery_start_utc"
    )
    if not frame.index.equals(expected_hours) and not allow_missing_hours:
        missing = expected_hours.difference(frame.index)
        raise ValueError(
            f"JAO {endpoint} is missing {len(missing)} hourly MTUs "
            f"between {start_date} and {end_date}."
        )
    return frame


def _fetch_jao_table_in_batches(
    endpoint: str,
    start_date: str,
    end_date: str,
    *,
    timezone: str,
    request_get,
    batch_days: int = 2,
    pause_seconds: float = 0.7,
    allow_missing_hours: bool = False,
) -> pd.DataFrame:
    """Download histories in ranges no longer than JAO's two-day limit."""
    first_day = pd.Timestamp(start_date)
    last_day = pd.Timestamp(end_date)
    batches: list[pd.DataFrame] = []

    while first_day <= last_day:
        batch_end = min(first_day + pd.Timedelta(days=batch_days - 1), last_day)

        # Two local days span 49 UTC hours around the autumn clock change,
        # which exceeds JAO's strict 48-hour limit. Fetch that day alone.
        utc_start, utc_end = _market_day_bounds(
            first_day.date().isoformat(), batch_end.date().isoformat(), timezone
        )
        if utc_end - utc_start > pd.Timedelta(days=2):
            batch_end = first_day

        batches.append(
            _fetch_jao_table(
                endpoint,
                first_day.date().isoformat(),
                batch_end.date().isoformat(),
                timezone=timezone,
                request_get=request_get,
                allow_missing_hours=allow_missing_hours,
            )
        )
        first_day = batch_end + pd.Timedelta(days=1)
        if first_day <= last_day:
            time.sleep(pause_seconds)

    # Some JAO responses contain optional columns that are entirely empty for
    # one batch. Dropping only those raw columns avoids a pandas dtype warning;
    # required DE columns are still checked explicitly after concatenation.
    non_empty_batches = [batch.dropna(axis=1, how="all") for batch in batches]
    return pd.concat(non_empty_batches).sort_index()


def _repeat_hourly_values_on_quarters(frame: pd.DataFrame) -> pd.DataFrame:
    """Repeat each hourly JAO value over its four delivery quarter-hours."""
    quarter_index = pd.date_range(
        frame.index[0],
        frame.index[-1] + pd.Timedelta(hours=1),
        freq="15min",
        inclusive="left",
        name="delivery_start_utc",
    )
    # ``limit=3`` fills only the three quarters following an observed hour.
    # A genuinely missing JAO hour therefore stays missing instead of silently
    # inheriting the preceding publication.
    return frame.reindex(quarter_index, method="ffill", limit=3)


def _select_and_rename(
    frame: pd.DataFrame, columns: Mapping[str, str], endpoint: str
) -> pd.DataFrame:
    """Keep the documented fields needed by the model and give them clear names."""
    missing = [source_name for source_name in columns if source_name not in frame]
    if missing:
        raise ValueError(f"JAO {endpoint} is missing columns: {missing}")
    return frame[list(columns)].rename(columns=columns).apply(pd.to_numeric)


def get_de_lu_jao_features(
    start_date: str,
    end_date: str,
    *,
    timezone: str = "Europe/Berlin",
    include_cross_border_limits: bool = True,
    allow_missing_hours: bool = False,
    request_get=requests.get,
) -> pd.DataFrame:
    """Return DE-LU JAO forecasts and capacity limits on a 15-minute grid.

    D2CF and capacity limits are publications known at 10:30 on D-1. Their
    historical publications are therefore used in training as they would have
    been known at the operational cutoff. They must not be replaced with
    cleared net positions or realized cross-border flows from delivery day D.

    JAO publishes hourly values. Repeating an hourly value over four quarters
    preserves its meaning; interpolation would invent a slope absent upstream.
    """
    d2cf = _fetch_jao_table_in_batches(
        "d2CF",
        start_date,
        end_date,
        timezone=timezone,
        request_get=request_get,
        allow_missing_hours=allow_missing_hours,
    )
    net_position_limits = _fetch_jao_table_in_batches(
        "maxNetPos",
        start_date,
        end_date,
        timezone=timezone,
        request_get=request_get,
        allow_missing_hours=allow_missing_hours,
    )

    selected_frames = [
        _select_and_rename(
            d2cf,
            {
                "verticalLoad_DE": "jao_vertical_load_mw",
                "generation_DE": "jao_generation_mw",
                "coreNetPosition_DE": "jao_net_position_mw",
            },
            "d2CF",
        ),
        _select_and_rename(
            net_position_limits,
            {
                "minDE": "jao_min_net_position_mw",
                "maxDE": "jao_max_net_position_mw",
            },
            "maxNetPos",
        ),
    ]

    if include_cross_border_limits:
        exchanges = _fetch_jao_table_in_batches(
            "maxExchanges",
            start_date,
            end_date,
            timezone=timezone,
            request_get=request_get,
            allow_missing_hours=allow_missing_hours,
        )
        exchange_columns: dict[str, str] = {}
        for neighbour in CORE_NEIGHBOURS:
            neighbour_name = neighbour.lower()
            exchange_columns[f"border_DE_{neighbour}"] = (
                f"jao_max_exchange_de_to_{neighbour_name}_mw"
            )
            exchange_columns[f"border_{neighbour}_DE"] = (
                f"jao_max_exchange_{neighbour_name}_to_de_mw"
            )
        selected_frames.append(
            _select_and_rename(exchanges, exchange_columns, "maxExchanges")
        )

    hourly = pd.concat(selected_frames, axis=1)
    result = _repeat_hourly_values_on_quarters(hourly)
    result.attrs["source"] = "jao_core_publication_tool"
    result.attrs["value_type"] = "pre_clearing_forecast_and_limits"
    result.attrs["native_resolution_minutes"] = 60
    result.attrs["publication_time_local_d_minus_1"] = "10:30"
    return result
