"""Collect and normalize official Day-Ahead prices for DE-LU and Poland.

Both sources are converted to the same schema. Delivery timestamps are stored
in UTC for joins and calculations, while a local timestamp is retained for
market-day analysis and trader-facing displays.
"""

from __future__ import annotations

from datetime import date
import time

import pandas as pd
import requests


SMARD_BASE_URL = "https://www.smard.de/app/chart_data"


def _request_with_retry(request_get, url: str, **kwargs):
    """Return one HTTP response, retrying only temporary server errors.

    Client errors such as 400 or 404 are raised immediately because repeating
    the same request would not correct them. A 5xx response is retried twice.
    """
    for attempt in range(3):
        response = request_get(url, **kwargs)
        try:
            response.raise_for_status()
            return response
        except requests.HTTPError:
            if attempt == 2 or response.status_code < 500:
                raise
            time.sleep(2 ** attempt)


def _local_delivery_bounds(start_date: str, end_date: str, timezone: str):
    """Convert inclusive local delivery dates to a UTC half-open interval.

    The two local midnights are localized separately. This matters on daylight
    saving transitions: the resulting interval naturally contains 23, 24 or 25
    hours instead of assuming that every market day lasts exactly 24 hours.
    """
    local_start = pd.Timestamp(start_date).tz_localize(timezone)
    next_local_date = pd.Timestamp(end_date) + pd.Timedelta(days=1)
    local_end = next_local_date.tz_localize(timezone)
    return local_start.tz_convert("UTC"), local_end.tz_convert("UTC")


def _add_common_columns(
    prices: pd.DataFrame,
    *,
    market_zone: str,
    timezone: str,
    currency: str,
    source: str,
    native_15m_start_date: str,
) -> pd.DataFrame:
    """Add shared metadata and identify the market-resolution regime.

    Prices before ``native_15m_start_date`` are hourly prices repeated on a
    15-minute grid. They remain in the dataset but are explicitly marked so the
    modeling step can assign them a lower training weight.
    """
    normalized = prices.copy()

    # UTC is the unambiguous timeline used for joins, lags and de-duplication.
    normalized["delivery_start_utc"] = pd.to_datetime(
        normalized["delivery_start_utc"], utc=True
    )

    # Local time is kept for market dates, calendar features and display.
    normalized["delivery_start_local"] = normalized["delivery_start_utc"].dt.tz_convert(
        timezone
    )
    normalized["delivery_date_local"] = normalized["delivery_start_local"].dt.date

    native_start = date.fromisoformat(native_15m_start_date)
    normalized["is_native_15m_target"] = (
        normalized["delivery_date_local"] >= native_start
    )
    normalized["source_market_resolution_minutes"] = normalized[
        "is_native_15m_target"
    ].map({True: 15, False: 60})
    normalized["market_zone"] = market_zone
    normalized["currency"] = currency
    normalized["source"] = source
    normalized["retrieved_at_utc"] = pd.Timestamp.now(tz="UTC")

    columns = [
        "market_zone",
        "delivery_start_utc",
        "delivery_start_local",
        "delivery_date_local",
        "day_ahead_price",
        "currency",
        "source_market_resolution_minutes",
        "is_native_15m_target",
        "price_publication_utc",
        "retrieved_at_utc",
        "source",
    ]
    return normalized[columns].sort_values("delivery_start_utc").reset_index(drop=True)


def fetch_smard_day_ahead_prices(
    start_date: str,
    end_date: str,
    config: dict,
    *,
    request_get=requests.get,
) -> pd.DataFrame:
    """Download and normalize DE-LU auction prices from SMARD.

    Parameters
    ----------
    start_date, end_date:
        Inclusive delivery dates expressed in the DE-LU market timezone.
    config:
        Market configuration containing the SMARD series identifiers, timezone,
        currency and date when native 15-minute prices began.
    request_get:
        HTTP function. It can be replaced by a fake function in unit tests.

    Returns
    -------
    pandas.DataFrame
        Prices restricted to the requested local delivery dates and expressed
        with the common market-data schema.
    """
    filter_id = str(config["filter_id"])
    region = config["region"]
    index_url = f"{SMARD_BASE_URL}/{filter_id}/{region}/index_quarterhour.json"
    index_response = _request_with_retry(request_get, index_url, timeout=30)
    block_starts = sorted(index_response.json()["timestamps"])

    utc_start, utc_end = _local_delivery_bounds(
        start_date, end_date, config["timezone"]
    )

    # The SMARD index lists every weekly file. Only download files whose time
    # interval overlaps the requested delivery interval.
    selected_blocks = []
    for position, block_start_ms in enumerate(block_starts):
        block_start = pd.to_datetime(block_start_ms, unit="ms", utc=True)
        next_start = (
            pd.to_datetime(block_starts[position + 1], unit="ms", utc=True)
            if position + 1 < len(block_starts)
            else block_start + pd.Timedelta(days=7)
        )
        if block_start < utc_end and next_start > utc_start:
            selected_blocks.append(block_start_ms)

    rows = []
    for block_start_ms in selected_blocks:
        filename = f"{filter_id}_{region}_quarterhour_{block_start_ms}.json"
        block_url = f"{SMARD_BASE_URL}/{filter_id}/{region}/{filename}"
        block_response = _request_with_retry(request_get, block_url, timeout=30)
        rows.extend(block_response.json()["series"])

    prices = pd.DataFrame(rows, columns=["delivery_start_ms", "day_ahead_price"])
    prices["delivery_start_utc"] = pd.to_datetime(
        prices.pop("delivery_start_ms"), unit="ms", utc=True
    )

    # SMARD exposes when a downloadable block was regenerated, but not the
    # original publication timestamp of each auction price.
    prices["price_publication_utc"] = pd.NaT
    prices = prices[
        prices["delivery_start_utc"].ge(utc_start)
        & prices["delivery_start_utc"].lt(utc_end)
        & prices["day_ahead_price"].notna()
    ]

    return _add_common_columns(
        prices,
        market_zone="de_lu",
        timezone=config["timezone"],
        currency=config["currency"],
        source="smard",
        native_15m_start_date=config["native_15m_start_date"],
    )


def fetch_pse_day_ahead_prices(
    start_date: str,
    end_date: str,
    config: dict,
    *,
    request_get=requests.get,
) -> pd.DataFrame:
    """Download and normalize Polish auction prices from PSE.

    PSE paginates long date ranges. Each ``nextLink`` already contains the
    complete continuation query, so query parameters are sent only with the
    first request.

    Parameters and return values follow :func:`fetch_smard_day_ahead_prices`,
    using the Polish market timezone and PSE-specific configuration instead.
    """
    url = config["endpoint"]
    params = {
        "$filter": (
            f"business_date ge '{start_date}' and business_date le '{end_date}'"
        )
    }
    rows = []

    while url:
        response = _request_with_retry(request_get, url, params=params, timeout=30)
        payload = response.json()
        rows.extend(payload["value"])
        url = payload.get("nextLink")
        params = None  # The nextLink already contains the complete query.

    prices = pd.DataFrame(rows)

    # PSE's dtime_utc marks the end of a delivery period. The project uses the
    # beginning of each 15-minute period as its canonical timestamp.
    prices["delivery_start_utc"] = (
        pd.to_datetime(prices["dtime_utc"], utc=True) - pd.Timedelta(minutes=15)
    )
    prices["day_ahead_price"] = pd.to_numeric(prices["csdac_pln"])
    prices["price_publication_utc"] = pd.to_datetime(
        prices["publication_ts_utc"], utc=True
    )

    utc_start, utc_end = _local_delivery_bounds(
        start_date, end_date, config["timezone"]
    )
    prices = prices[
        prices["delivery_start_utc"].ge(utc_start)
        & prices["delivery_start_utc"].lt(utc_end)
    ]

    return _add_common_columns(
        prices,
        market_zone="pl",
        timezone=config["timezone"],
        currency=config["currency"],
        source="pse",
        native_15m_start_date=config["native_15m_start_date"],
    )


def validate_price_continuity(
    prices: pd.DataFrame, start_date: str, end_date: str, timezone: str
) -> None:
    """Check that each expected delivery quarter-hour occurs exactly once.

    The expected UTC grid is derived from local-day boundaries. It therefore
    contains 92, 96 or 100 periods on daylight-saving, normal or winter-time
    transition days respectively.

    Raises
    ------
    ValueError
        If delivery timestamps are duplicated, missing or outside the requested
        interval.
    """
    utc_start, utc_end = _local_delivery_bounds(start_date, end_date, timezone)
    expected = pd.date_range(
        utc_start, utc_end, freq="15min", inclusive="left", name="delivery_start_utc"
    )
    observed = pd.DatetimeIndex(prices["delivery_start_utc"])

    if observed.has_duplicates:
        raise ValueError("Duplicate delivery timestamps found in Day-Ahead prices.")
    # Set differences give a concise error without altering or filling the data.
    missing = expected.difference(observed)
    unexpected = observed.difference(expected)
    if len(missing) or len(unexpected):
        raise ValueError(
            f"Price series is not continuous: {len(missing)} missing and "
            f"{len(unexpected)} unexpected timestamps."
        )
