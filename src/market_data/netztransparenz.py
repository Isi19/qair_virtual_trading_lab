"""Collect Germany's official quarter-hourly imbalance settlement prices.

The four German TSOs publish the cross-control-area balancing energy price
(reBAP) through Netztransparenz.  The API requires OAuth credentials; this
module reads them from a local ``refab.env`` file and never stores them in a
dataset or prints them.
"""

from __future__ import annotations

from io import StringIO
from pathlib import Path

import pandas as pd
import requests


TOKEN_URL = "https://identity.netztransparenz.de/users/connect/token"
REBAP_URL = "https://ds.netztransparenz.de/api/v1/data/NrvSaldo/reBAP/Qualitaetsgesichert"


def _read_credentials(env_path: str | Path) -> tuple[str, str]:
    """Read the two API credentials from a local dotenv-style file.

    Only ``NETZTRANSPARENZ_CLIENT_ID`` and ``NETZTRANSPARENZ_CLIENT_SECRET``
    are read.  Values stay in memory for the duration of the HTTP call.
    """
    values: dict[str, str] = {}
    for line in Path(env_path).read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", maxsplit=1)
        values[key.strip()] = value.strip().strip("\"").strip("'")

    required = ["NETZTRANSPARENZ_CLIENT_ID", "NETZTRANSPARENZ_CLIENT_SECRET"]
    missing = [key for key in required if not values.get(key)]
    if missing:
        raise ValueError(f"Missing Netztransparenz credentials: {', '.join(missing)}")
    return values[required[0]], values[required[1]]


def _get_access_token(
    client_id: str,
    client_secret: str,
    *,
    request_post=requests.post,
) -> str:
    """Request one OAuth access token without exposing it in logs."""
    response = request_post(
        TOKEN_URL,
        data={"grant_type": "client_credentials"},
        auth=(client_id, client_secret),
        timeout=30,
    )
    response.raise_for_status()
    token = response.json().get("access_token")
    if not token:
        raise ValueError("Netztransparenz returned no OAuth access token.")
    return token


def _local_day_bounds(
    start_date: str, end_date: str, timezone: str = "Europe/Berlin"
) -> tuple[pd.Timestamp, pd.Timestamp]:
    """Return the UTC bounds of inclusive German delivery dates.

    Local midnights are converted separately so a spring or autumn clock
    change naturally yields 92 or 100 quarter-hours.
    """
    start = pd.Timestamp(start_date).tz_localize(timezone).tz_convert("UTC")
    end = (
        (pd.Timestamp(end_date) + pd.Timedelta(days=1))
        .tz_localize(timezone)
        .tz_convert("UTC")
    )
    return start, end


def _format_api_timestamp(timestamp: pd.Timestamp) -> str:
    """Format a UTC timestamp for Netztransparenz's path-based API."""
    return timestamp.strftime("%Y-%m-%dT%H:%M:%S")


def _parse_rebap_csv(csv_text: str) -> pd.DataFrame:
    """Convert the documented reBAP CSV format into the project schema."""
    raw = pd.read_csv(StringIO(csv_text), sep=";", decimal=",")
    raw.columns = [str(column).strip().lstrip("\ufeff") for column in raw.columns]

    undercovered = next(
        (column for column in raw.columns if "unterdeckt" in column.lower()), None
    )
    overcovered = next(
        (column for column in raw.columns if "ueberdeckt" in column.lower()), None
    )
    if not undercovered or not overcovered:
        raise ValueError("Unexpected reBAP CSV columns from Netztransparenz.")

    # Format 9 reports the delivery period in UTC as a date and a start/end time.
    delivery_start = pd.to_datetime(
        raw["Datum"].astype(str) + " " + raw["von"].astype(str),
        format="%d.%m.%Y %H:%M",
    )
    delivery_end = pd.to_datetime(
        raw["Datum"].astype(str) + " " + raw["bis"].astype(str),
        format="%d.%m.%Y %H:%M",
    )
    # A period ending at 00:00 belongs to the following UTC date even though
    # the CSV repeats the delivery-start date in its separate date column.
    delivery_end = delivery_end.where(
        delivery_end > delivery_start, delivery_end + pd.Timedelta(days=1)
    )
    delivery_start = delivery_start.dt.tz_localize("UTC")
    delivery_end = delivery_end.dt.tz_localize("UTC")
    result = pd.DataFrame(
        {
            "delivery_start_utc": delivery_start,
            "delivery_end_utc": delivery_end,
            "rebap_undercovered_eur_mwh": pd.to_numeric(
                raw[undercovered], errors="coerce"
            ),
            "rebap_overcovered_eur_mwh": pd.to_numeric(
                raw[overcovered], errors="coerce"
            ),
        }
    )
    if result[["rebap_undercovered_eur_mwh", "rebap_overcovered_eur_mwh"]].isna().any().any():
        raise ValueError("reBAP CSV contains a non-numeric settlement price.")
    if not result["delivery_end_utc"].sub(result["delivery_start_utc"]).eq(
        pd.Timedelta(minutes=15)
    ).all():
        raise ValueError("reBAP CSV contains a period that is not 15 minutes long.")
    return result


def fetch_de_rebap(
    start_date: str,
    end_date: str,
    *,
    env_path: str | Path = "refab.env",
    request_post=requests.post,
    request_get=requests.get,
) -> pd.DataFrame:
    """Download final German reBAP values for inclusive local delivery dates.

    ``rebap_undercovered_eur_mwh`` settles a production shortfall
    (actual generation below the nominated quantity).  The overcovered price
    settles an excess of generation.  Both prices are retained even though
    they are usually identical.
    """
    client_id, client_secret = _read_credentials(env_path)
    token = _get_access_token(
        client_id, client_secret, request_post=request_post
    )
    utc_start, utc_end = _local_day_bounds(start_date, end_date)
    url = "/".join(
        [
            REBAP_URL,
            _format_api_timestamp(utc_start),
            _format_api_timestamp(utc_end),
        ]
    )
    response = request_get(
        url,
        headers={"Authorization": f"Bearer {token}"},
        timeout=60,
    )
    response.raise_for_status()
    result = _parse_rebap_csv(response.text)

    # The endpoint is UTC; restrict again to the requested local delivery days.
    result = result[
        result["delivery_start_utc"].ge(utc_start)
        & result["delivery_start_utc"].lt(utc_end)
    ].copy()
    result["settlement_area"] = "de"
    result["currency"] = "EUR"
    result["quality_status"] = "quality_assured"
    result["source"] = "netztransparenz"
    result["retrieved_at_utc"] = pd.Timestamp.now(tz="UTC")
    return result.sort_values("delivery_start_utc").reset_index(drop=True)
