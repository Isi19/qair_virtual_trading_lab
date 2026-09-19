"""Leakage-safe features for Day-Ahead electricity price forecasting."""

from __future__ import annotations

import numpy as np
import pandas as pd


CALENDAR_FEATURES = (
    "local_time_sin",
    "local_time_cos",
    "day_of_week_sin",
    "day_of_week_cos",
    "day_of_year_sin",
    "day_of_year_cos",
    "is_weekend",
    "utc_offset_hours",
)

PRICE_HISTORY_FEATURES = (
    "price_lag_1d",
    "price_lag_7d",
    "price_rolling_mean_7d",
    "price_rolling_mean_28d",
    "previous_day_mean_price",
    "previous_day_min_price",
    "previous_day_max_price",
    "previous_day_std_price",
)

WEATHER_SUFFIXES = (
    "_temperature_2m",
    "_cloud_cover",
    "_wind_speed_100m",
    "_shortwave_radiation",
    "_diffuse_radiation",
)

NEIGHBOUR_MARKETS = ("at", "be", "cz", "fr", "nl", "pl")


def _prepare_price_history(prices: pd.DataFrame, timezone: str) -> pd.DataFrame:
    """Return price history with a local date and quarter-of-day identifier."""
    history = prices[["delivery_start_utc", "day_ahead_price"]].copy()
    history["delivery_start_utc"] = pd.to_datetime(
        history["delivery_start_utc"], utc=True
    )
    local_time = history["delivery_start_utc"].dt.tz_convert(timezone)
    history["delivery_date_local"] = local_time.dt.tz_localize(None).dt.normalize()
    history["local_quarter"] = local_time.dt.hour * 4 + local_time.dt.minute // 15
    return history


def _complete_daily_quarter_clock(price_history: pd.DataFrame) -> pd.DataFrame:
    """Return one price per local date and quarter, including DST handling."""
    daily_quarter_prices = (
        price_history.groupby(
            ["delivery_date_local", "local_quarter"], as_index=False
        )["day_ahead_price"]
        .mean()
        .sort_values(["local_quarter", "delivery_date_local"])
    )

    # Complete the 96-position local clock used for daily lag matching. This
    # only creates values inside an observed day, notably the four nonexistent
    # spring-clock quarters between 01:45 and 03:00.
    local_dates = daily_quarter_prices["delivery_date_local"].unique()
    complete_clock = pd.MultiIndex.from_product(
        [local_dates, range(96)],
        names=["delivery_date_local", "local_quarter"],
    )
    daily_quarter_prices = (
        daily_quarter_prices.set_index(["delivery_date_local", "local_quarter"])
        .reindex(complete_clock)
    )
    daily_quarter_prices["day_ahead_price"] = daily_quarter_prices.groupby(
        level="delivery_date_local"
    )["day_ahead_price"].transform(
        lambda values: values.interpolate(limit_area="inside")
    )
    return daily_quarter_prices.dropna().reset_index()


def _add_historical_price_features(
    feature_rows: pd.DataFrame,
    price_history: pd.DataFrame,
    *,
    prefix: str,
) -> pd.DataFrame:
    """Add daily lags and past rollings for one price series.

    The autumn clock change contains two occurrences of the same local quarter;
    their mean gives one unambiguous lag value. The spring clock change has no
    02:00--02:45 quarters, so only those missing positions are interpolated in
    the historical lag curve. Observed target prices are never altered.
    """
    daily_quarter_prices = _complete_daily_quarter_clock(price_history)

    result = feature_rows
    for lag_days in (1, 7):
        lagged_prices = daily_quarter_prices.copy()
        lagged_prices["delivery_date_local"] += pd.Timedelta(days=lag_days)
        lagged_prices = lagged_prices.rename(
            columns={"day_ahead_price": f"{prefix}price_lag_{lag_days}d"}
        )
        result = result.merge(
            lagged_prices,
            on=["delivery_date_local", "local_quarter"],
            how="left",
        )

    # shift(1) excludes the current delivery date before calculating statistics.
    for window_days in (7, 28):
        rolling_name = f"{prefix}price_rolling_mean_{window_days}d"
        daily_quarter_prices[rolling_name] = daily_quarter_prices.groupby(
            "local_quarter"
        )["day_ahead_price"].transform(
            lambda values: values.shift(1).rolling(window_days, min_periods=2).mean()
        )
        result = result.merge(
            daily_quarter_prices[
                ["delivery_date_local", "local_quarter", rolling_name]
            ],
            on=["delivery_date_local", "local_quarter"],
            how="left",
        )

    # The complete D-1 curve is already published before the run for delivery D.
    daily_summary = (
        price_history.groupby("delivery_date_local")["day_ahead_price"]
        .agg(["mean", "min", "max", "std"])
        .reset_index()
    )
    daily_summary["delivery_date_local"] += pd.Timedelta(days=1)
    daily_summary = daily_summary.rename(
        columns={
            statistic: f"{prefix}previous_day_{statistic}_price"
            for statistic in ("mean", "min", "max", "std")
        }
    )
    return result.merge(daily_summary, on="delivery_date_local", how="left")


def add_neighbour_price_lags(
    feature_rows: pd.DataFrame,
    neighbour_prices: pd.DataFrame,
    *,
    timezone: str,
) -> pd.DataFrame:
    """Add D-1 and D-7 price lags for each configured neighbouring market.

    The neighbours' prices for delivery day D are unknown at the operational
    cutoff. Matching by local delivery date and quarter therefore uses only
    complete curves published one or seven days earlier.
    """
    required_time_columns = {"delivery_date_local", "local_quarter"}
    missing_time_columns = required_time_columns.difference(feature_rows.columns)
    if missing_time_columns:
        raise ValueError(
            "Feature rows are missing local time columns: "
            f"{sorted(missing_time_columns)}"
        )

    result = feature_rows.copy()
    for market in NEIGHBOUR_MARKETS:
        price_column = f"neighbor_price_{market}_eur_per_mwh"
        if price_column not in neighbour_prices:
            raise ValueError(f"Neighbour prices are missing column: {price_column}")

        market_prices = neighbour_prices[
            ["delivery_start_utc", price_column]
        ].rename(columns={price_column: "day_ahead_price"})
        history = _prepare_price_history(market_prices, timezone)

        daily_prices = _complete_daily_quarter_clock(history)
        for lag_days in (1, 7):
            lag_column = f"neighbor_{market}_price_lag_{lag_days}d"
            lagged_prices = daily_prices.copy()
            lagged_prices["delivery_date_local"] += pd.Timedelta(days=lag_days)
            lagged_prices = lagged_prices.rename(
                columns={"day_ahead_price": lag_column}
            )
            result = result.merge(
                lagged_prices[
                    ["delivery_date_local", "local_quarter", lag_column]
                ],
                on=["delivery_date_local", "local_quarter"],
                how="left",
            )

    return result.sort_values("delivery_start_utc").reset_index(drop=True)


def de_lu_model_feature_groups(model_table: pd.DataFrame) -> dict[str, list[str]]:
    """Return the readable feature groups retained for the DE-LU model.

    The model uses the calendar, price history, load, regional weather, JAO
    D-1 publications and delayed neighbour prices.
    """
    groups = {
        "calendrier": list(CALENDAR_FEATURES),
        "historique_prix_de_lu": list(PRICE_HISTORY_FEATURES),
        "charge": ["national_load_mw"],
        "meteo_regionale": [
            column
            for column in model_table
            if column.endswith(WEATHER_SUFFIXES)
        ],
        "jao_pre_clearing": [
            column for column in model_table if column.startswith("jao_")
        ],
        "prix_voisins_retardes": [
            column
            for column in model_table
            if column.startswith("neighbor_") and "_price_lag_" in column
        ],
    }
    missing = [
        column
        for columns in groups.values()
        for column in columns
        if column not in model_table
    ]
    if missing:
        raise ValueError(f"Model table is missing features: {missing}")
    return groups


def de_lu_operational_feature_columns(model_table: pd.DataFrame) -> list[str]:
    """Return the single feature set used by the operational price model.

    Historical training uses realised weather and load. Forecast replay replaces
    them with their J-1 counterparts. JAO and realised renewable generation
    remain available for EDA but are deliberately not model inputs.
    """
    groups = de_lu_model_feature_groups(model_table)
    selected_groups = (
        "calendrier",
        "historique_prix_de_lu",
        "charge",
        "meteo_regionale",
        "jao_pre_clearing",
        "prix_voisins_retardes",
    )
    return [column for name in selected_groups for column in groups[name]]


def build_day_ahead_price_features(
    target_prices: pd.DataFrame,
    *,
    timezone: str,
    other_zone_prices: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """Build a model-ready table without using information from delivery day D.

    Calendar variables describe the future delivery timestamp and are known in
    advance. Price features use only curves from D-1 or earlier, which have
    already been published when the forecast for D is produced.

    The returned dataframe still contains ``day_ahead_price`` as the supervised
    learning target. Feature creation itself does not drop early rows with
    incomplete lags and does not write another dataset to disk.
    """
    target_history = _prepare_price_history(target_prices, timezone)
    features = target_prices.copy()
    features["delivery_start_utc"] = target_history["delivery_start_utc"]
    features["delivery_date_local"] = target_history["delivery_date_local"]
    features["local_quarter"] = target_history["local_quarter"]

    local_time = features["delivery_start_utc"].dt.tz_convert(timezone)
    features["local_time_sin"] = np.sin(2 * np.pi * features["local_quarter"] / 96)
    features["local_time_cos"] = np.cos(2 * np.pi * features["local_quarter"] / 96)
    features["day_of_week_sin"] = np.sin(2 * np.pi * local_time.dt.dayofweek / 7)
    features["day_of_week_cos"] = np.cos(2 * np.pi * local_time.dt.dayofweek / 7)
    features["day_of_year_sin"] = np.sin(2 * np.pi * local_time.dt.dayofyear / 365.25)
    features["day_of_year_cos"] = np.cos(2 * np.pi * local_time.dt.dayofyear / 365.25)
    features["is_weekend"] = local_time.dt.dayofweek.ge(5).astype(int)
    features["utc_offset_hours"] = local_time.map(
        lambda timestamp: timestamp.utcoffset().total_seconds() / 3600
    )

    features = _add_historical_price_features(
        features, target_history, prefix=""
    )
    if other_zone_prices is not None:
        other_history = _prepare_price_history(other_zone_prices, timezone)
        features = _add_historical_price_features(
            features, other_history, prefix="other_zone_"
        )

    return features.sort_values("delivery_start_utc").reset_index(drop=True)
