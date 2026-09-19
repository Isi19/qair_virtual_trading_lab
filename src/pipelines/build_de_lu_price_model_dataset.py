"""Build the DE-LU Day-Ahead price training table at 15-minute resolution."""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd
import yaml

from src.features.day_ahead_price import (
    add_neighbour_price_lags,
    build_day_ahead_price_features,
)
from src.pipelines.storage import atomic_write_parquet


DEFAULT_CONFIG_PATH = Path("config/day_ahead_price_features.yaml")
DEFAULT_PRICE_PATH = Path("data/market_15m/de_lu_day_ahead_prices.parquet")
DEFAULT_SYSTEM_PATH = Path("data/market_15m/de_lu_system_actual_15m.parquet")
DEFAULT_WEATHER_PATH = Path("data/market_15m/de_lu_regional_reanalysis_weather_15m.parquet")
DEFAULT_JAO_PATH = Path("data/market_15m/de_lu_jao_historical_features_15m.parquet")
DEFAULT_NEIGHBOUR_PATH = Path("data/market_15m/de_lu_neighbor_day_ahead_prices.parquet")
DEFAULT_OUTPUT_PATH = Path("data/modeling/de_lu_day_ahead_price_training_full.parquet")


def _utc_bounds(
    start_date: str, end_date: str, timezone: str
) -> tuple[pd.Timestamp, pd.Timestamp]:
    """Return UTC boundaries for inclusive local delivery dates."""
    utc_start = pd.Timestamp(start_date).tz_localize(timezone).tz_convert("UTC")
    utc_end = (
        (pd.Timestamp(end_date) + pd.Timedelta(days=1))
        .tz_localize(timezone)
        .tz_convert("UTC")
    )
    return utc_start, utc_end


def assemble_de_lu_price_training_dataset(
    price_history: pd.DataFrame,
    national_load: pd.DataFrame,
    weather: pd.DataFrame,
    jao_features: pd.DataFrame,
    *,
    start_date: str,
    end_date: str,
    timezone: str = "Europe/Berlin",
    drop_missing_exogenous: bool = False,
) -> pd.DataFrame:
    """Join price, load, weather and JAO data on their UTC delivery timestamp.

    Price features are calculated before filtering the study period so that
    the first retained days can use price lags from the preceding history.
    """
    price_features = build_day_ahead_price_features(
        price_history, timezone=timezone
    ).drop(columns=["price_publication_utc"], errors="ignore")
    price_features = price_features.set_index("delivery_start_utc")

    model_inputs = [national_load, weather, jao_features]
    exogenous_columns = [column for frame in model_inputs for column in frame]
    dataset = price_features.join(model_inputs, how="left")

    utc_start, utc_end = _utc_bounds(start_date, end_date, timezone)
    dataset = dataset[(dataset.index >= utc_start) & (dataset.index < utc_end)]

    # Missing exogenous values would mean that the feature table no longer
    # describes the same delivery periods as the price target. Price-lag NaNs
    # are expected at the beginning of history and handled by the notebook.
    missing = dataset[exogenous_columns].isna().sum()
    missing = missing[missing.gt(0)]
    if not missing.empty:
        if not drop_missing_exogenous:
            raise ValueError(f"Missing exogenous values: {missing.to_dict()}")
        dataset = dataset.dropna(subset=exogenous_columns)

    return dataset.reset_index().sort_values("delivery_start_utc").reset_index(drop=True)


def _load_feature_table(path: str | Path, columns: list[str]) -> pd.DataFrame:
    """Read selected columns on their UTC delivery timestamp."""
    frame = pd.read_parquet(path, columns=["delivery_start_utc", *columns])
    return frame.set_index("delivery_start_utc").sort_index()


def build_de_lu_price_training_dataset(
    start_date: str,
    end_date: str,
    *,
    config_path: str | Path = DEFAULT_CONFIG_PATH,
    price_path: str | Path = DEFAULT_PRICE_PATH,
    output_path: str | Path = DEFAULT_OUTPUT_PATH,
    drop_missing_exogenous: bool = False,
) -> dict:
    """Assemble the training table from the collected Parquet files."""
    with Path(config_path).open(encoding="utf-8") as config_file:
        source_config = yaml.safe_load(config_file)["de_lu_sources"]

    prices = pd.read_parquet(price_path)
    timezone = source_config["timezone"]
    system = _load_feature_table(DEFAULT_SYSTEM_PATH, ["national_load_mw"])
    weather_columns = [
        name
        for name in pd.read_parquet(DEFAULT_WEATHER_PATH).columns
        if name not in {
            "delivery_start_utc",
            "weather_source",
            "weather_model",
            "weather_source_resolution_minutes",
            "weather_resolution_minutes",
            "weather_conversion",
        }
    ]
    weather = _load_feature_table(DEFAULT_WEATHER_PATH, weather_columns)
    jao_columns = [
        name
        for name in pd.read_parquet(DEFAULT_JAO_PATH).columns
        if name.startswith("jao_")
    ]
    jao_features = _load_feature_table(DEFAULT_JAO_PATH, jao_columns)
    dataset = assemble_de_lu_price_training_dataset(
        prices,
        system,
        weather,
        jao_features,
        start_date=start_date,
        end_date=end_date,
        timezone=timezone,
        drop_missing_exogenous=drop_missing_exogenous,
    )
    neighbours = pd.read_parquet(DEFAULT_NEIGHBOUR_PATH)
    dataset = add_neighbour_price_lags(dataset, neighbours, timezone=timezone)
    atomic_write_parquet(dataset, Path(output_path))
    return {
        "rows": len(dataset),
        "columns": len(dataset.columns),
        "output_path": str(output_path),
    }


def main() -> None:
    """Parse dates and build the DE-LU price model table."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--start-date", required=True)
    parser.add_argument("--end-date", required=True)
    parser.add_argument("--config", default=str(DEFAULT_CONFIG_PATH))
    parser.add_argument("--price-path", default=str(DEFAULT_PRICE_PATH))
    parser.add_argument("--output-path", default=str(DEFAULT_OUTPUT_PATH))
    parser.add_argument(
        "--drop-missing-exogenous",
        action="store_true",
        help="Drop rows with a missing load, weather or JAO value.",
    )
    args = parser.parse_args()

    result = build_de_lu_price_training_dataset(
        args.start_date,
        args.end_date,
        config_path=args.config,
        price_path=args.price_path,
        output_path=args.output_path,
        drop_missing_exogenous=args.drop_missing_exogenous,
    )
    print(
        f"Dataset DE-LU créé : {result['rows']} lignes, "
        f"{result['columns']} colonnes -> {result['output_path']}"
    )


if __name__ == "__main__":
    main()
