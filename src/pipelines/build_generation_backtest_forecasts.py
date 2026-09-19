"""Replay production forecasts with archived J-1 weather, without model leakage."""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from src.config import load_assets
from src.features.solar import build_solar_forecast_frame, build_solar_training_frame, solar_feature_columns
from src.features.wind import build_wind_forecast_frame, build_wind_training_frame, wind_feature_columns
from src.modeling.common import (
    build_quantile_lightgbm_models,
    fit_quantile_models,
)
from src.modeling.solar import (
    apply_conformal_adjustment,
    conformal_interval_adjustment,
    predict_ordered_quantiles,
)
from src.modeling.wind import (
    apply_wind_conformal_adjustment,
    build_wind_quantile_models,
    predict_wind_quantiles,
    wind_conformal_adjustment,
)
from src.pipelines.storage import atomic_write_parquet


PROJECT_ROOT = Path(__file__).resolve().parents[2]
ASSET_DATA_DIRECTORY = PROJECT_ROOT / "data" / "asset_data"
OUTPUT_DIRECTORY = PROJECT_ROOT / "data" / "backtesting" / "production_forecasts"
CALIBRATION_START_LOCAL = "2026-06-01"
BACKTEST_START_LOCAL = "2026-07-01"
BACKTEST_END_LOCAL = "2026-08-31"


def _active_assets(assets: dict[str, dict]) -> dict[str, dict]:
    """Keep this first replay limited to the active DE-LU portfolio."""
    return {
        asset_id: config
        for asset_id, config in assets.items()
        if config.get("portfolio_status", "active") == "active"
    }


def _model_training_masks(
    timestamps: pd.Series, market_timezone: str
) -> tuple[pd.Series, pd.Series]:
    """Return the fit rows and the June calibration rows from realised history."""
    local_time = pd.to_datetime(timestamps, utc=True).dt.tz_convert(market_timezone)
    calibration_start = pd.Timestamp(CALIBRATION_START_LOCAL, tz=market_timezone)
    backtest_start = pd.Timestamp(BACKTEST_START_LOCAL, tz=market_timezone)
    training = local_time < calibration_start
    calibration = (local_time >= calibration_start) & (local_time < backtest_start)
    if not training.any() or not calibration.any():
        raise ValueError("The fixed replay split does not contain training and calibration rows.")
    return training, calibration


def _select_local_window(
    backtest: pd.DataFrame,
    market_timezone: str,
    start_local: str,
    end_local: str,
) -> pd.DataFrame:
    """Keep an inclusive local delivery window from archived J-1 forecasts."""
    local_dates = pd.to_datetime(backtest["timestamp"], utc=True).dt.tz_convert(
        market_timezone
    ).dt.date
    start = pd.Timestamp(start_local).date()
    end = pd.Timestamp(end_local).date()
    selected = backtest.loc[(local_dates >= start) & (local_dates <= end)].copy()
    if selected.empty:
        raise ValueError("The configured backtest delivery window has no rows.")
    return selected


def _forecast_solar(
    historical: pd.DataFrame,
    calibration_weather: pd.DataFrame,
    j1_weather: pd.DataFrame,
    asset_config: dict,
) -> tuple[pd.DataFrame, float, int]:
    """Replay solar P10/P50/P90 from the model frozen before June."""
    columns = solar_feature_columns(asset_config)
    model_data = build_solar_training_frame(historical, asset_config).dropna(
        subset=[*columns, "production_mw"]
    )
    training_rows, _ = _model_training_masks(
        model_data["timestamp"], asset_config["market_timezone"]
    )
    models = fit_quantile_models(
        build_quantile_lightgbm_models(random_state=42),
        model_data.loc[training_rows, columns],
        model_data.loc[training_rows, "production_mw"],
    )
    calibration_predictors = build_solar_forecast_frame(
        calibration_weather, historical, asset_config
    )
    calibration_quantiles, _ = predict_ordered_quantiles(
        models,
        calibration_predictors[columns],
        calibration_predictors["is_day"],
        float(asset_config["capacity_mw"]),
    )
    calibration_actual = historical.set_index("timestamp")["production_mw"].reindex(
        calibration_weather["timestamp"]
    )
    adjustment = conformal_interval_adjustment(
        calibration_actual,
        calibration_quantiles,
        calibration_predictors["is_day"],
    )
    predictors = build_solar_forecast_frame(j1_weather, historical, asset_config)
    raw_quantiles, _ = predict_ordered_quantiles(
        models, predictors[columns], predictors["is_day"], float(asset_config["capacity_mw"])
    )
    return (
        apply_conformal_adjustment(
            raw_quantiles, adjustment, predictors["is_day"], float(asset_config["capacity_mw"])
        ),
        adjustment,
        int(training_rows.sum()),
    )


def _forecast_wind(
    historical: pd.DataFrame,
    calibration_weather: pd.DataFrame,
    j1_weather: pd.DataFrame,
    asset_config: dict,
) -> tuple[pd.DataFrame, float, int]:
    """Replay wind P10/P50/P90 from the model frozen before June."""
    columns = wind_feature_columns(asset_config)
    model_data = build_wind_training_frame(historical, asset_config).dropna(
        subset=[*columns, "production_mw"]
    )
    training_rows, _ = _model_training_masks(
        model_data["timestamp"], asset_config["market_timezone"]
    )
    models = fit_quantile_models(
        build_wind_quantile_models(random_state=42),
        model_data.loc[training_rows, columns],
        model_data.loc[training_rows, "production_mw"],
    )
    calibration_predictors = build_wind_forecast_frame(
        calibration_weather, historical, asset_config
    )
    calibration_quantiles, _ = predict_wind_quantiles(
        models, calibration_predictors[columns], float(asset_config["capacity_mw"])
    )
    calibration_actual = historical.set_index("timestamp")["production_mw"].reindex(
        calibration_weather["timestamp"]
    )
    adjustment = wind_conformal_adjustment(
        calibration_actual, calibration_quantiles
    )
    predictors = build_wind_forecast_frame(j1_weather, historical, asset_config)
    raw_quantiles, _ = predict_wind_quantiles(
        models, predictors[columns], float(asset_config["capacity_mw"])
    )
    return (
        apply_wind_conformal_adjustment(raw_quantiles, adjustment, float(asset_config["capacity_mw"])),
        adjustment,
        int(training_rows.sum()),
    )


def build_asset_generation_backtest(
    asset_id: str, asset_config: dict
) -> dict[str, str | int | float]:
    """Write point-in-time production forecasts and their later realised labels."""
    historical = pd.read_parquet(
        ASSET_DATA_DIRECTORY / f"{asset_id}_historical_reanalysis_15m.parquet"
    )
    j1_weather = pd.read_parquet(
        ASSET_DATA_DIRECTORY / f"{asset_id}_j1_forecast_weather_15m.parquet"
    )
    historical["timestamp"] = pd.to_datetime(historical["timestamp"], utc=True)
    j1_weather["timestamp"] = pd.to_datetime(j1_weather["timestamp"], utc=True)
    calibration_weather = _select_local_window(
        j1_weather,
        asset_config["market_timezone"],
        CALIBRATION_START_LOCAL,
        "2026-06-30",
    )
    backtest = _select_local_window(
        j1_weather,
        asset_config["market_timezone"],
        BACKTEST_START_LOCAL,
        BACKTEST_END_LOCAL,
    )
    if asset_config["technology"] == "solar":
        quantiles, adjustment, training_rows = _forecast_solar(
            historical, calibration_weather, backtest, asset_config
        )
    elif asset_config["technology"] == "wind":
        quantiles, adjustment, training_rows = _forecast_wind(
            historical, calibration_weather, backtest, asset_config
        )
    else:
        raise ValueError(f"Unsupported technology: {asset_config['technology']}")

    actual_generation = historical.set_index("timestamp")["production_mw"].reindex(
        backtest["timestamp"]
    )
    if actual_generation.isna().any():
        raise ValueError("Some J-1 deliveries have no realised virtual production.")

    result = pd.DataFrame(
        {
            "delivery_start_utc": pd.to_datetime(backtest["timestamp"], utc=True),
            "asset_id": asset_id,
            "technology": asset_config["technology"],
            "actual_generation_mw": actual_generation.to_numpy(),
            "generation_p10_mw": quantiles["p10_mw"].to_numpy(),
            "generation_p50_mw": quantiles["p50_mw"].to_numpy(),
            "generation_p90_mw": quantiles["p90_mw"].to_numpy(),
            "forecast_kind": "archived_j1_weather_replay",
            "forecast_run_timestamp_utc": backtest["forecast_run_timestamp_utc"].to_numpy(),
            "forecast_cutoff_timestamp_utc": backtest[
                "forecast_cutoff_timestamp_utc"
            ].to_numpy(),
            "forecast_model": backtest["forecast_model"].to_numpy(),
            "model_name": "lightgbm_quantile",
            "model_training_end_local": "2026-05-31",
            "calibration_start_local": CALIBRATION_START_LOCAL,
            "conformal_adjustment_mw": adjustment,
            "unit": "MW",
        }
    )
    output_path = OUTPUT_DIRECTORY / f"{asset_id}_j1_forecasts_20260701_20260831.parquet"
    atomic_write_parquet(result, output_path)
    return {
        "asset_id": asset_id,
        "rows": len(result),
        "training_rows": training_rows,
        "conformal_adjustment_mw": adjustment,
        "output_path": str(output_path),
    }


def main() -> None:
    """Build the production replay for every active asset."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config", type=Path, default=PROJECT_ROOT / "config" / "assets.yaml"
    )
    parser.add_argument("--asset", help="Replay only one configured active asset.")
    args = parser.parse_args()
    assets = _active_assets(load_assets(args.config))
    if args.asset:
        if args.asset not in assets:
            raise ValueError(f"Unknown active asset '{args.asset}'.")
        assets = {args.asset: assets[args.asset]}
    summaries = [build_asset_generation_backtest(asset_id, config) for asset_id, config in assets.items()]
    print(pd.DataFrame(summaries).to_string(index=False))


if __name__ == "__main__":
    main()
