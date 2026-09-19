"""Contraintes, évaluation et calibration propres à la production solaire."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from time import perf_counter

import numpy as np
import pandas as pd
from lightgbm import LGBMRegressor

from src.modeling.common import (
    temporal_split_labels,
    build_candidate_models,
    best_candidate_per_family,
    build_quantile_lightgbm_models,
    fit_quantile_models,
    pinball_loss,
    grouped_permutation_importance,
    lightgbm_feature_contributions,
    group_feature_contributions,
)


ENERGY_INTERVAL_HOURS = 0.25
LOWER_QUANTILE = 0.10
MEDIAN_QUANTILE = 0.50
UPPER_QUANTILE = 0.90
CENTRAL_INTERVAL_COVERAGE = UPPER_QUANTILE - LOWER_QUANTILE


def apply_solar_constraints(
    predictions: Sequence[float], is_day: Sequence, capacity_mw: float
) -> np.ndarray:
    """Clip forecasts to [0, capacity] and force every night value to zero."""
    bounded_predictions = np.asarray(predictions, dtype=float).reshape(-1).copy()
    daylight_mask = np.asarray(is_day, dtype=bool).reshape(-1)
    if bounded_predictions.shape != daylight_mask.shape:
        raise ValueError("Predictions and is_day must have the same length.")
    if capacity_mw <= 0:
        raise ValueError("capacity_mw must be positive.")

    bounded_predictions = np.clip(bounded_predictions, 0.0, float(capacity_mw))
    bounded_predictions[~daylight_mask] = 0.0
    return bounded_predictions


def solar_regression_metrics(
    y_true: Sequence[float],
    y_pred: Sequence[float],
    timestamps: Sequence,
    is_day: Sequence,
    *,
    capacity_mw: float,
    market_timezone: str,
) -> dict[str, float]:
    """Compute quarter-hour power errors and local-day energy errors."""
    actual_production = np.asarray(y_true, dtype=float).reshape(-1)
    predicted_production = np.asarray(y_pred, dtype=float).reshape(-1)
    daylight_mask = np.asarray(is_day, dtype=bool).reshape(-1)
    utc_timestamps = pd.DatetimeIndex(pd.to_datetime(timestamps))


    power_errors = predicted_production - actual_production
    absolute_power_errors = np.abs(power_errors)
    mae_mw = float(absolute_power_errors.mean())
    rmse_mw = float(np.sqrt(np.mean(np.square(power_errors))))
    daylight_mae_mw = (
        float(absolute_power_errors[daylight_mask].mean())
        if daylight_mask.any()
        else np.nan
    )

    # Power is an average over 15 minutes. Multiplying by 0.25 hour converts
    # each quarter-hour from MW to MWh before aggregating by local market day.
    local_date = utc_timestamps.tz_convert(market_timezone).date
    daily_energy = pd.DataFrame(
        {
            "local_date": local_date,
            "actual_mwh": actual_production * ENERGY_INTERVAL_HOURS,
            "predicted_mwh": predicted_production * ENERGY_INTERVAL_HOURS,
        }
    ).groupby("local_date", sort=True)[["actual_mwh", "predicted_mwh"]].sum()
    daily_energy_errors = daily_energy["predicted_mwh"] - daily_energy["actual_mwh"]

    return {
        "mae_mw": mae_mw,
        "rmse_mw": rmse_mw,
        # The MAE as a percentage of capacity is a simple, interpretable scale 
        "nmae_capacity_pct": 100 * mae_mw / float(capacity_mw),
        # Bias indicates whether the model tends to over- or under-produce.
        "bias_mw": float(power_errors.mean()),
        "daylight_mae_mw": daylight_mae_mw,
        "daily_energy_mae_mwh": float(daily_energy_errors.abs().mean()),
        "daily_energy_bias_mwh": float(daily_energy_errors.mean()),
    }


def hourly_error_profile(
    y_true: Sequence[float],
    y_pred: Sequence[float],
    timestamps: Sequence,
    market_timezone: str,
) -> pd.DataFrame:
    """Show where MAE and over/under-production occur in the local day."""
    actual_production = np.asarray(y_true, dtype=float).reshape(-1)
    predicted_production = np.asarray(y_pred, dtype=float).reshape(-1)
    utc_timestamps = pd.DatetimeIndex(pd.to_datetime(timestamps))

    hourly_errors = pd.DataFrame(
        {
            "local_hour": utc_timestamps.tz_convert(market_timezone).hour,
            "error_mw": predicted_production - actual_production,
        }
    )
    hourly_errors["absolute_error_mw"] = hourly_errors["error_mw"].abs()
    return hourly_errors.groupby("local_hour", sort=True).agg(
        mae_mw=("absolute_error_mw", "mean"),
        bias_mw=("error_mw", "mean"),
        rows=("error_mw", "size"),
    )


def evaluate_candidate_models(
    candidate_models: Mapping[str, Mapping[str, object]],
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_validation: pd.DataFrame,
    y_validation: pd.Series,
    validation_timestamps: Sequence,
    validation_is_day: Sequence,
    *,
    capacity_mw: float,
    market_timezone: str,
) -> tuple[pd.DataFrame, dict[tuple[str, str], object]]:
    """Fit every small-grid candidate and score it on temporal validation."""
    evaluation_records: list[dict[str, object]] = []
    fitted_models: dict[tuple[str, str], object] = {}

    for model_family, configurations in candidate_models.items():
        for configuration_name, estimator in configurations.items():
            fit_started_at = perf_counter()
            estimator.fit(X_train, y_train)
            fit_seconds = perf_counter() - fit_started_at

            constrained_prediction = apply_solar_constraints(
                estimator.predict(X_validation),
                validation_is_day,
                capacity_mw,
            )
            metrics = solar_regression_metrics(
                y_validation,
                constrained_prediction,
                validation_timestamps,
                validation_is_day,
                capacity_mw=capacity_mw,
                market_timezone=market_timezone,
            )
            evaluation_records.append(
                {
                    "family": model_family,
                    "configuration": configuration_name,
                    "fit_seconds": fit_seconds,
                    **metrics,
                }
            )
            fitted_models[(model_family, configuration_name)] = estimator

    evaluation_table = (
        pd.DataFrame(evaluation_records)
        .sort_values(["mae_mw", "daily_energy_mae_mwh"])
        .reset_index(drop=True)
    )
    return evaluation_table, fitted_models


def predict_ordered_quantiles(
    quantile_models: Mapping[str, LGBMRegressor],
    predictors: pd.DataFrame,
    is_day: Sequence,
    capacity_mw: float,
) -> tuple[pd.DataFrame, float]:
    """Predict P10/P50/P90, report crossings, then enforce physical ordering."""
    required_columns = ["p10_mw", "p50_mw", "p90_mw"]
    missing_models = [name for name in required_columns if name not in quantile_models]
    if missing_models:
        raise ValueError(f"Missing quantile models: {missing_models}")

    constrained_quantiles = pd.DataFrame(index=predictors.index)
    for quantile_column in required_columns:
        constrained_quantiles[quantile_column] = apply_solar_constraints(
            quantile_models[quantile_column].predict(predictors),
            is_day,
            capacity_mw,
        )

    crossing_mask = (
        constrained_quantiles["p10_mw"] > constrained_quantiles["p50_mw"]
    ) | (constrained_quantiles["p50_mw"] > constrained_quantiles["p90_mw"])
    crossing_rate = float(crossing_mask.mean())

    # Sorting is a transparent post-processing safeguard. It changes values only
    # on rows where independently trained quantile models cross one another.
    ordered_values = np.sort(constrained_quantiles.to_numpy(), axis=1)
    ordered_quantiles = pd.DataFrame(
        ordered_values,
        columns=required_columns,
        index=predictors.index,
    )
    return ordered_quantiles, crossing_rate


def conformal_interval_adjustment(
    actual_production: Sequence[float],
    calibration_quantiles: pd.DataFrame,
    is_day: Sequence,
    *,
    target_coverage: float = CENTRAL_INTERVAL_COVERAGE,
) -> float:
    """Return a daylight-only expansion that calibrates the central interval.

    The validation period is used only to measure how far observations fall
    outside raw P10-P90. A finite-sample 'higher' quantile produces a single MW
    adjustment applied symmetrically to future lower and upper bounds.
    """
    if not 0 < target_coverage < 1:
        raise ValueError("target_coverage must lie strictly between 0 and 1.")
    actual = np.asarray(actual_production, dtype=float).reshape(-1)
    daylight_mask = np.asarray(is_day, dtype=bool).reshape(-1)
    if len(actual) != len(calibration_quantiles) or len(actual) != len(daylight_mask):
        raise ValueError("Calibration inputs must have the same length.")
    if not daylight_mask.any():
        raise ValueError("Calibration requires at least one daylight row.")

    lower_error = calibration_quantiles["p10_mw"].to_numpy() - actual
    upper_error = actual - calibration_quantiles["p90_mw"].to_numpy()
    # A zero floor means calibration can widen, but never shrink, the raw band.
    nonconformity = np.maximum.reduce(
        [lower_error, upper_error, np.zeros_like(actual)]
    )[daylight_mask]
    finite_sample_level = min(
        1.0,
        np.ceil((len(nonconformity) + 1) * target_coverage) / len(nonconformity),
    )
    return float(np.quantile(nonconformity, finite_sample_level, method="higher"))


def apply_conformal_adjustment(
    ordered_quantiles: pd.DataFrame,
    adjustment_mw: float,
    is_day: Sequence,
    capacity_mw: float,
) -> pd.DataFrame:
    """Expand P10-P90 by the calibrated MW amount and reapply solar bounds."""
    adjusted_quantiles = ordered_quantiles.copy()
    adjusted_quantiles["p10_mw"] = apply_solar_constraints(
        adjusted_quantiles["p10_mw"] - adjustment_mw,
        is_day,
        capacity_mw,
    )
    adjusted_quantiles["p50_mw"] = apply_solar_constraints(
        adjusted_quantiles["p50_mw"], is_day, capacity_mw
    )
    adjusted_quantiles["p90_mw"] = apply_solar_constraints(
        adjusted_quantiles["p90_mw"] + adjustment_mw,
        is_day,
        capacity_mw,
    )
    ordered_values = np.sort(adjusted_quantiles.to_numpy(), axis=1)
    return pd.DataFrame(
        ordered_values,
        columns=["p10_mw", "p50_mw", "p90_mw"],
        index=ordered_quantiles.index,
    )


def probabilistic_forecast_metrics(
    y_true: Sequence[float],
    quantile_predictions: pd.DataFrame,
    timestamps: Sequence,
    is_day: Sequence,
    *,
    capacity_mw: float,
    market_timezone: str,
) -> dict[str, float]:
    """Evaluate P10/P50/P90 sharpness, calibration and median point accuracy."""
    actual = np.asarray(y_true, dtype=float).reshape(-1)
    daylight_mask = np.asarray(is_day, dtype=bool).reshape(-1)
    lower = quantile_predictions["p10_mw"].to_numpy()
    median = quantile_predictions["p50_mw"].to_numpy()
    upper = quantile_predictions["p90_mw"].to_numpy()
    if not (
        len(actual)
        == len(daylight_mask)
        == len(lower)
        == len(median)
        == len(upper)
    ):
        raise ValueError("Probabilistic metric inputs must have the same length.")

    inside_interval = (actual >= lower) & (actual <= upper)
    interval_width = upper - lower
    interval_miss_probability = 1 - CENTRAL_INTERVAL_COVERAGE
    interval_score = interval_width.copy()
    interval_score += (
        2 / interval_miss_probability * (lower - actual) * (actual < lower)
    )
    interval_score += (
        2 / interval_miss_probability * (actual - upper) * (actual > upper)
    )

    median_metrics = solar_regression_metrics(
        actual,
        median,
        timestamps,
        daylight_mask,
        capacity_mw=capacity_mw,
        market_timezone=market_timezone,
    )
    return {
        **{f"p50_{name}": value for name, value in median_metrics.items()},
        "p10_pinball_loss_mw": pinball_loss(actual, lower, LOWER_QUANTILE),
        "p50_pinball_loss_mw": pinball_loss(actual, median, MEDIAN_QUANTILE),
        "p90_pinball_loss_mw": pinball_loss(actual, upper, UPPER_QUANTILE),
        "p10_p90_coverage_pct": 100 * float(inside_interval.mean()),
        "daylight_coverage_pct": (
            100 * float(inside_interval[daylight_mask].mean())
            if daylight_mask.any()
            else np.nan
        ),
        "mean_interval_width_mw": float(interval_width.mean()),
        "daylight_interval_width_mw": (
            float(interval_width[daylight_mask].mean())
            if daylight_mask.any()
            else np.nan
        ),
        "mean_interval_score_mw": float(interval_score.mean()),
        "daylight_interval_score_mw": (
            float(interval_score[daylight_mask].mean())
            if daylight_mask.any()
            else np.nan
        ),
    }
