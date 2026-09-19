"""Train and evaluate direct-global wind forecasts for Langer Wald."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from time import perf_counter

import numpy as np
import pandas as pd
from lightgbm import LGBMRegressor
from sklearn.base import clone

from src.modeling.common import (
    best_candidate_per_family,
    build_candidate_models,
    build_quantile_lightgbm_models,
    fit_quantile_models,
    group_feature_contributions,
    grouped_permutation_importance,
    lightgbm_feature_contributions,
    pinball_loss,
    temporal_split_labels,
)


INTERVAL_HOURS = 0.25
LOWER_QUANTILE = 0.10
MEDIAN_QUANTILE = 0.50
UPPER_QUANTILE = 0.90
CENTRAL_COVERAGE = UPPER_QUANTILE - LOWER_QUANTILE


def apply_wind_constraints(predictions: Sequence[float], capacity_mw: float) -> np.ndarray:
    """Keep wind forecasts between zero and installed capacity."""
    if capacity_mw <= 0:
        raise ValueError("capacity_mw must be positive.")
    values = np.asarray(predictions, dtype=float).reshape(-1)
    return np.clip(values, 0.0, float(capacity_mw))


def wind_regression_metrics(
    y_true: Sequence[float],
    y_pred: Sequence[float],
    timestamps: Sequence,
    *,
    capacity_mw: float,
    market_timezone: str,
) -> dict[str, float]:
    """Measure quarter-hour power errors and local-day energy errors."""
    actual = np.asarray(y_true, dtype=float).reshape(-1)
    predicted = np.asarray(y_pred, dtype=float).reshape(-1)
    if len(actual) != len(predicted):
        raise ValueError("Actual and predicted production must have the same length.")

    errors = predicted - actual
    local_dates = pd.DatetimeIndex(pd.to_datetime(timestamps)).tz_convert(
        market_timezone
    ).date
    daily_energy = pd.DataFrame(
        {
            "local_date": local_dates,
            "actual_mwh": actual * INTERVAL_HOURS,
            "predicted_mwh": predicted * INTERVAL_HOURS,
        }
    ).groupby("local_date", sort=True)[["actual_mwh", "predicted_mwh"]].sum()
    daily_errors = daily_energy["predicted_mwh"] - daily_energy["actual_mwh"]

    mae_mw = float(np.mean(np.abs(errors)))
    return {
        "mae_mw": mae_mw,
        "rmse_mw": float(np.sqrt(np.mean(np.square(errors)))),
        "nmae_capacity_pct": 100 * mae_mw / float(capacity_mw),
        "bias_mw": float(np.mean(errors)),
        "daily_energy_mae_mwh": float(np.mean(np.abs(daily_errors))),
        "daily_energy_bias_mwh": float(np.mean(daily_errors)),
    }


def build_wind_candidate_models(random_state: int = 42) -> dict[str, dict[str, object]]:
    """Return the candidate configurations for the wind study."""
    return build_candidate_models(random_state=random_state)


def evaluate_wind_models(
    candidate_models: Mapping[str, Mapping[str, object]],
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_validation: pd.DataFrame,
    y_validation: pd.Series,
    validation_timestamps: Sequence,
    *,
    capacity_mw: float,
    market_timezone: str,
) -> tuple[pd.DataFrame, dict[tuple[str, str], object]]:
    """Fit each small candidate and rank it on the validation MAE."""
    records = []
    fitted_models = {}
    for family, configurations in candidate_models.items():
        for configuration, model in configurations.items():
            started_at = perf_counter()
            fitted_model = model.fit(X_train, y_train)
            fit_seconds = perf_counter() - started_at
            prediction = apply_wind_constraints(
                fitted_model.predict(X_validation), capacity_mw
            )
            metrics = wind_regression_metrics(
                y_validation,
                prediction,
                validation_timestamps,
                capacity_mw=capacity_mw,
                market_timezone=market_timezone,
            )
            records.append(
                {
                    "family": family,
                    "configuration": configuration,
                    "fit_seconds": fit_seconds,
                    **metrics,
                }
            )
            fitted_models[(family, configuration)] = fitted_model

    results = pd.DataFrame(records).sort_values(
        ["mae_mw", "daily_energy_mae_mwh"], ignore_index=True
    )
    return results, fitted_models


def evaluate_wind_temporal_cv(
    candidate_models: Mapping[str, Mapping[str, object]],
    model_data: pd.DataFrame,
    predictor_columns: list[str],
    validation_windows: pd.DataFrame,
    *,
    capacity_mw: float,
    market_timezone: str,
) -> pd.DataFrame:
    """Évalue chaque configuration sur des fenêtres temporelles successives.

    Chaque fenêtre contient start_local et end_local (borne de fin exclue).
    Un nouveau modèle apprend sur tout le passé antérieur à start_local.
    La persistance 48 h est évaluée sur exactement les mêmes lignes.
    """
    results = []
    local_time = model_data["timestamp"].dt.tz_convert(market_timezone)
    for window in validation_windows.itertuples(index=False):
        start = pd.Timestamp(window.start_local, tz=market_timezone)
        end = pd.Timestamp(window.end_local, tz=market_timezone)
        train = model_data.loc[local_time < start]
        validation = model_data.loc[(local_time >= start) & (local_time < end)]
        if train.empty or validation.empty:
            raise ValueError("Each temporal fold needs training and validation rows.")

        # Le scaler de Ridge est lui aussi réajusté uniquement sur le train.
        fresh_models = {
            family: {name: clone(model) for name, model in configurations.items()}
            for family, configurations in candidate_models.items()
        }
        scores, _ = evaluate_wind_models(
            fresh_models,
            train[predictor_columns], train["production_mw"],
            validation[predictor_columns], validation["production_mw"],
            validation["timestamp"],
            capacity_mw=capacity_mw,
            market_timezone=market_timezone,
        )
        baseline = wind_regression_metrics(
            validation["production_mw"],
            apply_wind_constraints(
                validation["production_lag_48h_mw"], capacity_mw
            ),
            validation["timestamp"],
            capacity_mw=capacity_mw,
            market_timezone=market_timezone,
        )
        scores = pd.concat([
            scores,
            pd.DataFrame([{
                "family": "Persistence 48h", "configuration": "fixed",
                "fit_seconds": 0.0, **baseline,
            }]),
        ], ignore_index=True)
        scores["fold"] = window.fold
        scores["training_rows"] = len(train)
        scores["validation_rows"] = len(validation)
        results.append(scores)
    return pd.concat(results, ignore_index=True)


def build_wind_quantile_models(random_state: int = 42) -> dict[str, LGBMRegressor]:
    """Build LightGBM P10, P50 and P90 models for wind production."""
    return build_quantile_lightgbm_models(random_state=random_state)


def predict_wind_quantiles(
    quantile_models: Mapping[str, LGBMRegressor],
    predictors: pd.DataFrame,
    capacity_mw: float,
) -> tuple[pd.DataFrame, float]:
    """Predict bounded P10/P50/P90 and correct any quantile crossing."""
    columns = ["p10_mw", "p50_mw", "p90_mw"]
    missing_models = [column for column in columns if column not in quantile_models]
    if missing_models:
        raise ValueError(f"Missing quantile models: {missing_models}")

    raw_predictions = np.column_stack(
        [
            apply_wind_constraints(quantile_models[column].predict(predictors), capacity_mw)
            for column in columns
        ]
    )
    crossing_rate = float(
        np.mean(
            (raw_predictions[:, 0] > raw_predictions[:, 1])
            | (raw_predictions[:, 1] > raw_predictions[:, 2])
        )
    )
    return (
        pd.DataFrame(
            np.sort(raw_predictions, axis=1),
            columns=columns,
            index=predictors.index,
        ),
        crossing_rate,
    )


def wind_conformal_adjustment(
    actual_production: Sequence[float],
    calibration_quantiles: pd.DataFrame,
    *,
    target_coverage: float = CENTRAL_COVERAGE,
) -> float:
    """Find an expansion that reaches the requested validation coverage."""
    actual = np.asarray(actual_production, dtype=float).reshape(-1)
    if len(actual) != len(calibration_quantiles):
        raise ValueError("Calibration inputs must have the same length.")
    lower_error = calibration_quantiles["p10_mw"].to_numpy() - actual
    upper_error = actual - calibration_quantiles["p90_mw"].to_numpy()
    nonconformity = np.maximum.reduce(
        [lower_error, upper_error, np.zeros_like(actual)]
    )
    finite_sample_level = min(
        1.0,
        np.ceil((len(nonconformity) + 1) * target_coverage) / len(nonconformity),
    )
    return float(np.quantile(nonconformity, finite_sample_level, method="higher"))


def apply_wind_conformal_adjustment(
    ordered_quantiles: pd.DataFrame,
    adjustment_mw: float,
    capacity_mw: float,
) -> pd.DataFrame:
    """Expand P10-P90 by the calibrated amount and keep physical bounds."""
    adjusted = ordered_quantiles.copy()
    adjusted["p10_mw"] = apply_wind_constraints(
        adjusted["p10_mw"] - adjustment_mw, capacity_mw
    )
    adjusted["p50_mw"] = apply_wind_constraints(adjusted["p50_mw"], capacity_mw)
    adjusted["p90_mw"] = apply_wind_constraints(
        adjusted["p90_mw"] + adjustment_mw, capacity_mw
    )
    return pd.DataFrame(
        np.sort(adjusted.to_numpy(), axis=1),
        columns=["p10_mw", "p50_mw", "p90_mw"],
        index=adjusted.index,
    )


def wind_probabilistic_metrics(
    y_true: Sequence[float],
    quantile_predictions: pd.DataFrame,
    timestamps: Sequence,
    *,
    capacity_mw: float,
    market_timezone: str,
) -> dict[str, float]:
    """Evaluate median accuracy, interval coverage, width and pinball loss."""
    actual = np.asarray(y_true, dtype=float).reshape(-1)
    lower = quantile_predictions["p10_mw"].to_numpy()
    median = quantile_predictions["p50_mw"].to_numpy()
    upper = quantile_predictions["p90_mw"].to_numpy()
    inside = (actual >= lower) & (actual <= upper)
    width = upper - lower

    miss_probability = 1 - CENTRAL_COVERAGE
    interval_score = width.copy()
    interval_score += 2 / miss_probability * (lower - actual) * (actual < lower)
    interval_score += 2 / miss_probability * (actual - upper) * (actual > upper)

    median_metrics = wind_regression_metrics(
        actual,
        median,
        timestamps,
        capacity_mw=capacity_mw,
        market_timezone=market_timezone,
    )
    return {
        **{f"p50_{name}": value for name, value in median_metrics.items()},
        "p10_pinball_loss_mw": pinball_loss(actual, lower, LOWER_QUANTILE),
        "p50_pinball_loss_mw": pinball_loss(actual, median, MEDIAN_QUANTILE),
        "p90_pinball_loss_mw": pinball_loss(actual, upper, UPPER_QUANTILE),
        "p10_p90_coverage_pct": 100 * float(np.mean(inside)),
        "mean_interval_width_mw": float(np.mean(width)),
        "mean_interval_score_mw": float(np.mean(interval_score)),
    }


__all__ = [
    "apply_wind_conformal_adjustment",
    "apply_wind_constraints",
    "best_candidate_per_family",
    "build_wind_candidate_models",
    "build_wind_quantile_models",
    "evaluate_wind_models",
    "evaluate_wind_temporal_cv",
    "fit_quantile_models",
    "group_feature_contributions",
    "grouped_permutation_importance",
    "lightgbm_feature_contributions",
    "predict_wind_quantiles",
    "temporal_split_labels",
    "wind_conformal_adjustment",
    "wind_probabilistic_metrics",
    "wind_regression_metrics",
]
