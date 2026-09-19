"""Simple models and metrics for DE-LU Day-Ahead price forecasting."""

from __future__ import annotations

from collections import OrderedDict
from sklearn.base import clone

import numpy as np
import pandas as pd
from lightgbm import LGBMRegressor
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.linear_model import Ridge
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from xgboost import XGBRegressor


LOWER_QUANTILE = 0.10
MEDIAN_QUANTILE = 0.50
UPPER_QUANTILE = 0.90
CENTRAL_INTERVAL_COVERAGE = UPPER_QUANTILE - LOWER_QUANTILE


def build_price_models(random_state: int = 42) -> OrderedDict:
    """Return the four deliberately small candidate models.

    Ridge receives standardized inputs because its penalty depends on feature
    scale. Tree models use the original units, which keeps their splits and
    later explanations easier to read.
    """
    return OrderedDict(
        {
            "Ridge": Pipeline(
                [
                    ("scale", StandardScaler()),
                    ("model", Ridge(alpha=10.0)),
                ]
            ),
            "HistGradientBoosting": HistGradientBoostingRegressor(
                learning_rate=0.05,
                max_iter=300,
                max_leaf_nodes=31,
                l2_regularization=1.0,
                random_state=random_state,
            ),
            "XGBoost": XGBRegressor(
                n_estimators=500,
                learning_rate=0.05,
                max_depth=6,
                subsample=0.8,
                colsample_bytree=0.8,
                objective="reg:squarederror",
                n_jobs=4,
                random_state=random_state,
            ),
            "LightGBM": LGBMRegressor(
                n_estimators=500,
                learning_rate=0.05,
                num_leaves=31,
                subsample=0.8,
                colsample_bytree=0.8,
                verbosity=-1,
                random_state=random_state,
            ),
        }
    )


def fit_price_model(model, features, target, sample_weight):
    """Fit one candidate while handling Ridge's pipeline parameter name."""
    if isinstance(model, Pipeline):
        model.fit(features, target, model__sample_weight=sample_weight)
    else:
        model.fit(features, target, sample_weight=sample_weight)
    return model


def price_forecast_metrics(actual, predicted) -> dict[str, float]:
    """Return MAE, RMSE and signed bias in EUR/MWh."""
    actual_values = np.asarray(actual, dtype=float)
    predicted_values = np.asarray(predicted, dtype=float)
    errors = predicted_values - actual_values
    return {
        "MAE": float(np.mean(np.abs(errors))),
        "RMSE": float(np.sqrt(np.mean(errors**2))),
        "biais": float(np.mean(errors)),
    }


def compare_predictions(
    actual: pd.Series,
    predictions: dict[str, np.ndarray | pd.Series],
) -> pd.DataFrame:
    """Build one sorted validation table for the baseline and all models."""
    rows = []
    for model_name, predicted in predictions.items():
        rows.append({"modèle": model_name, **price_forecast_metrics(actual, predicted)})
    return pd.DataFrame(rows).sort_values(["MAE", "RMSE"]).reset_index(drop=True)


def build_price_quantile_models(point_model) -> dict[str, object]:
    """Reuse the selected tree configuration for P10, P50 and P90.

    Only the loss changes. Ridge has no equivalent quantile implementation in
    this small project, so it remains a point-forecast benchmark.
    """
    quantile_levels = {
        "p10_eur_mwh": LOWER_QUANTILE,
        "p50_eur_mwh": MEDIAN_QUANTILE,
        "p90_eur_mwh": UPPER_QUANTILE,
    }
    if isinstance(point_model, XGBRegressor):
        return {
            name: clone(point_model).set_params(
                objective="reg:quantileerror", quantile_alpha=level
            )
            for name, level in quantile_levels.items()
        }
    if isinstance(point_model, LGBMRegressor):
        return {
            name: clone(point_model).set_params(objective="quantile", alpha=level)
            for name, level in quantile_levels.items()
        }
    if isinstance(point_model, HistGradientBoostingRegressor):
        return {
            name: clone(point_model).set_params(loss="quantile", quantile=level)
            for name, level in quantile_levels.items()
        }
    raise ValueError("P10/P50/P90 requires a selected tree model.")


def fit_price_quantile_models(
    quantile_models: dict[str, object],
    features: pd.DataFrame,
    target: pd.Series,
    sample_weight: pd.Series,
) -> dict[str, object]:
    """Fit the three quantile models on the same historical observations."""
    return {
        name: fit_price_model(model, features, target, sample_weight)
        for name, model in quantile_models.items()
    }


def predict_ordered_price_quantiles(
    quantile_models: dict[str, object], features: pd.DataFrame
) -> tuple[pd.DataFrame, float]:
    """Predict P10/P50/P90 and restore their order if independent fits cross."""
    columns = ["p10_eur_mwh", "p50_eur_mwh", "p90_eur_mwh"]
    missing = [column for column in columns if column not in quantile_models]
    if missing:
        raise ValueError(f"Missing quantile models: {missing}")

    raw_quantiles = pd.DataFrame(
        {
            column: quantile_models[column].predict(features)
            for column in columns
        },
        index=features.index,
    )
    crossing = (raw_quantiles[columns[0]] > raw_quantiles[columns[1]]) | (
        raw_quantiles[columns[1]] > raw_quantiles[columns[2]]
    )
    ordered = pd.DataFrame(
        np.sort(raw_quantiles.to_numpy(), axis=1), columns=columns, index=features.index
    )
    return ordered, float(crossing.mean())


def conformal_price_interval_adjustment(
    actual: pd.Series, quantiles: pd.DataFrame
) -> float:
    """Measure one symmetric P10-P90 widening from the validation period.

    The adjustment makes the central interval target 80% coverage. It is
    measured on validation only and then applied unchanged to the test/live
    interval; no realised future price is used at prediction time.
    """
    actual_values = np.asarray(actual, dtype=float)
    lower_error = quantiles["p10_eur_mwh"].to_numpy() - actual_values
    upper_error = actual_values - quantiles["p90_eur_mwh"].to_numpy()
    nonconformity = np.maximum.reduce(
        [lower_error, upper_error, np.zeros_like(actual_values)]
    )
    finite_sample_level = min(
        1.0,
        np.ceil((len(nonconformity) + 1) * CENTRAL_INTERVAL_COVERAGE)
        / len(nonconformity),
    )
    return float(np.quantile(nonconformity, finite_sample_level, method="higher"))


def apply_price_interval_adjustment(
    quantiles: pd.DataFrame, adjustment_eur_mwh: float
) -> pd.DataFrame:
    """Apply the validation-derived widening without moving the median."""
    adjusted = quantiles.copy()
    adjusted["p10_eur_mwh"] -= adjustment_eur_mwh
    adjusted["p90_eur_mwh"] += adjustment_eur_mwh
    return adjusted


def price_probabilistic_metrics(
    actual: pd.Series, quantiles: pd.DataFrame
) -> dict[str, float]:
    """Return quantile losses, interval coverage and width in EUR/MWh."""
    actual_values = np.asarray(actual, dtype=float)
    lower = quantiles["p10_eur_mwh"].to_numpy()
    median = quantiles["p50_eur_mwh"].to_numpy()
    upper = quantiles["p90_eur_mwh"].to_numpy()

    def pinball(prediction: np.ndarray, level: float) -> float:
        residual = actual_values - prediction
        return float(np.maximum(level * residual, (level - 1) * residual).mean())

    inside = (actual_values >= lower) & (actual_values <= upper)
    width = upper - lower
    return {
        "p10_pinball_loss": pinball(lower, LOWER_QUANTILE),
        "p50_pinball_loss": pinball(median, MEDIAN_QUANTILE),
        "p90_pinball_loss": pinball(upper, UPPER_QUANTILE),
        "p10_p90_coverage_pct": 100 * float(inside.mean()),
        "mean_interval_width_eur_mwh": float(width.mean()),
    }
