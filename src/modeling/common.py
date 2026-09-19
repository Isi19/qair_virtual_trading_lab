"""Modèles et outils statistiques communs aux études éolienne et solaire."""

from __future__ import annotations

from collections.abc import Mapping, Sequence

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


def temporal_split_labels(
    timestamps: Sequence,
    market_timezone: str,
    *,
    train_end_local: str = "2026-01-01",
    validation_end_local: str = "2026-05-01",
    test_end_local: str = "2026-08-23",
) -> np.ndarray:
    """Assign ordered train, validation, test and outside labels in market time."""
    utc_timestamps = pd.DatetimeIndex(pd.to_datetime(timestamps))
    if utc_timestamps.tz is None:
        raise ValueError("Temporal split timestamps must be timezone-aware.")
    local_timestamps = utc_timestamps.tz_convert(market_timezone).tz_localize(None)
    return np.select(
        [
            local_timestamps < pd.Timestamp(train_end_local),
            local_timestamps < pd.Timestamp(validation_end_local),
            local_timestamps < pd.Timestamp(test_end_local),
        ],
        ["train", "validation", "test"],
        default="outside",
    )


def build_candidate_models(random_state: int = 42) -> dict[str, dict[str, object]]:
    """Return the intentionally small validation grid used in the notebook."""
    # Ridge needs scaling so its L2 penalty treats every physical unit fairly.
    ridge_candidates = {
        f"alpha_{alpha:g}": Pipeline(
            [
                ("scaler", StandardScaler()),
                ("model", Ridge(alpha=alpha)),
            ]
        )
        for alpha in (1.0, 10.0, 100.0)
    }

    # Early stopping is disabled because its automatic random holdout would
    # violate the explicit temporal validation used .
    hist_gradient_boosting_candidates = {
        f"leaves_{leaves}": HistGradientBoostingRegressor(
            loss="squared_error",
            learning_rate=0.05,
            max_iter=300,
            max_leaf_nodes=leaves,
            min_samples_leaf=40,
            l2_regularization=1.0,
            early_stopping=False,
            random_state=random_state,
        )
        for leaves in (15, 31)
    }
    xgboost_candidates = {
        f"depth_{depth}": XGBRegressor(
            objective="reg:squarederror",
            n_estimators=400,
            learning_rate=0.05,
            max_depth=depth,
            min_child_weight=5,
            subsample=0.9,
            colsample_bytree=0.9,
            reg_lambda=1.0,
            tree_method="hist",
            n_jobs=4,
            random_state=random_state,
        )
        for depth in (4, 6)
    }
    lightgbm_candidates = {
        f"leaves_{leaves}": LGBMRegressor(
            objective="regression",
            n_estimators=400,
            learning_rate=0.05,
            num_leaves=leaves,
            min_child_samples=40,
            subsample=0.9,
            colsample_bytree=0.9,
            reg_lambda=1.0,
            verbosity=-1,
            deterministic=True,
            force_col_wise=True,
            n_jobs=4,
            random_state=random_state,
        )
        for leaves in (31, 63)
    }
    return {
        "Ridge": ridge_candidates,
        "HistGradientBoosting": hist_gradient_boosting_candidates,
        "XGBoost": xgboost_candidates,
        "LightGBM": lightgbm_candidates,
    }


def best_candidate_per_family(candidate_results: pd.DataFrame) -> pd.DataFrame:
    """Keep the lowest-validation-MAE configuration from each model family."""
    return (
        candidate_results
        .sort_values(["family", "mae_mw", "daily_energy_mae_mwh"])
        .groupby("family", as_index=False, sort=False)
        .first()
    )


def build_quantile_lightgbm_models(
    random_state: int = 42,
) -> dict[str, LGBMRegressor]:
    """Build P10, P50 and P90 with the same fixed LightGBM configuration."""
    quantile_levels = {
        "p10_mw": LOWER_QUANTILE,
        "p50_mw": MEDIAN_QUANTILE,
        "p90_mw": UPPER_QUANTILE,
    }
    return {
        column: LGBMRegressor(
            objective="quantile",
            alpha=quantile_level,
            n_estimators=400,
            learning_rate=0.05,
            num_leaves=63,
            min_child_samples=40,
            subsample=0.9,
            colsample_bytree=0.9,
            reg_lambda=1.0,
            verbosity=-1,
            deterministic=True,
            force_col_wise=True,
            n_jobs=4,
            random_state=random_state,
        )
        for column, quantile_level in quantile_levels.items()
    }


def fit_quantile_models(
    quantile_models: Mapping[str, LGBMRegressor],
    predictors: pd.DataFrame,
    target: pd.Series,
) -> dict[str, LGBMRegressor]:
    """Fit the three quantiles on the same historical rows and features."""
    fitted_models: dict[str, LGBMRegressor] = {}
    for quantile_column, model in quantile_models.items():
        fitted_models[quantile_column] = model.fit(predictors, target)
    return fitted_models


def pinball_loss(
    actual_production: Sequence[float],
    quantile_prediction: Sequence[float],
    quantile_level: float,
) -> float:
    """Measure asymmetric quantile error; lower is better calibrated and sharper."""
    actual = np.asarray(actual_production, dtype=float).reshape(-1)
    predicted_quantile = np.asarray(quantile_prediction, dtype=float).reshape(-1)
    if len(actual) != len(predicted_quantile):
        raise ValueError("Pinball-loss inputs must have the same length.")
    if not 0 < quantile_level < 1:
        raise ValueError("quantile_level must lie strictly between 0 and 1.")
    residual = actual - predicted_quantile
    loss = np.maximum(quantile_level * residual, (quantile_level - 1) * residual)
    return float(loss.mean())


def grouped_permutation_importance(
    model: object,
    predictors: pd.DataFrame,
    target: Sequence[float],
    feature_groups: Mapping[str, Sequence[str]],
    *,
    n_repeats: int = 5,
    random_state: int = 42,
) -> pd.DataFrame:
    """Measure the MAE increase when a whole business feature group is shuffled.

    Every column in one group receives the same row permutation. This preserves
    relationships inside correlated groups such as the radiation variables while
    removing the link between that group and the production target.
    """
    if n_repeats < 1:
        raise ValueError("n_repeats must be at least one.")

    actual = np.asarray(target, dtype=float).reshape(-1)
    if len(actual) != len(predictors):
        raise ValueError("Predictors and target must have the same length.")

    grouped_columns = [
        column for columns in feature_groups.values() for column in columns
    ]
    missing_columns = sorted(set(grouped_columns) - set(predictors.columns))
    if missing_columns:
        raise ValueError(f"Unknown grouped feature columns: {missing_columns}")
    if len(grouped_columns) != len(set(grouped_columns)):
        raise ValueError("A feature cannot belong to more than one group.")

    # The reference error is computed once on the untouched validation sample.
    baseline_prediction = np.asarray(
        model.predict(predictors), dtype=float
    ).reshape(-1)
    baseline_mae = float(np.mean(np.abs(baseline_prediction - actual)))
    random_generator = np.random.default_rng(random_state)
    importance_records: list[dict[str, float | str | int]] = []

    for group_name, group_columns in feature_groups.items():
        if not group_columns:
            raise ValueError(f"Feature group {group_name!r} cannot be empty.")

        mae_increases = []
        for _ in range(n_repeats):
            shuffled_predictors = predictors.copy()
            shuffled_rows = random_generator.permutation(len(predictors))
            # One common permutation keeps correlations within the group intact.
            shuffled_predictors.loc[:, group_columns] = predictors.loc[
                :, group_columns
            ].to_numpy()[shuffled_rows]
            shuffled_prediction = np.asarray(
                model.predict(shuffled_predictors), dtype=float
            ).reshape(-1)
            shuffled_mae = float(np.mean(np.abs(shuffled_prediction - actual)))
            mae_increases.append(shuffled_mae - baseline_mae)

        importance_records.append(
            {
                "group": group_name,
                "feature_count": len(group_columns),
                "mae_increase_mw": float(np.mean(mae_increases)),
                "std_mw": float(np.std(mae_increases, ddof=0)),
            }
        )

    return pd.DataFrame(importance_records).sort_values(
        "mae_increase_mw", ascending=False, ignore_index=True
    )


def lightgbm_feature_contributions(
    model: LGBMRegressor, predictors: pd.DataFrame
) -> tuple[pd.DataFrame, pd.Series]:
    """Return TreeSHAP contributions and the model baseline in MW.

    LightGBM's ``pred_contrib=True`` implements its native TreeSHAP calculation.
    For each row, the baseline plus all feature contributions reconstructs the
    raw model prediction before clipping, night forcing or quantile reordering.
    """
    raw_contributions = np.asarray(model.predict(predictors, pred_contrib=True))
    expected_columns = predictors.shape[1] + 1
    if raw_contributions.ndim != 2 or raw_contributions.shape[1] != expected_columns:
        raise ValueError("Unexpected LightGBM contribution matrix shape.")
    feature_contributions = pd.DataFrame(
        raw_contributions[:, :-1],
        columns=predictors.columns,
        index=predictors.index,
    )
    baseline_contribution = pd.Series(
        raw_contributions[:, -1],
        index=predictors.index,
        name="model_baseline_mw",
    )
    return feature_contributions, baseline_contribution


def group_feature_contributions(
    feature_contributions: pd.DataFrame,
    feature_to_group: Mapping[str, str],
) -> pd.DataFrame:
    """Sum row-level TreeSHAP values into business-readable feature groups."""
    missing_features = sorted(
        set(feature_contributions.columns) - set(feature_to_group)
    )
    if missing_features:
        raise ValueError(f"Features without an explanation group: {missing_features}")

    grouped_contributions: dict[str, pd.Series] = {}
    for feature_name in feature_contributions.columns:
        group_name = feature_to_group[feature_name]
        if group_name not in grouped_contributions:
            grouped_contributions[group_name] = feature_contributions[
                feature_name
            ].copy()
        else:
            grouped_contributions[group_name] += feature_contributions[feature_name]

    return pd.DataFrame(grouped_contributions, index=feature_contributions.index)
