"""Model training and evaluation helpers."""

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

from src.modeling.solar import (
    apply_conformal_adjustment,
    apply_solar_constraints,
    conformal_interval_adjustment,
    evaluate_candidate_models,
    hourly_error_profile,
    predict_ordered_quantiles,
    probabilistic_forecast_metrics,
    solar_regression_metrics,
)
from src.modeling.wind import (
    apply_wind_conformal_adjustment,
    apply_wind_constraints,
    build_wind_candidate_models,
    build_wind_quantile_models,
    evaluate_wind_models,
    predict_wind_quantiles,
    wind_conformal_adjustment,
    wind_probabilistic_metrics,
    wind_regression_metrics,
)

__all__ = [
    "apply_conformal_adjustment",
    "apply_solar_constraints",
    "best_candidate_per_family",
    "build_candidate_models",
    "build_quantile_lightgbm_models",
    "conformal_interval_adjustment",
    "evaluate_candidate_models",
    "fit_quantile_models",
    "group_feature_contributions",
    "grouped_permutation_importance",
    "hourly_error_profile",
    "lightgbm_feature_contributions",
    "pinball_loss",
    "predict_ordered_quantiles",
    "probabilistic_forecast_metrics",
    "solar_regression_metrics",
    "temporal_split_labels",
    "apply_wind_conformal_adjustment",
    "apply_wind_constraints",
    "build_wind_candidate_models",
    "build_wind_quantile_models",
    "evaluate_wind_models",
    "predict_wind_quantiles",
    "wind_conformal_adjustment",
    "wind_probabilistic_metrics",
    "wind_regression_metrics",
]
