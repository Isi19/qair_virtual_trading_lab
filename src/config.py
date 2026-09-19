"""Asset configuration loading and validation."""

from __future__ import annotations

from pathlib import Path
from zoneinfo import ZoneInfo, ZoneInfoNotFoundError

import yaml


def load_assets(config_path: Path) -> dict[str, dict]:
    """Load and validate virtual-asset definitions from YAML."""
    with config_path.open(encoding="utf-8") as stream:
        document = yaml.safe_load(stream) or {}
        assets = document.get("assets", {})
    if not assets:
        raise ValueError("No assets found in configuration.")
    for asset_id, config in assets.items():
        missing = [
            field
            for field in (
                "technology",
                "market_timezone",
                "latitude",
                "longitude",
                "capacity_mw",
            )
            if config.get(field) is None
        ]
        if missing:
            raise ValueError(f"Asset '{asset_id}' is missing required fields: {missing}")
        if config["technology"] not in {"solar", "wind"}:
            raise ValueError(f"Asset '{asset_id}' has unsupported technology.")
        if not -90 <= float(config["latitude"]) <= 90:
            raise ValueError(f"Asset '{asset_id}' has an invalid latitude.")
        if not -180 <= float(config["longitude"]) <= 180:
            raise ValueError(f"Asset '{asset_id}' has an invalid longitude.")
        if float(config["capacity_mw"]) <= 0:
            raise ValueError(f"Asset '{asset_id}' must have a positive capacity_mw.")
        try:
            ZoneInfo(str(config["market_timezone"]))
        except ZoneInfoNotFoundError as exc:
            raise ValueError(
                f"Asset '{asset_id}' has an invalid market_timezone."
            ) from exc
        if config["technology"] == "solar":
            solar_missing = [
                field for field in ("tilt", "azimuth") if config.get(field) is None
            ]
            if solar_missing:
                raise ValueError(
                    f"Asset '{asset_id}' is missing solar fields: {solar_missing}"
                )
        else:
            wind_missing = [
                field
                for field in (
                    "number_of_turbines",
                    "turbine_rated_power_mw",
                    "hub_height_m",
                    "windpowerlib_turbine_type",
                )
                if config.get(field) is None
            ]
            if wind_missing:
                raise ValueError(
                    f"Asset '{asset_id}' is missing wind fields: {wind_missing}"
                )

        reanalysis = config.get("historical_reanalysis", {})
        if not config.get("historical_start_date") or not reanalysis.get("model"):
            raise ValueError(f"Asset '{asset_id}' is missing historical reanalysis settings.")
        for variable_field in ("target_variables", "feature_variables"):
            values = reanalysis.get(variable_field)
            if not isinstance(values, list) or not all(
                isinstance(value, str) and value for value in values
            ):
                raise ValueError(
                    f"Asset '{asset_id}' must define {variable_field} as a list of names."
                )
            if len(values) != len(set(values)):
                raise ValueError(
                    f"Asset '{asset_id}' has duplicate names in {variable_field}."
                )
    return assets
