"""Utilities for loading and saving experiment configurations."""

from __future__ import annotations

import argparse
from copy import deepcopy
from pathlib import Path
from typing import Any, Mapping

import yaml

from .base_config import ExperimentConfig

DEFAULT_CONFIG_PATH = Path(__file__).with_name("default.yaml")


def _load_yaml_dict(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as fp:
        data = yaml.safe_load(fp) or {}
    if not isinstance(data, dict):
        raise ValueError(f"Configuration at {path} must be a mapping, got {type(data)}")
    return data


def _deep_update(base: dict[str, Any], updates: Mapping[str, Any]) -> dict[str, Any]:
    """Recursively merge mapping updates into base, returning a new dict."""
    merged: dict[str, Any] = deepcopy(base)
    for key, value in updates.items():
        if (
            isinstance(value, Mapping)
            and key in merged
            and isinstance(merged[key], Mapping)
        ):
            merged[key] = _deep_update(merged[key], value)
        else:
            merged[key] = deepcopy(value)
    return merged


def load_config(
    path: str | Path | None = None,
    overrides: Mapping[str, Any] | None = None,
) -> ExperimentConfig:
    """Load an experiment configuration, layering defaults, file, and overrides."""
    config_data = _load_yaml_dict(DEFAULT_CONFIG_PATH)

    if path is not None:
        config_path = Path(path)
        if not config_path.exists():
            raise FileNotFoundError(f"Config file not found: {config_path}")
        file_data = _load_yaml_dict(config_path)
        config_data = _deep_update(config_data, file_data)

    if overrides:
        config_data = _deep_update(config_data, overrides)

    try:
        return ExperimentConfig.model_validate(config_data)
    except Exception as exc:  # pragma: no cover - provide context rich error
        origin = path or DEFAULT_CONFIG_PATH
        raise ValueError(f"Invalid configuration at {origin}") from exc


def save_config(config: ExperimentConfig, path: str | Path) -> None:
    """Persist an experiment configuration to a YAML file."""
    target_path = Path(path)
    target_path.parent.mkdir(parents=True, exist_ok=True)

    payload = config.model_dump(mode="json", exclude_none=True)

    with target_path.open("w", encoding="utf-8") as fp:
        yaml.safe_dump(payload, fp, sort_keys=False)


def _cli() -> None:
    parser = argparse.ArgumentParser(description="Config loader smoke test.")
    parser.add_argument(
        "config_path",
        type=Path,
        help="Path to the YAML configuration file to load.",
    )
    args = parser.parse_args()

    config = load_config(args.config_path)
    config.print_config()


if __name__ == "__main__":
    _cli()
