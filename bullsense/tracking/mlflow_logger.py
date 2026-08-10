from __future__ import annotations

import json
from contextlib import AbstractContextManager
from pathlib import Path
from typing import Any, Iterable, Mapping, MutableMapping, Optional


def _flatten_dict(
    data: Mapping[str, Any],
    parent_key: str = "",
    separator: str = ".",
) -> MutableMapping[str, Any]:
    """Recursively flatten nested dictionaries using ``separator`` separated keys."""
    items: dict[str, Any] = {}
    for key, value in data.items():
        new_key = f"{parent_key}{separator}{key}" if parent_key else str(key)
        if isinstance(value, Mapping):
            items.update(_flatten_dict(value, new_key, separator))
        else:
            items[new_key] = value
    return items


def _stringify_param_value(value: Any) -> Any:
    """
    Convert complex parameter values into MLflow friendly representations.

    MLflow allows strings, numbers, and booleans for parameters. Lists or other
    iterables are converted to JSON strings for readability.
    """
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    if isinstance(value, (set, frozenset)):
        return json.dumps(sorted(value), default=str)
    if isinstance(value, Iterable) and not isinstance(value, (bytes, bytearray)):
        return json.dumps(list(value), default=str)
    return json.dumps(value, default=str)


class MLflowLogger(AbstractContextManager["MLflowLogger"]):
    """
    Lightweight wrapper around MLflow tracking APIs.

    Ensures all calls are no-ops when MLflow is disabled or unavailable.
    """

    def __init__(
        self,
        *,
        enabled: bool = True,
        tracking_uri: Optional[str] = None,
        registry_uri: Optional[str] = None,
        experiment_name: Optional[str] = None,
        run_name: Optional[str] = None,
        tags: Optional[Mapping[str, Any]] = None,
    ) -> None:
        self.enabled = bool(enabled)
        self.tracking_uri = tracking_uri
        self.registry_uri = registry_uri
        self.experiment_name = experiment_name
        self.run_name = run_name
        self.tags = dict(tags or {})

        self._mlflow = None
        self._active_run = None
        self._active = False

    # --------------------------------------------------------------------- #
    # Factory helpers
    # --------------------------------------------------------------------- #
    @classmethod
    def from_config(
        cls,
        cfg: Any | None,
        *,
        run_name: Optional[str] = None,
        tags: Optional[Mapping[str, Any]] = None,
    ) -> "MLflowLogger":
        if cfg is None:
            return cls(enabled=False)

        enable_mlflow = getattr(cfg, "enable_mlflow", False)
        tracking_uri = getattr(cfg, "tracking_uri", None)
        registry_uri = getattr(cfg, "registry_uri", None)
        experiment_name = getattr(cfg, "experiment_name", None)
        cfg_run_name = getattr(cfg, "run_name", None)
        cfg_tags = getattr(cfg, "tags", {}) or {}

        combined_tags: dict[str, Any] = {}
        combined_tags.update(cfg_tags)
        if tags:
            combined_tags.update(tags)

        return cls(
            enabled=enable_mlflow,
            tracking_uri=tracking_uri,
            registry_uri=registry_uri,
            experiment_name=experiment_name,
            run_name=run_name or cfg_run_name,
            tags=combined_tags,
        )

    # --------------------------------------------------------------------- #
    # Context manager
    # --------------------------------------------------------------------- #
    def __enter__(self) -> "MLflowLogger":
        if not self.enabled:
            return self

        try:
            import mlflow
        except ImportError:
            # MLflow not installed. Disable silently.
            self.enabled = False
            return self

        self._mlflow = mlflow
        if self.tracking_uri:
            mlflow.set_tracking_uri(self.tracking_uri)
        if self.registry_uri:
            mlflow.set_registry_uri(self.registry_uri)
        if self.experiment_name:
            mlflow.set_experiment(self.experiment_name)

        self._active_run = mlflow.start_run(run_name=self.run_name, tags=self._stringify_tags(self.tags))
        self._active = True
        return self

    def __exit__(self, exc_type, exc, tb) -> None:  # type: ignore[override]
        if not self.active:
            return None

        status = "FINISHED" if exc is None else "FAILED"
        assert self._mlflow is not None
        self._mlflow.end_run(status=status)
        self._active = False
        self._active_run = None
        return None

    # --------------------------------------------------------------------- #
    # Properties & utilities
    # --------------------------------------------------------------------- #
    @property
    def active(self) -> bool:
        return self.enabled and self._active

    @staticmethod
    def _stringify_tags(tags: Mapping[str, Any]) -> dict[str, str]:
        return {str(k): str(v) for k, v in tags.items()}

    # --------------------------------------------------------------------- #
    # Logging hooks
    # --------------------------------------------------------------------- #
    def log_params(self, params: Mapping[str, Any]) -> None:
        if not self.active or not params:
            return
        assert self._mlflow is not None
        flat = _flatten_dict(params)
        sanitized = {key: _stringify_param_value(value) for key, value in flat.items()}
        self._mlflow.log_params(sanitized)

    def log_metrics(self, metrics: Mapping[str, Any], step: Optional[int] = None) -> None:
        if not self.active or not metrics:
            return
        assert self._mlflow is not None

        numeric_metrics: dict[str, float] = {}
        for key, value in metrics.items():
            try:
                numeric_metrics[key] = float(value)
            except (TypeError, ValueError):
                continue

        if numeric_metrics:
            self._mlflow.log_metrics(numeric_metrics, step=step)

    def log_metric(self, key: str, value: Any, step: Optional[int] = None) -> None:
        if not self.active:
            return
        assert self._mlflow is not None

        try:
            numeric_value = float(value)
        except (TypeError, ValueError):
            return
        self._mlflow.log_metric(key, numeric_value, step=step)

    def log_dict(self, data: Mapping[str, Any], artifact_file: str) -> None:
        if not self.active:
            return
        assert self._mlflow is not None
        self._mlflow.log_dict(data, artifact_file)

    def log_artifact(self, path: str | Path, artifact_path: Optional[str] = None) -> None:
        if not self.active:
            return
        assert self._mlflow is not None

        path_obj = Path(path)
        if path_obj.is_dir():
            self._mlflow.log_artifacts(str(path_obj), artifact_path=artifact_path)
        elif path_obj.exists():
            self._mlflow.log_artifact(str(path_obj), artifact_path=artifact_path)

    def set_tags(self, tags: Mapping[str, Any]) -> None:
        if not self.active or not tags:
            return
        assert self._mlflow is not None

        merged = self.tags.copy()
        merged.update(tags)
        str_tags = self._stringify_tags(tags)
        self._mlflow.set_tags(str_tags)
        self.tags = merged
