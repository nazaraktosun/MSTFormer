"""Typed experiment configuration models built on Pydantic."""

from __future__ import annotations

import math
from pathlib import Path
from typing import Any, Dict, Literal, Optional, Union

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator


class DataConfig(BaseModel):
    """Dataset related configuration."""

    model_config = ConfigDict(validate_assignment=True, extra="forbid")

    symbol: str = Field(
        default="UNSPECIFIED", description="Instrument ticker or identifier."
    )
    obid: Optional[int] = Field(
        default=None, ge=0, description="Internal order book identifier."
    )
    train_ratio: float = Field(default=0.7, gt=0, lt=1)
    val_ratio: float = Field(default=0.15, gt=0, lt=1)
    sequence_length: int = Field(default=40, gt=0)
    sequence_stride: int = Field(default=30, gt=0)
    bucket_ms: Optional[int] = Field(
        default=500, description="Bucket size in ms when time bucketing is enabled."
    )
    use_time_bucketing: bool = Field(
        default=True,
        description="If False, sequences use raw event timestamps without bucketing.",
    )
    use_temporal_features: bool = Field(
        default=True,
        description=(
            "If True, sequencers append delta_t_s/tod_sin/tod_cos to model inputs. "
            "Set False for strict raw LOB snapshot baselines."
        ),
    )
    feature_dim: Optional[int] = Field(
        default=None,
        gt=0,
        description="Number of engineered features. If None it will be inferred.",
    )
    # --- Session / timezone (market-specific) ---
    session_tz: str = Field(
        default="Europe/Istanbul",
        description="Timezone for the trading-session filter (e.g. America/New_York for US).",
    )
    session_start: str = Field(
        default="09:55",
        description="Local session start (HH:MM). Rows before this are dropped.",
    )
    session_end: str = Field(
        default="18:00",
        description="Local session end (HH:MM), exclusive.",
    )
    session_weekday_only: bool = Field(
        default=True, description="Keep Mon-Fri only when True."
    )
    # --- Optional ingest-time resample (scale control for high-volume names) ---
    ingest_resample_ms: Optional[int] = Field(
        default=None,
        gt=0,
        description=(
            "If set, lazily resample the ingested LOB to this ms grid (last observation) "
            "at read time, before per-row processing. Use for high-volume symbols (e.g. SPY) "
            "so raw event rows never fully materialize. Should be <= bucket_ms."
        ),
    )
    ingest_batch_collapse: bool = Field(
        default=False,
        description=(
            "If True, collapse the ingested stream to one snapshot per unique "
            "timestamp (book=last, msgf_*=sum) instead of a fixed ms grid. This is "
            "the event/snapshot grid: an event-count horizon of h then means h book "
            "updates. Requires use_time_bucketing=False (rows ARE events) and is "
            "mutually exclusive with ingest_resample_ms."
        ),
    )
    lob_npy: Optional[Path] = Field(
        default=None,
        description="Optional local .npy LOB input path used by scripts/prepare_data.py.",
    )
    lob_parquet: Optional[Path] = Field(
        default=None,
        description="Optional local parquet LOB input path used by scripts/prepare_data.py.",
    )
    msg_parquet: Optional[Path] = Field(
        default=None,
        description="Optional message parquet paired with lob_parquet.",
    )
    parquet_symbol: Optional[str] = Field(
        default=None,
        description="Symbol stamped into parquet-ingested frames; defaults to data.symbol.",
    )
    price_scale: float = Field(
        default=1000.0,
        gt=0,
        description="Divisor for parquet LOB price columns. Use 1.0 if already scaled.",
    )

    @model_validator(mode="after")
    def _check_splits(self) -> "DataConfig":
        if self.train_ratio + self.val_ratio >= 1:
            raise ValueError(
                "train_ratio + val_ratio must be less than 1 to leave room for test set."
            )
        return self

    @model_validator(mode="after")
    def _validate_bucketing(self) -> "DataConfig":
        if self.use_time_bucketing:
            if self.bucket_ms is None:
                raise ValueError("bucket_ms must be set when use_time_bucketing is True.")
            if self.bucket_ms <= 0:
                raise ValueError("bucket_ms must be > 0 when use_time_bucketing is True.")
        elif self.bucket_ms is not None and self.bucket_ms <= 0:
            raise ValueError("bucket_ms must be > 0 when provided.")
        return self

    @model_validator(mode="after")
    def _validate_batch_collapse(self) -> "DataConfig":
        if not self.ingest_batch_collapse:
            return self
        if self.ingest_resample_ms is not None:
            raise ValueError(
                "ingest_batch_collapse and ingest_resample_ms are mutually exclusive: "
                "pick either the snapshot grid or a fixed ms grid."
            )
        if self.use_time_bucketing:
            raise ValueError(
                "ingest_batch_collapse requires use_time_bucketing=False: the collapsed "
                "rows already ARE the snapshot grid, and an event-count horizon counts "
                "those rows. Re-bucketing them by time would double-grid the data."
            )
        return self


class LabelConfig(BaseModel):
    """Label generation configuration."""

    model_config = ConfigDict(validate_assignment=True, extra="forbid")

    price_col: str = Field(
        default="mid_price",
        description="Column used for price-based labelling (time-horizon).",
    )
    label_strategy: Literal[
        "fixed",
        "fixed_vol",
        "time",
        "zret",
        "triple_barrier",
        "triple_barrier_cost",
        "triple_barrier_cost_aware",
        "five_class",
    ] = Field(
        default="fixed",
        description="Which labelling routine to apply.",
    )
    use_time_based_labelling: bool = Field(
        default=False,
        description="Deprecated: prefer label_strategy='time'.",
    )
    time_horizon: str = Field(default="3m")
    time_threshold: float = Field(default=0.0007, gt=0)
    label_h_events: int = Field(default=100, gt=0)
    label_k: int = Field(default=5, ge=0)
    label_theta: float = Field(default=0.00020, gt=0)
    label_vol_window: int = Field(default=300, gt=1)
    label_vol_k: float = Field(default=1.0, gt=0)
    label_max_theta: Optional[float] = Field(default=None, gt=0)
    zret_h_events: int = Field(default=30, gt=0)
    zret_vol_window: int = Field(default=300, gt=0)
    zret_k: float = Field(default=1.0, gt=0)
    zret_clip: Optional[float] = Field(default=10.0)
    zret_timestamp_col: str = Field(default="timestamp")
    triple_horizon_ticks: int = Field(default=50, gt=0)
    triple_volatility_window: int = Field(default=2000, gt=0)
    triple_barrier_multiplier: float = Field(default=1.0, gt=0)
    triple_min_threshold_pct: float = Field(default=0.0001, gt=0)
    triple_max_threshold_pct: float = Field(default=0.005, gt=0)
    triple_tick_size: float = Field(default=0.01, gt=0)
    triple_spread_ticks: Optional[float] = Field(default=None, gt=0)
    triple_commission_rate: float = Field(default=0.0003, ge=0)
    triple_min_profit_ticks: float = Field(default=0.3, ge=0)
    five_horizon_ticks: int = Field(default=200, gt=0)
    five_volatility_window: int = Field(default=2000, gt=0)
    five_num_std: float = Field(default=1.0, gt=0)
    five_strong_multiplier: float = Field(
        default=2.0,
        gt=1.0,
        description="Strong Buy/Sell threshold multiplier relative to the base threshold.",
    )
    reg_target: Literal["mid", "micro", "avg"] = Field(default="mid")
    reg_label_basis: Literal["return_bps", "price"] = Field(default="return_bps")
    reg_normalization: Literal["none", "vol"] = Field(default="none")
    reg_vol_window: int = Field(default=100, gt=1)
    reg_vol_source: Literal["mid", "micro", "avg"] = Field(default="mid")
    reg_horizon_ticks: int = Field(default=10, gt=0)
    reg_clip_value: Optional[float] = Field(default=None, gt=0)


class FeatureConfig(BaseModel):
    """Feature engineering related toggles."""

    model_config = ConfigDict(validate_assignment=True, extra="forbid")

    use_message_features: bool = Field(
        default=False,
        description=(
            "Derive causal trailing message features (msgf_*) from the per-event "
            "columns attached by the Databento LOBSTER ingest. When False, msgf_* "
            "columns are also excluded from the model features so LOB baselines "
            "stay message-free."
        ),
    )
    msg_time_windows: list[str] = Field(
        default_factory=lambda: ["1s", "10s"],
        description="Trailing wall-clock windows for message aggregates (polars durations).",
    )
    msg_event_windows: list[int] = Field(
        default_factory=lambda: [100],
        description="Trailing event-count windows for message aggregates (rows, per day).",
    )
    use_message_flow_features: bool = Field(
        default=False,
        description=(
            "Add the 4 normalized message-flow features (msgf_ofi_norm_e*, "
            "msgf_exec_flow_norm_e*, msgf_touch_add_imb_e*, msgf_touch_cancel_imb_e*) "
            "over msg_flow_windows. Independent of use_message_features so the 4 can be "
            "ablated on their own against a pure-LOB baseline. Requires the ingest to "
            "have attached msgf_* + touch columns (load_messages=True)."
        ),
    )
    msg_flow_windows: list[int] = Field(
        default_factory=lambda: [10, 50, 100],
        description=(
            "Trailing event/snapshot-count windows (rows, per day) for the normalized "
            "message-flow features. On the batch/snapshot grid these are book-update counts."
        ),
    )
    orderbook_features: list[dict[str, Any]] = Field(
        default_factory=list, description="Ordered feature pipeline definitions."
    )


class ModelConfig(BaseModel):
    """Model architecture configuration."""

    model_config = ConfigDict(validate_assignment=True, extra="forbid")

    model_type: Literal[
        "tlob",
        "tlob_spatiotemporal",
        "bullsense_tlob",
        "mlplob",
        "tlob_original",
        "original_tlob",
        "mlplob_original",
        "original_mlplob",
        "deeplob",
        "binctabl",
        "binctabl_original",
    ] = Field(default="tlob")
    num_classes: int = Field(default=3, gt=0)
    use_bilinear_norm: bool = Field(
        default=True, description="Enable per-sample bilinear normalization inside the model."
    )
    bilinear_norm_type: Literal["bin"] = Field(
        default="bin",
        description=(
            "Input normalizer used when use_bilinear_norm is enabled. Bullsense TLOB "
            "uses original BiN so comparisons with tlob_original share normalization."
        ),
    )
    temporal_readout: Literal[
        "attention",
        "mean",
        "last",
    ] = Field(
        default="last",
        description="How Bullsense TLOB summarizes the temporal dimension before classification. "
        "'last' reads the final bar (coherent with predicting the post-window return).",
    )
    attn_dropout: float = Field(
        default=0.0,
        ge=0,
        lt=1,
        description="Dropout inside attention (decoupled from FFN dropout). Keep low; the model underfits.",
    )
    temporal_causal: bool = Field(
        default=False,
        description="Causal-mask temporal attention. False = bidirectional over the observed "
        "window (no leakage; the label is a future return after the window).",
    )
    gradient_checkpointing: bool = Field(
        default=False,
        description="Checkpoint Bullsense TLOB blocks during training to reduce activation memory. "
        "Enable only for large sizes that OOM.",
    )

    # Transformer LOB (TLOB) specific parameters
    d_model: int = Field(default=64, gt=0)
    t_layers: int = Field(default=2, gt=0)
    n_heads: int = Field(default=4, gt=0)
    dropout: float = Field(default=0.25, ge=0, lt=1)

    # MLP-LOB specific parameters
    hidden_dim: int = Field(default=256, gt=0)
    num_layers: int = Field(default=3, gt=0)

    # Original TLOB-family compatibility parameters
    is_sin_emb: bool = Field(default=True)
    order_type_idx: Optional[int] = Field(default=41, ge=0)

    # Optional input shift normalization. Keep disabled for classification baselines.
    # For dain/grouped_dain (and any wrapper norm) set use_bilinear_norm=false so
    # exactly one normalizer is active per run.
    input_shift_norm: Literal["none", "dish", "revin", "dain", "grouped_dain"] = Field(
        default="none"
    )
    dish_init: Literal["standard", "avg", "uniform"] = Field(default="standard")
    dish_activate: bool = Field(default=True)
    use_target_shift_for_price: bool = Field(default=True)
    revin_affine: bool = Field(default=True)
    revin_detach_stats: bool = Field(default=True)
    dain_mode: Literal["avg", "adaptive_avg", "adaptive_scale", "full"] = Field(
        default="adaptive_scale",
        description="DAIN sublayer ladder (arXiv:1902.07892): adaptive_avg=DAIN(1), "
        "adaptive_scale=DAIN(1+2), full=DAIN(1+2+3, the paper's headline method).",
    )
    dain_mean_lr: float = Field(default=1e-5, gt=0)
    dain_gate_lr: float = Field(default=1e-3, gt=0)
    dain_scale_lr: float = Field(default=1e-5, gt=0)
    grouped_dain_groups: Optional[Dict[str, Any]] = Field(
        default=None,
        description="grouped_dain only: {name: {indices: [...], mode: ...}} feature-column "
        "groups (e.g. price vs size blocks). Groups must not overlap; ungrouped columns "
        "pass through unchanged.",
    )


class TrainingConfig(BaseModel):
    """Training procedure hyperparameters."""

    model_config = ConfigDict(validate_assignment=True, extra="forbid")

    use_lob_normalizer: bool = Field(
        default=True,
        description="Apply LOBNormalizer on input tensors before the model.",
    )
    lob_normalizer_style: Literal["bullsense", "tlob", "features_only", "full_zscore"] = Field(default="bullsense")
    batch_size: int = Field(default=128, gt=0)
    learning_rate: float = Field(default=2e-5, gt=0)
    weight_decay: float = Field(default=0.01, ge=0)
    epochs: int = Field(default=100, gt=0)

    early_stopping_patience: int = Field(default=15, ge=0)
    grad_clip: Optional[float] = Field(default=1.0, ge=0)

    use_focal_loss: bool = Field(default=False)
    focal_gamma: float = Field(default=2.5, ge=0)
    focal_alpha: Optional[Union[float, list[float]]] = Field(default_factory=lambda: [0.2, 0.6, 0.2])
    use_auto_focal_alpha: bool = Field(default=False)
    use_class_weights: bool = Field(default=True)
    label_smoothing: float = Field(default=0.05, ge=0, lt=1)
    max_grad_norm: float = Field(default=1.0, ge=0)
    regression_loss: Literal["huber", "mse", "mae", "weighted_huber"] = Field(
        default="huber"
    )
    huber_delta: float = Field(default=1.0, gt=0)
    weighted_huber_alpha: float = Field(default=1.0, ge=0)
    weighted_huber_power: float = Field(default=1.0, gt=0)
    weighted_huber_scale: float = Field(default=1.0, gt=0)

    device: Literal["cpu", "cuda"] = Field(default="cuda")

    @field_validator("focal_alpha")
    @classmethod
    def _ensure_alpha_in_range(
        cls, value: Optional[Union[float, list[float]]]
    ) -> Optional[Union[float, list[float]]]:
        if value is None:
            return value
        values = value if isinstance(value, list) else [value]
        for alpha in values:
            alpha_value = float(alpha)
            if not math.isfinite(alpha_value) or alpha_value < 0:
                raise ValueError("focal_alpha values must be finite and non-negative.")
        return value


class PathsConfig(BaseModel):
    """Filesystem locations for data and outputs."""

    model_config = ConfigDict(validate_assignment=True, extra="forbid")

    data_dir: Path = Field(default=Path("data/processed_lob"))
    output_dir: Path = Field(default=Path("runs"))


class TrackingConfig(BaseModel):
    """Experiment tracking configuration."""

    model_config = ConfigDict(validate_assignment=True, extra="forbid")

    enable_mlflow: bool = Field(
        default=False, description="Enable MLflow tracking for the run."
    )
    tracking_uri: Optional[str] = Field(
        default=None,
        description="Tracking server URI; defaults to local file backend when omitted.",
    )
    registry_uri: Optional[str] = Field(
        default=None,
        description="Optional registry URI for model registration.",
    )
    experiment_name: Optional[str] = Field(
        default="bullsense",
        description="MLflow experiment name. Created if it does not exist.",
    )
    run_name: Optional[str] = Field(
        default=None,
        description="Preferred MLflow run name. Overrides generated names when set.",
    )
    tags: Dict[str, str] = Field(
        default_factory=dict,
        description="Static tags applied to the MLflow run.",
    )


class ExperimentConfig(BaseModel):
    """Top-level experiment configuration."""

    model_config = ConfigDict(validate_assignment=True, extra="forbid")

    name: str = Field(default="tlob_experiment")
    seed: Optional[int] = Field(default=42)
    task: Literal["classification", "regression"] = Field(default="classification")

    data: DataConfig = Field(default_factory=DataConfig)
    label: LabelConfig = Field(default_factory=LabelConfig)
    feature: FeatureConfig = Field(default_factory=FeatureConfig)
    model: ModelConfig = Field(default_factory=ModelConfig)
    training: TrainingConfig = Field(default_factory=TrainingConfig)
    paths: PathsConfig = Field(default_factory=PathsConfig)
    tracking: TrackingConfig = Field(default_factory=TrackingConfig)

    def summary(self) -> str:
        """Return a human readable configuration summary."""
        lines = [
            f"Experiment: {self.name}",
            f"Seed: {self.seed}",
            f"Data: symbol={self.data.symbol}, seq_len={self.data.sequence_length}",
            f"Model: {self.model.model_type}, d_model={self.model.d_model}",
            f"Training: batch={self.training.batch_size}, lr={self.training.learning_rate}",
            f"Output dir: {self.paths.output_dir}",
        ]
        return "\n".join(lines)

    def to_dict(self) -> Dict[str, Any]:
        """Return a dictionary representation suitable for logging frameworks."""
        return self.model_dump(mode="json", by_alias=True)

    def print_config(self) -> None:
        """Pretty-print the configuration summary."""
        divider = "=" * 60
        print(divider)
        print(self.summary())
        print(divider)
