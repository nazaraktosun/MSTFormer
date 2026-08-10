
from __future__ import annotations

from copy import deepcopy
from pathlib import Path
from typing import Mapping

if __package__ is None or __package__ == "":
    import sys

    sys.path.append(str(Path(__file__).resolve().parents[2]))

from bullsense.config.base_config import ExperimentConfig
from bullsense.config.config_loader import DEFAULT_CONFIG_PATH, load_config


class Config:
    """Backward-compatible facade around YAML-backed ExperimentConfig defaults."""

    DEFAULT_PATH: Path = DEFAULT_CONFIG_PATH
    _experiment: ExperimentConfig = load_config()

    @classmethod
    def experiment(cls) -> ExperimentConfig:
        """Return the underlying ExperimentConfig instance."""
        return cls._experiment

    @classmethod
    def reload(
        cls,
        path: str | Path | None = None,
        overrides: Mapping[str, object] | None = None,
    ) -> ExperimentConfig:
        """Reload configuration from disk (used for transitional compatibility)."""
        target_path = Path(path) if path is not None else cls.DEFAULT_PATH
        cls._experiment = load_config(target_path, overrides)
        cls._apply_experiment(cls._experiment)
        return cls._experiment

    @classmethod
    def _apply_experiment(cls, experiment: ExperimentConfig) -> None:
        """Sync class-level constants from the ExperimentConfig instance."""
        data = experiment.data
        label = experiment.label
        feature = experiment.feature
        model = experiment.model
        training = experiment.training
        paths = experiment.paths

        # ========== META ==========
        cls.NAME = experiment.name
        cls.SEED = experiment.seed
        cls.TASK = experiment.task

        # ========== DATA & SYMBOL ==========
        cls.SYMBOL = data.symbol
        cls.OBID = data.obid
        cls.TRAIN_RATIO = data.train_ratio
        cls.VAL_RATIO = data.val_ratio
        cls.SEQUENCE_LENGTH = data.sequence_length
        cls.SEQUENCE_STRIDE = data.sequence_stride
        cls.BUCKET_MS = data.bucket_ms
        cls.USE_TIME_BUCKETING = data.use_time_bucketing
        cls.FEATURE_DIM = data.feature_dim

        # ========== LABEL PARAMETERS ==========
        cls.LABEL_STRATEGY = label.label_strategy
        cls.LABEL_H_EVENTS = label.label_h_events
        cls.LABEL_K = label.label_k
        cls.LABEL_THETA = label.label_theta
        cls.LABEL_VOL_WINDOW = label.label_vol_window
        cls.LABEL_VOL_K = label.label_vol_k
        cls.LABEL_MAX_THETA = label.label_max_theta
        cls.LABEL_PRICE_COL = label.price_col
        cls.USE_TIME_BASED_LABELLING = label.use_time_based_labelling
        cls.TIME_HORIZON = label.time_horizon
        cls.TIME_THRESHOLD = label.time_threshold
        cls.ZRET_H_EVENTS = label.zret_h_events
        cls.ZRET_VOL_WINDOW = label.zret_vol_window
        cls.ZRET_K = label.zret_k
        cls.ZRET_CLIP = label.zret_clip
        cls.ZRET_TIMESTAMP_COL = label.zret_timestamp_col
        cls.TRIPLE_H_EVENTS = label.triple_horizon_ticks
        cls.TRIPLE_VOL_WINDOW = label.triple_volatility_window
        cls.TRIPLE_BARRIER_MULTIPLIER = label.triple_barrier_multiplier
        cls.TRIPLE_MIN_THRESHOLD = label.triple_min_threshold_pct
        cls.TRIPLE_MAX_THRESHOLD = label.triple_max_threshold_pct
        cls.FIVE_HORIZON_TICKS = label.five_horizon_ticks
        cls.FIVE_VOLATILITY_WINDOW = label.five_volatility_window
        cls.FIVE_NUM_STD = label.five_num_std
        cls.FIVE_STRONG_MULTIPLIER = label.five_strong_multiplier
        cls.REG_TARGET = label.reg_target
        cls.REG_LABEL_BASIS = label.reg_label_basis
        cls.REG_NORMALIZATION = label.reg_normalization
        cls.REG_VOL_WINDOW = label.reg_vol_window
        cls.REG_VOL_SOURCE = label.reg_vol_source
        cls.REG_HORIZON_TICKS = label.reg_horizon_ticks
        cls.REG_CLIP_VALUE = label.reg_clip_value

        # ========== FEATURE CONFIGURATION ==========
        cls.USE_MESSAGE_FEATURES = feature.use_message_features
        cls.ORDERBOOK_FEATURES = deepcopy(feature.orderbook_features)

        # ========== MODEL ==========
        cls.MODEL_TYPE = model.model_type
        cls.NUM_CLASSES = model.num_classes
        cls.D_MODEL = model.d_model
        cls.T_LAYERS = model.t_layers
        cls.N_HEADS = model.n_heads
        cls.DROPOUT = model.dropout
        cls.HIDDEN_DIM = model.hidden_dim
        cls.NUM_LAYERS = model.num_layers
        cls.USE_BILINEAR_NORM = model.use_bilinear_norm
        cls.BILINEAR_NORM_TYPE = model.bilinear_norm_type
        cls.TEMPORAL_READOUT = model.temporal_readout
        cls.INPUT_SHIFT_NORM = model.input_shift_norm
        cls.DISH_INIT = model.dish_init
        cls.DISH_ACTIVATE = model.dish_activate
        cls.USE_TARGET_SHIFT_FOR_PRICE = model.use_target_shift_for_price
        cls.REVIN_AFFINE = model.revin_affine
        cls.REVIN_DETACH_STATS = model.revin_detach_stats

        # ========== TRAINING CONTROL ==========
        cls.BATCH_SIZE = training.batch_size
        cls.LOB_NORMALIZER_STYLE = training.lob_normalizer_style
        cls.LEARNING_RATE = training.learning_rate
        cls.WEIGHT_DECAY = training.weight_decay
        cls.EPOCHS = training.epochs
        cls.EARLY_STOPPING_PATIENCE = training.early_stopping_patience
        cls.GRAD_CLIP = training.grad_clip
        cls.USE_FOCAL_LOSS = training.use_focal_loss
        cls.FOCAL_GAMMA = training.focal_gamma
        cls.FOCAL_ALPHA = deepcopy(training.focal_alpha)
        cls.USE_AUTO_FOCAL_ALPHA = training.use_auto_focal_alpha
        cls.USE_CLASS_WEIGHTS = training.use_class_weights
        cls.LABEL_SMOOTHING = training.label_smoothing
        cls.MAX_GRAD_NORM = training.max_grad_norm
        cls.REGRESSION_LOSS = training.regression_loss
        cls.HUBER_DELTA = training.huber_delta
        cls.WEIGHTED_HUBER_ALPHA = training.weighted_huber_alpha
        cls.WEIGHTED_HUBER_POWER = training.weighted_huber_power
        cls.WEIGHTED_HUBER_SCALE = training.weighted_huber_scale
        cls.DEVICE = training.device

        # ========== PATHS ==========
        cls.DATA_DIR = paths.data_dir
        cls.OUTPUT_DIR = paths.output_dir

    @classmethod
    def print_config(cls) -> None:
        """Config özetini yazdır."""
        print("\n" + "=" * 70)
        print(" CONFIGURATION (Sweep parametreleri override edecek)")
        print("=" * 70)

        print("\n Labelling:")
        strategy = getattr(cls, "LABEL_STRATEGY", "fixed")
        if strategy == "time" or cls.USE_TIME_BASED_LABELLING:
            print(
                f"    Time-based horizon (horizon={cls.TIME_HORIZON}, threshold={cls.TIME_THRESHOLD})"
            )
        elif strategy == "zret":
            print(
                "    Z-RET "
                f"(h_events={cls.ZRET_H_EVENTS}, vol_win={cls.ZRET_VOL_WINDOW}, "
                f"k={cls.ZRET_K}, clip={cls.ZRET_CLIP})"
            )
        elif strategy == "fixed_vol":
            print(
                "    Fixed volatility horizon "
                f"(theta_floor={cls.LABEL_THETA}, vol_win={cls.LABEL_VOL_WINDOW}, "
                f"vol_k={cls.LABEL_VOL_K}, max_theta={cls.LABEL_MAX_THETA})"
            )
        else:
            print(f"    Fixed horizon (theta={cls.LABEL_THETA})")

        print("\nTraining Defaults:")
        print(f"  BATCH_SIZE: {cls.BATCH_SIZE}")
        print(f"  LEARNING_RATE: {cls.LEARNING_RATE}")
        print(f"  DROPOUT: {cls.DROPOUT}")
        print(f"  WEIGHT_DECAY: {cls.WEIGHT_DECAY}")
        print(f"  EPOCHS: {cls.EPOCHS}")
        print(f"  PATIENCE: {cls.EARLY_STOPPING_PATIENCE}")

        print("\n Loss:")
        print(f"  USE_FOCAL_LOSS: {cls.USE_FOCAL_LOSS}")
        if cls.USE_FOCAL_LOSS:
            print(f"  FOCAL_GAMMA: {cls.FOCAL_GAMMA}")
            print(f"  USE_AUTO_FOCAL_ALPHA: {cls.USE_AUTO_FOCAL_ALPHA}")

        print("\n Data:")
        print(f"  NUM_CLASSES: {cls.NUM_CLASSES}")
        print(f"  SEQUENCE_LENGTH: {cls.SEQUENCE_LENGTH}")
        print("=" * 70)


# Populate class constants on import for backward compatibility.
Config._apply_experiment(Config._experiment)


if __name__ == "__main__":
    Config.print_config()
