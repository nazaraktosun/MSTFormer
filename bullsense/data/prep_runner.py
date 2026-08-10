"""Modular data preparation orchestrator for Bullsense."""

from __future__ import annotations

import re
from dataclasses import dataclass
import copy
from pathlib import Path
from typing import Callable, Literal, Sequence

import numpy as np
import polars as pl

from bullsense.config.base_config import ExperimentConfig
from bullsense.data.feature_layout import build_lob_feature_map, canonicalize_feature_columns
from bullsense.data.metadata import (
    DatasetMetadata,
    LabelMetadata,
    SplitClassCounts,
    SplitRatios,
    SplitShapes,
)
from bullsense.features.pipeline import (
    add_message_flow_features,
    add_message_trailing_features,
    apply_feature_pipeline,
    prepare_basic_columns,
)
from bullsense.io.ingest import IngestionResult, ingest_clickhouse
from bullsense.labeling.labels import (
    LabelSummary,
    apply_fixed_horizon_labels,
    apply_fixed_volatility_labels,
    apply_five_class_dynamic_labels,
    apply_regression_targets,
    apply_time_based_labels,
    apply_zret_labels,
    apply_triple_barrier_labels,
    apply_triple_barrier_cost_aware,
)
from bullsense.data.sequences import EventSequencer, TimeAwareSequencer  

Mode = Literal["lob", "feature", "fusion"]

_METADATA_COLS = {
    "target_class",
    "pcap_timestamp",
    "datetime",
    "readable_timestamp",
    "order_book_id",
    "symbol",
    "unique_message_id",
    "metadata_id",
    "seq_num",
    "msg_idx",
    "target_reg",
    "target_reg_raw",
    "target_reg_scale",
    # Raw per-event message fields (Databento LOBSTER ingest). These carry
    # the events that message FEATURES are derived from; they must never be
    # model inputs themselves (order_id is overfit bait, raw price/size are
    # non-stationary duplicates of derived features).
    "time",
    "order_id",
    "msg_size",
    "msg_price",
    "event_type",
    "direction",
}

#: label strategies whose horizon is counted in ROWS of the current grid
_ROW_HORIZON_STRATEGIES = {
    "fixed",
    "fixed_vol",
    "zret",
    "triple_barrier",
    "triple_barrier_cost",
    "triple_barrier_cost_aware",
    "five_class",
}


def _default_ingest(obid: int) -> IngestionResult:
    return ingest_clickhouse(obid)


def _select_numeric_feature_columns(df: pl.DataFrame) -> list[str]:
    numeric = [c for c in df.columns if c not in _METADATA_COLS and not c.startswith("_")]
    string_cols = df.select(pl.selectors.string()).columns
    return [c for c in numeric if c not in string_cols]


def _is_raw_lob_column(name: str) -> bool:
    return bool(re.match(r"^(p|q)(a|b)\d+$", name, flags=re.IGNORECASE))


def _pick_columns_by_mode(
    df: pl.DataFrame, mode: Mode, *, include_msg_features: bool = True
) -> list[str]:
    numeric_cols = _select_numeric_feature_columns(df)
    if not include_msg_features:
        # Keep the pure-LOB baseline honest: without the flag, message-derived
        # columns must not leak into the feature set just because they are
        # numeric and present on the frame.
        numeric_cols = [c for c in numeric_cols if not c.startswith("msgf_")]
    if mode == "lob":
        return [c for c in numeric_cols if _is_raw_lob_column(c)]
    if mode == "feature":
        return [c for c in numeric_cols if not _is_raw_lob_column(c)]
    if mode == "fusion":
        return numeric_cols
    raise ValueError(f"Unknown mode: {mode}")


def _estimate_avg_liquidity(df: pl.DataFrame, train_ratio: float = 0.7) -> float:
    """
    Estimate average liquidity from a TRAIN-period day's book depth.
    Uses (qb1 + qa1) / 2 grouped by date and picks the last day of the train
    slice. The split is chronological, so picking the penultimate day of the
    whole frame (old behavior) meant scaling features with a TEST-period
    constant -- a train->test leak. Falls back to 250_000 when unavailable.
    """
    fallback = 250_000.0
    if not {"qb1", "qa1"}.issubset(set(df.columns)):
        return fallback
    try:
        liq = ((pl.col("qb1") + pl.col("qa1")) / 2.0).alias("_liq")
        dated = df.with_columns(liq)
        dated = dated.with_columns(pl.col("datetime").dt.date().alias("_date"))
        agg = dated.group_by("_date").agg(pl.col("_liq").mean().alias("mean_liq")).sort("_date")
        dates = agg["_date"].to_list()
        if not dates:
            return fallback
        # last day inside the train fraction (chronological split)
        idx = min(max(int(len(dates) * train_ratio) - 1, 0), len(dates) - 1)
        val = float(agg["mean_liq"][idx])
        if np.isfinite(val) and val > 0:
            return val
    except Exception:
        pass
    return fallback


def _split_temporal(
    X: np.ndarray,
    y: np.ndarray,
    train_ratio: float,
    val_ratio: float,
    *,
    timestamps: np.ndarray | None = None,
) -> dict[str, tuple[np.ndarray, np.ndarray, np.ndarray | None]]:
    """Temporal split that respects calendar-day boundaries.

    When `timestamps` is provided, unique dates are split by the requested
    ratios and every window goes to its date's bucket — no calendar day is
    cut across train/val/test. This prevents the same-day leakage you get
    from raw row-index splits when window stride is dense.

    Falls back to row-index splitting if no timestamps are available
    (e.g. legacy paths).
    """
    total = X.shape[0]

    if timestamps is None:
        train_end = int(total * train_ratio)
        val_end = int(total * (train_ratio + val_ratio))
        return {
            "train": (X[:train_end], y[:train_end], None),
            "val": (X[train_end:val_end], y[train_end:val_end], None),
            "test": (X[val_end:], y[val_end:], None),
        }

    # Day key per window (truncated to date). datetime64[D] gives us a fast,
    # numpy-native day comparison without going through Python objects.
    dates_per_window = timestamps.astype("datetime64[D]")
    unique_dates = np.unique(dates_per_window)
    unique_dates.sort()
    n_dates = unique_dates.shape[0]

    if n_dates == 0:
        empty = X[:0]
        empty_y = y[:0]
        empty_ts = timestamps[:0]
        return {
            "train": (empty, empty_y, empty_ts),
            "val": (empty, empty_y, empty_ts),
            "test": (empty, empty_y, empty_ts),
        }

    train_cut = int(n_dates * train_ratio)
    val_cut = int(n_dates * (train_ratio + val_ratio))

    train_dates = unique_dates[:train_cut]
    val_dates = unique_dates[train_cut:val_cut]
    test_dates = unique_dates[val_cut:]

    train_mask = np.isin(dates_per_window, train_dates)
    val_mask = np.isin(dates_per_window, val_dates)
    test_mask = np.isin(dates_per_window, test_dates)

    return {
        "train": (X[train_mask], y[train_mask], timestamps[train_mask]),
        "val": (X[val_mask], y[val_mask], timestamps[val_mask]),
        "test": (X[test_mask], y[test_mask], timestamps[test_mask]),
    }


def _compute_split_counts(
    splits: dict[str, tuple[np.ndarray, np.ndarray, np.ndarray | None]],
    num_classes: int,
) -> dict[str, list[int]]:
    counts: dict[str, list[int]] = {}
    for name, (_, labels, _) in splits.items():
        hist = np.bincount(labels, minlength=num_classes)
        counts[name] = hist.astype(int).tolist()
    return counts


def _compute_split_shapes(
    splits: dict[str, tuple[np.ndarray, np.ndarray, np.ndarray | None]]
) -> dict[str, list[int]]:
    shapes: dict[str, list[int]] = {}
    for name, (features, _, _) in splits.items():
        shapes[name] = list(features.shape)
    return shapes


@dataclass(slots=True)
class DataPrepResult:
    """Summary of a preparation run."""

    mode: Mode
    feature_names: list[str]
    label_summary: LabelSummary
    split_counts: dict[str, list[int]]
    split_shapes: dict[str, list[int]]
    output_dir: Path
    metadata: DatasetMetadata

    @property
    def total_sequences(self) -> int:
        return sum(self.split_counts.get("train", [])) + sum(
            self.split_counts.get("val", [])
        ) + sum(self.split_counts.get("test", []))


class DataPrepRunner:
    """Coordinate ingestion, 
    feature engineering, labelling, and persistence."""

    def __init__(
        self,
        config: ExperimentConfig,
        *,
        feature_pipeline_cls,
        feature_registry_factory: Callable[[], Sequence[dict]],
        ingest_fn: Callable[[int], IngestionResult] = _default_ingest,
    ) -> None:
        self.config = config
        self._feature_pipeline_cls = feature_pipeline_cls
        self._feature_registry_factory = feature_registry_factory
        self._ingest_fn = ingest_fn

    def run(
        self,
        mode: Mode,
        output_dir: Path,
        *,
        price_col: str | None = None,
    ) -> DataPrepResult:
        self._validate_mode(mode)
        self._validate_label_grid()
        # obid is only meaningful for the ClickHouse ingest path; file/parquet
        # ingest ignores it. Default to 0 so it is no longer a hard requirement.
        obid = self.config.data.obid if self.config.data.obid is not None else 0

        # 1) Ingest
        ingestion = self._ingest_fn(obid)
        data_cfg = self.config.data
        lob_df = prepare_basic_columns(
            ingestion.orderbook,
            session_tz=data_cfg.session_tz,
            session_start=data_cfg.session_start,
            session_end=data_cfg.session_end,
            weekday_only=data_cfg.session_weekday_only,
        )

        # 1a) Message features: CAUSAL trailing aggregates over the per-event
        # msgf_* columns the ingest attached by row index. Raw event fields
        # stay metadata; only the derived features become model inputs.
        feature_cfg = self.config.feature
        if feature_cfg.use_message_features:
            lob_df = add_message_trailing_features(
                lob_df,
                time_windows=tuple(feature_cfg.msg_time_windows),
                event_windows=tuple(feature_cfg.msg_event_windows),
            )
        if getattr(feature_cfg, "use_message_flow_features", False):
            lob_df = add_message_flow_features(
                lob_df,
                windows=tuple(feature_cfg.msg_flow_windows),
            )

        # 1b) Auto average liquidity from a train-period day for volume features
        avg_liq = _estimate_avg_liquidity(lob_df, train_ratio=data_cfg.train_ratio)
        feature_cfgs = copy.deepcopy(self.config.feature.orderbook_features)
        for spec in feature_cfgs:
            if spec.get("name") == "volume_features_scaled":
                params = spec.setdefault("params", {})
                params["average_liquidity"] = avg_liq

        # 2) Feature engineering
        # fusion: ham LOB kalsın (eski usul), feature: ham LOB düşsün
        drop_raw = mode == "feature"
        feature_df = apply_feature_pipeline(
            lob_df,
            feature_cfgs,
            pipeline_cls=self._feature_pipeline_cls,
            registry_factory=self._feature_registry_factory,
            drop_raw_lob=drop_raw,
            log=None,  # sessiz
        )

        # 3) Labelling
        price_col = price_col or getattr(self.config.label, "price_col", "mid_price")
        task = str(getattr(self.config, "task", "classification")).lower()
        labelled_df, label_summary, label_meta = self._generate_labels(
            feature_df, price_col=price_col
        )
        label_col = "target_reg" if task == "regression" else "target_class"
        labelled_df = labelled_df.drop_nulls(subset=[label_col])

        # 4) Feature seçimi
        # Any msgf_* column (trailing block OR the 4 flow features) counts as a
        # message feature for selection; the pure-LOB baseline has both off.
        include_msg = feature_cfg.use_message_features or getattr(
            feature_cfg, "use_message_flow_features", False
        )
        feature_cols = _pick_columns_by_mode(
            labelled_df, mode, include_msg_features=include_msg
        )
        feature_cols = canonicalize_feature_columns(feature_cols)

        # 5) Sequencer (tek kaynak)
        seq_len = int(self.config.data.sequence_length)
        stride = int(self.config.data.sequence_stride)
        bucket_ms = getattr(self.config.data, "bucket_ms", 500)
        use_time_bucketing = bool(getattr(self.config.data, "use_time_bucketing", True))
        use_temporal_features = bool(getattr(self.config.data, "use_temporal_features", True))

        if use_time_bucketing:
            sequencer = TimeAwareSequencer(
                bucket_ms=bucket_ms,
                seq_len=seq_len,
                stride=stride,
                time_col="readable_timestamp",
                group_col="symbol",
                use_time_bucketing=True,
                use_temporal_features=use_temporal_features,
            )
        else:
            sequencer = EventSequencer(
                seq_len=seq_len,
                stride=stride,
                time_col="readable_timestamp",
                group_col="symbol",
                use_temporal_features=use_temporal_features,
            )
        seq_res = sequencer.build(
            df=labelled_df,
            feature_cols=feature_cols,
            label_col=label_col,
        )
        X_seq, y_seq = seq_res.X, seq_res.y
        ts_seq = seq_res.timestamps
        if ts_seq is None or ts_seq.shape[0] != X_seq.shape[0]:
            raise ValueError(
                "Sequencer did not provide per-window timestamps; required to map predictions back to time."
            )
        feature_cols = seq_res.used_features  # gerçekten kullanılan set

        # 6) Split & persist
        splits = _split_temporal(
            X_seq,
            y_seq,
            train_ratio=self.config.data.train_ratio,
            val_ratio=self.config.data.val_ratio,
            timestamps=ts_seq,
        )

        if task == "regression":
            counts = {name: [int(labels.shape[0])] for name, (_, labels, _) in splits.items()}
        else:
            counts = _compute_split_counts(splits, self.config.model.num_classes)
        shapes = _compute_split_shapes(splits)

        output_dir.mkdir(parents=True, exist_ok=True)
        self._save_numpy_split(
            splits["train"],
            output_dir / "X_train.npy",
            output_dir / "y_train.npy",
            output_dir / "ts_train.npy",
        )
        self._save_numpy_split(
            splits["val"],
            output_dir / "X_val.npy",
            output_dir / "y_val.npy",
            output_dir / "ts_val.npy",
        )
        self._save_numpy_split(
            splits["test"],
            output_dir / "X_test.npy",
            output_dir / "y_test.npy",
            output_dir / "ts_test.npy",
        )

        # 7) Metadata
        metadata = self._build_metadata(
            mode=mode,
            feature_names=feature_cols,
            seq_len=seq_len,
            feature_dim=X_seq.shape[-1],
            label_meta=label_meta,
            counts=counts,
            shapes=shapes,
            feature_index_map=build_lob_feature_map(feature_cols),
        )
        metadata.save_json(output_dir / "metadata.json")

        return DataPrepResult(
            mode=mode,
            feature_names=feature_cols,
            label_summary=label_summary,
            split_counts=counts,
            split_shapes=shapes,
            output_dir=output_dir,
            metadata=metadata,
        )

    def _generate_labels(
        self,
        frame: pl.DataFrame,
        *,
        price_col: str,
    ) -> tuple[pl.DataFrame, LabelSummary, LabelMetadata]:
        label_cfg = self.config.label

        # After prepare_basic_columns the time column is `readable_timestamp`
        # (the raw `timestamp` column is consumed by the parquet ingest). Resolve
        # to whatever time column actually exists so time/zret labelling works on
        # both BIST and US-parquet inputs instead of hard-failing on "timestamp".
        ts_col = "readable_timestamp" if "readable_timestamp" in frame.columns else "datetime"

        strategy = getattr(label_cfg, "label_strategy", None)
        if not strategy:
            strategy = "time" if getattr(label_cfg, "use_time_based_labelling", False) else "fixed"
        strategy = strategy.lower()

        task = str(getattr(self.config, "task", "classification")).lower()
        if task == "regression":
            labelled, summary = apply_regression_targets(
                frame,
                target=label_cfg.reg_target,
                label_basis=label_cfg.reg_label_basis,
                horizon_ticks=label_cfg.reg_horizon_ticks,
                normalization=label_cfg.reg_normalization,
                vol_window=label_cfg.reg_vol_window,
                vol_source=label_cfg.reg_vol_source,
                clip_value=label_cfg.reg_clip_value,
            )
            meta = LabelMetadata(
                strategy="regression",
                params={
                    "target": label_cfg.reg_target,
                    "label_basis": label_cfg.reg_label_basis,
                    "horizon_ticks": label_cfg.reg_horizon_ticks,
                    "normalization": label_cfg.reg_normalization,
                    "vol_window": label_cfg.reg_vol_window,
                    "vol_source": label_cfg.reg_vol_source,
                    "clip_value": label_cfg.reg_clip_value,
                },
            )
            return labelled, summary, meta

        if strategy == "time":
            labelled, summary = apply_time_based_labels(
                frame,
                price_col=price_col,
                time_horizon=label_cfg.time_horizon,
                threshold=label_cfg.time_threshold,
                timestamp_col=ts_col,
            )
            meta = LabelMetadata(
                strategy="time_horizon",
                params={
                    "price_col": price_col,
                    "time_horizon": label_cfg.time_horizon,
                    "threshold": label_cfg.time_threshold,
                    "timestamp_col": ts_col,
                },
            )
            return labelled, summary, meta

        if strategy == "zret":
            zret_ts_col = label_cfg.zret_timestamp_col
            if zret_ts_col not in frame.columns:
                zret_ts_col = ts_col
            labelled, summary = apply_zret_labels(
                frame,
                price_col=price_col,
                h_events=label_cfg.zret_h_events,
                vol_win=label_cfg.zret_vol_window,
                k_z=label_cfg.zret_k,
                clip_z=label_cfg.zret_clip,
                timestamp_col=zret_ts_col,
            )
            meta = LabelMetadata(
                strategy="zret",
                params={
                    "price_col": price_col,
                    "h_events": label_cfg.zret_h_events,
                    "vol_win": label_cfg.zret_vol_window,
                    "k_z": label_cfg.zret_k,
                    "clip_z": label_cfg.zret_clip,
                    "timestamp_col": zret_ts_col,
                },
            )
            return labelled, summary, meta

        if strategy == "triple_barrier":
            labelled, summary = apply_triple_barrier_labels(
                frame,
                price_col=price_col,
                horizon_ticks=label_cfg.triple_horizon_ticks,
                volatility_window=label_cfg.triple_volatility_window,
                barrier_multiplier=label_cfg.triple_barrier_multiplier,
                min_threshold_pct=label_cfg.triple_min_threshold_pct,
                max_threshold_pct=label_cfg.triple_max_threshold_pct,
            )
            meta = LabelMetadata(
                strategy="triple_barrier",
                params={
                    "price_col": price_col,
                    "horizon_ticks": label_cfg.triple_horizon_ticks,
                    "volatility_window": label_cfg.triple_volatility_window,
                    "barrier_multiplier": label_cfg.triple_barrier_multiplier,
                    "min_threshold_pct": label_cfg.triple_min_threshold_pct,
                    "max_threshold_pct": label_cfg.triple_max_threshold_pct,
                },
            )
            return labelled, summary, meta

        if strategy in {"triple_barrier_cost", "triple_barrier_cost_aware"}:
            labelled, summary = apply_triple_barrier_cost_aware(
                frame,
                price_col=price_col,
                horizon_ticks=label_cfg.triple_horizon_ticks,
                volatility_window=label_cfg.triple_volatility_window,
                barrier_multiplier=label_cfg.triple_barrier_multiplier,
                tick_size=label_cfg.triple_tick_size,
                spread_ticks=label_cfg.triple_spread_ticks,
                commission_rate=label_cfg.triple_commission_rate,
                min_profit_ticks=label_cfg.triple_min_profit_ticks,
                max_pct=label_cfg.triple_max_threshold_pct,
            )
            meta = LabelMetadata(
                strategy="triple_barrier_cost",
                params={
                    "price_col": price_col,
                    "horizon_ticks": label_cfg.triple_horizon_ticks,
                    "volatility_window": label_cfg.triple_volatility_window,
                    "barrier_multiplier": label_cfg.triple_barrier_multiplier,
                    "tick_size": label_cfg.triple_tick_size,
                    "spread_ticks": label_cfg.triple_spread_ticks,
                    "commission_rate": label_cfg.triple_commission_rate,
                    "min_profit_ticks": label_cfg.triple_min_profit_ticks,
                    "max_pct": label_cfg.triple_max_threshold_pct,
                },
            )
            return labelled, summary, meta

        if strategy == "five_class":
            labelled, summary = apply_five_class_dynamic_labels(
                frame,
                price_col=price_col,
                horizon_ticks=label_cfg.five_horizon_ticks,
                volatility_window=label_cfg.five_volatility_window,
                num_std=label_cfg.five_num_std,
                strong_multiplier=label_cfg.five_strong_multiplier,
            )
            meta = LabelMetadata(
                strategy="five_class_dynamic",
                params={
                    "price_col": price_col,
                    "horizon_ticks": label_cfg.five_horizon_ticks,
                    "volatility_window": label_cfg.five_volatility_window,
                    "num_std": label_cfg.five_num_std,
                    "strong_multiplier": label_cfg.five_strong_multiplier,
                },
            )
            return labelled, summary, meta

        if strategy == "fixed_vol":
            labelled, summary = apply_fixed_volatility_labels(
                frame,
                price_col=price_col,
                h_events=label_cfg.label_h_events,
                k=label_cfg.label_k,
                theta=label_cfg.label_theta,
                vol_window=label_cfg.label_vol_window,
                vol_k=label_cfg.label_vol_k,
                max_theta=label_cfg.label_max_theta,
                timestamp_col=ts_col,
            )
            meta = LabelMetadata(
                strategy="fixed_volatility",
                params={
                    "price_col": price_col,
                    "h_events": label_cfg.label_h_events,
                    "k": label_cfg.label_k,
                    "theta_floor": label_cfg.label_theta,
                    "vol_window": label_cfg.label_vol_window,
                    "vol_k": label_cfg.label_vol_k,
                    "max_theta": label_cfg.label_max_theta,
                    "timestamp_col": ts_col,
                },
            )
            return labelled, summary, meta

        if strategy == "fixed":
            labelled, summary = apply_fixed_horizon_labels(
                frame,
                price_col=price_col,
                h_events=label_cfg.label_h_events,
                k=label_cfg.label_k,
                theta=label_cfg.label_theta,
                timestamp_col=ts_col,
            )
            meta = LabelMetadata(
                strategy="fixed_theta",
                params={
                    "price_col": price_col,
                    "h_events": label_cfg.label_h_events,
                    "k": label_cfg.label_k,
                    "theta": label_cfg.label_theta,
                    "timestamp_col": ts_col,
                },
            )
            return labelled, summary, meta

        raise ValueError(f"Unknown label strategy '{strategy}'")

    def _build_metadata(
        self,
        *,
        mode: Mode,
        feature_names: Sequence[str],
        seq_len: int,
        feature_dim: int,
        label_meta: LabelMetadata,
        counts: dict[str, list[int]],
        shapes: dict[str, list[int]],
        feature_index_map: dict,
    ) -> DatasetMetadata:
        ratios = SplitRatios(
            train=float(self.config.data.train_ratio),
            val=float(self.config.data.val_ratio),
            test=float(1 - self.config.data.train_ratio - self.config.data.val_ratio),
        )
        return DatasetMetadata(
            symbol=self.config.data.symbol,
            obid=int(self.config.data.obid or 0),
            seq_len=int(seq_len),
            feature_dim=int(feature_dim),
            feature_names=list(feature_names),
            setting=mode,
            label=label_meta,
            split_ratios=ratios,
            shapes=SplitShapes(
                X_train=shapes["train"],
                X_val=shapes["val"],
                X_test=shapes["test"],
            ),
            class_counts=SplitClassCounts(
                train=counts["train"],
                val=counts["val"],
                test=counts["test"],
            ),
            feature_layout=str(feature_index_map.get("layout", "unknown")),
            feature_index_map=feature_index_map,
        )

    @staticmethod
    def _save_numpy_split(
        split: tuple[np.ndarray, np.ndarray, np.ndarray | None],
        X_path: Path,
        y_path: Path,
        ts_path: Path | None = None,
    ) -> None:
        X, y, timestamps = split
        np.save(X_path, X)
        np.save(y_path, y)
        if ts_path is not None and timestamps is not None:
            np.save(ts_path, timestamps)

    def _validate_label_grid(self) -> None:
        """Fail fast when a row-horizon label would run on a different grid
        than the model steps on.

        Row-horizon strategies (fixed/zret/triple_*/regression) count rows of
        the frame they are computed on. Labels are computed BEFORE the
        sequencer buckets, so with time bucketing enabled the two grids only
        agree when the ingest already resampled to the same grid
        (ingest_resample_ms == bucket_ms). Event-count horizons are exactly
        the use_time_bucketing=False path. Wall-clock ('time') labels are
        as-of joins and are grid-independent.
        """
        label_cfg = self.config.label
        strategy = getattr(label_cfg, "label_strategy", None)
        if not strategy:
            strategy = "time" if getattr(label_cfg, "use_time_based_labelling", False) else "fixed"
        strategy = strategy.lower()

        task = str(getattr(self.config, "task", "classification")).lower()
        row_horizon = strategy in _ROW_HORIZON_STRATEGIES or task == "regression"
        if not row_horizon:
            return

        data_cfg = self.config.data
        if not data_cfg.use_time_bucketing:
            return  # event grid: rows ARE events; the horizon is an event count

        resample = data_cfg.ingest_resample_ms
        bucket = data_cfg.bucket_ms
        if resample is None or int(resample) != int(bucket):
            raise ValueError(
                f"Label strategy '{strategy}' counts its horizon in rows, but labels are "
                f"computed on the ingest grid (ingest_resample_ms={resample}) while the "
                f"sequencer steps on a {bucket}ms grid. The horizon would mean different "
                "things at label time and at model time. Either set "
                "ingest_resample_ms == bucket_ms, or use use_time_bucketing=false "
                "(event grid), or switch to label_strategy='time' (wall-clock, "
                "grid-independent)."
            )

    @staticmethod
    def _validate_mode(mode: Mode) -> None:
        if mode not in {"lob", "feature", "fusion"}:
            raise ValueError(f"Unknown mode: {mode}")


__all__ = ["DataPrepRunner", "DataPrepResult"]
