"""
Time-aware sequence builder for LOB data using Polars.

Pipeline:
  1) (Optional) Bucketize timestamps by `bucket_ms` and aggregate rows by last-observation.
  2) Optionally enrich with temporal features: delta_t_s, time-of-day sin/cos.
  3) Respect day boundaries: build windows per (symbol, date) only.
  4) Slide windows with configurable `stride`.

Returns windows X [N, T, F], labels y [N], and the used feature list.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Iterable, List, Sequence

import numpy as np
import polars as pl


@dataclass
class SequenceBuildResult:
    X: np.ndarray  # [N, T, F]
    y: np.ndarray  # [N]
    used_features: List[str]
    timestamps: np.ndarray | None = None  # window end timestamps (datetime64[ns])


def _add_temporal_features(
    df: pl.DataFrame, ts_col: str, group_col: str | None
) -> pl.DataFrame:
    """Add delta_t_s (reset at day/symbol boundaries) and time-of-day sin/cos.

    Shared by the event-driven and time-bucketed sequencers: in event mode each
    step covers a variable amount of wall-clock time, so the model must be told
    the inter-event gap explicitly or it cannot distinguish a burst from a lull.
    """
    two_pi = 2.0 * math.pi

    delta_s = (
        (pl.col(ts_col).dt.epoch("ns") - pl.col(ts_col).shift(1).dt.epoch("ns"))
        .fill_null(0)
        / 1e9
    )

    group_keys = [c for c in (group_col, "_date") if c and c in df.columns]
    if group_keys:
        delta_s = delta_s.over(group_keys)

    tod_seconds = (
        pl.col(ts_col).dt.hour() * 3600
        + pl.col(ts_col).dt.minute() * 60
        + pl.col(ts_col).dt.second()
    )
    tod_angle = tod_seconds / 86400.0 * two_pi

    return df.with_columns(
        [
            delta_s.alias("delta_t_s"),
            tod_angle.sin().alias("tod_sin"),
            tod_angle.cos().alias("tod_cos"),
        ]
    )


def _ensure_datetime_column(df: pl.DataFrame, time_col: str) -> pl.DataFrame:
    dt_col = time_col if time_col in df.columns else "datetime"
    s = df.get_column(dt_col)
    if s.dtype == pl.Datetime:
        return df.with_columns(s.alias("_dt"))
    if s.dtype == pl.Utf8:
        return df.with_columns(pl.col(dt_col).str.to_datetime(strict=False).alias("_dt"))
    if dt_col != "datetime" and "datetime" in df.columns:
        return df.with_columns(pl.col("datetime").alias("_dt"))
    return df.with_columns(pl.col(dt_col).alias("_dt"))


def _select_numeric_features(
    df: pl.DataFrame, feature_cols: Sequence[str], extras: Iterable[str] = ()
) -> List[str]:
    available = set(df.columns)
    string_cols = set(df.select(pl.selectors.string()).columns)
    numeric = [c for c in feature_cols if c in available and c not in string_cols]
    for extra in extras:
        if extra in df.columns:
            numeric.append(extra)
    seen: set[str] = set()
    used: list[str] = []
    for c in numeric:
        if c not in seen:
            used.append(c)
            seen.add(c)
    return used


def _build_windows(
    gdf: pl.DataFrame,
    used_features: Sequence[str],
    label_col: str,
    seq_len: int,
    stride: int,
    ts_col: str | None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray | None]:
    gdf = gdf.drop_nulls(subset=[label_col])
    if gdf.height < seq_len:
        return (
            np.zeros((0, seq_len, len(used_features)), dtype=np.float32),
            np.zeros((0,), dtype=np.int64),
            None if ts_col is None else np.zeros((0,), dtype="datetime64[ns]"),
        )

    feats = gdf.select(used_features).to_numpy()
    feats = np.nan_to_num(feats, nan=0.0, posinf=0.0, neginf=0.0)
    labels = gdf.get_column(label_col).to_numpy()
    ts_values = gdf.get_column(ts_col).to_numpy() if ts_col else None

    N = feats.shape[0]
    idxs = range(0, N - seq_len + 1, stride)
    num_windows = len(idxs)
    X = np.zeros((num_windows, seq_len, feats.shape[1]), dtype=np.float32)
    labels_dtype = np.float32 if np.issubdtype(labels.dtype, np.floating) else np.int64
    y = np.zeros((num_windows,), dtype=labels_dtype)
    ts = None
    if ts_values is not None:
        ts = np.zeros((num_windows,), dtype=ts_values.dtype)

    k = 0
    for i in idxs:
        X[k] = feats[i : i + seq_len]
        y[k] = labels[i + seq_len - 1]
        if ts is not None:
            ts[k] = ts_values[i + seq_len - 1]
        k += 1
    return X, y, ts


class EventSequencer:
    """Classic event-driven window builder without temporal bucket aggregation."""

    def __init__(
        self,
        *,
        seq_len: int = 360,
        stride: int = 60,
        time_col: str = "readable_timestamp",
        group_col: str = "symbol",
        use_temporal_features: bool = True,
    ) -> None:
        if seq_len <= 0:
            raise ValueError("seq_len must be > 0")
        if stride <= 0:
            raise ValueError("stride must be > 0")
        self.seq_len = int(seq_len)
        self.stride = int(stride)
        self.time_col = time_col
        self.group_col = group_col
        self.use_temporal_features = bool(use_temporal_features)

    def build(
        self, df: pl.DataFrame, feature_cols: Sequence[str], label_col: str
    ) -> SequenceBuildResult:
        self._validate_inputs(df, feature_cols, label_col)
        df2 = _ensure_datetime_column(df, self.time_col)
        df_e = df2.with_columns(pl.col("_dt").dt.date().alias("_date"))
        df_e = df_e.sort("_dt")
        if self.use_temporal_features:
            df_e = _add_temporal_features(df_e, "_dt", self.group_col)

        temporal_extras = ("delta_t_s", "tod_sin", "tod_cos") if self.use_temporal_features else ()
        used_features = _select_numeric_features(df_e, feature_cols, extras=temporal_extras)

        X_list: list[np.ndarray] = []
        y_list: list[np.ndarray] = []
        ts_list: list[np.ndarray] = []

        group_keys = [c for c in (self.group_col, "_date") if c in df_e.columns]
        if not group_keys:
            group_keys = ["_date"]

        for _, gdf in df_e.group_by(group_keys, maintain_order=True):
            gdf = gdf.sort("_dt")
            Xg, yg, tg = _build_windows(
                gdf,
                used_features,
                label_col,
                self.seq_len,
                self.stride,
                ts_col="_dt",
            )
            if Xg.size:
                X_list.append(Xg)
                y_list.append(yg)
                if tg is not None:
                    ts_list.append(tg)

        if not X_list:
            print("[EventSequencer] No windows produced; check parameters and data.")
            return SequenceBuildResult(
                X=np.zeros((0, self.seq_len, len(used_features)), dtype=np.float32),
                y=np.zeros((0,), dtype=np.float32 if label_col == "target_reg" else np.int64),
                used_features=used_features,
                timestamps=np.zeros((0,), dtype="datetime64[ns]"),
            )

        X = np.concatenate(X_list, axis=0).astype(np.float32, copy=False)
        y_dtype = np.float32 if label_col == "target_reg" else np.int64
        y = np.concatenate(y_list, axis=0).astype(y_dtype, copy=False)
        timestamps = (
            np.concatenate(ts_list, axis=0).astype(ts_list[0].dtype, copy=False)
            if ts_list
            else None
        )

        try:
            first_ts = df_e["_dt"][0]
            last_ts = df_e["_dt"][-1]
            print(
                f"[EventSequencer] Windows: {X.shape[0]:,}  "
                f"SeqLen: {self.seq_len}  Feats: {X.shape[2]}  "
                f"First ts: {first_ts}  Last ts: {last_ts}"
            )
        except Exception:
            pass

        return SequenceBuildResult(X=X, y=y, used_features=used_features, timestamps=timestamps)

    def _validate_inputs(
        self, df: pl.DataFrame, feature_cols: Sequence[str], label_col: str
    ) -> None:
        if self.time_col not in df.columns and "datetime" not in df.columns:
            raise ValueError(
                f"Neither {self.time_col!r} nor 'datetime' found in DataFrame."
            )
        if label_col not in df.columns:
            raise ValueError(f"Label column {label_col!r} not found in DataFrame.")
        if not feature_cols:
            raise ValueError("feature_cols must contain at least one column name.")


class TimeAwareSequencer:
    def __init__(
        self,
        bucket_ms: int | None = 100,
        seq_len: int = 360,
        stride: int = 60,
        time_col: str = "readable_timestamp",
        group_col: str = "symbol",
        use_time_bucketing: bool = True,
        use_temporal_features: bool = True,
    ) -> None:
        if seq_len <= 0:
            raise ValueError("seq_len must be > 0")
        if stride <= 0:
            raise ValueError("stride must be > 0")
        self.use_time_bucketing = bool(use_time_bucketing)
        self.bucket_ms = int(bucket_ms) if bucket_ms is not None else None
        if self.use_time_bucketing:
            if self.bucket_ms is None or self.bucket_ms <= 0:
                raise ValueError("bucket_ms must be > 0 when use_time_bucketing is True")
        elif self.bucket_ms is not None and self.bucket_ms <= 0:
            raise ValueError("bucket_ms must be > 0 when provided.")
        self.seq_len = int(seq_len)
        self.stride = int(stride)
        self.time_col = time_col
        self.group_col = group_col
        self.use_temporal_features = bool(use_temporal_features)


    def build(
        self, df: pl.DataFrame, feature_cols: Sequence[str], label_col: str
    ) -> SequenceBuildResult:
        """Build time-aware windows.

        Args:
            df: Input Polars DataFrame containing features and label.
            feature_cols: Candidate feature columns to use.
            label_col: Name of the label column (target_class).

        Returns:
            SequenceBuildResult with X [N, T, F], y [N], and used feature names.
        """
        self._validate_inputs(df, feature_cols, label_col)

        df2 = _ensure_datetime_column(df, self.time_col)
        df_b = self._bucketize(df2)
        df_e = self._add_temporal_features(df_b) if self.use_temporal_features else df_b

        temporal_extras = ("delta_t_s", "tod_sin", "tod_cos") if self.use_temporal_features else ()
        used_features = _select_numeric_features(df_e, feature_cols, extras=temporal_extras)

        X_list: List[np.ndarray] = []
        y_list: List[np.ndarray] = []
        ts_list: List[np.ndarray] = []

        # Group by symbol (if exists) and date; enforce no cross-day windows
        group_keys = [c for c in (self.group_col, "_date") if c in df_e.columns]
        if not group_keys:
            group_keys = ["_date"]

        for _, gdf in df_e.group_by(group_keys, maintain_order=True):
            # Sort within each group to ensure time order
            gdf = gdf.sort("_bucket_dt")
            Xg, yg, tg = _build_windows(
                gdf,
                used_features,
                label_col,
                self.seq_len,
                self.stride,
                ts_col="_bucket_dt",
            )
            if Xg.size:
                X_list.append(Xg)
                y_list.append(yg)
                if tg is not None:
                    ts_list.append(tg)

        if not X_list:
            print("[TimeAwareSequencer] No windows produced; check parameters and data.")
            return SequenceBuildResult(
                X=np.zeros((0, self.seq_len, len(used_features)), dtype=np.float32),
                y=np.zeros((0,), dtype=np.float32 if label_col == "target_reg" else np.int64),
                used_features=used_features,
                timestamps=np.zeros((0,), dtype="datetime64[ns]"),
            )

        X = np.concatenate(X_list, axis=0).astype(np.float32, copy=False)
        y_dtype = np.float32 if label_col == "target_reg" else np.int64
        y = np.concatenate(y_list, axis=0).astype(y_dtype, copy=False)
        timestamps = (
            np.concatenate(ts_list, axis=0).astype(ts_list[0].dtype, copy=False)
            if ts_list
            else None
        )

        # Quick validations
        assert X.ndim == 3 and y.ndim == 1, "Output shapes must be [N,T,F] and [N]"

        # Print a small summary (first/last timestamps in first group)
        try:
            first_ts = df_e["_bucket_dt"][0]
            last_ts = df_e["_bucket_dt"][-1]
            print(
                f"[TimeAwareSequencer] Windows: {X.shape[0]:,}  "
                f"SeqLen: {self.seq_len}  Feats: {X.shape[2]}  "
                f"First ts: {first_ts}  Last ts: {last_ts}"
            )
        except Exception:
            pass

        return SequenceBuildResult(X=X, y=y, used_features=used_features, timestamps=timestamps)

    def _validate_inputs(
        self, df: pl.DataFrame, feature_cols: Sequence[str], label_col: str
    ) -> None:
        if self.time_col not in df.columns and "datetime" not in df.columns:
            raise ValueError(
                f"Neither {self.time_col!r} nor 'datetime' found in DataFrame."
            )
        if label_col not in df.columns:
            raise ValueError(f"Label column {label_col!r} not found in DataFrame.")
        if not feature_cols:
            raise ValueError("feature_cols must contain at least one column name.")

    def _bucketize(self, df: pl.DataFrame) -> pl.DataFrame:
        if self.use_time_bucketing:
            every = f"{self.bucket_ms}ms"
            bucket_expr = pl.col("_dt").dt.truncate(every).alias("_bucket_dt")
        else:
            bucket_expr = pl.col("_dt").alias("_bucket_dt")

        df_b = df.with_columns([bucket_expr])

        if self.use_time_bucketing:
            # Collapse rows within the same (group, bucket) by last observation
            keys = [c for c in (self.group_col,) if c in df_b.columns] + ["_bucket_dt"]
            agg_exprs = [pl.all().last()]
            df_b = df_b.group_by(keys, maintain_order=True).agg(agg_exprs).sort("_bucket_dt")
            df_b = df_b.unnest("all") if "all" in df_b.columns else df_b
        else:
            df_b = df_b.sort("_bucket_dt")

        # Add date for day boundary handling
        df_b = df_b.with_columns(pl.col("_bucket_dt").dt.date().alias("_date"))
        return df_b

    def _add_temporal_features(self, df: pl.DataFrame) -> pl.DataFrame:
        return _add_temporal_features(df, "_bucket_dt", self.group_col)

__all__ = ["TimeAwareSequencer", "EventSequencer", "SequenceBuildResult"]
