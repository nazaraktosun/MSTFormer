"""Label generation utilities used across Bullsense data prep."""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Any, Dict, Optional

import numpy as np
import pandas as pd
import polars as pl


@dataclass(slots=True)
class LabelSummary:
    """Metadata describing class distribution for generated labels."""

    counts: Dict[int, int]
    ratios: Dict[int, float]
    total: int

    @property
    def imbalance_ratio(self) -> float:
        """Return max/min class ratio (ignoring empty classes)."""
        non_zero = [c for c in self.counts.values() if c > 0]
        if not non_zero:
            return 0.0
        return max(non_zero) / min(non_zero)

    def to_dict(self) -> Dict[str, Any]:
        """Serialize summary for persistence."""
        return {
            "counts": self.counts,
            "ratios": self.ratios,
            "total": self.total,
            "imbalance_ratio": self.imbalance_ratio,
        }


def _resolve_ts_col(frame: pl.DataFrame, timestamp_col: str) -> str:
    """Resolve the timestamp column, falling back to the standard aliases."""
    for cand in (timestamp_col, "datetime", "readable_timestamp"):
        if cand in frame.columns:
            return cand
    raise ValueError(
        f"No timestamp column found (tried {timestamp_col!r}, 'datetime', "
        "'readable_timestamp'); needed for per-day label computation."
    )


def _map_per_day(
    frame: pl.DataFrame, ts_col: str, fn
) -> pl.DataFrame:
    """Apply ``fn`` to each calendar day independently and re-concatenate.

    Forward-looking label computations must never cross a day boundary --
    otherwise the last `horizon` rows of every day get labelled with the
    overnight gap return.
    """
    frame = frame.sort(ts_col)
    parts: list[pl.DataFrame] = []
    for _, day_df in frame.group_by(
        pl.col(ts_col).dt.date().alias("_label_day"), maintain_order=True
    ):
        out = fn(day_df.drop("_label_day", strict=False))
        if out.height:
            parts.append(out)
    if not parts:
        raise ValueError("Labelling produced no rows for any day.")
    return pl.concat(parts, how="vertical_relaxed")


def _compute_summary(labels: pl.Series) -> LabelSummary:
    counts: Dict[int, int] = {}
    ratios: Dict[int, float] = {}

    labels = labels.drop_nulls()
    counts_df = labels.value_counts().sort(labels.name)
    total = int(labels.len())

    for cls, count in counts_df.iter_rows():
        cls_id = int(cls)
        counts[cls_id] = int(count)
        ratios[cls_id] = (int(count) / total) if total else 0.0

    max_cls = max(counts.keys(), default=-1)
    for cls in range(max_cls + 1):
        counts.setdefault(cls, 0)
        ratios.setdefault(cls, 0.0)

    return LabelSummary(counts=counts, ratios=ratios, total=total)


def _empty_regression_summary(total: int) -> LabelSummary:
    return LabelSummary(counts={0: int(total)}, ratios={0: 1.0 if total else 0.0}, total=int(total))


def _price_streams_from_frame(frame: pl.DataFrame) -> dict[str, np.ndarray]:
    cols = set(frame.columns)
    required = {"pa1", "qa1", "pb1", "qb1"}
    missing = sorted(required - cols)
    if missing:
        raise ValueError(
            "Regression targets require top-of-book columns "
            f"{sorted(required)}; missing {missing}."
        )

    ask_price = frame.get_column("pa1").to_numpy().astype(np.float64, copy=False)
    ask_size = frame.get_column("qa1").to_numpy().astype(np.float64, copy=False)
    bid_price = frame.get_column("pb1").to_numpy().astype(np.float64, copy=False)
    bid_size = frame.get_column("qb1").to_numpy().astype(np.float64, copy=False)

    mid = (ask_price + bid_price) / 2.0
    micro = (bid_price * ask_size + ask_price * bid_size) / (ask_size + bid_size + 1e-9)
    return {"mid": mid, "micro": micro, "avg": (mid + micro) / 2.0}


def _forward_target_from_series(
    series: np.ndarray,
    horizon_ticks: int,
    label_basis: str,
) -> np.ndarray:
    target = np.full(series.shape[0], np.nan, dtype=np.float64)
    if label_basis == "return_bps":
        safe_now = np.clip(series[:-horizon_ticks], 1e-12, None)
        safe_future = np.clip(series[horizon_ticks:], 1e-12, None)
        target[:-horizon_ticks] = np.log(safe_future / safe_now) * 10000.0
    elif label_basis == "price":
        target[:-horizon_ticks] = series[horizon_ticks:]
    else:
        raise ValueError(f"Unknown regression label basis: {label_basis}")
    return target


def _realized_volatility(series: np.ndarray, window: int) -> np.ndarray:
    returns = np.full(series.shape[0], np.nan, dtype=np.float64)
    returns[1:] = np.diff(np.log(np.clip(series, 1e-12, None))) * 10000.0
    vol = pd.Series(returns).rolling(window=window, min_periods=window).std().shift(1)
    values = vol.to_numpy(dtype=np.float64)
    values[values <= 1e-12] = np.nan
    return values


def apply_regression_targets(
    lob_df: pl.DataFrame,
    *,
    target: str = "mid",
    label_basis: str = "return_bps",
    horizon_ticks: int = 10,
    normalization: str = "none",
    vol_window: int = 100,
    vol_source: str = "mid",
    clip_value: float | None = None,
    timestamp_col: str = "datetime",
) -> tuple[pl.DataFrame, LabelSummary]:
    """Build TLOB-style scalar regression targets.

    TLOB computes mid/micro/avg streams from top-of-book and supports:
    - ``return_bps``: log forward return in basis points
    - ``price``: raw future price
    - ``normalization='vol'``: divide target by lagged realized volatility

    Targets are computed PER CALENDAR DAY so the forward horizon and the
    volatility window never cross the overnight gap; the last `horizon_ticks`
    rows of each day are dropped (NaN target) instead of being labelled with
    the next day's open.
    """

    target = (target or "mid").lower()
    label_basis = (label_basis or "return_bps").lower()
    normalization = (normalization or "none").lower()
    vol_source = (vol_source or "mid").lower()
    if target not in {"mid", "micro", "avg"}:
        raise ValueError(f"Unknown regression target: {target}")
    if vol_source not in {"mid", "micro", "avg"}:
        raise ValueError(f"Unknown regression vol_source: {vol_source}")
    if horizon_ticks <= 0:
        raise ValueError(f"horizon_ticks must be positive, got {horizon_ticks}")
    if normalization not in {"none", "vol"}:
        raise ValueError(f"Unknown regression normalization: {normalization}")

    ts_col = _resolve_ts_col(lob_df, timestamp_col)

    def _label_day(day_df: pl.DataFrame) -> pl.DataFrame:
        if day_df.height <= horizon_ticks:
            return day_df.head(0)
        streams = _price_streams_from_frame(day_df)
        raw_target = _forward_target_from_series(
            streams[target], horizon_ticks, label_basis
        )
        reg_target = raw_target.copy()

        scale = None
        if normalization == "vol":
            scale = _realized_volatility(streams[vol_source], vol_window)
            reg_target = reg_target / scale

        if clip_value is not None:
            reg_target = np.clip(reg_target, -float(clip_value), float(clip_value))

        columns = [
            pl.Series("target_reg", reg_target),
            pl.Series("target_reg_raw", raw_target),
        ]
        if scale is not None:
            columns.append(pl.Series("target_reg_scale", scale))
        return day_df.with_columns(columns)

    labelled = _map_per_day(lob_df, ts_col, _label_day)
    labelled = labelled.drop_nulls(subset=["target_reg"]).filter(
        pl.col("target_reg").is_finite()
    )
    return labelled, _empty_regression_summary(labelled.height)


def apply_fixed_horizon_labels(
    lob_df: pl.DataFrame,
    *,
    price_col: str,
    h_events: int,
    k: int,
    theta: float,
    timestamp_col: str = "datetime",
) -> tuple[pl.DataFrame, LabelSummary]:
    """Fixed EVENT/ROW-horizon labelling (h_events rows of the current grid).

    On the event grid (no bucketing) `h_events` is a true event-count horizon;
    on a resampled grid it counts buckets. All windows are computed per
    calendar day (`.over(day)`) so neither the smoothing nor the forward shift
    crosses the overnight gap. Rows whose windows are incomplete (day head or
    day tail) get a NULL label and are dropped downstream -- never a silent
    'hold' (class 0).
    """
    if k >= h_events:
        raise ValueError(
            f"label_k ({k}) must be < label_h_events ({h_events}); otherwise the "
            "future smoothing window reaches back to (or before) the anchor row."
        )
    ts_col = _resolve_ts_col(lob_df, timestamp_col)
    day = pl.col(ts_col).dt.date()

    df = lob_df.sort(ts_col).with_columns(
        [
            pl.col(price_col)
            .rolling_mean(window_size=k + 1, min_periods=k + 1)
            .over(day)
            .alias("w_minus"),
            pl.col(price_col)
            .shift(-h_events)
            .rolling_mean(window_size=k + 1, min_periods=k + 1)
            .over(day)
            .alias("w_plus"),
        ]
    )

    df = df.with_columns(
        ((pl.col("w_plus") - pl.col("w_minus")) / pl.col("w_minus")).alias("l_val")
    )

    df = df.with_columns(
        pl.when(pl.col("l_val").is_null() | pl.col("l_val").is_nan())
        .then(pl.lit(None, dtype=pl.Int32))
        .when(pl.col("l_val") > theta)
        .then(pl.lit(1, dtype=pl.Int32))
        .when(pl.col("l_val") < -theta)
        .then(pl.lit(2, dtype=pl.Int32))
        .otherwise(pl.lit(0, dtype=pl.Int32))
        .alias("target_class")
    ).drop(["w_minus", "w_plus", "l_val"])

    summary = _compute_summary(df["target_class"])
    return df, summary


def apply_fixed_volatility_labels(
    lob_df: pl.DataFrame,
    *,
    price_col: str,
    h_events: int,
    k: int,
    theta: float,
    vol_window: int,
    vol_k: float,
    max_theta: Optional[float] = None,
    timestamp_col: str = "datetime",
) -> tuple[pl.DataFrame, LabelSummary]:
    """Fixed horizon label with a causal volatility-scaled threshold.

    This keeps the current TLOB-style event/row horizon and smoothed
    past/future prices, but replaces the single global threshold with:

        threshold_t = max(theta, vol_k * trailing_vol_t * sqrt(h_events))

    The volatility is computed from one-row log returns using only data before
    the anchor row, independently per calendar day. During volatility warm-up,
    the label falls back to the fixed threshold. This makes the target more
    comparable across names and intraday regimes without changing the horizon.
    """
    if k >= h_events:
        raise ValueError(
            f"label_k ({k}) must be < label_h_events ({h_events}); otherwise the "
            "future smoothing window reaches back to (or before) the anchor row."
        )
    if vol_window <= 1:
        raise ValueError(f"label_vol_window must be > 1, got {vol_window}")
    if vol_k <= 0:
        raise ValueError(f"label_vol_k must be > 0, got {vol_k}")
    if max_theta is not None and max_theta <= theta:
        raise ValueError(
            f"label_max_theta ({max_theta}) must be greater than label_theta ({theta})."
        )

    ts_col = _resolve_ts_col(lob_df, timestamp_col)
    day = pl.col(ts_col).dt.date()

    df = lob_df.sort(ts_col).with_columns(
        [
            pl.col(price_col)
            .rolling_mean(window_size=k + 1, min_periods=k + 1)
            .over(day)
            .alias("w_minus"),
            pl.col(price_col)
            .shift(-h_events)
            .rolling_mean(window_size=k + 1, min_periods=k + 1)
            .over(day)
            .alias("w_plus"),
            pl.when(pl.col(price_col) > 1e-12)
            .then(pl.col(price_col))
            .otherwise(pl.lit(1e-12))
            .log()
            .alias("_log_price"),
        ]
    )

    df = df.with_columns(
        (pl.col("_log_price") - pl.col("_log_price").shift(1).over(day)).alias("_ret_1")
    )
    df = df.with_columns(
        pl.col("_ret_1")
        .rolling_std(window_size=vol_window, min_periods=vol_window)
        .shift(1)
        .over(day)
        .alias("_vol_1")
    )
    df = df.with_columns(
        ((pl.col("w_plus") - pl.col("w_minus")) / pl.col("w_minus")).alias("l_val")
    )

    threshold_expr = pl.max_horizontal(
        pl.lit(theta),
        pl.lit(vol_k * math.sqrt(h_events))
        * pl.col("_vol_1").fill_null(0.0).fill_nan(0.0),
    )
    if max_theta is not None:
        threshold_expr = pl.min_horizontal(pl.lit(max_theta), threshold_expr)

    df = df.with_columns(threshold_expr.alias("l_threshold"))
    df = df.with_columns(
        pl.when(
            pl.col("l_val").is_null()
            | pl.col("l_val").is_nan()
            | pl.col("l_threshold").is_null()
            | pl.col("l_threshold").is_nan()
        )
        .then(pl.lit(None, dtype=pl.Int32))
        .when(pl.col("l_val") > pl.col("l_threshold"))
        .then(pl.lit(1, dtype=pl.Int32))
        .when(pl.col("l_val") < -pl.col("l_threshold"))
        .then(pl.lit(2, dtype=pl.Int32))
        .otherwise(pl.lit(0, dtype=pl.Int32))
        .alias("target_class")
    ).drop(
        ["w_minus", "w_plus", "l_val", "l_threshold", "_log_price", "_ret_1", "_vol_1"],
        strict=False,
    )

    summary = _compute_summary(df["target_class"])
    return df, summary


def _parse_time_horizon(spec: str) -> pl.Expr:
    """Return a Polars duration expression for shorthand specs like '3m' or '30s'."""
    norm = spec.strip().lower()
    if norm.endswith("m"):
        minutes = int(norm[:-1])
        return pl.duration(minutes=minutes)
    if norm.endswith("s"):
        seconds = int(norm[:-1])
        return pl.duration(seconds=seconds)
    msg = "time_horizon must end with 'm' (minutes) or 's' (seconds), e.g. '3m' or '30s'."
    raise ValueError(msg)


def apply_time_based_labels(
    lob_df: pl.DataFrame,
    *,
    price_col: str = "mid_micro",
    time_horizon: str = "5s",
    threshold: float = 0.0002,
    timestamp_col: str = "timestamp",
) -> tuple[pl.DataFrame, LabelSummary]:
    """
    Time-based horizon labelling using an ASOF lookup on forward timestamps.

    Args:
        lob_df: Polars DataFrame containing at least the timestamp column and `price_col`.
        price_col: Column used for return calculation (default mid_micro).
        time_horizon: Horizon shorthand, e.g. "3m" or "30s".
        threshold: Return threshold that defines UP/DOWN assignments.
        timestamp_col: Column name containing event timestamps (default "timestamp").
    """

    timestamp_col = _resolve_ts_col(lob_df, timestamp_col)

    df = lob_df.sort(timestamp_col)

    # join_asof requires both keys share a time unit. Adding a polars duration
    # can promote the result to us, while the source may be ns (parquet ingest),
    # so pin both join keys to ns explicitly.
    future_lookup = df.select(
        [
            pl.col(timestamp_col).dt.cast_time_unit("ns").alias("lookup_ts"),
            pl.col(price_col).alias("future_price"),
        ]
    )

    duration = _parse_time_horizon(time_horizon)
    df = df.with_columns(
        (pl.col(timestamp_col) + duration).dt.cast_time_unit("ns").alias("target_time")
    )

    # Mask rows whose horizon reaches past the last observation of their own
    # calendar day. Without this, the backward as-of silently matches the
    # session close: near-close rows get a shrunken horizon biased toward
    # 'hold', and the label no longer means what the config says it means.
    day = pl.col(timestamp_col).dt.date()
    df = df.with_columns(
        pl.col(timestamp_col).dt.cast_time_unit("ns").max().over(day).alias("_day_end")
    )
    df = df.filter(pl.col("target_time") <= pl.col("_day_end")).drop("_day_end")

    df = df.join_asof(
        future_lookup,
        left_on="target_time",
        right_on="lookup_ts",
        strategy="backward",
    )

    df = df.with_columns(
        (
            (pl.col("future_price") - pl.col(price_col)) / pl.col(price_col)
        ).alias("future_ret")
    )

    df = df.with_columns(
        pl.when(pl.col("future_ret") > threshold)
        .then(1)
        .when(pl.col("future_ret") < -threshold)
        .then(2)
        .otherwise(0)
        .cast(pl.Int32)
        .alias("target_class")
    )

    df = df.drop_nulls(subset=["future_price"])
    df = df.drop(["target_time", "lookup_ts", "future_price", "future_ret"], strict=False)

    summary = _compute_summary(df["target_class"])
    return df, summary


def apply_zret_labels(
    lob_df: pl.DataFrame,
    *,
    price_col: str = "mid_price",
    h_events: int = 30,
    vol_win: int = 300,
    k_z: float = 1.0,
    clip_z: Optional[float] = 10.0,
    timestamp_col: str = "timestamp",
) -> tuple[pl.DataFrame, LabelSummary]:
    """
    Z-score return labelling.
    Labels:
      0 -> hold
      1 -> up
      2 -> down
    """
    if price_col not in lob_df.columns:
        if {"bid_px_1", "ask_px_1"} <= set(lob_df.columns):
            lob_df = lob_df.with_columns(
                ((pl.col("bid_px_1") + pl.col("ask_px_1")) / 2).alias("mid_price")
            )
            price_col = "mid_price"
        else:
            msg = f"Input DataFrame must include '{price_col}' or bid/ask columns to derive mid_price."
            raise ValueError(msg)

    ts_col = _resolve_ts_col(lob_df, timestamp_col)

    def _label_day(day_df: pl.DataFrame) -> pl.DataFrame:
        if day_df.height <= max(h_events, vol_win):
            return day_df.head(0)
        pdf = day_df.to_pandas()

        prices = pdf[price_col].to_numpy()
        future = np.roll(prices, -h_events)
        future[-h_events:] = np.nan

        ret = (future - prices) / prices
        log_ret = np.log(prices / np.roll(prices, 1))
        log_ret[0] = 0.0

        vol = (
            pd.Series(log_ret)
            .rolling(vol_win, min_periods=vol_win)
            .std()
            .to_numpy()
        )
        denom = vol * np.sqrt(h_events)
        with np.errstate(divide="ignore", invalid="ignore"):
            zret = ret / denom
        if clip_z is not None:
            zret = np.clip(zret, -clip_z, clip_z)

        # Undefined zret (warm-up vol window, day tail, zero vol) -> NULL
        # label, dropped downstream. Never a silent 'hold'.
        labels = np.full(len(prices), np.nan)
        labels[zret > k_z] = 1
        labels[zret < -k_z] = 2
        labels[np.abs(zret) <= k_z] = 0

        pdf["target_class"] = labels
        pdf = pdf.dropna(subset=["target_class", price_col])
        out = pl.from_pandas(pdf, include_index=False)
        return out.with_columns(pl.col("target_class").cast(pl.Int32))

    labelled = _map_per_day(lob_df, ts_col, _label_day)
    summary = _compute_summary(labelled["target_class"])
    return labelled, summary


def apply_triple_barrier_labels(
    lob_df: pl.DataFrame,
    *,
    price_col: str = "mid_price",
    horizon_ticks: int = 50,
    volatility_window: int = 2000,
    barrier_multiplier: float = 1.0,
    min_threshold_pct: float = 0.0001,
    max_threshold_pct: float = 0.005,
    timestamp_col: str = "datetime",
) -> tuple[pl.DataFrame, LabelSummary]:
    """
    Triple barrier (UP/DOWN/HOLD) labelling implemented with numpy/pandas.
    price_col yoksa bid/ask ile mid_price hesaplanır.

    Computed per calendar day: the barrier scan and the volatility window
    never cross the overnight gap, and each day's last `horizon_ticks` rows
    are dropped rather than labelled against the next day's open.
    """

    if price_col not in lob_df.columns:
        if {"bid_px_1", "ask_px_1"} <= set(lob_df.columns):
            lob_df = lob_df.with_columns(
                ((pl.col("bid_px_1") + pl.col("ask_px_1")) / 2).alias("mid_price")
            )
            price_col = "mid_price"
        else:
            msg = f"Input DataFrame must include '{price_col}' or bid/ask columns to derive mid_price."
            raise ValueError(msg)

    ts_col = _resolve_ts_col(lob_df, timestamp_col)

    def _label_day(day_df: pl.DataFrame) -> pl.DataFrame:
        pdf = day_df.to_pandas()
        pdf = pdf[pdf[price_col] > 0].copy()
        if len(pdf) <= max(horizon_ticks, volatility_window):
            return day_df.head(0)

        prices = pdf[price_col].to_numpy()
        log_ret = np.log(prices / np.roll(prices, 1))
        log_ret[0] = 0.0

        vol = (
            pd.Series(log_ret)
            .rolling(volatility_window, min_periods=volatility_window)
            .std()
            .to_numpy()
        )
        threshold = vol * barrier_multiplier * np.sqrt(horizon_ticks)
        threshold = np.clip(threshold, min_threshold_pct, max_threshold_pct)

        barrier_up = prices * (1 + threshold)
        barrier_down = prices * (1 - threshold)

        sentinel = np.int16(999)
        first_up = np.full(len(prices), sentinel, dtype=np.int16)
        first_down = np.full(len(prices), sentinel, dtype=np.int16)

        for step in range(1, horizon_ticks + 1):
            future = prices[step:]
            current_up = barrier_up[:-step]
            current_down = barrier_down[:-step]

            active = slice(0, len(prices) - step)
            upd_up = (first_up[active] == sentinel) & (future >= current_up)
            upd_down = (first_down[active] == sentinel) & (future <= current_down)

            first_up[active] = np.where(upd_up, step, first_up[active])
            first_down[active] = np.where(upd_down, step, first_down[active])

        labels = np.zeros(len(prices), dtype=np.int8)
        labels[(first_up < first_down) & (first_up != sentinel)] = 1
        labels[(first_down < first_up) & (first_down != sentinel)] = 2

        pdf["target_class"] = labels
        pdf["volatility"] = vol
        pdf["threshold"] = threshold

        pdf = pdf.dropna(subset=["volatility", "threshold"])
        if horizon_ticks > 0:
            pdf = pdf.iloc[:-horizon_ticks]

        pdf = pdf.drop(columns=["log_ret", "volatility", "threshold"], errors="ignore")
        return pl.from_pandas(pdf, include_index=False)

    labelled = _map_per_day(lob_df, ts_col, _label_day)
    summary = _compute_summary(labelled["target_class"])
    return labelled, summary


def apply_triple_barrier_cost_aware(
    lob_df: pl.DataFrame,
    *,
    price_col: str = "mid_price",
    horizon_ticks: int = 100,
    volatility_window: int = 2000,
    barrier_multiplier: float = 1.0,
    tick_size: float = 0.01,
    spread_ticks: float | None = None,
    commission_rate: float = 0.0003,
    min_profit_ticks: float = 0.3,
    max_pct: float = 0.005,
    timestamp_col: str = "datetime",
) -> tuple[pl.DataFrame, LabelSummary]:
    """
    Cost-aware triple barrier labelling.
    threshold = max(vol_based, spread + commission + min_profit), capped at max_pct.

    Computed per calendar day (see apply_triple_barrier_labels).
    """
    cols = set(lob_df.columns)
    if price_col not in lob_df.columns:
        raise ValueError(f"{price_col} not found for labelling.")

    ts_col = _resolve_ts_col(lob_df, timestamp_col)

    def _label_day(day_df: pl.DataFrame) -> pl.DataFrame:
        pdf = day_df.to_pandas()
        pdf = pdf[pdf[price_col] > 0].copy()

        prices = pdf[price_col].to_numpy()
        if len(prices) <= horizon_ticks + volatility_window:
            return day_df.head(0)

        if spread_ticks is None:
            if {"pa1", "pb1"} <= cols:
                spread = (pdf["pa1"] - pdf["pb1"]).to_numpy()
            elif "spread" in pdf.columns:
                spread = pdf["spread"].to_numpy()
            else:
                spread = np.zeros_like(prices)
            spread_ticks_arr = np.clip(spread / tick_size, 0, None)
        else:
            spread_ticks_arr = np.full_like(prices, fill_value=spread_ticks, dtype=float)

        log_ret = np.zeros_like(prices, dtype=float)
        log_ret[1:] = np.log(prices[1:] / prices[:-1])
        vol = (
            pd.Series(log_ret)
            .rolling(volatility_window, min_periods=volatility_window)
            .std()
            .to_numpy()
        )
        vol = np.nan_to_num(vol, nan=0.0001)

        dynamic_threshold = vol * barrier_multiplier * np.sqrt(horizon_ticks)

        spread_cost_pct = (spread_ticks_arr * tick_size) / prices
        commission_cost_pct = 2 * commission_rate
        min_profit_pct = (min_profit_ticks * tick_size) / prices

        total_min_pct = spread_cost_pct + commission_cost_pct + min_profit_pct
        threshold = np.maximum(dynamic_threshold, total_min_pct)
        threshold = np.minimum(threshold, max_pct)

        thr_ticks = (threshold * prices) / tick_size
        upper = prices * (1 + threshold)
        lower = prices * (1 - threshold)

        sentinel = np.int16(999)
        first_up = np.full(len(prices), sentinel, dtype=np.int16)
        first_down = np.full(len(prices), sentinel, dtype=np.int16)

        for step in range(1, horizon_ticks + 1):
            future = prices[step:]
            active = slice(0, len(prices) - step)

            upd_up = (first_up[active] == sentinel) & (future >= upper[:-step])
            upd_down = (first_down[active] == sentinel) & (future <= lower[:-step])

            first_up[active] = np.where(upd_up, step, first_up[active])
            first_down[active] = np.where(upd_down, step, first_down[active])

        labels = np.zeros(len(prices), dtype=np.int8)
        labels[(first_up < first_down) & (first_up != sentinel)] = 1
        labels[(first_down < first_up) & (first_down != sentinel)] = 2

        pdf["target_class"] = labels
        pdf["threshold_ticks"] = thr_ticks

        pdf = pdf.dropna(subset=["target_class"])
        if horizon_ticks > 0:
            pdf = pdf.iloc[:-horizon_ticks]

        return pl.from_pandas(pdf, include_index=False)

    labelled = _map_per_day(lob_df, ts_col, _label_day)
    summary = _compute_summary(labelled["target_class"])
    return labelled, summary


def apply_five_class_dynamic_labels(*args, **kwargs):
    """
    Placeholder for five-class dynamic labels.
    Currently not implemented; kept for backward compatibility with configs.
    """
    raise NotImplementedError("apply_five_class_dynamic_labels is not implemented in this build.")




__all__ = [
    "LabelSummary",
    "apply_regression_targets",
    "apply_fixed_horizon_labels",
    "apply_fixed_volatility_labels",
    "apply_time_based_labels",
    "apply_zret_labels",
    "apply_triple_barrier_labels",
    "apply_triple_barrier_cost_aware",
]
