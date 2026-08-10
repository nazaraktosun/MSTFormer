"""Feature engineering helpers for Bullsense order book data."""

from __future__ import annotations

import re
from datetime import time
from typing import Any, Callable, Iterable, Sequence

import polars as pl

LogFn = Callable[[str], None]


def _emit(message: str, log: LogFn | None) -> None:
    if log is not None:
        log(message)


def _parse_hhmm(value: str) -> time:
    """Parse an 'HH:MM' (or 'HH:MM:SS') string into a datetime.time."""
    parts = [int(p) for p in str(value).split(":")]
    parts += [0] * (3 - len(parts))
    hh, mm, ss = parts[:3]
    return time(hh, mm, ss)


def prepare_basic_columns(
    lob_df: pl.DataFrame,
    *,
    session_tz: str = "Europe/Istanbul",
    session_start: str = "09:55",
    session_end: str = "18:00",
    weekday_only: bool = True,
) -> pl.DataFrame:
    """
    Prepare the basic columns:
    1. Build a `datetime` column and convert it to `session_tz`.
    2. Keep only the trading session window [session_start, session_end).
    3. Compute mid_price / mid_micro.

    Market-agnostic: pass America/New_York + 09:30/16:00 for US equities,
    Europe/Istanbul + 09:55/18:00 for BIST.
    """
    df = lob_df.clone()

    # 1. Timestamp -> datetime, then convert to the session timezone.
    #    If `datetime` is absent, derive it from `pcap_timestamp` (assumed UTC, us).
    if "datetime" not in df.columns:
        df = df.with_columns(
            pl.from_epoch(pl.col("pcap_timestamp"), time_unit="us")
            .dt.replace_time_zone("UTC")
            .alias("datetime")
        )
    # Convert to the session tz (no-op if already there; robust if the ingest
    # produced a different tz).
    df = df.with_columns(pl.col("datetime").dt.convert_time_zone(session_tz))
    df = df.with_columns(pl.col("datetime").alias("readable_timestamp"))

    df = df.sort("datetime")

    # 2. Session filter [start, end) in local (session_tz) time, optionally weekdays only.
    start_t = _parse_hhmm(session_start)
    end_t = _parse_hhmm(session_end)
    initial_len = df.height
    session_mask = pl.col("datetime").dt.time().is_between(start_t, end_t, closed="left")
    if weekday_only:
        session_mask = session_mask & (pl.col("datetime").dt.weekday() <= 5)
    df = df.filter(session_mask)
    print(f"Session filter [{session_start},{session_end}) {session_tz}: dropped {initial_len - df.height} rows.")

    # 3. Fiyat Hesaplamaları
    if "mid_price" not in df.columns:
        df = df.with_columns(
            ((pl.col("pb1") + pl.col("pa1")) / 2.0).alias("mid_price")
        )

    if "mid_micro" not in df.columns:
        eps = 1e-9
        df = df.with_columns(
            (
                (pl.col("pa1") * pl.col("qb1") + pl.col("pb1") * pl.col("qa1"))
                / (pl.col("qa1") + pl.col("qb1") + eps)
            ).alias("mid_micro")
        )

    return df


#: per-event message columns (from bullsense.io.lobster_parquet) that trailing
#: windows aggregate over. Counts first, then unsigned/signed volumes.
_MSG_BASE_COLS = (
    "msgf_is_add",
    "msgf_is_del",
    "msgf_is_exec",
    "msgf_is_modify",
    "msgf_is_trade",
    "msgf_is_flush",
    "msgf_add_vol",
    "msgf_del_vol",
    "msgf_exec_vol",
    "msgf_modify_vol",
    "msgf_trade_vol",
    "msgf_flush_vol",
    "msgf_sadd_vol",
    "msgf_sdel_vol",
    "msgf_sexec_vol",
    "msgf_smodify_vol",
    "msgf_strade_vol",
)


def add_message_trailing_features(
    df: pl.DataFrame,
    *,
    time_windows: Sequence[str] = ("1s", "10s"),
    event_windows: Sequence[int] = (100,),
    log: LogFn | None = None,
) -> pl.DataFrame:
    """Add CAUSAL trailing message features to the combined book+message frame.

    Every feature attached to row t aggregates events over a window that ends
    AT t (``closed='right'`` / trailing row window including the current row).
    Nothing after t ever enters the features for t -- this is the as-of
    guarantee for the whole message-feature block.

    Time windows use the row's datetime (works on the event grid and on a
    resampled grid alike; a window never spans the overnight gap because the
    gap is far larger than any sane window). Event-count windows are computed
    per calendar day so a window never crosses a day boundary.
    """
    base = [c for c in _MSG_BASE_COLS if c in df.columns]
    if not base:
        raise ValueError(
            "add_message_trailing_features: no msgf_* columns found. "
            "Ingest with load_messages=True (bullsense.io.lobster_parquet)."
        )
    if "datetime" not in df.columns:
        raise ValueError("add_message_trailing_features requires a 'datetime' column.")

    df = df.sort("datetime").with_columns(
        pl.col("datetime").dt.date().alias("_msg_day")
    )

    exprs: list[pl.Expr] = []
    for window in time_windows:
        for col in base:
            exprs.append(
                pl.col(col)
                .rolling_sum_by("datetime", window_size=window, closed="right")
                .alias(f"{col}_{window}")
            )
    for k in event_windows:
        for col in base:
            exprs.append(
                pl.col(col)
                .rolling_sum(window_size=int(k), min_samples=1)
                .over("_msg_day")
                .alias(f"{col}_e{int(k)}")
            )

    # Instantaneous event-price distance from the prevailing mid (bps).
    # Only meaningful on the event grid (raw fields are dropped on resample).
    if "msg_price" in df.columns and "mid_price" in df.columns:
        exprs.append(
            pl.when(pl.col("msg_price") > 0)
            .then((pl.col("msg_price") - pl.col("mid_price")) / pl.col("mid_price") * 1e4)
            .otherwise(0.0)
            .alias("msgf_px_bps")
        )

    df = df.with_columns(exprs).drop("_msg_day")
    _emit(
        f"Added {len(exprs)} trailing message features "
        f"(time={list(time_windows)}, events={list(event_windows)})",
        log,
    )
    return df


#: per-event columns the 4 normalized flow features are built from. All are
#: msgf_* so they survive batch-collapse / ms-resample as summed aggregates.
_MSG_FLOW_INPUTS = (
    "msgf_sadd_vol",
    "msgf_sdel_vol",
    "msgf_sexec_vol",
    "msgf_add_vol",
    "msgf_del_vol",
    "msgf_exec_vol",
    "msgf_add_bid_touch_vol",
    "msgf_add_ask_touch_vol",
    "msgf_del_bid_touch_vol",
    "msgf_del_ask_touch_vol",
)


def add_message_flow_features(
    df: pl.DataFrame,
    *,
    windows: Sequence[int] = (10, 50, 100),
    eps: float = 1e-9,
    log: LogFn | None = None,
) -> pl.DataFrame:
    """Add the 4 normalized, causal message-flow features per event window.

    For each window ``k`` (rows = events on the event grid, = book updates on
    the batch/snapshot grid), over a TRAILING window ending at the current row,
    computed per calendar day so no window spans the overnight gap:

      msgf_ofi_norm_e{k}        signed order-flow imbalance
                                (Σ sadd − Σ sdel + Σ sexec) / Σ(add+del+exec vol)
      msgf_exec_flow_norm_e{k}  aggressor-signed exec flow  Σ sexec / Σ exec_vol
      msgf_touch_add_imb_e{k}   (Σ add@bid − Σ add@ask) / (Σ add@bid + Σ add@ask)
      msgf_touch_cancel_imb_e{k}(Σ del@ask − Σ del@bid) / (Σ del@ask + Σ del@bid)

    Signs follow the microstructure prior that predicted positive forward
    returns in feature_research: buy-side pressure / ask-side depletion → +.
    These are ratios of causal sums, so they inherit the as-of guarantee of the
    trailing window — nothing after row t enters row t's value.
    """
    missing = [c for c in _MSG_FLOW_INPUTS if c not in df.columns]
    if missing:
        raise ValueError(
            "add_message_flow_features: missing input columns "
            f"{missing}. Ingest with load_messages=True so msgf_* and touch "
            "columns are attached (bullsense.io.lobster_parquet)."
        )
    if "datetime" not in df.columns:
        raise ValueError("add_message_flow_features requires a 'datetime' column.")

    df = df.sort("datetime").with_columns(
        pl.col("datetime").dt.date().alias("_flow_day")
    )

    def rs(col: str, k: int) -> pl.Expr:
        return (
            pl.col(col)
            .rolling_sum(window_size=int(k), min_samples=1)
            .over("_flow_day")
        )

    exprs: list[pl.Expr] = []
    for k in windows:
        ofi_num = rs("msgf_sadd_vol", k) - rs("msgf_sdel_vol", k) + rs("msgf_sexec_vol", k)
        ofi_den = rs("msgf_add_vol", k) + rs("msgf_del_vol", k) + rs("msgf_exec_vol", k)
        add_bid, add_ask = rs("msgf_add_bid_touch_vol", k), rs("msgf_add_ask_touch_vol", k)
        del_bid, del_ask = rs("msgf_del_bid_touch_vol", k), rs("msgf_del_ask_touch_vol", k)
        exprs += [
            (ofi_num / (ofi_den + eps)).alias(f"msgf_ofi_norm_e{int(k)}"),
            (rs("msgf_sexec_vol", k) / (rs("msgf_exec_vol", k) + eps)).alias(
                f"msgf_exec_flow_norm_e{int(k)}"
            ),
            ((add_bid - add_ask) / (add_bid + add_ask + eps)).alias(
                f"msgf_touch_add_imb_e{int(k)}"
            ),
            ((del_ask - del_bid) / (del_ask + del_bid + eps)).alias(
                f"msgf_touch_cancel_imb_e{int(k)}"
            ),
        ]

    df = df.with_columns(exprs).drop("_flow_day")
    _emit(f"Added {len(exprs)} message-flow features (windows={list(windows)})", log)
    return df


def apply_feature_pipeline(
    lob_df: pl.DataFrame,
    feature_configs: Sequence[dict[str, Any]],
    pipeline_cls: Callable[..., Any],
    registry_factory: Callable[[], Any],
    *,
    drop_raw_lob: bool = True,
    log: LogFn | None = None,
) -> pl.DataFrame:
    """Apply the financial feature pipeline and optionally drop raw LOB columns."""
    df = lob_df.clone()
    registry = registry_factory()
    _emit(f"Bootstrapped registry with {len(registry)} entries", log)

    pipeline = pipeline_cls(feature_configs, cache_dir=None)
    df = pipeline.fit_transform(df)
    _emit(f"Applied {len(feature_configs)} feature transforms", log)

    if drop_raw_lob:
        drop_pattern = re.compile(r"^(p|q)(a|b)\d+$", flags=re.IGNORECASE)
        to_drop = [c for c in df.columns if drop_pattern.search(c)]
        if to_drop:
            df = df.drop(to_drop)
            _emit(f"Dropped {len(to_drop)} raw LOB columns", log)
    else:
        _emit("Retained raw LOB columns", log)

    return df


__all__ = [
    "add_message_flow_features",
    "add_message_trailing_features",
    "apply_feature_pipeline",
    "prepare_basic_columns",
]
