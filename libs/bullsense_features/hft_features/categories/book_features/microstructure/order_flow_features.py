import polars as pl
from hft_features.core.base import feature

##### düzeltmedik

@feature
def total_depth_volume(df: pl.DataFrame, levels: int = 3) -> pl.DataFrame:
    bid_cols = [f"qb{i}" for i in range(1, levels + 1)]
    ask_cols = [f"qa{i}" for i in range(1, levels + 1)]

    total_bid = pl.sum_horizontal([pl.col(c) for c in bid_cols]).alias("total_bid_volume")
    total_ask = pl.sum_horizontal([pl.col(c) for c in ask_cols]).alias("total_ask_volume")

    df = df.with_columns([total_bid, total_ask])
    return df.with_columns(
        (pl.col("total_bid_volume") + pl.col("total_ask_volume"))
        .alias("total_depth_volume")
    )


# -------------------------
# Hybrid OFI
# -------------------------



@feature
def ofi_hybrid(df: pl.DataFrame) -> pl.DataFrame:
    """
    Hybrid Order Flow Imbalance (Cont–Kukanov):
      - ofi_delta: simple L1 volume delta
      - ofi_flip: flip contribution when bid/ask price flips
      - ofi_hybrid: ofi_flip if flip happened, else ofi_delta
    """
    # Current and previous L1 state
    bid_px = pl.col("pb1")
    ask_px = pl.col("pa1")
    bid_q = pl.col("qb1")
    ask_q = pl.col("qa1")

    bid_px_prev = bid_px.shift(1)
    ask_px_prev = ask_px.shift(1)
    bid_q_prev = bid_q.shift(1)
    ask_q_prev = ask_q.shift(1)

    # 1) Simple delta (always defined; first satır 0)
    ofi_delta = (
        (bid_q - bid_q_prev) - (ask_q - ask_q_prev)
    ).fill_null(0.0)

    # 2) Cont–Kukanov flip contribution (non-zero only on flips)
    ofi_flip = (
        pl.when(bid_px > bid_px_prev).then(bid_q).otherwise(0)
        - pl.when(bid_px < bid_px_prev).then(bid_q_prev).otherwise(0)
        - pl.when(ask_px < ask_px_prev).then(ask_q).otherwise(0)
        + pl.when(ask_px > ask_px_prev).then(ask_q_prev).otherwise(0)
    ).fill_null(0.0)

    # 3) Choose: flip value when a flip happened, else delta
    did_flip = (
        (bid_px != bid_px_prev) | (ask_px != ask_px_prev)
    ).fill_null(False)

    hybrid = (
        pl.when(did_flip)
          .then(ofi_flip)
          .otherwise(ofi_delta)
          .alias("ofi_hybrid")
    )

    return df.with_columns(hybrid)


@feature
def f_rolling_ofi_hybrid(
    df: pl.DataFrame,
    window: int = 50
) -> pl.DataFrame:
    """
    Rolling-sum of ofi_hybrid over the last `window` bars.
    """
    return df.with_columns(
        pl.col("ofi_hybrid")
          .rolling_sum(window_size=window, min_periods=1)
          .alias("f_rolling_ofi_hybrid")
    )

@feature(deps={"total_depth_volume", "ofi_hybrid"})
def ofi_hybrid_depth_scaled(df: pl.DataFrame) -> pl.DataFrame:
    """
    Snapshot-relative OFI:
      ofi_hybrid_depth_scaled = ofi_hybrid / total_depth_volume
    """
    return df.with_columns(
        (pl.col("ofi_hybrid") / (pl.col("total_depth_volume") + 1e-8))
        .alias("ofi_hybrid_depth_scaled")
    )


@feature(deps={"ofi_hybrid_depth_scaled"})
def f_rolling_ofi_hybrid_scaled(
    df: pl.DataFrame,
    window: int = 50
) -> pl.DataFrame:
    return df.with_columns(
        pl.col("ofi_hybrid_depth_scaled")
          .rolling_sum(window_size=window, min_periods=1)
          .alias(f"f_rolling_ofi_hybrid_scaled_{window}")
    )

# -------------------------
# Basit delta & flip indikatorleri
# -------------------------

@feature
def bidqty_delta(df: pl.DataFrame) -> pl.DataFrame:
    """Change in bid quantity since previous bar."""
    return df.with_columns(
        pl.col("qb1").diff().fill_null(0.0).alias("bidqty_delta")
    )


@feature
def askqty_delta(df: pl.DataFrame) -> pl.DataFrame:
    """Change in ask quantity since previous bar."""
    return df.with_columns(
        pl.col("qa1").diff().fill_null(0.0).alias("askqty_delta")
    )

@feature(deps={"total_depth_volume"})
def bidqty_delta_rel(df: pl.DataFrame) -> pl.DataFrame:
    delta = pl.col("qb1").diff().fill_null(0.0)
    return df.with_columns(
        (delta / (pl.col("total_depth_volume") + 1e-8))
        .alias("bidqty_delta_rel")
    )


@feature(deps={"total_depth_volume"})
def askqty_delta_rel(df: pl.DataFrame) -> pl.DataFrame:
    delta = pl.col("qa1").diff().fill_null(0.0)
    return df.with_columns(
        (delta / (pl.col("total_depth_volume") + 1e-8))
        .alias("askqty_delta_rel")
    )
 
@feature
def bid_flip(df: pl.DataFrame) -> pl.DataFrame:
    """Indicator (0/1) if bid price changed since previous bar."""
    return df.with_columns(
        (pl.col("pb1") != pl.col("pb1").shift(1))
        .fill_null(False)
        .cast(pl.Int64)
        .alias("bid_flip")
    )


@feature
def ask_flip(df: pl.DataFrame) -> pl.DataFrame:
    """Indicator (0/1) if ask price changed since previous bar."""
    return df.with_columns(
        (pl.col("pa1") != pl.col("pa1").shift(1))
        .fill_null(False)
        .cast(pl.Int64)
        .alias("ask_flip")
    )


@feature
def f_bid_flip_count(
    df: pl.DataFrame,
    window: int = 50
) -> pl.DataFrame:
    """Rolling count of bid_flip over the last `window` bars."""
    return df.with_columns(
        pl.col("bid_flip")
          .rolling_sum(window_size=window, min_periods=1)
          .alias("f_bid_flip_count")
    )


@feature
def f_ask_flip_count(
    df: pl.DataFrame,
    window: int = 50
) -> pl.DataFrame:
    """Rolling count of ask_flip over the last `window` bars."""
    return df.with_columns(
        pl.col("ask_flip")
          .rolling_sum(window_size=window, min_periods=1)
          .alias("f_ask_flip_count")
    )

@feature(deps={"bid_flip"})
def f_bid_flip_freq(
    df: pl.DataFrame,
    window: int = 50
) -> pl.DataFrame:
    return df.with_columns(
        (
            pl.col("bid_flip").rolling_sum(window_size=window, min_periods=1)
            / window
        ).alias(f"f_bid_flip_freq_{window}")
    )


@feature(deps={"ask_flip"})
def f_ask_flip_freq(
    df: pl.DataFrame,
    window: int = 50
) -> pl.DataFrame:
    return df.with_columns(
        (
            pl.col("ask_flip").rolling_sum(window_size=window, min_periods=1)
            / window
        ).alias(f"f_ask_flip_freq_{window}")
    )




# -------------------------
# Trade Flow Imbalance (TFI)
# -------------------------

@feature
def trade_flow_imbalance(
    df: pl.DataFrame,
    window: int = 50
) -> pl.DataFrame:
    """
    Trade Flow Imbalance (TFI).
    Requires execution data (msg_type='E', exec_side in {B,S}, exec_qty).
      TFI = rolling_sum(V_buy - V_sell, window)
    """
    required_cols = ["msg_type", "exec_side", "exec_qty"]
    if not all(col in df.columns for col in required_cols):
        # If no execution data, return zero TFI
        return df.with_columns(pl.lit(0.0).alias("tfi"))

    is_exec = pl.col("msg_type") == "E"

    v_buy = (
        pl.when(is_exec & (pl.col("exec_side") == "B"))
          .then(pl.col("exec_qty"))
          .otherwise(0.0)
          .rolling_sum(window_size=window, min_periods=1)
    )

    v_sell = (
        pl.when(is_exec & (pl.col("exec_side") == "S"))
          .then(pl.col("exec_qty"))
          .otherwise(0.0)
          .rolling_sum(window_size=window, min_periods=1)
    )

    tfi = (v_buy - v_sell).alias("tfi")
    return df.with_columns(tfi)

@feature
def tfi_ratio(
    df: pl.DataFrame,
    window: int = 50
) -> pl.DataFrame:
    required_cols = ["msg_type", "exec_side", "exec_qty"]
    if not all(col in df.columns for col in required_cols):
        return df.with_columns(pl.lit(0.0).alias(f"tfi_ratio_{window}"))

    is_exec = pl.col("msg_type") == "E"

    v_buy = (
        pl.when(is_exec & (pl.col("exec_side") == "B"))
          .then(pl.col("exec_qty"))
          .otherwise(0.0)
          .rolling_sum(window_size=window, min_periods=1)
    )

    v_sell = (
        pl.when(is_exec & (pl.col("exec_side") == "S"))
          .then(pl.col("exec_qty"))
          .otherwise(0.0)
          .rolling_sum(window_size=window, min_periods=1)
    )

    num = v_buy - v_sell
    den = v_buy + v_sell + 1e-8

    return df.with_columns(
        (num / den).alias(f"tfi_ratio_{window}")
    )


# -------------------------
# Wrapper
# -------------------------
@feature(deps={
    "ofi_hybrid",
    "f_rolling_ofi_hybrid",
    "bidqty_delta", "askqty_delta",
    "bid_flip", "ask_flip",
    "f_bid_flip_count", "f_ask_flip_count",
    "bid_time_in_book", "ask_time_in_book",
    "trade_flow_imbalance",

    "total_depth_volume",
    "ofi_hybrid_depth_scaled", "f_rolling_ofi_hybrid_scaled",
    "bidqty_delta_rel", "askqty_delta_rel",
    "f_bid_flip_freq", "f_ask_flip_freq",
    "tfi_ratio",
    # opsiyonel:
    # "bid_time_in_book_log", "ask_time_in_book_log",
})
def order_flow_features_scaled(
    df: pl.DataFrame,
    window: int = 50,
    **params
) -> pl.DataFrame:
    df = ofi_hybrid(df)
    df = f_rolling_ofi_hybrid(df, window=window)

    df = total_depth_volume(df)
    df = ofi_hybrid_depth_scaled(df)
    df = f_rolling_ofi_hybrid_scaled(df, window=window)

    df = bidqty_delta(df)
    df = askqty_delta(df)
    df = bidqty_delta_rel(df)
    df = askqty_delta_rel(df)

    df = bid_flip(df)
    df = ask_flip(df)
    df = f_bid_flip_count(df, window=window)
    df = f_ask_flip_count(df, window=window)
    df = f_bid_flip_freq(df, window=window)
    df = f_ask_flip_freq(df, window=window)


    df = trade_flow_imbalance(df, window=window)
    df = tfi_ratio(df, window=window)

    return df
