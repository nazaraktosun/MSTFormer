from hft_features.core.base import feature
import polars as pl



@feature
def add_obi_and_ema_delta(df: pl.DataFrame, ema_window: int = 10) -> pl.DataFrame:
    """
    OBI = (sum qb – sum qa) / (sum qb + sum qa)
    ve bu OBI’nin EMA farkı.
    Eklenen: obi, obi_ema_delta
    """
    total_b = pl.sum_horizontal([pl.col(f"qb{i}") for i in range(1,4)])
    total_a = pl.sum_horizontal([pl.col(f"qa{i}") for i in range(1,4)])
    obi_expr = (total_b - total_a) / (total_b + total_a).fill_null(1).replace(0, 1)
    obi = obi_expr.alias("obi")
    # Add `obi` first to avoid referencing a just-created column in the same call.
    df = df.with_columns(obi)
    obi_ema = pl.col("obi").ewm_mean(alpha=2 / (ema_window + 1))
    delta = (pl.col("obi") - obi_ema).alias("obi_ema_delta")
    return df.with_columns(delta)

@feature
def add_price_vs_ewm(df: pl.DataFrame, span: int = 20) -> pl.DataFrame:
    """
    mid_price – mid_price’in EWM(span)
    Eklenen: price_vs_ewm
    """
    mid = pl.col("mid_price") if "mid_price" in df.columns else (pl.col("pb1") + pl.col("pa1")) / 2
    vs = (mid - mid.ewm_mean(span=span)).alias("price_vs_ewm")
    return df.with_columns(vs)



@feature
def depth_ratio(df: pl.DataFrame, levels: int = 3) -> pl.DataFrame:
    """Ratio of cumulative bid to ask volume over top N levels."""
    bid = pl.sum_horizontal([pl.col(f"qb{i}") for i in range(1, levels + 1)])
    ask = pl.sum_horizontal([pl.col(f"qa{i}") for i in range(1, levels + 1)])
    return df.with_columns(((bid + 1) / (ask + 1)).alias("depth_ratio"))

@feature
def depth_imbalance(df: pl.DataFrame, levels: int = 3) -> pl.DataFrame:
    """Normalized difference between cumulative bid and ask volumes."""
    bid = pl.sum_horizontal([pl.col(f"qb{i}") for i in range(1, levels + 1)])
    ask = pl.sum_horizontal([pl.col(f"qa{i}") for i in range(1, levels + 1)])
    return df.with_columns(((bid - ask) / (bid + ask + 1e-6)).alias("depth_imbalance"))

@feature
def weighted_imbalance(df: pl.DataFrame, levels: int = 3) -> pl.DataFrame:
    """Depth imbalance weighted by level index (more weight to inner book)."""
    bid = pl.sum_horizontal([pl.col(f"qb{i}") * i for i in range(1, levels + 1)])
    ask = pl.sum_horizontal([pl.col(f"qa{i}") * i for i in range(1, levels + 1)])
    return df.with_columns(((bid - ask) / (bid + ask + 1e-6)).alias("weighted_imbalance"))

@feature
def weighted_depth_metrics(
    df: pl.DataFrame,
    w1: float = 1.0,
    w2: float = 1.0,
    w3: float = 1.0
) -> pl.DataFrame:
    """
    Custom-weight depth ratio & imbalance using weights w1,w2,w3 for levels 1–3.
    Adds columns: weighted_depth_ratio, weighted_depth_imbalance
    """
    ask_w = (w1 * pl.col("qa1") +
             w2 * pl.col("qa2") +
             w3 * pl.col("qa3"))
    bid_w = (w1 * pl.col("qb1") +
             w2 * pl.col("qb2") +
             w3 * pl.col("qb3"))

    weighted_depth_ratio = (ask_w / (bid_w + 1e-6)).alias("weighted_depth_ratio")
    weighted_depth_imbalance = (
        (ask_w - bid_w) / (ask_w + bid_w + 1e-6)
    ).alias("weighted_depth_imbalance")

    return df.with_columns([weighted_depth_ratio, weighted_depth_imbalance])
@feature
def f_range_ticks(
    df: pl.DataFrame,
    lookback_window: int | None = None,
    tick_size: float = 1.0,
    windows: list[int] | None = None,
) -> pl.DataFrame:
    """
    Son lookback_window bar içindeki highest mid_price ile lowest mid_price
    arasını tick_size ile böler.
    Alias: "f_range_ticks"
    """
    mid = pl.col("mid_price")
    if windows is None:
        lookback = lookback_window if lookback_window is not None else 50
        hi = mid.rolling_max(lookback, min_periods=1)
        lo = mid.rolling_min(lookback, min_periods=1)
        return df.with_columns(((hi - lo) / tick_size).alias("f_range_ticks"))

    out = []
    for w in windows:
        hi = mid.rolling_max(w, min_periods=1)
        lo = mid.rolling_min(w, min_periods=1)
        out.append(((hi - lo) / tick_size).alias(f"f_range_ticks_{w}"))
    return df.with_columns(out)

def add_imbalance_persistence(
    df: pl.DataFrame,
    quantile_prob: float = 0.75,
    decay_span: int = 50,
    history_window: int = 100,
    quantile_imbalance: float | None = None,
) -> pl.DataFrame:
    """
    Imbalance persistence feature.
    """
    if quantile_imbalance is not None:
        quantile_prob = quantile_imbalance

    if "ofi_hybrid" in df.columns:
        ofi_expr = pl.col("ofi_hybrid")
    elif "f_obi_1" in df.columns:
        ofi_expr = pl.col("f_obi_1")
    else:
        ofi_expr = pl.col("qb1").diff().fill_null(0) - pl.col("qa1").diff().fill_null(0)

    ofi_abs = ofi_expr.abs()
    threshold = ofi_abs.rolling_quantile(
        quantile_prob,
        window_size=history_window,
        min_periods=history_window // 2,
    )

    is_high_imbalance = ofi_abs > threshold
    current_sign = ofi_expr.sign()
    prev_sign = ofi_expr.shift(1).sign()
    same_sign = (current_sign == prev_sign).cast(pl.Float64)

    persistence_stream = pl.when(is_high_imbalance).then(same_sign).otherwise(None)
    persistence_feature = (
        persistence_stream
        .ewm_mean(span=decay_span, ignore_nulls=True, min_periods=1)
        .fill_null(0.5)
        .alias("feat_imbalance_persistence")
    )

    return df.with_columns(persistence_feature)


#TODO bid time in book and ask time in book is computed at 1 level currently we can develop it further if needede
@feature(deps={"add_obi_and_ema_delta", "add_price_vs_ewm"})
def advanced_orderbook_features(df: pl.DataFrame, **params) -> pl.DataFrame:
    """
    Wrapper: Advanced Orderbook kategorisi.
    """
    df = add_obi_and_ema_delta(df, ema_window=params.get("ema_window", 10))
    df = add_price_vs_ewm(df, span=params.get("span", 20))

    lookback_window = params.get("lookback_window")
    windows = params.get("windows")
    if lookback_window is not None or windows is not None:
        df = f_range_ticks(
            df,
            lookback_window=lookback_window,
            tick_size=params.get("tick_size", 1.0),
            windows=windows,
        )

    df = add_imbalance_persistence(
        df,
        quantile_prob=params.get("quantile_prob", 0.75),
        decay_span=params.get("decay_span", 50),
        history_window=params.get("history_window", 100),
        quantile_imbalance=params.get("quantile_imbalance"),
    )
    return df
