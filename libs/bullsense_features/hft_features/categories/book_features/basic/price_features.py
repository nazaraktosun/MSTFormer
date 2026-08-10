"""
Required Inputs:
- average_liquidity: Pre-computed average (bid_qty + ask_qty) / 2
- tick_size: Minimum price increment for the asset
"""

import polars as pl
import numpy as np
from hft_features.core.base import feature


EPS = 1e-9

# Clipping bounds
RETURN_CLIP_PCT = 5.0       # ±5% max return
SPREAD_CLIP_TICKS = 50      # Max 50 ticks spread
ZSCORE_CLIP = 5.0           # ±5 std max
QTY_SCALED_CLIP = 10.0      # Max 10x average liquidity

# Default windows (in ticks, assuming 100ms grid)
WINDOW_SHORT = 10           # 1 second
WINDOW_MED = 50             # 5 seconds
WINDOW_LONG = 200           # 20 seconds
WINDOW_VLONG = 1000         # 100 seconds



@feature
def base_prices(df: pl.DataFrame, level: int = 1) -> pl.DataFrame:
    """
    Calculate base price values (internal use).
    
    Outputs:
    - _mid: Mid price (internal)
    - _spread: Spread (internal)
    - mid_price: Mid price (for reference, may be dropped later)
    """
    bid = pl.col(f"pb{level}")
    ask = pl.col(f"pa{level}")
    
    mid = ((bid + ask) / 2)
    spread = (ask - bid)
    
    return df.with_columns([
        mid.alias("_mid"),
        spread.alias("_spread"),
        mid.alias("mid_price"),  # Keep for labeling
    ])




@feature(deps={"base_prices"})
def spread_features(
    df: pl.DataFrame,
    tick_size: float,
    clip_ticks: float = SPREAD_CLIP_TICKS,
) -> pl.DataFrame:
    """
    Spread features scaled by tick size.
    
    Args:
        tick_size: Minimum price increment (e.g., 0.02 for THYAO)
    
    Outputs:
        f_spread_ticks: Spread in ticks [1, 50]
        f_spread_bps: Spread in basis points (for reference)
    """
    spread = pl.col("_spread")
    mid = pl.col("_mid")
    
    # Spread in ticks (primary - cross-asset comparable)
    spread_ticks = (spread / tick_size).round().clip(0, clip_ticks).alias("f_spread_ticks")
    
    # Spread in BPS (secondary - for reference)
    spread_bps = ((spread / (mid + EPS)) * 1e4).clip(0, 200).alias("f_spread_bps")
    
    return df.with_columns([spread_ticks, spread_bps])



@feature(deps={"base_prices"})
def return_features(
    df: pl.DataFrame,
    steps: list[int] = None,
    clip_pct: float = RETURN_CLIP_PCT,
) -> pl.DataFrame:
    """
    Return features as percentages.
    
    Following Senior's approach: (price / past_price - 1) * 100
    
    Outputs:
        f_ret_1: 1-step return in % [-5, 5]
        f_ret_5: 5-step return in %
        f_ret_10: 10-step return in %
        f_ret_50: 50-step return in %
    """
    if steps is None:
        steps = [1, 5, 10, 20, 50]
    
    mid = pl.col("_mid")
    
    out = []
    for k in steps:
        # Percentage return (Senior's style)
        ret_pct = ((mid / mid.shift(k) - 1) * 100).clip(-clip_pct, clip_pct).alias(f"f_ret_{k}")
        out.append(ret_pct)
    
    return df.with_columns(out)


@feature(deps={"base_prices"})
def daily_return(df: pl.DataFrame) -> pl.DataFrame:
    """
    Daily return from session open.
    
    Requires: First row's mid_price as reference (open_mid_price)
    
    Output:
        f_daily_return: Return since open in %
    """
    mid = pl.col("_mid")
    
    # Get first mid price of the session
    open_mid = mid.first()
    
    daily_ret = ((mid / open_mid - 1) * 100).alias("f_daily_return")
    
    return df.with_columns(daily_ret)


@feature(deps={"return_features"})
def volatility_features(
    df: pl.DataFrame,
    windows: list[int] = None,
) -> pl.DataFrame:
    """
    Volatility features.
    
    Raw volatility is already small (std of % returns).
    Also compute relative volatility (current vs longer-term).
    
    Outputs:
        f_vol_10: 10-step rolling volatility
        f_vol_50: 50-step rolling volatility
        f_vol_200: 200-step rolling volatility
        f_vol_ratio: Short-term / Long-term volatility [0.2, 5.0]
    """
    if windows is None:
        windows = [WINDOW_SHORT, WINDOW_MED, WINDOW_LONG]
    
    ret1 = pl.col("f_ret_1")
    
    out = []
    for w in windows:
        vol = ret1.rolling_std(w).fill_null(0.0).alias(f"f_vol_{w}")
        out.append(vol)
    
    df = df.with_columns(out)
    
    # Volatility ratio (short-term vs long-term)
    if len(windows) >= 2:
        vol_short = pl.col(f"f_vol_{windows[0]}")
        vol_long = pl.col(f"f_vol_{windows[-1]}")
        vol_ratio = (vol_short / (vol_long + EPS)).clip(0.1, 5.0).alias("f_vol_ratio")
        df = df.with_columns(vol_ratio)
    
    return df



@feature(deps={"return_features", "volatility_features"})
def momentum_features(
    df: pl.DataFrame,
    steps: list[int] = None,
    clip: float = ZSCORE_CLIP,
) -> pl.DataFrame:
    """
    Momentum = Return / Volatility (Sharpe-like).
    
    This is naturally scaled as a z-score-like measure.
    
    Outputs:
        f_mom_10: 10-step momentum [-5, 5]
        f_mom_50: 50-step momentum [-5, 5]
    """
    if steps is None:
        steps = [10, 50]
    
    # Use medium-term vol as reference
    vol_ref = pl.col(f"f_vol_{WINDOW_MED}")
    
    out = []
    for k in steps:
        ret_col = f"f_ret_{k}"
        
        # Check if return column exists
        if ret_col in df.columns:
            ret = pl.col(ret_col)
        else:
            # Calculate return if not exists
            mid = pl.col("_mid")
            ret = (mid / mid.shift(k) - 1) * 100
        
        # Vol-adjusted momentum
        mom = (ret / (vol_ref + EPS)).clip(-clip, clip).alias(f"f_mom_{k}")
        out.append(mom)
    
    return df.with_columns(out)



@feature(deps={"base_prices"})
def micro_price_features(
    df: pl.DataFrame,
    tick_size: float,
    level: int = 1,
) -> pl.DataFrame:
    """
    Micro price and its deviation from mid.
    
    Micro price = volume-weighted average of best bid/ask.
    Deviation scaled by:
    1. Half-spread (Senior's implicit approach) → [-1, 1]
    2. Ticks (explicit) → [-5, 5]
    
    Outputs:
        microprice: Raw microprice (for reference)
        f_micro_off_hs: Micro offset in half-spreads [-2, 2]
        f_micro_off_ticks: Micro offset in ticks [-5, 5]
    """
    bid_p = pl.col(f"pb{level}")
    ask_p = pl.col(f"pa{level}")
    bid_q = pl.col(f"qb{level}")
    ask_q = pl.col(f"qa{level}")
    
    mid = pl.col("_mid")
    spread = pl.col("_spread")
    
    # Microprice calculation
    micro = (bid_p * ask_q + ask_p * bid_q) / (bid_q + ask_q + EPS)
    
    # Deviation from mid
    micro_dev = micro - mid
    
    # Scaled by half-spread (naturally bounded ~[-1, 1])
    micro_off_hs = (micro_dev / (spread / 2 + EPS)).clip(-2, 2).alias("f_micro_off_hs")
    
    # Scaled by tick size
    micro_off_ticks = (micro_dev / tick_size).clip(-5, 5).alias("f_micro_off_ticks")
    
    return df.with_columns([
        micro.alias("microprice"),
        micro_off_hs,
        micro_off_ticks,
    ])


@feature(deps={"base_prices"})
def weighted_price_features(df: pl.DataFrame, levels: int = 3) -> pl.DataFrame:
    """
    Volume-weighted average prices across levels.
    
    Senior's clever scaling: weighted_bid_price / askpx
    This gives a ratio ~0.99-1.01 (naturally bounded).
    
    Outputs:
        f_bid_weighted_2: Weighted bid (L1+L2) / ask price
        f_ask_weighted_2: Weighted ask (L1+L2) / bid price
        f_bid_weighted_3: Weighted bid (L1+L2+L3) / ask price
        f_ask_weighted_3: Weighted ask (L1+L2+L3) / bid price
    """
    out = []
    
    for k in [2, 3]:
        if k > levels:
            continue
            
        # Weighted bid price
        bid_num = pl.sum_horizontal([pl.col(f"pb{i}") * pl.col(f"qb{i}") for i in range(1, k + 1)])
        bid_den = pl.sum_horizontal([pl.col(f"qb{i}") for i in range(1, k + 1)])
        bid_weighted = bid_num / (bid_den + EPS)
        
        # Weighted ask price
        ask_num = pl.sum_horizontal([pl.col(f"pa{i}") * pl.col(f"qa{i}") for i in range(1, k + 1)])
        ask_den = pl.sum_horizontal([pl.col(f"qa{i}") for i in range(1, k + 1)])
        ask_weighted = ask_num / (ask_den + EPS)
        
        # Senior's scaling: ratio to opposite side
        f_bid_weighted = (bid_weighted / pl.col("pa1")).alias(f"f_bid_weighted_{k}")
        f_ask_weighted = (ask_weighted / pl.col("pb1")).alias(f"f_ask_weighted_{k}")
        
        out.extend([f_bid_weighted, f_ask_weighted])
    
    return df.with_columns(out)



@feature(deps={"base_prices"})
def range_features(
    df: pl.DataFrame,
    tick_size: float,
    window: int = WINDOW_MED,
) -> pl.DataFrame:
    """
    Price range in ticks.
    
    Output:
        f_range_ticks: High-Low range in ticks
    """
    mid = pl.col("_mid")
    
    hi = mid.rolling_max(window)
    lo = mid.rolling_min(window)
    
    range_ticks = ((hi - lo) / tick_size).alias("f_range_ticks")
    
    return df.with_columns(range_ticks)



@feature(deps={"base_prices"})
def trend_features(
    df: pl.DataFrame,
    windows: list[int] = None,
) -> pl.DataFrame:
    """
    Price direction trend.
    
    Sum of price directions over window, normalized by window size.
    
    Outputs:
        f_trend_10: Normalized trend [-1, 1]
        f_trend_20: Normalized trend [-1, 1]
        f_trend_50: Normalized trend [-1, 1]
    """
    if windows is None:
        windows = [10, 20, 50]
    
    mid = pl.col("_mid")
    mid_change = mid - mid.shift(1)
    
    # Price direction: +1, 0, -1
    price_dir = (
        pl.when(mid_change > 0).then(1)
          .when(mid_change < 0).then(-1)
          .otherwise(0)
    )
    
    out = []
    for w in windows:
        # Normalized trend (sum / window) → [-1, 1]
        trend_norm = (price_dir.rolling_sum(w) / w).alias(f"f_trend_{w}")
        out.append(trend_norm)
    
    return df.with_columns(out)


@feature(deps={"base_prices"})
def monotonicity_features(
    df: pl.DataFrame,
    windows: list[int] = None,
) -> pl.DataFrame:
    """
    Price path monotonicity.
    
    monotonicity = |net_change| / sum(|changes|)
    
    1.0 = perfectly monotonic (straight line)
    0.0 = highly oscillating
    
    Outputs:
        f_mono_10: Monotonicity [0, 1]
        f_mono_50: Monotonicity [0, 1]
    """
    if windows is None:
        windows = [WINDOW_SHORT, WINDOW_MED]
    
    mid = pl.col("_mid")
    
    out = []
    for w in windows:
        abs_diff = mid.diff().abs()
        tot_abs = abs_diff.rolling_sum(w) + EPS
        net_chg = (mid - mid.shift(w)).abs()
        
        mono = (net_chg / tot_abs).clip(0, 1).alias(f"f_mono_{w}")
        out.append(mono)
    
    return df.with_columns(out)


@feature(deps={"micro_price_features"})
def fair_value_features(
    df: pl.DataFrame,
    span: int = 100,
    tick_size: float = None,
) -> pl.DataFrame:
    """
    Fair value = EWM of microprice.
    
    Deviation from fair value is a mean-reversion signal.
    
    Outputs:
        fair_value: EWM of microprice
        f_dist_to_fv_ticks: Distance to fair value in ticks
    """
    micro = pl.col("microprice")
    mid = pl.col("_mid")
    
    # EWM of microprice
    alpha = 2.0 / (span + 1)
    fv = micro.ewm_mean(alpha=alpha, adjust=False).alias("fair_value")
    
    df = df.with_columns(fv)
    
    # Distance to fair value
    if tick_size is not None:
        dist_fv = ((mid - pl.col("fair_value")) / tick_size).clip(-10, 10).alias("f_dist_to_fv_ticks")
        df = df.with_columns(dist_fv)
    
    return df

# price_features.py'ye ekle:

@feature(deps={"base_prices"})
def price_slope_features(
    df: pl.DataFrame,
    tick_size: float,
    window: int = 5,
) -> pl.DataFrame:
    """
    Price slope (momentum) in ticks per step.
    
    Outputs:
        f_bid_slope: Bid price slope (ticks/step)
        f_ask_slope: Ask price slope (ticks/step)
        f_mid_slope: Mid price slope (ticks/step)
    """
    w = max(2, window)
    denom = float(w - 1)
    
    pb = pl.col("pb1")
    pa = pl.col("pa1")
    mid = pl.col("_mid")
    
    bid_slope = ((pb - pb.shift(w - 1)) / (tick_size * denom)).fill_null(0).clip(-10, 10).alias("f_bid_slope")
    ask_slope = ((pa - pa.shift(w - 1)) / (tick_size * denom)).fill_null(0).clip(-10, 10).alias("f_ask_slope")
    mid_slope = ((mid - mid.shift(w - 1)) / (tick_size * denom)).fill_null(0).clip(-10, 10).alias("f_mid_slope")
    
    return df.with_columns([bid_slope, ask_slope, mid_slope])



@feature(deps={
    "base_prices",
    "spread_features",
    "return_features",
    "daily_return",
    "volatility_features",
    "momentum_features",
    "micro_price_features",
    "weighted_price_features",
    "range_features",
    "trend_features",
    "monotonicity_features",
    "fair_value_features",
})
def price_features_scaled(
    df: pl.DataFrame,
    tick_size: float,
    levels: int = 3,
    return_steps: list[int] = None,
    vol_windows: list[int] = None,
    trend_windows: list[int] = None,
) -> pl.DataFrame:
    """
    All price features with proper scaling.
    
    Args:
        tick_size: Minimum price increment for the asset
        levels: Number of LOB levels to use
        return_steps: Steps for return calculation
        vol_windows: Windows for volatility calculation
        trend_windows: Windows for trend calculation
    
    Returns:
        DataFrame with all price features:
        - f_spread_ticks, f_spread_bps
        - f_ret_1, f_ret_5, f_ret_10, f_ret_20, f_ret_50
        - f_daily_return
        - f_vol_10, f_vol_50, f_vol_200, f_vol_ratio
        - f_mom_10, f_mom_50
        - f_micro_off_hs, f_micro_off_ticks
        - f_bid_weighted_2/3, f_ask_weighted_2/3
        - f_range_ticks
        - f_trend_10, f_trend_20, f_trend_50
        - f_mono_10, f_mono_50
        - fair_value, f_dist_to_fv_ticks
    """
    # Base calculations
    df = base_prices(df, level=1)
    
    # Spread (tick scaled)
    df = spread_features(df, tick_size=tick_size)
    
    # Returns (percentage)
    df = return_features(df, steps=return_steps)
    df = daily_return(df)
    
    # Volatility
    df = volatility_features(df, windows=vol_windows)
    
    # Momentum (vol-adjusted)
    df = momentum_features(df)
    
    # Micro price
    df = micro_price_features(df, tick_size=tick_size)
    
    # Weighted prices (Senior's approach)
    df = weighted_price_features(df, levels=levels)
    
    # Range
    df = range_features(df, tick_size=tick_size)
    
    # Trend
    df = trend_features(df, windows=trend_windows)
    
    # Monotonicity
    df = monotonicity_features(df)
    
    # Fair value
    df = fair_value_features(df, tick_size=tick_size)
    
    # Drop internal columns
    internal_cols = ["_mid", "_spread"]
    df = df.drop([c for c in internal_cols if c in df.columns])
    
    return df
