"""
momentum_features.py - Price Momentum & Trend Features

Scaling Philosophy:
1. Price differences → scale by tick_size (ticks) or percentage
2. Oscillators → naturally bounded [-100, 100] or [-1, 1]
3. Trend strength → naturally bounded [-1, 1]

All features should be cross-asset comparable.
"""

import polars as pl
import numpy as np
from hft_features.core.base import feature


EPS = 1e-9

# Clipping bounds
MOMENTUM_TICKS_CLIP = 50.0    
PPO_CLIP = 10.0               
SLOPE_CLIP = 10.0 
            
# Windows
WINDOW_SHORT = 10
WINDOW_MED = 50
WINDOW_LONG = 100


@feature
def momentum_ticks(
    df: pl.DataFrame,
    tick_size: float,
    windows: list[int] = None,
    clip: float = MOMENTUM_TICKS_CLIP,
) -> pl.DataFrame:
    """
    Price momentum in ticks.
    
    momentum = (mid_now - mid_past) / tick_size
    
    Positive = price went up
    Negative = price went down
    
    Outputs:
        f_mom_ticks_10: 10-step momentum in ticks
        f_mom_ticks_50: 50-step momentum in ticks
    """
    if windows is None:
        windows = [WINDOW_SHORT, WINDOW_MED]
    
    mid = pl.col("mid_price")
    
    out = []
    for w in windows:
        mom = ((mid - mid.shift(w)) / tick_size).fill_null(0).clip(-clip, clip)
        out.append(mom.alias(f"f_mom_ticks_{w}"))
    
    return df.with_columns(out)


# ============================================================
# 2) MOMENTUM SLOPE (Ticks per Step)
# ============================================================

@feature
def momentum_slope(
    df: pl.DataFrame,
    tick_size: float,
    windows: list[int] = None,
    clip: float = SLOPE_CLIP,
) -> pl.DataFrame:
    """
    Price slope (momentum per step) in ticks.
    
    slope = (mid_now - mid_past) / (tick_size * steps)
    
    This is the "velocity" of price movement.
    
    Outputs:
        f_slope_5: 5-step slope (ticks/step)
        f_slope_10: 10-step slope (ticks/step)
    """
    if windows is None:
        windows = [5, WINDOW_SHORT]
    
    mid = pl.col("mid_price")
    
    out = []
    for w in windows:
        denom = tick_size * max(w - 1, 1)
        slope = ((mid - mid.shift(w - 1)) / denom).fill_null(0).clip(-clip, clip)
        out.append(slope.alias(f"f_slope_{w}"))
    
    return df.with_columns(out)




@feature
def ppo_features(
    df: pl.DataFrame,
    fast: int = 12,
    slow: int = 26,
    clip: float = PPO_CLIP,
) -> pl.DataFrame:
    """
    Percentage Price Oscillator.
    
    PPO = 100 * (EMA_fast - EMA_slow) / EMA_slow
    
    Similar to MACD but percentage-based (cross-asset comparable).
    
    Outputs:
        f_ppo: PPO value [-10, 10]
        f_ppo_signal: Signal line (EMA of PPO)
        f_ppo_hist: PPO - Signal (histogram)
    """
    mid = pl.col("mid_price")
    
    # EMA calculations
    alpha_fast = 2.0 / (fast + 1)
    alpha_slow = 2.0 / (slow + 1)
    
    ema_fast = mid.ewm_mean(alpha=alpha_fast, adjust=False)
    ema_slow = mid.ewm_mean(alpha=alpha_slow, adjust=False)
    
    # PPO
    ppo = (100 * (ema_fast - ema_slow) / (ema_slow + EPS)).clip(-clip, clip).alias("f_ppo")
    
    df = df.with_columns(ppo)
    
    # Signal line (9-period EMA of PPO)
    alpha_signal = 2.0 / 10
    ppo_signal = pl.col("f_ppo").ewm_mean(alpha=alpha_signal, adjust=False).alias("f_ppo_signal")
    
    df = df.with_columns(ppo_signal)
    
    # Histogram
    ppo_hist = (pl.col("f_ppo") - pl.col("f_ppo_signal")).alias("f_ppo_hist")
    
    return df.with_columns(ppo_hist)




@feature
def trend_strength(
    df: pl.DataFrame,
    windows: list[int] = None,
) -> pl.DataFrame:
    """
    Trend strength indicator.
    
    Measures the ratio of up moves to total moves.
    
    trend_strength = (avg_up - avg_down) / (avg_up + avg_down)
    
    +1 = All up moves (strong uptrend)
    -1 = All down moves (strong downtrend)
     0 = Equal up and down (no trend)
    
    Outputs:
        f_trend_strength_10: 10-step trend strength [-1, 1]
        f_trend_strength_50: 50-step trend strength [-1, 1]
    """
    if windows is None:
        windows = [WINDOW_SHORT, WINDOW_MED]
    
    mid = pl.col("mid_price")
    diff = mid.diff()
    
    # Up and down components
    up = diff.clip(lower_bound=0)
    dn = (-diff).clip(lower_bound=0)
    
    out = []
    for w in windows:
        avg_up = up.rolling_mean(w)
        avg_dn = dn.rolling_mean(w)
        
        ts = ((avg_up - avg_dn) / (avg_up + avg_dn + EPS)).alias(f"f_trend_strength_{w}")
        out.append(ts)
    
    return df.with_columns(out)



@feature
def direction_imbalance(
    df: pl.DataFrame,
    windows: list[int] = None,
) -> pl.DataFrame:
    """
    Imbalance between up and down moves (count-based).
    
    dir_imb = (up_count - down_count) / (up_count + down_count)
    
    This is similar to trend_strength but count-based, not magnitude-based.
    
    Outputs:
        f_dir_imb_10: 10-step direction imbalance [-1, 1]
        f_dir_imb_50: 50-step direction imbalance [-1, 1]
    """
    if windows is None:
        windows = [WINDOW_SHORT, WINDOW_MED]
    
    mid = pl.col("mid_price")
    
    # Direction indicators
    is_up = (mid > mid.shift(1)).cast(pl.Float64)
    is_dn = (mid < mid.shift(1)).cast(pl.Float64)
    
    out = []
    for w in windows:
        up_count = is_up.rolling_sum(w)
        dn_count = is_dn.rolling_sum(w)
        
        dir_imb = ((up_count - dn_count) / (up_count + dn_count + EPS)).alias(f"f_dir_imb_{w}")
        out.append(dir_imb)
    
    return df.with_columns(out)



@feature(deps={"momentum_ticks"})
def momentum_acceleration(
    df: pl.DataFrame,
    base_window: int = WINDOW_SHORT,
) -> pl.DataFrame:
    """
    Momentum acceleration (second derivative of price).
    
    Is momentum increasing or decreasing?
    
    Outputs:
        f_mom_accel: Change in momentum
        f_mom_accel_sign: Sign of acceleration (-1, 0, +1)
    """
    mom = pl.col(f"f_mom_ticks_{base_window}")
    
    # Acceleration = change in momentum
    accel = (mom - mom.shift(1)).fill_null(0).clip(-10, 10).alias("f_mom_accel")
    
    # Acceleration sign
    accel_sign = (
        pl.when(mom > mom.shift(1)).then(1)
          .when(mom < mom.shift(1)).then(-1)
          .otherwise(0)
    ).alias("f_mom_accel_sign")
    
    return df.with_columns([accel, accel_sign])




@feature(deps={"momentum_ticks"})
def momentum_divergence(df: pl.DataFrame) -> pl.DataFrame:
    """
    Divergence between short-term and long-term momentum.
    
    Positive divergence = short-term stronger than long-term (acceleration)
    Negative divergence = short-term weaker than long-term (deceleration)
    
    Outputs:
        f_mom_divergence: Short-term mom - Long-term mom
    """
    mom_short = pl.col(f"f_mom_ticks_{WINDOW_SHORT}")
    mom_long = pl.col(f"f_mom_ticks_{WINDOW_MED}")
    
    # Normalize by long-term magnitude for comparability
    divergence = (mom_short - mom_long).alias("f_mom_divergence")
    
    return df.with_columns(divergence)


@feature(deps={
    "momentum_ticks",
    "momentum_slope",
    "ppo_features",
    "trend_strength",
    "direction_imbalance",
    "momentum_acceleration",
    "momentum_divergence",
})
def momentum_features_scaled(
    df: pl.DataFrame,
    tick_size: float,
    mom_windows: list[int] = None,
    slope_windows: list[int] = None,
    trend_windows: list[int] = None,
    ppo_fast: int = 12,
    ppo_slow: int = 26,
) -> pl.DataFrame:

    # Momentum in ticks
    df = momentum_ticks(df, tick_size=tick_size, windows=mom_windows)
    
    # Slope
    df = momentum_slope(df, tick_size=tick_size, windows=slope_windows)
    
    # PPO
    df = ppo_features(df, fast=ppo_fast, slow=ppo_slow)
    
    # Trend strength
    df = trend_strength(df, windows=trend_windows)
    
    # Direction imbalance
    df = direction_imbalance(df, windows=trend_windows)
    
    # Momentum dynamics
    df = momentum_acceleration(df)
    df = momentum_divergence(df)
    
    return df
