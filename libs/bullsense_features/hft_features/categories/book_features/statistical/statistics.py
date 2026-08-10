

import polars as pl
import numpy as np
from hft_features.core.base import feature



EPS = 1e-9

# Clipping bounds
SKEW_CLIP = 3.0             # Skewness typically [-3, 3]
KURTOSIS_CLIP = 10.0        # Excess kurtosis, clip extreme values
ZSCORE_CLIP = 5.0           # Z-score clip
RANGE_TICKS_CLIP = 100.0    # Max 100 ticks range

# Windows
WINDOW_SHORT = 10
WINDOW_MED = 50
WINDOW_LONG = 100




@feature
def rolling_volatility(
    df: pl.DataFrame,
    windows: list[int] = None,
) -> pl.DataFrame:
    """    
    This is the standard deviation of percentage returns.
    Already cross-asset comparable since it's in percentage terms.
    
    Outputs:
        f_vol_10: 10-step volatility (std of returns)
        f_vol_50: 50-step volatility
        f_vol_100: 100-step volatility
        f_vol_ratio: Short-term / Long-term volatility
    """
    if windows is None:
        windows = [WINDOW_SHORT, WINDOW_MED, WINDOW_LONG]
    
    mid = pl.col("mid_price")
    
    # Calculate return first (percentage)
    ret = (mid / mid.shift(1) - 1) * 100  # Percentage return
    
    out = []
    for w in windows:
        vol = ret.rolling_std(w, min_periods=1).fill_null(0).alias(f"f_vol_{w}")
        out.append(vol)
    
    df = df.with_columns(out)
    
    # Volatility ratio (regime indicator)
    if len(windows) >= 2:
        vol_short = pl.col(f"f_vol_{windows[0]}")
        vol_long = pl.col(f"f_vol_{windows[-1]}")
        vol_ratio = (vol_short / (vol_long + EPS)).clip(0.1, 5.0).alias("f_vol_ratio")
        df = df.with_columns(vol_ratio)
    
    return df



@feature
def rolling_range_ticks(
    df: pl.DataFrame,
    tick_size: float,
    windows: list[int] = None,
    clip: float = RANGE_TICKS_CLIP,
) -> pl.DataFrame:
    """
    Rolling price range in TICKS (not raw price!).
    
    range_ticks = (rolling_max - rolling_min) / tick_size
    
    Cross-asset comparable: "How many ticks did price move?"
    
    Outputs:
        f_range_ticks_10: 10-step range in ticks
        f_range_ticks_50: 50-step range in ticks
    """
    if windows is None:
        windows = [WINDOW_SHORT, WINDOW_MED]
    
    mid = pl.col("mid_price")
    
    out = []
    for w in windows:
        rolling_max = mid.rolling_max(w, min_periods=1)
        rolling_min = mid.rolling_min(w, min_periods=1)
        
        range_ticks = ((rolling_max - rolling_min) / tick_size).clip(0, clip)
        out.append(range_ticks.alias(f"f_range_ticks_{w}"))
    
    return df.with_columns(out)



@feature
def price_position(
    df: pl.DataFrame,
    windows: list[int] = None,
) -> pl.DataFrame:
    """
    Where is current price within recent range?
    
    position = (mid - rolling_min) / (rolling_max - rolling_min)
    
    0 = At the bottom of range (oversold?)
    1 = At the top of range (overbought?)
    0.5 = Middle of range
    
    This is like a stochastic oscillator.
    
    Outputs:
        f_price_pos_10: Position within 10-step range [0, 1]
        f_price_pos_50: Position within 50-step range [0, 1]
    """
    if windows is None:
        windows = [WINDOW_SHORT, WINDOW_MED]
    
    mid = pl.col("mid_price")
    
    out = []
    for w in windows:
        rolling_max = mid.rolling_max(w, min_periods=1)
        rolling_min = mid.rolling_min(w, min_periods=1)
        range_size = rolling_max - rolling_min
        
        # Position within range [0, 1]
        pos = ((mid - rolling_min) / (range_size + EPS)).clip(0, 1)
        out.append(pos.alias(f"f_price_pos_{w}"))
    
    return df.with_columns(out)



@feature
def price_zscore(
    df: pl.DataFrame,
    windows: list[int] = None,
    clip: float = ZSCORE_CLIP,
) -> pl.DataFrame:
    """
    Z-score: How far is current price from rolling mean?
    
    zscore = (mid - rolling_mean) / rolling_std
    
    This is mean-reversion signal:
    - High positive = price above average (might revert down)
    - High negative = price below average (might revert up)
    
    Outputs:
        f_zscore_10: 10-step z-score [-5, 5]
        f_zscore_50: 50-step z-score [-5, 5]
    """
    if windows is None:
        windows = [WINDOW_SHORT, WINDOW_MED]
    
    mid = pl.col("mid_price")
    
    out = []
    for w in windows:
        rolling_mean = mid.rolling_mean(w, min_periods=1)
        rolling_std = mid.rolling_std(w, min_periods=1)
        
        zscore = ((mid - rolling_mean) / (rolling_std + EPS)).clip(-clip, clip)
        out.append(zscore.alias(f"f_zscore_{w}"))
    
    return df.with_columns(out)



@feature
def return_distribution(
    df: pl.DataFrame,
    window: int = WINDOW_MED,
    skew_clip: float = SKEW_CLIP,
    kurt_clip: float = KURTOSIS_CLIP,
) -> pl.DataFrame:
    """
    Distribution characteristics of recent RETURNS.
    
    Skewness:
    - Positive = More extreme up moves (right tail)
    - Negative = More extreme down moves (left tail)
    
    Kurtosis:
    - High = Fat tails (more extreme moves)
    - Low = Thin tails (normal-ish)
    
    Outputs:
        f_ret_skew: Skewness of returns [-3, 3]
        f_ret_kurtosis: Excess kurtosis of returns [0, 10]
    """
    mid = pl.col("mid_price")
    
    # Calculate return
    ret = (mid / mid.shift(1) - 1) * 100
    
    # Skewness
    ret_skew = ret.rolling_skew(window).fill_null(0).clip(-skew_clip, skew_clip).alias("f_ret_skew")
    
    # Kurtosis (excess kurtosis, so normal = 0)
    ret_kurt = ret.rolling_kurtosis(window).fill_null(0).clip(0, kurt_clip).alias("f_ret_kurtosis")
    
    return df.with_columns([ret_skew, ret_kurt])




@feature(deps={"rolling_volatility"})
def variance_ratio(
    df: pl.DataFrame,
) -> pl.DataFrame:
    """
    Variance ratio test for mean reversion vs momentum.
    
    VR = Var(k-period returns) / (k * Var(1-period returns))
    
    VR > 1: Momentum (trends persist)
    VR < 1: Mean reversion (trends reverse)
    VR = 1: Random walk
    
    Outputs:
        f_var_ratio: Variance ratio [0.5, 2.0]
    """
    # Use volatilities we already calculated
    vol_short = pl.col(f"f_vol_{WINDOW_SHORT}")
    vol_long = pl.col(f"f_vol_{WINDOW_MED}")
    
    # Simplified variance ratio approximation
    # True VR requires more complex calculation
    var_ratio = ((vol_long.pow(2)) / (vol_short.pow(2) * (WINDOW_MED / WINDOW_SHORT) + EPS))
    var_ratio = var_ratio.clip(0.2, 5.0).alias("f_var_ratio")
    
    return df.with_columns(var_ratio)


# ============================================================
# 7) JUMP DETECTION (Quarticity-based)
# ============================================================

@feature
def jump_indicator(
    df: pl.DataFrame,
    window: int = WINDOW_MED,
    threshold: float = 3.0,
) -> pl.DataFrame:
    """
    Jump/spike detection based on return magnitude.
    
    A "jump" is when |return| > threshold * rolling_std
    
    Outputs:
        f_jump_intensity: How many std is current return? [0, 5]
        f_jump_count: Rolling count of jumps [0, 1] (frequency)
    """
    mid = pl.col("mid_price")
    
    # Return and volatility
    ret = (mid / mid.shift(1) - 1) * 100
    vol = ret.rolling_std(window, min_periods=1)
    
    # Jump intensity (how many std?)
    jump_intensity = (ret.abs() / (vol + EPS)).clip(0, 5).alias("f_jump_intensity")
    
    # Jump indicator (binary)
    is_jump = (ret.abs() > threshold * vol).cast(pl.Float64)
    
    # Jump frequency
    jump_freq = is_jump.rolling_mean(window).alias("f_jump_freq")
    
    return df.with_columns([jump_intensity, jump_freq])


# ============================================================
# WRAPPER: ALL ROLLING STATS FEATURES
# ============================================================

@feature(deps={
    "rolling_volatility",
    "rolling_range_ticks",
    "price_position",
    "price_zscore",
    "return_distribution",
    "variance_ratio",
    "jump_indicator",
})
def rolling_stats_features_scaled(
    df: pl.DataFrame,
    tick_size: float,
    vol_windows: list[int] = None,
    range_windows: list[int] = None,
    zscore_windows: list[int] = None,
    dist_window: int = WINDOW_MED,
) -> pl.DataFrame:
    """
    All rolling statistical features with proper scaling.
    
    
    Args:
        tick_size: Minimum price increment
        vol_windows: Windows for volatility calculation
        range_windows: Windows for range calculation
        zscore_windows: Windows for z-score calculation
        dist_window: Window for distribution features
    
    Returns:
        DataFrame with all rolling stats features:
        
        Volatility:
        - f_vol_10, f_vol_50, f_vol_100
        - f_vol_ratio
        
        Range (Ticks):
        - f_range_ticks_10, f_range_ticks_50
        
        Price Position [0, 1]:
        - f_price_pos_10, f_price_pos_50
        
        Z-Score [-5, 5]:
        - f_zscore_10, f_zscore_50
        
        Return Distribution:
        - f_ret_skew [-3, 3]
        - f_ret_kurtosis [0, 10]
        
        Variance Ratio:
        - f_var_ratio [0.2, 5]
        
        Jump Detection:
        - f_jump_intensity [0, 5]
        - f_jump_freq [0, 1]
    """
    # Volatility
    df = rolling_volatility(df, windows=vol_windows)
    
    # Range in ticks
    df = rolling_range_ticks(df, tick_size=tick_size, windows=range_windows)
    
    # Price position
    df = price_position(df, windows=zscore_windows)
    
    # Z-score
    df = price_zscore(df, windows=zscore_windows)
    
    # Return distribution
    df = return_distribution(df, window=dist_window)
    
    # Variance ratio
    df = variance_ratio(df)
    
    # Jump detection
    df = jump_indicator(df, window=dist_window)
    
    return df
