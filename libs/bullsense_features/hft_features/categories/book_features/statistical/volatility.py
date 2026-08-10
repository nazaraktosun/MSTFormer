"""
volatility_features.py - Advanced Volatility Features

Basic volatility (rolling std) is in rolling_stats_features.py.
This file contains ADVANCED volatility features:
- EWM volatility
- Volatility of volatility
- Volatility regime detection
- Parkinson/Garman-Klass estimators (if OHLC available)

All features properly scaled.
"""

import polars as pl
import numpy as np
from hft_features.core.base import feature

# ============================================================
# CONSTANTS
# ============================================================

EPS = 1e-9

# Scaling: Convert log returns to percentage/BPS for readability
LOG_TO_PCT = 100.0      # Multiply log return by 100 for percentage
LOG_TO_BPS = 10000.0    # Multiply log return by 10000 for BPS

# Windows
WINDOW_SHORT = 10
WINDOW_MED = 50
WINDOW_LONG = 100

# Clipping
VOL_CLIP = 5.0          # Max 5% volatility
VOL_RATIO_CLIP = 5.0    # Max 5x vol ratio


# ============================================================
# 1) LOG RETURN (Base - Scaled to Percentage)
# ============================================================

@feature
def log_return_pct(df: pl.DataFrame) -> pl.DataFrame:
    """
    Log return scaled to percentage.
    
    log_return_pct = log(mid / mid_prev) * 100
    
    This makes values more readable (~0.1% instead of 0.001).
    
    Output:
        f_log_ret: Log return in percentage [-5, 5]
    """
    mid = pl.col("mid_price")
    
    log_ret = ((mid / mid.shift(1)).log() * LOG_TO_PCT).clip(-5, 5).alias("f_log_ret")
    
    return df.with_columns(log_ret)


# ============================================================
# 2) EWM VOLATILITY (Exponentially Weighted)
# ============================================================

@feature(deps={"log_return_pct"})
def ewm_volatility(
    df: pl.DataFrame,
    spans: list[int] = None,
    clip: float = VOL_CLIP,
) -> pl.DataFrame:
    """
    Exponentially weighted moving volatility.
    
    EWM gives more weight to recent observations.
    Better for detecting volatility regime changes.
    
    Outputs:
        f_ewm_vol_10: EWM volatility (span=10)
        f_ewm_vol_50: EWM volatility (span=50)
    """
    if spans is None:
        spans = [WINDOW_SHORT, WINDOW_MED]
    
    ret = pl.col("f_log_ret")
    
    out = []
    for span in spans:
        alpha = 2.0 / (span + 1)
        
        # EWM mean
        ewm_mean = ret.ewm_mean(alpha=alpha, adjust=False)
        
        # EWM variance (manual calculation)
        squared_diff = (ret - ewm_mean).pow(2)
        ewm_var = squared_diff.ewm_mean(alpha=alpha, adjust=False)
        
        # EWM volatility (std)
        ewm_vol = ewm_var.sqrt().clip(0, clip).alias(f"f_ewm_vol_{span}")
        out.append(ewm_vol)
    
    return df.with_columns(out)


# ============================================================
# 3) ABSOLUTE RETURN (Alternative Vol Measure)
# ============================================================

@feature(deps={"log_return_pct"})
def absolute_return_features(
    df: pl.DataFrame,
    windows: list[int] = None,
    clip: float = VOL_CLIP,
) -> pl.DataFrame:
    """
    Rolling mean of absolute returns.
    
    This is a robust volatility measure (less sensitive to outliers than std).
    
    Outputs:
        f_abs_ret_10: Mean absolute return (10-step)
        f_abs_ret_50: Mean absolute return (50-step)
    """
    if windows is None:
        windows = [WINDOW_SHORT, WINDOW_MED]
    
    ret = pl.col("f_log_ret")
    
    out = []
    for w in windows:
        abs_ret = ret.abs().rolling_mean(w, min_periods=1).clip(0, clip).alias(f"f_abs_ret_{w}")
        out.append(abs_ret)
    
    return df.with_columns(out)


# ============================================================
# 4) VOLATILITY OF VOLATILITY (Vol Clustering)
# ============================================================

@feature(deps={"log_return_pct"})
def vol_of_vol(
    df: pl.DataFrame,
    inner_window: int = WINDOW_SHORT,
    outer_window: int = WINDOW_MED,
) -> pl.DataFrame:
    """
    Volatility of volatility (vol clustering indicator).
    
    High vol-of-vol = Volatility is unstable (regime changes)
    Low vol-of-vol = Volatility is stable
    
    Output:
        f_vol_of_vol: Std of rolling volatility
    """
    ret = pl.col("f_log_ret")
    
    # Inner volatility
    inner_vol = ret.rolling_std(inner_window, min_periods=1)
    
    # Volatility of volatility
    vol_of_vol = inner_vol.rolling_std(outer_window, min_periods=1).fill_null(0).alias("f_vol_of_vol")
    
    return df.with_columns(vol_of_vol)





# ============================================================
# 6) REALIZED VOLATILITY (Sum of Squared Returns)
# ============================================================

@feature(deps={"log_return_pct"})
def realized_volatility(
    df: pl.DataFrame,
    windows: list[int] = None,
) -> pl.DataFrame:
    """
    Realized volatility (sum of squared returns).
    
    RV = sqrt(sum(ret^2))
    
    This is the standard measure in academic literature.
    
    Outputs:
        f_rv_10: Realized volatility (10-step)
        f_rv_50: Realized volatility (50-step)
    """
    if windows is None:
        windows = [WINDOW_SHORT, WINDOW_MED]
    
    ret = pl.col("f_log_ret")
    ret_sq = ret.pow(2)
    
    out = []
    for w in windows:
        rv = ret_sq.rolling_sum(w, min_periods=1).sqrt().alias(f"f_rv_{w}")
        out.append(rv)
    
    return df.with_columns(out)


# ============================================================
# 7) RETURN ASYMMETRY (Up vs Down Volatility)
# ============================================================

@feature(deps={"log_return_pct"})
def return_asymmetry(
    df: pl.DataFrame,
    window: int = WINDOW_MED,
) -> pl.DataFrame:
    """
    Asymmetry between up and down move volatility.
    
    up_vol = std of positive returns
    down_vol = std of negative returns
    asymmetry = (up_vol - down_vol) / (up_vol + down_vol)
    
    Positive = Up moves more volatile
    Negative = Down moves more volatile (fear/panic)
    
    Output:
        f_vol_asymmetry: [-1, 1]
    """
    ret = pl.col("f_log_ret")
    
    # Up and down returns
    up_ret = pl.when(ret > 0).then(ret).otherwise(0)
    dn_ret = pl.when(ret < 0).then(ret.abs()).otherwise(0)
    
    # Rolling std of each
    up_vol = up_ret.rolling_std(window, min_periods=1).fill_null(0)
    dn_vol = dn_ret.rolling_std(window, min_periods=1).fill_null(0)
    
    # Asymmetry
    asymmetry = ((up_vol - dn_vol) / (up_vol + dn_vol + EPS)).clip(-1, 1).alias("f_vol_asymmetry")
    
    return df.with_columns(asymmetry)


# ============================================================
# WRAPPER: ALL VOLATILITY FEATURES
# ============================================================

@feature(deps={
    "log_return_pct",
    "ewm_volatility",
    "absolute_return_features",
    "vol_of_vol",
    "vol_regime",
    "realized_volatility",
    "return_asymmetry",
})
def volatility_features_scaled(
    df: pl.DataFrame,
    ewm_spans: list[int] = None,
    abs_ret_windows: list[int] = None,
    rv_windows: list[int] = None,
) -> pl.DataFrame:
    """
    All advanced volatility features with proper scaling.
    
    Note: Basic volatility (rolling std) is in rolling_stats_features.py.
    This file contains advanced/specialized volatility features.
    
    Returns:
        DataFrame with all volatility features:
        
        Base:
        - f_log_ret: Log return in % [-5, 5]
        
        EWM Volatility:
        - f_ewm_vol_10, f_ewm_vol_50
        
        Absolute Return:
        - f_abs_ret_10, f_abs_ret_50
        
        Vol of Vol:
        - f_vol_of_vol
        
        Vol Regime:
        - f_vol_regime [0.2, 5]
        - f_vol_percentile [0, 1]
        
        Realized Volatility:
        - f_rv_10, f_rv_50
        
        Return Asymmetry:
        - f_vol_asymmetry [-1, 1]
    """
    # Base log return
    df = log_return_pct(df)
    
    # EWM volatility
    df = ewm_volatility(df, spans=ewm_spans)
    
    # Absolute return
    df = absolute_return_features(df, windows=abs_ret_windows)
    
    # Vol of vol
    df = vol_of_vol(df)
    
    # Realized volatility
    df = realized_volatility(df, windows=rv_windows)
    
    # Return asymmetry
    df = return_asymmetry(df)
    
    return df
