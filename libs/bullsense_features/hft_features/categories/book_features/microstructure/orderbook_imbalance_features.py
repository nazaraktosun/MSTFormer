"""
order_imbalance_features.py - Order Book Imbalance & Pressure Features

Features are naturally bounded [-1, 1] or [0, 1].

Key additions based on XGBoost results:
- pressure_1/2/3 (log-scaled bid/ask ratio) - MOST IMPORTANT!
- Level-specific imbalances (imb_1, imb_2, imb_3)
- Imbalance lags and rolling means
"""

import polars as pl
import numpy as np
from hft_features.core.base import feature

# ============================================================
# CONSTANTS
# ============================================================

EPS = 1e-9

# Clipping
PRESSURE_CLIP = 3.0     # log(ratio) clip for pressure
ZSCORE_CLIP = 5.0

# Windows
WINDOW_SHORT = 10
WINDOW_MED = 50
WINDOW_LONG = 100


# ============================================================
# 1) LEVEL-SPECIFIC IMBALANCE (imb_1, imb_2, imb_3)
# ============================================================

@feature
def imbalance_per_level(df: pl.DataFrame, levels: int = 5) -> pl.DataFrame:
    """
    Imbalance at each individual level.
    
    imb_i = (qb_i - qa_i) / (qb_i + qa_i)
        
    Outputs:
        f_imb_1, f_imb_2, ..., f_imb_5: Level-specific imbalance [-1, 1]
    """
    out = []
    
    for i in range(1, levels + 1):
        qb = pl.col(f"qb{i}")
        qa = pl.col(f"qa{i}")
        denom = qb + qa
        
        imb = (
            pl.when(denom > 0)
              .then((qb - qa) / denom)
              .otherwise(0.0)
              .alias(f"f_imb_{i}")
        )
        out.append(imb)
    
    return df.with_columns(out)


# ============================================================
# 2) CUMULATIVE IMBALANCE (OBI - Order Book Imbalance)
# ============================================================

@feature
def obi_cumulative(df: pl.DataFrame, max_level: int = 3) -> pl.DataFrame:
    """
    Cumulative Order Book Imbalance across levels.
    
    f_obi_k = (sum_{i=1..k} qb_i - sum_{i=1..k} qa_i) / (sum + sum)
    
    Outputs:
        f_obi_1: Level 1 only (same as f_imb_1)
        f_obi_2: Levels 1+2 combined
        f_obi_3: Levels 1+2+3 combined
    """
    out = []
    
    for k in range(1, max_level + 1):
        bid_sum = pl.sum_horizontal([pl.col(f"qb{i}") for i in range(1, k + 1)])
        ask_sum = pl.sum_horizontal([pl.col(f"qa{i}") for i in range(1, k + 1)])
        denom = bid_sum + ask_sum
        
        obi = (
            pl.when(denom > 0)
              .then((bid_sum - ask_sum) / denom)
              .otherwise(0.0)
              .alias(f"f_obi_{k}")
        )
        out.append(obi)
    
    return df.with_columns(out)


# ============================================================
# 3) PRESSURE FEATURES (Most Important from XGBoost!)
# ============================================================

@feature
def pressure_features(df: pl.DataFrame, levels: int = 3) -> pl.DataFrame:
    """
    Bid/Ask pressure ratio - THE MOST IMPORTANT FEATURE!
    
    XGBoost showed pressure_1 has 15.52% importance!
    
    pressure = log(qb / qa)
    - Positive: More bid pressure (buyers)
    - Negative: More ask pressure (sellers)
    
    Log transform makes it:
    - Symmetric around 0
    - Bounded with clipping
    - More normally distributed
    
    Outputs:
        f_pressure_1: log(qb1/qa1) [-3, 3]
        f_pressure_2: log(qb2/qa2) [-3, 3]
        f_pressure_3: log(qb3/qa3) [-3, 3]
        f_pressure_cum_2: log((qb1+qb2)/(qa1+qa2)) [-3, 3]
        f_pressure_cum_3: log((qb1+qb2+qb3)/(qa1+qa2+qa3)) [-3, 3]
    """
    out = []
    
    # Level-specific pressure
    for i in range(1, levels + 1):
        qb = pl.col(f"qb{i}")
        qa = pl.col(f"qa{i}")
        
        # Log pressure (bounded, symmetric)
        pressure = ((qb + EPS) / (qa + EPS)).log().clip(-PRESSURE_CLIP, PRESSURE_CLIP)
        out.append(pressure.alias(f"f_pressure_{i}"))
    
    # Cumulative pressure
    for k in [2, 3]:
        if k <= levels:
            qb_sum = pl.sum_horizontal([pl.col(f"qb{i}") for i in range(1, k + 1)])
            qa_sum = pl.sum_horizontal([pl.col(f"qa{i}") for i in range(1, k + 1)])
            
            pressure_cum = ((qb_sum + EPS) / (qa_sum + EPS)).log().clip(-PRESSURE_CLIP, PRESSURE_CLIP)
            out.append(pressure_cum.alias(f"f_pressure_cum_{k}"))
    
    return df.with_columns(out)


# ============================================================
# 4) WEIGHTED IMBALANCE (Inner levels weighted more)
# ============================================================

@feature
def weighted_imbalance(df: pl.DataFrame, levels: int = 3) -> pl.DataFrame:
    """
    Depth imbalance weighted by level (inner levels more important).
    
    weight_i = levels + 1 - i
    So L1 has weight=levels, L2 has weight=levels-1, etc.
    
    Output:
        f_weighted_imb: Weighted imbalance [-1, 1]
    """
    bid_w = pl.sum_horizontal([
        pl.col(f"qb{i}") * (levels + 1 - i)
        for i in range(1, levels + 1)
    ])
    ask_w = pl.sum_horizontal([
        pl.col(f"qa{i}") * (levels + 1 - i)
        for i in range(1, levels + 1)
    ])
    denom = bid_w + ask_w
    
    weighted_imb = (
        pl.when(denom > 0)
          .then((bid_w - ask_w) / denom)
          .otherwise(0.0)
          .alias("f_weighted_imb")
    )
    
    return df.with_columns(weighted_imb)


# ============================================================
# 5) IMBALANCE DYNAMICS (EMA, Lags, Rolling)
# ============================================================

@feature(deps={"imbalance_per_level"})
def imbalance_ema(df: pl.DataFrame, span: int = 10) -> pl.DataFrame:
    """
    EMA of imbalance and delta from EMA.
    
    Outputs:
        f_imb_ema: EMA of f_imb_1 [-1, 1]
        f_imb_ema_delta: f_imb_1 - f_imb_ema [-2, 2]
    """
    alpha = 2.0 / (span + 1.0)
    
    imb = pl.col("f_imb_1")
    imb_ema = imb.ewm_mean(alpha=alpha, adjust=False).alias("f_imb_ema")
    
    df = df.with_columns(imb_ema)
    
    imb_delta = (pl.col("f_imb_1") - pl.col("f_imb_ema")).alias("f_imb_ema_delta")
    
    return df.with_columns(imb_delta)


@feature(deps={"imbalance_per_level"})
def imbalance_lags(df: pl.DataFrame, lags: list[int] = None) -> pl.DataFrame:
    """
    Lagged imbalance values.
    
    Past imbalance is predictive of future price moves.
    
    Outputs:
        f_imb_lag_1, f_imb_lag_2, f_imb_lag_5, f_imb_lag_10
    """
    if lags is None:
        lags = [1, 2, 5, 10]
    
    out = []
    for lag in lags:
        lagged = pl.col("f_imb_1").shift(lag).alias(f"f_imb_lag_{lag}")
        out.append(lagged)
    
    return df.with_columns(out)


@feature(deps={"imbalance_per_level"})
def imbalance_rolling(df: pl.DataFrame, windows: list[int] = None) -> pl.DataFrame:
    """
    Rolling statistics of imbalance.
    
    Outputs:
        f_imb_ma_10/50/100: Rolling mean of imbalance
        f_imb_std_50: Rolling std of imbalance (volatility of imbalance)
    """
    if windows is None:
        windows = [WINDOW_SHORT, WINDOW_MED, WINDOW_LONG]
    
    imb = pl.col("f_imb_1")
    
    out = []
    for w in windows:
        ma = imb.rolling_mean(w, min_periods=1).alias(f"f_imb_ma_{w}")
        out.append(ma)
    
    # Imbalance volatility (using medium window)
    imb_std = imb.rolling_std(WINDOW_MED, min_periods=1).alias(f"f_imb_std_{WINDOW_MED}")
    out.append(imb_std)
    
    return df.with_columns(out)


# ============================================================
# 6) IMBALANCE GRADIENT (How imbalance changes across levels)
# ============================================================

@feature(deps={"imbalance_per_level"})
def imbalance_gradient(df: pl.DataFrame, levels: int = 3) -> pl.DataFrame:
    """
    Gradient of imbalance across levels.
    
    Shows if imbalance is increasing or decreasing as we go deeper in the book.
    
    Outputs:
        f_imb_grad: (imb_outer - imb_inner) / (levels - 1)
        f_imb_slope: Linear regression slope of imbalance across levels
    """
    # Simple gradient: outer - inner
    imb_inner = pl.col("f_imb_1")
    imb_outer = pl.col(f"f_imb_{levels}")
    
    grad = ((imb_outer - imb_inner) / (levels - 1)).alias("f_imb_grad")
    
    return df.with_columns(grad)


# ============================================================
# 7) DEPTH CONCENTRATION (Where is liquidity?)
# ============================================================

@feature
def depth_concentration(df: pl.DataFrame, levels: int = 5) -> pl.DataFrame:
    """
    Measures how concentrated liquidity is at top of book.
    
    concentration = L1_qty / total_qty
    
    High concentration = liquidity at top (tight market)
    Low concentration = liquidity spread out (deep market)
    
    Outputs:
        f_bid_concentration: qb1 / sum(qb) [0, 1]
        f_ask_concentration: qa1 / sum(qa) [0, 1]
        f_concentration_imb: bid_conc - ask_conc [-1, 1]
    """
    total_bid = pl.sum_horizontal([pl.col(f"qb{i}") for i in range(1, levels + 1)])
    total_ask = pl.sum_horizontal([pl.col(f"qa{i}") for i in range(1, levels + 1)])
    
    bid_conc = (pl.col("qb1") / (total_bid + EPS)).alias("f_bid_concentration")
    ask_conc = (pl.col("qa1") / (total_ask + EPS)).alias("f_ask_concentration")
    
    df = df.with_columns([bid_conc, ask_conc])
    
    # Concentration imbalance
    conc_imb = (pl.col("f_bid_concentration") - pl.col("f_ask_concentration")).alias("f_concentration_imb")
    
    return df.with_columns(conc_imb)


# ============================================================
# 8) RELATIVE DEPTH (Bid vs Ask total)
# ============================================================

@feature
def relative_depth(df: pl.DataFrame, levels: int = 3) -> pl.DataFrame:
    """
    Relative depth between bid and ask sides.
    
    Outputs:
        f_depth_ratio_log: log(total_bid / total_ask) [-3, 3]
        f_bid_depth_share: total_bid / (total_bid + total_ask) [0, 1]
    """
    total_bid = pl.sum_horizontal([pl.col(f"qb{i}") for i in range(1, levels + 1)])
    total_ask = pl.sum_horizontal([pl.col(f"qa{i}") for i in range(1, levels + 1)])
    
    # Log ratio (bounded)
    depth_ratio_log = ((total_bid + EPS) / (total_ask + EPS)).log().clip(-PRESSURE_CLIP, PRESSURE_CLIP)
    
    # Bid share
    bid_share = total_bid / (total_bid + total_ask + EPS)
    
    return df.with_columns([
        depth_ratio_log.alias("f_depth_ratio_log"),
        bid_share.alias("f_bid_depth_share"),
    ])


# ============================================================
# WRAPPER: ALL ORDER IMBALANCE FEATURES
# ============================================================

@feature(deps={
    "imbalance_per_level",
    "obi_cumulative",
    "pressure_features",
    "weighted_imbalance",
    "imbalance_ema",
    "imbalance_lags",
    "imbalance_rolling",
    "imbalance_gradient",
    "depth_concentration",
    "relative_depth",
})
def order_imbalance_features_scaled(
    df: pl.DataFrame,
    levels: int = 5,
    ema_span: int = 10,
    lag_steps: list[int] = None,
    rolling_windows: list[int] = None,
) -> pl.DataFrame:
    """
    All order book imbalance features with proper scaling.
    
    All features are naturally bounded (no external scaling needed).
    
    Args:
        levels: Number of LOB levels to use
        ema_span: Span for EMA calculation
        lag_steps: Lag steps for imbalance lags
        rolling_windows: Windows for rolling statistics
    
    Returns:
        DataFrame with all imbalance features:
        
        Level-Specific Imbalance [-1, 1]:
        - f_imb_1, f_imb_2, f_imb_3, f_imb_4, f_imb_5
        
        Cumulative Imbalance (OBI) [-1, 1]:
        - f_obi_1, f_obi_2, f_obi_3
        
        Pressure (Log Ratio) [-3, 3]:
        - f_pressure_1, f_pressure_2, f_pressure_3
        - f_pressure_cum_2, f_pressure_cum_3
        
        Weighted Imbalance [-1, 1]:
        - f_weighted_imb
        
        Imbalance Dynamics:
        - f_imb_ema, f_imb_ema_delta
        - f_imb_lag_1/2/5/10
        - f_imb_ma_10/50/100, f_imb_std_50
        
        Imbalance Gradient:
        - f_imb_grad
        
        Depth Concentration [0, 1]:
        - f_bid_concentration, f_ask_concentration
        - f_concentration_imb [-1, 1]
        
        Relative Depth:
        - f_depth_ratio_log [-3, 3]
        - f_bid_depth_share [0, 1]
    """
    # Level-specific imbalance
    df = imbalance_per_level(df, levels=levels)
    
    # Cumulative OBI
    df = obi_cumulative(df, max_level=min(levels, 3))
    
    # Pressure (MOST IMPORTANT!)
    df = pressure_features(df, levels=min(levels, 3))
    
    # Weighted imbalance
    df = weighted_imbalance(df, levels=min(levels, 3))
    
    # Imbalance dynamics
    df = imbalance_ema(df, span=ema_span)
    df = imbalance_lags(df, lags=lag_steps)
    df = imbalance_rolling(df, windows=rolling_windows)
    
    # Imbalance gradient
    df = imbalance_gradient(df, levels=min(levels, 3))
    
    # Depth metrics
    df = depth_concentration(df, levels=levels)
    df = relative_depth(df, levels=min(levels, 3))
    
    return df
