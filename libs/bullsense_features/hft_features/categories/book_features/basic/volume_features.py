"""
volume_features.py - Properly Scaled Volume/Quantity Features for HFT

Scaling Philosophy (from Senior's approach):
1. Raw quantities → scale by average_liquidity
2. Depth shares → naturally bounded [0, 1]
3. Log quantities → centered by rolling mean
4. Pressure ratios → log transform for boundedness

Required Inputs:
- average_liquidity: Pre-computed average (bid_qty + ask_qty) / 2 from previous day
"""

import polars as pl
import numpy as np
from hft_features.core.base import feature

# ============================================================
# CONSTANTS
# ============================================================

EPS = 1e-9

# Clipping bounds
QTY_SCALED_CLIP = 10.0      # Max 10x average liquidity
PRESSURE_LOG_CLIP = 3.0     # log(ratio) clip
SHARE_MIN = 0.001           # Min share to avoid log(0)

# Windows
WINDOW_SHORT = 10
WINDOW_MED = 50
WINDOW_LONG = 200
RVOL_WINDOW = 600           # ~60 seconds at 100ms




@feature
def total_volumes(df: pl.DataFrame, levels: int = 5) -> pl.DataFrame:
    """
    Calculate total bid/ask volumes across levels.
    These are intermediate values for other features.
    
    Outputs:
        _total_bid: Total bid volume (internal)
        _total_ask: Total ask volume (internal)
        _total_depth: Total depth (internal)
    """
    bid_cols = [f"qb{i}" for i in range(1, levels + 1)]
    ask_cols = [f"qa{i}" for i in range(1, levels + 1)]
    
    total_bid = pl.sum_horizontal([pl.col(c) for c in bid_cols]).alias("_total_bid")
    total_ask = pl.sum_horizontal([pl.col(c) for c in ask_cols]).alias("_total_ask")
    
    df = df.with_columns([total_bid, total_ask])
    
    total_depth = (pl.col("_total_bid") + pl.col("_total_ask")).alias("_total_depth")
    
    return df.with_columns(total_depth)



@feature(deps={"total_volumes"})
def scaled_quantities(
    df: pl.DataFrame,
    average_liquidity: float,
    levels: int = 3,
    clip: float = QTY_SCALED_CLIP,
) -> pl.DataFrame:
    """
    Quantities scaled by average_liquidity .
    
    This is the KEY scaling for cross-asset comparability!
    
    Args:
        average_liquidity: Pre-computed average (qb1 + qa1) / 2 from previous day
    
    Outputs:
        f_bidqty_scaled1: qb1 / average_liquidity [0, 10]
        f_askqty_scaled1: qa1 / average_liquidity [0, 10]
        f_bidqty_scaled2: (qb1+qb2) / average_liquidity [0, 10]
        f_askqty_scaled2: (qa1+qa2) / average_liquidity [0, 10]
        f_bidqty_scaled3: (qb1+qb2+qb3) / average_liquidity [0, 10]
        f_askqty_scaled3: (qa1+qa2+qa3) / average_liquidity [0, 10]
        f_total_depth_scaled: total_depth / (2 * average_liquidity) [0, 10]
    """
    out = []
    
    # Level 1
    f_bid1 = (pl.col("qb1") / average_liquidity).clip(0, clip).alias("f_bidqty_scaled1")
    f_ask1 = (pl.col("qa1") / average_liquidity).clip(0, clip).alias("f_askqty_scaled1")
    out.extend([f_bid1, f_ask1])
    
    # Level 1+2
    if levels >= 2:
        bid2 = pl.col("qb1") + pl.col("qb2")
        ask2 = pl.col("qa1") + pl.col("qa2")
        f_bid2 = (bid2 / average_liquidity).clip(0, clip).alias("f_bidqty_scaled2")
        f_ask2 = (ask2 / average_liquidity).clip(0, clip).alias("f_askqty_scaled2")
        out.extend([f_bid2, f_ask2])
    
    # Level 1+2+3
    if levels >= 3:
        bid3 = pl.col("qb1") + pl.col("qb2") + pl.col("qb3")
        ask3 = pl.col("qa1") + pl.col("qa2") + pl.col("qa3")
        f_bid3 = (bid3 / average_liquidity).clip(0, clip).alias("f_bidqty_scaled3")
        f_ask3 = (ask3 / average_liquidity).clip(0, clip).alias("f_askqty_scaled3")
        out.extend([f_bid3, f_ask3])
    
    # Total depth scaled
    f_depth = (pl.col("_total_depth") / (2 * average_liquidity)).clip(0, clip).alias("f_total_depth_scaled")
    out.append(f_depth)
    
    return df.with_columns(out)



@feature
def log_quantities(
    df: pl.DataFrame,
    levels: int = 5,
    center_window: int = WINDOW_LONG,
) -> pl.DataFrame:
    """
    Log-transformed quantities, centered by rolling mean.
        
    Centering: log(qty) - rolling_mean(log(qty))
    This makes it ~0-centered and cross-asset comparable.
    
    Outputs:
        f_qb1_log, f_qa1_log, ..., f_qb5_log, f_qa5_log
    """
    out = []
    
    for i in range(1, levels + 1):
        # Log transform
        qb_log = (pl.col(f"qb{i}") + 1).log()
        qa_log = (pl.col(f"qa{i}") + 1).log()
        
        # Center by rolling mean (makes it ~0 centered)
        qb_centered = (qb_log - qb_log.rolling_mean(center_window)).alias(f"f_qb{i}_log")
        qa_centered = (qa_log - qa_log.rolling_mean(center_window)).alias(f"f_qa{i}_log")
        
        out.extend([qb_centered, qa_centered])
    
    return df.with_columns(out)




@feature(deps={"total_volumes"})
def depth_shares(df: pl.DataFrame, levels: int = 3) -> pl.DataFrame:
    """
    Each level's share of total depth.
    
    Naturally bounded [0, 1].
    Shows concentration of liquidity.
    
    Outputs:
        f_qb1_share, f_qa1_share, ...: Level share of own side
        f_qb1_depth_share, f_qa1_depth_share, ...: Level share of total depth
    """
    out = []
    
    for i in range(1, levels + 1):
        qb = pl.col(f"qb{i}")
        qa = pl.col(f"qa{i}")
        
        # Share of own side
        qb_share = (qb / (pl.col("_total_bid") + EPS)).alias(f"f_qb{i}_share")
        qa_share = (qa / (pl.col("_total_ask") + EPS)).alias(f"f_qa{i}_share")
        
        # Share of total depth
        qb_depth_share = (qb / (pl.col("_total_depth") + EPS)).alias(f"f_qb{i}_depth_share")
        qa_depth_share = (qa / (pl.col("_total_depth") + EPS)).alias(f"f_qa{i}_depth_share")
        
        out.extend([qb_share, qa_share, qb_depth_share, qa_depth_share])
    
    return df.with_columns(out)


# ============================================================
# 5) PRESSURE FEATURES (XGBoost #1 Feature!)
# ============================================================

@feature
def pressure_features(
    df: pl.DataFrame,
    levels: int = 3,
    clip: float = PRESSURE_LOG_CLIP,
) -> pl.DataFrame:
    """    
    
    Raw pressure = qb / qa (unbounded)
    Log pressure = log(qb / qa) (bounded, symmetric)
    
    Outputs:
        f_pressure_1: log(qb1/qa1) [-3, 3]
        f_pressure_2: log(qb2/qa2) [-3, 3]
        f_pressure_3: log(qb3/qa3) [-3, 3]
        f_pressure_12: log((qb1+qb2)/(qa1+qa2)) [-3, 3]
        f_pressure_123: log((qb1+qb2+qb3)/(qa1+qa2+qa3)) [-3, 3]
    """
    out = []
    
    # Level-specific pressure
    for i in range(1, levels + 1):
        qb = pl.col(f"qb{i}")
        qa = pl.col(f"qa{i}")
        
        # Log pressure (bounded, symmetric around 0)
        pressure_log = (qb / (qa + EPS)).log().clip(-clip, clip).alias(f"f_pressure_{i}")
        out.append(pressure_log)
    
    # Cumulative pressure
    if levels >= 2:
        qb_12 = pl.col("qb1") + pl.col("qb2")
        qa_12 = pl.col("qa1") + pl.col("qa2")
        pressure_12 = (qb_12 / (qa_12 + EPS)).log().clip(-clip, clip).alias("f_pressure_12")
        out.append(pressure_12)
    
    if levels >= 3:
        qb_123 = pl.col("qb1") + pl.col("qb2") + pl.col("qb3")
        qa_123 = pl.col("qa1") + pl.col("qa2") + pl.col("qa3")
        pressure_123 = (qb_123 / (qa_123 + EPS)).log().clip(-clip, clip).alias("f_pressure_123")
        out.append(pressure_123)
    
    return df.with_columns(out)




@feature(deps={"total_volumes"})
def relative_volume(
    df: pl.DataFrame,
    window: int = RVOL_WINDOW,
    clip: float = 5.0,
) -> pl.DataFrame:
    """
    Relative volume = current / rolling_mean.
    
    Shows if current liquidity is normal or unusual.
    
    Outputs:
        f_rvol_bid: Relative bid volume [0.2, 5]
        f_rvol_ask: Relative ask volume [0.2, 5]
        f_rvol_depth: Relative total depth [0.2, 5]
    """
    total_bid = pl.col("_total_bid")
    total_ask = pl.col("_total_ask")
    total_depth = pl.col("_total_depth")
    
    rvol_bid = (total_bid / (total_bid.rolling_mean(window) + EPS)).clip(0.1, clip).alias("f_rvol_bid")
    rvol_ask = (total_ask / (total_ask.rolling_mean(window) + EPS)).clip(0.1, clip).alias("f_rvol_ask")
    rvol_depth = (total_depth / (total_depth.rolling_mean(window) + EPS)).clip(0.1, clip).alias("f_rvol_depth")
    
    return df.with_columns([rvol_bid, rvol_ask, rvol_depth])


@feature
def quantity_changes(
    df: pl.DataFrame,
    average_liquidity: float,
    levels: int = 3,
    clip: float = 2.0,
) -> pl.DataFrame:
    """
    Quantity changes scaled by average_liquidity.
    
    Shows order additions/cancellations relative to typical size.
    
    Outputs:
        f_qb1_change, f_qa1_change, ...: Scaled quantity changes [-2, 2]
    """
    out = []
    
    for i in range(1, levels + 1):
        qb = pl.col(f"qb{i}")
        qa = pl.col(f"qa{i}")
        
        # Raw change
        qb_change = qb - qb.shift(1)
        qa_change = qa - qa.shift(1)
        
        # Scaled by average_liquidity
        qb_change_scaled = (qb_change / average_liquidity).clip(-clip, clip).alias(f"f_qb{i}_change")
        qa_change_scaled = (qa_change / average_liquidity).clip(-clip, clip).alias(f"f_qa{i}_change")
        
        out.extend([qb_change_scaled, qa_change_scaled])
    
    return df.with_columns(out)





# ============================================================
# WRAPPER: ALL VOLUME FEATURES
# ============================================================

@feature(deps={
    "total_volumes",
    "scaled_quantities",
    "log_quantities",
    "depth_shares",
    "pressure_features",
    "relative_volume",
    "quantity_changes",
    "depth_profile",
})
def volume_features_scaled(
    df: pl.DataFrame,
    average_liquidity: float,
    levels: int = 5,
    rvol_window: int = RVOL_WINDOW,
) -> pl.DataFrame:
    """
    All volume/quantity features with proper scaling.
    
    Args:
        average_liquidity: Pre-computed average (qb1 + qa1) / 2 from previous day
        levels: Number of LOB levels to use
        rvol_window: Window for relative volume calculation
    
    Returns:
        DataFrame with all volume features:
        
        Scaled Quantities (by average_liquidity):
        - f_bidqty_scaled1/2/3, f_askqty_scaled1/2/3
        - f_total_depth_scaled
        
        Log Quantities (centered):
        - f_qb1_log, f_qa1_log, ..., f_qb5_log, f_qa5_log
        
        Depth Shares [0, 1]:
        - f_qb1_share, f_qa1_share, ...
        - f_qb1_depth_share, f_qa1_depth_share, ...
        
        Pressure (log-scaled) [-3, 3]:
        - f_pressure_1, f_pressure_2, f_pressure_3
        - f_pressure_12, f_pressure_123
        
        Relative Volume [0.2, 5]:
        - f_rvol_bid, f_rvol_ask, f_rvol_depth
        
        Quantity Changes [-2, 2]:
        - f_qb1_change, f_qa1_change, ...
        
        Depth Profile [0, 1]:
        - f_bid_concentration, f_ask_concentration
        - f_depth_hhi
    """
    # Base totals (internal)
    df = total_volumes(df, levels=levels)
    df = scaled_quantities(df, average_liquidity=average_liquidity, levels=min(levels, 3))
    df = log_quantities(df, levels=levels)
    df = depth_shares(df, levels=min(levels, 3))
    df = pressure_features(df, levels=min(levels, 3))
    df = relative_volume(df, window=rvol_window)
    df = quantity_changes(df, average_liquidity=average_liquidity, levels=min(levels, 3))
    
    
    internal_cols = ["_total_bid", "_total_ask", "_total_depth"]
    df = df.drop([c for c in internal_cols if c in df.columns])
    
    return df
