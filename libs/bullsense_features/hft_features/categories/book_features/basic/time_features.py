"""
temporal_features.py - Time-Based Features (Properly Scaled)

All features bounded and cross-asset comparable.
"""

import polars as pl
import numpy as np
from hft_features.core.base import feature

EPS = 1e-12


def _time_to_secs(hhmmss: str) -> int:
    h, m, s = hhmmss.split(":")
    return int(h) * 3600 + int(m) * 60 + int(s)


# ============================================================
# 1) TIME OF DAY (Cyclical Encoding)
# ============================================================

@feature
def time_of_day_cyclical(
    df: pl.DataFrame,
    datetime_col: str = "datetime",
    session_hours: float = 8.0,  # 8 saat seans (10:00-18:00)
) -> pl.DataFrame:
    """
    Cyclical encoding of time of day using sin/cos.
    
    Uses session hours (not 24h) for better resolution.
    
    Outputs:
        f_tod_sin: Sin component [-1, 1]
        f_tod_cos: Cos component [-1, 1]
    """
    secs = (
        pl.col(datetime_col).dt.hour() * 3600
        + pl.col(datetime_col).dt.minute() * 60
        + pl.col(datetime_col).dt.second()
        + (pl.col(datetime_col).dt.nanosecond() * 1e-9)
    )
    
    # Normalize to session (8 hours = 28800 seconds)
    session_secs = session_hours * 3600
    angle = secs * (2 * np.pi / session_secs)
    
    return df.with_columns([
        angle.sin().alias("f_tod_sin"),
        angle.cos().alias("f_tod_cos"),
    ])


# ============================================================
# 2) SESSION PROGRESS
# ============================================================

@feature
def session_progress(
    df: pl.DataFrame,
    datetime_col: str = "datetime",
    market_open_time: str = "10:00:00",
    market_close_time: str = "18:00:00",
) -> pl.DataFrame:
    """
    Session progress: How far into the trading day are we?
    
    Output:
        f_session_progress: [0, 1] where 0=open, 1=close
    """
    open_s = _time_to_secs(market_open_time)
    close_s = _time_to_secs(market_close_time)
    denom = max(1, close_s - open_s)

    t = (
        pl.col(datetime_col).dt.hour() * 3600
        + pl.col(datetime_col).dt.minute() * 60
        + pl.col(datetime_col).dt.second()
        + (pl.col(datetime_col).dt.nanosecond() * 1e-9)
    )
    
    prog = ((t - open_s) / denom).clip(0.0, 1.0).alias("f_session_progress")
    
    return df.with_columns(prog)


# ============================================================
# 3) DISTANCE TO OPEN/CLOSE (Exponential Decay)
# ============================================================

@feature(deps={"session_progress"})
def distance_to_boundaries(
    df: pl.DataFrame,
    tau: float = 0.10,
) -> pl.DataFrame:
    """
    Exponential distance to session open/close.
    
    Captures the "urgency" near open/close times.
    
    Args:
        tau: Decay rate (smaller = faster decay)
    
    Outputs:
        f_dist_to_open: exp(-progress / tau) - High at open (0, 1]
        f_dist_to_close: exp(-(1-progress) / tau) - High at close (0, 1]
    """
    p = pl.col("f_session_progress")
    tau_val = max(tau, 1e-6)
    
    dist_open = (-(p / tau_val)).exp().alias("f_dist_to_open")
    dist_close = (-((1.0 - p) / tau_val)).exp().alias("f_dist_to_close")
    
    return df.with_columns([dist_open, dist_close])


# ============================================================
# 4) TIME-IN-BOOK (Bounded)
# ============================================================

@feature
def time_in_book(
    df: pl.DataFrame,
    grid_ms: int = 100,
    tau_sec: float = 1.0,
    bid_price_col: str = "pb1",
    ask_price_col: str = "pa1",
) -> pl.DataFrame:
    """
    Time-in-book: How long has best bid/ask price stayed unchanged?
    
    Uses exponential saturation for bounded output.
    
    Args:
        grid_ms: Grid resolution in milliseconds
        tau_sec: Saturation time constant
    
    Outputs:
        f_bid_tib: Bid time-in-book [0, 1)
        f_ask_tib: Ask time-in-book [0, 1)
        f_tib_imbalance: Bid TIB - Ask TIB [-1, 1]
    """
    # Detect price changes
    bid_changed = (pl.col(bid_price_col) != pl.col(bid_price_col).shift(1)).cast(pl.Int64).fill_null(1)
    ask_changed = (pl.col(ask_price_col) != pl.col(ask_price_col).shift(1)).cast(pl.Int64).fill_null(1)

    # Segment IDs
    bid_seg = bid_changed.cum_sum().alias("_bid_seg")
    ask_seg = ask_changed.cum_sum().alias("_ask_seg")

    df = df.with_columns([bid_seg, ask_seg])

    # Count steps within each segment
    bid_steps = pl.col(bid_price_col).cum_count().over("_bid_seg")
    ask_steps = pl.col(ask_price_col).cum_count().over("_ask_seg")

    # Convert to seconds
    step_sec = grid_ms / 1000.0
    bid_sec = bid_steps * step_sec
    ask_sec = ask_steps * step_sec

    # Bounded normalization: 1 - exp(-t/τ) → [0, 1)
    tau = max(tau_sec, 1e-6)
    f_bid_tib = (1.0 - (-(bid_sec / tau)).exp()).alias("f_bid_tib")
    f_ask_tib = (1.0 - (-(ask_sec / tau)).exp()).alias("f_ask_tib")

    df = df.with_columns([f_bid_tib, f_ask_tib])
    
    # TIB imbalance
    f_tib_imbalance = (pl.col("f_bid_tib") - pl.col("f_ask_tib")).alias("f_tib_imbalance")

    return df.with_columns(f_tib_imbalance).drop(["_bid_seg", "_ask_seg"])


# ============================================================
# 5) OPENING/CLOSING FLAGS (Optional)
# ============================================================

@feature(deps={"session_progress"})
def session_boundary_flags(
    df: pl.DataFrame,
    opening_threshold: float = 0.05,   # First 5% of session (~24 min)
    closing_threshold: float = 0.95,   # Last 5% of session (~24 min)
) -> pl.DataFrame:
    """
    Binary flags for opening/closing periods.
    
    Outputs:
        f_is_opening: 1 if in opening period, else 0
        f_is_closing: 1 if in closing period, else 0
    """
    p = pl.col("f_session_progress")
    
    is_opening = (p < opening_threshold).cast(pl.Float64).alias("f_is_opening")
    is_closing = (p > closing_threshold).cast(pl.Float64).alias("f_is_closing")
    
    return df.with_columns([is_opening, is_closing])


# ============================================================
# WRAPPER: ALL TEMPORAL FEATURES
# ============================================================

@feature(deps={
    "time_of_day_cyclical",
    "session_progress",
    "distance_to_boundaries",
    "time_in_book",
    "session_boundary_flags",
})
def temporal_features_scaled(
    df: pl.DataFrame,
    datetime_col: str = "datetime",
    market_open_time: str = "10:00:00",
    market_close_time: str = "18:00:00",
    grid_ms: int = 100,
    tib_tau_sec: float = 1.0,
) -> pl.DataFrame:
    """
    All temporal features with proper scaling.
    
    All features bounded and cross-asset comparable.
    
    Returns:
        DataFrame with temporal features:
        
        Time of Day (Cyclical):
        - f_tod_sin [-1, 1]
        - f_tod_cos [-1, 1]
        
        Session Progress:
        - f_session_progress [0, 1]
        - f_dist_to_open (0, 1]
        - f_dist_to_close (0, 1]
        
        Time-in-Book:
        - f_bid_tib [0, 1)
        - f_ask_tib [0, 1)
        - f_tib_imbalance [-1, 1]
        
        Session Boundaries:
        - f_is_opening {0, 1}
        - f_is_closing {0, 1}
    """
    # Time of day
    df = time_of_day_cyclical(df, datetime_col=datetime_col)
    
    # Session progress
    df = session_progress(
        df,
        datetime_col=datetime_col,
        market_open_time=market_open_time,
        market_close_time=market_close_time,
    )
    
    # Distance to boundaries
    df = distance_to_boundaries(df)
    
    # Time-in-book
    df = time_in_book(df, grid_ms=grid_ms, tau_sec=tib_tau_sec)
    
    # Session boundary flags
    df = session_boundary_flags(df)
    
    return df
