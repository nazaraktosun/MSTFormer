

import polars as pl
import numpy as np


EPS = 1e-9

# Message types (after preprocessing)
MSG_ADD = "A"
MSG_DELETE = "D"
MSG_EXECUTE = "E"

# Side (after preprocessing)
SIDE_BID = "B"
SIDE_ASK = "S"

# Clipping bounds
RATIO_CLIP = 10.0
CV_CLIP = 5.0
INTENSITY_CLIP = 5.0

# Default windows (message counts)
WINDOW_SHORT = 50
WINDOW_MED = 100
WINDOW_LONG = 500



def order_flow_counts(
    df: pl.DataFrame,
    windows: list[int] = None,
) -> pl.DataFrame:

    if windows is None:
        windows = [WINDOW_SHORT, WINDOW_MED]

    # Event indicators
    is_add = (pl.col("message_type") == MSG_ADD).cast(pl.Float64)
    is_delete = (pl.col("message_type") == MSG_DELETE).cast(pl.Float64)
    is_execute = (pl.col("message_type") == MSG_EXECUTE).cast(pl.Float64)

    out = []
    for w in windows:
        # Frequencies (count / window)
        add_freq = is_add.rolling_sum(w, min_periods=1) / w
        delete_freq = is_delete.rolling_sum(w, min_periods=1) / w
        execute_freq = is_execute.rolling_sum(w, min_periods=1) / w

        out.extend([
            add_freq.alias(f"f_add_freq_{w}"),
            delete_freq.alias(f"f_delete_freq_{w}"),
            execute_freq.alias(f"f_execute_freq_{w}"),
        ])

        # Add/Delete ratio
        add_count = is_add.rolling_sum(w, min_periods=1)
        delete_count = is_delete.rolling_sum(w, min_periods=1)
        add_delete_ratio = (add_count / (add_count + delete_count + EPS)).clip(0, 1)
        out.append(add_delete_ratio.alias(f"f_add_delete_ratio_{w}"))

    return df.with_columns(out)




def order_flow_by_side(
    df: pl.DataFrame,
    windows: list[int] = None,
) -> pl.DataFrame:
    """
    Order activity split by side.

    Args:
        df: Preprocessed DataFrame with side in ['B', 'S']
        windows: Rolling window sizes (default: [50, 100])

    Outputs:
        f_bid_add_freq_{w}: Bid add frequency [0, 1]
        f_ask_add_freq_{w}: Ask add frequency [0, 1]
        f_add_side_imb_{w}: (bid_add - ask_add) / total [-1, 1]
        f_bid_delete_freq_{w}: Bid delete frequency [0, 1]
        f_ask_delete_freq_{w}: Ask delete frequency [0, 1]
        f_delete_side_imb_{w}: (bid_del - ask_del) / total [-1, 1]
    """
    if windows is None:
        windows = [WINDOW_SHORT, WINDOW_MED]

    # Side indicators
    is_bid = pl.col("side") == SIDE_BID
    is_ask = pl.col("side") == SIDE_ASK

    # Add by side
    bid_add = ((pl.col("message_type") == MSG_ADD) & is_bid).cast(pl.Float64)
    ask_add = ((pl.col("message_type") == MSG_ADD) & is_ask).cast(pl.Float64)

    # Delete by side
    bid_delete = ((pl.col("message_type") == MSG_DELETE) & is_bid).cast(pl.Float64)
    ask_delete = ((pl.col("message_type") == MSG_DELETE) & is_ask).cast(pl.Float64)

    out = []
    for w in windows:
        # Add frequencies
        bid_add_sum = bid_add.rolling_sum(w, min_periods=1)
        ask_add_sum = ask_add.rolling_sum(w, min_periods=1)

        out.extend([
            (bid_add_sum / w).alias(f"f_bid_add_freq_{w}"),
            (ask_add_sum / w).alias(f"f_ask_add_freq_{w}"),
        ])

        # Add side imbalance
        add_imb = ((bid_add_sum - ask_add_sum) / (bid_add_sum + ask_add_sum + EPS)).clip(-1, 1)
        out.append(add_imb.alias(f"f_add_side_imb_{w}"))

        # Delete frequencies
        bid_del_sum = bid_delete.rolling_sum(w, min_periods=1)
        ask_del_sum = ask_delete.rolling_sum(w, min_periods=1)

        out.extend([
            (bid_del_sum / w).alias(f"f_bid_delete_freq_{w}"),
            (ask_del_sum / w).alias(f"f_ask_delete_freq_{w}"),
        ])

        # Delete side imbalance
        del_imb = ((bid_del_sum - ask_del_sum) / (bid_del_sum + ask_del_sum + EPS)).clip(-1, 1)
        out.append(del_imb.alias(f"f_delete_side_imb_{w}"))

    return df.with_columns(out)


def cancel_trade_ratio(
    df: pl.DataFrame,
    windows: list[int] = None,
    clip: float = RATIO_CLIP,
) -> pl.DataFrame:
    """
    Cancel to trade ratio: Indicator of HFT/quote stuffing.

    High ratio = Many cancels per trade (potential manipulation)
    Low ratio = Orders mostly getting filled

    Args:
        df: Preprocessed DataFrame
        windows: Rolling window sizes
        clip: Maximum ratio value

    Outputs:
        f_cancel_trade_ratio_{w}: Delete count / Execute count [0, 10]
        f_cancel_rate_{w}: Delete / (Delete + Execute) [0, 1]
    """
    if windows is None:
        windows = [WINDOW_SHORT, WINDOW_MED]

    is_delete = (pl.col("message_type") == MSG_DELETE).cast(pl.Float64)
    is_execute = (pl.col("message_type") == MSG_EXECUTE).cast(pl.Float64)

    out = []
    for w in windows:
        delete_count = is_delete.rolling_sum(w, min_periods=1)
        execute_count = is_execute.rolling_sum(w, min_periods=1)

        # Ratio (clipped)
        ratio = (delete_count / (execute_count + 1)).clip(0, clip)
        out.append(ratio.alias(f"f_cancel_trade_ratio_{w}"))

        # Rate (bounded)
        rate = (delete_count / (delete_count + execute_count + EPS)).clip(0, 1)
        out.append(rate.alias(f"f_cancel_rate_{w}"))

    return df.with_columns(out)



def order_volume_flow(
    df: pl.DataFrame,
    average_order_volume: float,
    windows: list[int] = None,
    clip: float = INTENSITY_CLIP,
) -> pl.DataFrame:
    """
    Order volume flow: Volume added/deleted relative to average.

    Args:
        df: Preprocessed DataFrame
        average_order_volume: Pre-computed average order volume (from stats)
        windows: Rolling window sizes
        clip: Maximum intensity value

    Outputs:
        f_add_vol_intensity_{w}: Added volume / expected [0, 5]
        f_delete_vol_intensity_{w}: Deleted volume / expected [0, 5]
        f_net_vol_flow_{w}: (Add - Delete) / (Add + Delete) [-1, 1]
    """
    if windows is None:
        windows = [WINDOW_SHORT, WINDOW_MED]

    # Volume by message type
    add_vol = pl.when(pl.col("message_type") == MSG_ADD).then(pl.col("quantity")).otherwise(0)
    delete_vol = pl.when(pl.col("message_type") == MSG_DELETE).then(pl.col("quantity")).otherwise(0)

    out = []
    for w in windows:
        add_sum = add_vol.rolling_sum(w, min_periods=1)
        delete_sum = delete_vol.rolling_sum(w, min_periods=1)

        expected_vol = average_order_volume * w

        # Intensities
        add_intensity = (add_sum / (expected_vol + EPS)).clip(0, clip)
        delete_intensity = (delete_sum / (expected_vol + EPS)).clip(0, clip)

        out.extend([
            add_intensity.alias(f"f_add_vol_intensity_{w}"),
            delete_intensity.alias(f"f_delete_vol_intensity_{w}"),
        ])

        # Net flow
        net_flow = ((add_sum - delete_sum) / (add_sum + delete_sum + EPS)).clip(-1, 1)
        out.append(net_flow.alias(f"f_net_vol_flow_{w}"))

    return df.with_columns(out)


# ============================================================
# 5) ORDER ARRIVAL REGULARITY
# ============================================================

def order_arrival_regularity(
    df: pl.DataFrame,
    window: int = WINDOW_MED,
    clip: float = CV_CLIP,
) -> pl.DataFrame:
    """
    Order arrival regularity: CV of inter-arrival times.

    Low CV (~0): Regular arrivals (algorithmic, TWAP/VWAP)
    CV ~1: Random arrivals (Poisson process, retail)
    High CV: Bursty arrivals (news, events)

    Outputs:
        f_arrival_cv: Coefficient of variation [0, 5]
        f_arrival_regularity: 1 / (1 + CV) - higher = more regular [0, 1]
    """
    if "timestamp" not in df.columns:
        return df.with_columns([
            pl.lit(1.0).alias("f_arrival_cv"),
            pl.lit(0.5).alias("f_arrival_regularity"),
        ])

    # Inter-arrival time (in milliseconds)
    iat = pl.col("timestamp").diff().dt.total_milliseconds().fill_null(0).cast(pl.Float64)

    # Rolling stats
    mean_iat = iat.rolling_mean(window, min_periods=1)
    std_iat = iat.rolling_std(window, min_periods=1)

    # CV
    cv = (std_iat / (mean_iat + EPS)).clip(0, clip).alias("f_arrival_cv")

    # Regularity score (inverse of CV, bounded)
    regularity = (1 / (1 + std_iat / (mean_iat + EPS))).clip(0, 1).alias("f_arrival_regularity")

    return df.with_columns([cv, regularity])


# ============================================================
# 6) ORDER SIZE FEATURES
# ============================================================

def order_size_features(
    df: pl.DataFrame,
    average_order_size: float,
    window: int = WINDOW_MED,
) -> pl.DataFrame:
    """
    Order size distribution features.

    Args:
        df: Preprocessed DataFrame
        average_order_size: Pre-computed average order size (from stats)
        window: Rolling window size

    Outputs:
        f_order_size_cv: Size coefficient of variation [0, 5]
        f_order_size_ratio: Current / average [0.1, 10]
        f_small_order_ratio: % orders < 0.5x average [0, 1]
        f_large_order_ratio: % orders > 2x average [0, 1]
    """
    qty = pl.col("quantity").cast(pl.Float64)

    # Rolling stats
    mean_qty = qty.rolling_mean(window, min_periods=1)
    std_qty = qty.rolling_std(window, min_periods=1)

    # CV
    cv = (std_qty / (mean_qty + EPS)).clip(0, CV_CLIP).alias("f_order_size_cv")

    # Size ratio
    size_ratio = (qty / average_order_size).clip(0.1, 10).alias("f_order_size_ratio")

    # Small/Large order indicators
    is_small = (qty < 0.5 * average_order_size).cast(pl.Float64)
    is_large = (qty > 2.0 * average_order_size).cast(pl.Float64)

    small_ratio = is_small.rolling_mean(window, min_periods=1).alias("f_small_order_ratio")
    large_ratio = is_large.rolling_mean(window, min_periods=1).alias("f_large_order_ratio")

    return df.with_columns([cv, size_ratio, small_ratio, large_ratio])


# ============================================================
# 7) PRICE LEVEL ACTIVITY
# ============================================================

def price_level_activity(
    df: pl.DataFrame,
    tick_size: float,
    window: int = WINDOW_MED,
) -> pl.DataFrame:
    """
    Activity at different price levels.

    Note: Requires best_bid_price and best_ask_price columns.
          If not available, returns zeros.

    Outputs:
        f_price_level_spread: Spread of active price levels (ticks)
        f_add_at_best_ratio: % adds at best bid/ask [0, 1]
        f_add_away_ratio: % adds away from best [0, 1]
    """
    required = ["price", "best_bid_price", "best_ask_price"]
    
    # Check if we have best bid/ask
    has_best = all(col in df.columns for col in required)
    
    if has_best:
        # Check if best_bid/ask are filled (not all zeros)
        bid_filled = df.select(pl.col("best_bid_price").sum()).item() > 0
        ask_filled = df.select(pl.col("best_ask_price").sum()).item() > 0
        has_best = bid_filled and ask_filled
    
    if not has_best:
        # Return placeholder features
        return df.with_columns([
            pl.lit(0.0).alias("f_price_level_spread"),
            pl.lit(0.5).alias("f_add_at_best_ratio"),
            pl.lit(0.5).alias("f_add_away_ratio"),
        ])

    is_add = pl.col("message_type") == MSG_ADD
    order_price = pl.col("price")
    best_bid = pl.col("best_bid_price")
    best_ask = pl.col("best_ask_price")
    side = pl.col("side")

    # Add at best (bid side at best_bid, ask side at best_ask)
    at_best_bid = is_add & (side == SIDE_BID) & (order_price >= best_bid)
    at_best_ask = is_add & (side == SIDE_ASK) & (order_price <= best_ask)
    at_best = (at_best_bid | at_best_ask).cast(pl.Float64)

    # Add away from best
    away_bid = is_add & (side == SIDE_BID) & (order_price < best_bid)
    away_ask = is_add & (side == SIDE_ASK) & (order_price > best_ask)
    away = (away_bid | away_ask).cast(pl.Float64)

    add_count = is_add.cast(pl.Float64).rolling_sum(window, min_periods=1) + EPS

    at_best_ratio = (at_best.rolling_sum(window, min_periods=1) / add_count).clip(0, 1).alias("f_add_at_best_ratio")
    away_ratio = (away.rolling_sum(window, min_periods=1) / add_count).clip(0, 1).alias("f_add_away_ratio")

    # Price level spread (distance from best for adds)
    dist_from_best = pl.when(is_add & (side == SIDE_BID)).then(
        (best_bid - order_price).abs() / tick_size
    ).when(is_add & (side == SIDE_ASK)).then(
        (order_price - best_ask).abs() / tick_size
    ).otherwise(None)

    level_spread = dist_from_best.rolling_mean(window, min_periods=1).fill_null(0).clip(0, 20).alias("f_price_level_spread")

    return df.with_columns([level_spread, at_best_ratio, away_ratio])


# ============================================================
# 8) ORDER FLOW MOMENTUM
# ============================================================

def order_flow_momentum(
    df: pl.DataFrame,
    windows: list[int] = None,
) -> pl.DataFrame:
    """
    Order flow momentum: Is order flow accelerating?

    Outputs:
        f_add_momentum_{w}: Change in add rate [-0.5, 0.5]
        f_order_flow_accel: Acceleration of net flow [-0.5, 0.5]
    """
    if windows is None:
        windows = [WINDOW_SHORT, WINDOW_MED]

    is_add = (pl.col("message_type") == MSG_ADD).cast(pl.Float64)
    is_delete = (pl.col("message_type") == MSG_DELETE).cast(pl.Float64)

    out = []
    for w in windows:
        add_rate = is_add.rolling_mean(w, min_periods=1)
        add_rate_prev = add_rate.shift(w // 2)

        # Momentum = current rate - past rate
        momentum = (add_rate - add_rate_prev).fill_null(0).clip(-0.5, 0.5)
        out.append(momentum.alias(f"f_add_momentum_{w}"))

    df = df.with_columns(out)

    # Net flow acceleration
    net_flow = is_add - is_delete
    net_flow_ma = net_flow.rolling_mean(WINDOW_SHORT, min_periods=1)
    net_flow_ma_prev = net_flow_ma.shift(WINDOW_SHORT)

    accel = (net_flow_ma - net_flow_ma_prev).fill_null(0).clip(-0.5, 0.5).alias("f_order_flow_accel")

    return df.with_columns(accel)


# ============================================================
# 9) FLEETING ORDER FEATURES
# ============================================================

def fleeting_order_features(
    df: pl.DataFrame,
    window: int = WINDOW_MED,
    fleeting_threshold_ms: float = 100,
) -> pl.DataFrame:
    """
    Fleeting orders: Orders that are quickly cancelled.

    High fleeting ratio = HFT activity, quote stuffing

    Note: Simplified version using time between consecutive messages.
          Full implementation would track order_id lifecycle.

    Outputs:
        f_fleeting_ratio: % of orders cancelled quickly [0, 1]
        f_order_lifetime_norm: Normalized order lifetime [0, 1]
    """
    if "timestamp" not in df.columns:
        return df.with_columns([
            pl.lit(0.0).alias("f_fleeting_ratio"),
            pl.lit(0.5).alias("f_order_lifetime_norm"),
        ])

    is_delete = pl.col("message_type") == MSG_DELETE

    # Time since last message (proxy for order lifetime)
    time_since_prev = pl.col("timestamp").diff().dt.total_milliseconds().fill_null(0)

    # Fleeting = delete within threshold
    is_fleeting = (is_delete & (time_since_prev < fleeting_threshold_ms)).cast(pl.Float64)

    delete_count = is_delete.cast(pl.Float64).rolling_sum(window, min_periods=1) + EPS
    fleeting_ratio = (is_fleeting.rolling_sum(window, min_periods=1) / delete_count).clip(0, 1).alias("f_fleeting_ratio")

    # Normalized lifetime (exponential saturation)
    tau = fleeting_threshold_ms * 10  # 1 second saturation
    lifetime_norm = (1 - (-time_since_prev / tau).exp()).clip(0, 1).alias("f_order_lifetime_norm")

    return df.with_columns([fleeting_ratio, lifetime_norm])


# ============================================================
# MAIN WRAPPER
# ============================================================


def order_activity_features_all(
    df: pl.DataFrame,
    average_order_volume: float,
    average_order_size: float,
    tick_size: float,
    windows: list[int] = None,
    fleeting_threshold_ms: float = 100,
    validate: bool = True,
) -> pl.DataFrame:
    
    if windows is None:
        windows = [WINDOW_SHORT, WINDOW_MED, WINDOW_LONG]



    # Calculate all features
    df = order_flow_counts(df, windows=windows)
    df = order_flow_by_side(df, windows=windows)
    df = cancel_trade_ratio(df, windows=windows)
    df = order_volume_flow(df, average_order_volume=average_order_volume, windows=windows)
    df = order_arrival_regularity(df)
    df = order_size_features(df, average_order_size=average_order_size)
    df = price_level_activity(df, tick_size=tick_size)
    df = order_flow_momentum(df, windows=windows[:2])  # Only short/med for momentum
    df = fleeting_order_features(df, fleeting_threshold_ms=fleeting_threshold_ms)

    return df


# ============================================================
# FEATURE LIST HELPER
# ============================================================

def get_feature_names(windows: list[int] = None) -> list[str]:
    """Get list of all feature names that will be generated."""
    if windows is None:
        windows = [WINDOW_SHORT, WINDOW_MED, WINDOW_LONG]
    
    features = []
    
    # Order flow counts
    for w in windows:
        features.extend([
            f"f_add_freq_{w}",
            f"f_delete_freq_{w}",
            f"f_execute_freq_{w}",
            f"f_add_delete_ratio_{w}",
        ])
    
    # Order flow by side
    for w in windows:
        features.extend([
            f"f_bid_add_freq_{w}",
            f"f_ask_add_freq_{w}",
            f"f_add_side_imb_{w}",
            f"f_bid_delete_freq_{w}",
            f"f_ask_delete_freq_{w}",
            f"f_delete_side_imb_{w}",
        ])
    
    # Cancel/Trade ratio
    for w in windows:
        features.extend([
            f"f_cancel_trade_ratio_{w}",
            f"f_cancel_rate_{w}",
        ])
    
    # Order volume flow
    for w in windows:
        features.extend([
            f"f_add_vol_intensity_{w}",
            f"f_delete_vol_intensity_{w}",
            f"f_net_vol_flow_{w}",
        ])
    
    # Arrival regularity
    features.extend([
        "f_arrival_cv",
        "f_arrival_regularity",
    ])
    
    # Order size
    features.extend([
        "f_order_size_cv",
        "f_order_size_ratio",
        "f_small_order_ratio",
        "f_large_order_ratio",
    ])
    
    # Price level activity
    features.extend([
        "f_price_level_spread",
        "f_add_at_best_ratio",
        "f_add_away_ratio",
    ])
    
    # Order flow momentum (only first 2 windows)
    for w in windows[:2]:
        features.append(f"f_add_momentum_{w}")
    features.append("f_order_flow_accel")
    
    # Fleeting orders
    features.extend([
        "f_fleeting_ratio",
        "f_order_lifetime_norm",
    ])
    
    return features
