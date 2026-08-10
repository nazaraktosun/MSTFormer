import polars as pl
from hft_features.core.base import feature

EPS = 1e-9

ADD_LABELS = ["A", "AddOrder"]
DELETE_LABELS = ["D", "OrderDelete"]
EXEC_LABELS = ["E", "OrderExecuted"]

SELECTED_WINDOW = 200
DEFAULT_WINDOWS = [SELECTED_WINDOW]

MIN_DEPTH = 1.0
FILL_HORIZON_S = 1.0
CANCEL_EFFICIENCY = 0.0

SELECTED_FEATURES = [
    f"f_qdepl_exec_bid_{SELECTED_WINDOW}",
    f"f_qdepl_exec_ask_{SELECTED_WINDOW}",
    f"f_qdepl_cxl_bid_{SELECTED_WINDOW}",
    f"f_qdepl_cxl_ask_{SELECTED_WINDOW}",
    f"f_qdepl_imb_net_{SELECTED_WINDOW}",
    f"f_qtouch_flip_{SELECTED_WINDOW}",
]

_REQUIRED = ["timestamp", "message_type", "price", "quantity", "pb1", "pa1", "qb1", "qa1"]
_TMP = [
    "_q_day", "_q_dt",
    "_q_exec_bid", "_q_exec_ask",
    "_q_cxl_bid", "_q_cxl_ask",
    "_q_add_bid", "_q_add_ask",
    "_q_flip",
]


def _prepare(df: pl.DataFrame, tick_size: float) -> pl.DataFrame:
    missing = [c for c in _REQUIRED if c not in df.columns]
    if missing:
        raise ValueError(f"queue_features requires missing columns {missing}")

    df = df.with_columns(pl.col("timestamp").dt.date().alias("_q_day"))

    tol = tick_size / 2.0
    px = pl.col("price").cast(pl.Float64)
    qty = pl.col("quantity").cast(pl.Float64)
    bid = pl.col("pb1").cast(pl.Float64)
    ask = pl.col("pa1").cast(pl.Float64)
    bid0 = bid.shift(1).over("_q_day")
    ask0 = ask.shift(1).over("_q_day")

    live = (bid0 > 0) & (ask0 > bid0)
    at_bid = live & ((px - bid0).abs() < tol)
    at_ask = live & ((px - ask0).abs() < tol)

    is_add = pl.col("message_type").is_in(ADD_LABELS)
    is_del = pl.col("message_type").is_in(DELETE_LABELS)
    is_exec = pl.col("message_type").is_in(EXEC_LABELS)

    df = df.with_columns(
        pl.when(is_exec & at_bid).then(qty).otherwise(0.0).alias("_q_exec_bid"),
        pl.when(is_exec & at_ask).then(qty).otherwise(0.0).alias("_q_exec_ask"),
        pl.when(is_del & at_bid).then(qty).otherwise(0.0).alias("_q_cxl_bid"),
        pl.when(is_del & at_ask).then(qty).otherwise(0.0).alias("_q_cxl_ask"),
        pl.when(is_add & at_bid).then(qty).otherwise(0.0).alias("_q_add_bid"),
        pl.when(is_add & at_ask).then(qty).otherwise(0.0).alias("_q_add_ask"),
        ((bid != bid0) | (ask != ask0)).fill_null(False).cast(pl.Float64).alias("_q_flip"),
    )

    return df.with_columns(
        (pl.col("timestamp").diff().over("_q_day").dt.total_microseconds().cast(pl.Float64) / 1e6)
        .fill_null(0.0)
        .clip(0.0, None)
        .alias("_q_dt")
    )


def _rates(window: int) -> dict[str, pl.Expr]:
    elapsed = pl.col("_q_dt").rolling_sum(window, min_samples=1).over("_q_day") + EPS
    cols = ("_q_exec_bid", "_q_exec_ask", "_q_cxl_bid", "_q_cxl_ask", "_q_add_bid", "_q_add_ask", "_q_flip")
    return {c: pl.col(c).rolling_sum(window, min_samples=1).over("_q_day") / elapsed for c in cols}


def _depths() -> tuple[pl.Expr, pl.Expr]:
    floor = pl.lit(MIN_DEPTH)
    return (
        pl.max_horizontal(pl.col("qb1").cast(pl.Float64), floor),
        pl.max_horizontal(pl.col("qa1").cast(pl.Float64), floor),
    )


@feature
def queue_depletion_rate(
    df: pl.DataFrame,
    tick_size: float,
    windows: list[int] = None,
) -> pl.DataFrame:
    if windows is None:
        windows = DEFAULT_WINDOWS

    df = _prepare(df, tick_size)
    qb, qa = _depths()

    out = []
    for w in windows:
        r = _rates(w)
        exec_b, exec_a = r["_q_exec_bid"] / qb, r["_q_exec_ask"] / qa
        cxl_b, cxl_a = r["_q_cxl_bid"] / qb, r["_q_cxl_ask"] / qa
        net_b = (r["_q_exec_bid"] + r["_q_cxl_bid"] - r["_q_add_bid"]) / qb
        net_a = (r["_q_exec_ask"] + r["_q_cxl_ask"] - r["_q_add_ask"]) / qa

        out.extend([
            exec_b.log1p().alias(f"f_qdepl_exec_bid_{w}"),
            exec_a.log1p().alias(f"f_qdepl_exec_ask_{w}"),
            cxl_b.log1p().alias(f"f_qdepl_cxl_bid_{w}"),
            cxl_a.log1p().alias(f"f_qdepl_cxl_ask_{w}"),
            ((net_a - net_b) / (net_a.abs() + net_b.abs() + EPS))
            .clip(-1.0, 1.0)
            .alias(f"f_qdepl_imb_net_{w}"),
        ])

    return df.with_columns(out).drop(_TMP)


@feature
def queue_fill_probability(
    df: pl.DataFrame,
    tick_size: float,
    windows: list[int] = None,
    horizon_s: float = FILL_HORIZON_S,
    cancel_efficiency: float = CANCEL_EFFICIENCY,
) -> pl.DataFrame:
    if windows is None:
        windows = DEFAULT_WINDOWS

    df = _prepare(df, tick_size)
    qb, qa = _depths()

    out = []
    for w in windows:
        r = _rates(w)
        ahead_b = (r["_q_exec_bid"] + cancel_efficiency * r["_q_cxl_bid"]) / qb
        ahead_a = (r["_q_exec_ask"] + cancel_efficiency * r["_q_cxl_ask"]) / qa

        out.extend([
            (1.0 - (-ahead_b * horizon_s).exp()).clip(0.0, 1.0).alias(f"f_qfill_bid_{w}"),
            (1.0 - (-ahead_a * horizon_s).exp()).clip(0.0, 1.0).alias(f"f_qfill_ask_{w}"),
            r["_q_flip"].log1p().alias(f"f_qtouch_flip_{w}"),
        ])

    return df.with_columns(out).drop(_TMP)


@feature(deps={"queue_depletion_rate", "queue_fill_probability"})
def queue_features_all(
    df: pl.DataFrame,
    tick_size: float,
    windows: list[int] = None,
    horizon_s: float = FILL_HORIZON_S,
    cancel_efficiency: float = CANCEL_EFFICIENCY,
) -> pl.DataFrame:
    df = queue_depletion_rate(df, tick_size=tick_size, windows=windows)
    return queue_fill_probability(
        df,
        tick_size=tick_size,
        windows=windows,
        horizon_s=horizon_s,
        cancel_efficiency=cancel_efficiency,
    )


def get_feature_names(windows: list[int] = None) -> list[str]:
    if windows is None:
        windows = DEFAULT_WINDOWS
    names = []
    for w in windows:
        names += [
            f"f_qdepl_exec_bid_{w}", f"f_qdepl_exec_ask_{w}",
            f"f_qdepl_cxl_bid_{w}", f"f_qdepl_cxl_ask_{w}",
            f"f_qdepl_imb_net_{w}",
            f"f_qfill_bid_{w}", f"f_qfill_ask_{w}",
            f"f_qtouch_flip_{w}",
        ]
    return names


__all__ = [
    "queue_depletion_rate",
    "queue_fill_probability",
    "queue_features_all",
    "get_feature_names",
    "SELECTED_FEATURES",
]
