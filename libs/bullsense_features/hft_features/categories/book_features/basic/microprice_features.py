"""Microprice and historical microprice returns (paper Sec. Historical Microprice-Return Features).

The microprice is the opposite-size-weighted best quote

    p_micro = (pa1 * qb1 + pb1 * qa1) / (qb1 + qa1 + eps)

i.e. each side's PRICE is weighted by the OPPOSITE side's displayed QUANTITY,
which pulls the reference price toward the side that is harder to consume. This
is the same quantity the ingest layer exposes as ``mid_micro``
(``bullsense.features.pipeline.prepare_basic_columns``); it is recomputed here so
the feature is usable standalone inside a registry pipeline.

Returns are causal log returns in basis points over row lookbacks W:

    r_{t,W} = 1e4 * ln(p_micro_t / p_micro_{t-W})

W counts ROWS, not wall-clock. On the 1s resampled grid used for the paper,
W in {1, 5, 10} is 1s/5s/10s. Shifts are taken within a calendar day so a
lookback never reaches across the overnight gap.
"""

from __future__ import annotations

import polars as pl

from hft_features.core.base import feature

EPS = 1e-9

#: paper lookbacks: 1s / 5s / 10s on the 1s grid
DEFAULT_RETURN_WINDOWS = (1, 5, 10)

#: clip in bps -- a 1s microprice move beyond this is a data fault, not signal
RETURN_CLIP_BPS = 200.0


def _day_col(df: pl.DataFrame) -> tuple[str, pl.Expr]:
    """Return (timestamp column name, day-key expression) for day-scoped windows."""
    ts = "datetime" if "datetime" in df.columns else "timestamp"
    if ts not in df.columns:
        raise ValueError(
            "microprice features require a 'datetime' (or 'timestamp') column "
            "to scope lookbacks per trading day"
        )
    return ts, pl.col(ts).dt.date()


@feature
def microprice(df: pl.DataFrame) -> pl.DataFrame:
    """Opposite-size-weighted best quote.

    Outputs:
        f_microprice: (pa1*qb1 + pb1*qa1) / (qb1+qa1)
    """
    missing = [c for c in ("pb1", "pa1", "qb1", "qa1") if c not in df.columns]
    if missing:
        raise ValueError(f"microprice requires top-of-book columns {missing}")

    pb1 = pl.col("pb1").cast(pl.Float64)
    pa1 = pl.col("pa1").cast(pl.Float64)
    qb1 = pl.col("qb1").cast(pl.Float64)
    qa1 = pl.col("qa1").cast(pl.Float64)

    return df.with_columns(
        ((pa1 * qb1 + pb1 * qa1) / (qb1 + qa1 + EPS)).alias("f_microprice")
    )


@feature(deps={"microprice"})
def micro_return_features(
    df: pl.DataFrame,
    windows: tuple[int, ...] | list[int] | None = None,
    clip_bps: float = RETURN_CLIP_BPS,
) -> pl.DataFrame:
    """Causal microprice log returns in basis points.

    Outputs, for each W in ``windows``:
        f_micro_ret_bps_{W}: 1e4 * ln(p_micro_t / p_micro_{t-W}), day-scoped
    """
    if windows is None:
        windows = DEFAULT_RETURN_WINDOWS

    if "f_microprice" not in df.columns:
        df = microprice(df)

    _, day = _day_col(df)
    df = df.with_columns(day.alias("_mret_day"))

    micro = pl.col("f_microprice")
    out = []
    for w in windows:
        past = micro.shift(int(w)).over("_mret_day")
        ret = (
            pl.when((micro > 0) & (past > 0))
            .then((micro / past).log() * 1e4)
            .otherwise(None)
            .clip(-clip_bps, clip_bps)
            .alias(f"f_micro_ret_bps_{int(w)}")
        )
        out.append(ret)

    return df.with_columns(out).drop("_mret_day")


def get_feature_names(windows: tuple[int, ...] | list[int] | None = None) -> list[str]:
    """Column names produced by :func:`micro_return_features` (incl. the level)."""
    if windows is None:
        windows = DEFAULT_RETURN_WINDOWS
    return ["f_microprice"] + [f"f_micro_ret_bps_{int(w)}" for w in windows]


__all__ = [
    "microprice",
    "micro_return_features",
    "get_feature_names",
    "DEFAULT_RETURN_WINDOWS",
]
