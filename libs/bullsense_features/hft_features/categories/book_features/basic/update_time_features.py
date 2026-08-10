"""Time since the last genuine market update (paper Sec. Temporal Context Features).

The paper's staleness feature is

    dt_t = tau_t - u_t

where ``tau_t`` is the timestamp represented at grid time t and ``u_t <= tau_t`` is
the timestamp of the most recent GENUINE market update available at t.

The distinction matters on a resampled grid. Consecutive rows of a 1s grid are
1s apart by construction, so a naive row-to-row time delta is the constant grid
step and carries no information. What we want is how stale the book state is:
if the top of book last changed 4.2s ago, every forward-filled row in between
should report its true age. A row that IS an update reports 0.

"Genuine update" is defined here as a change in the top-of-book vector
(pb1, qb1, pa1, qa1). Deeper-level-only changes are not counted -- they do not
move the quantity the labels are built from. Pass ``levels`` to widen the
definition to the top L levels.

The companion cyclical time-of-day features (f_tod_sin / f_tod_cos) already live
in ``time_features.time_of_day_cyclical``; together the three make up the
paper's temporal context block.
"""

from __future__ import annotations

import polars as pl

from hft_features.core.base import feature

#: reference scale for the bounded transform, in seconds
TAU_0 = 1.0


def _book_cols(df: pl.DataFrame, levels: int) -> list[str]:
    cols = [
        f"{p}{i}"
        for i in range(1, int(levels) + 1)
        for p in ("pb", "qb", "pa", "qa")
    ]
    missing = [c for c in cols if c not in df.columns]
    if missing:
        raise ValueError(f"update_time_features requires book columns {missing}")
    return cols


@feature
def time_since_update(
    df: pl.DataFrame,
    levels: int = 1,
    tau_0: float = TAU_0,
) -> pl.DataFrame:
    """Seconds since the most recent genuine book update, scoped per trading day.

    Outputs:
        f_dt_since_update: raw age of the book state in seconds (0 on an update)
        f_dt_since_update_log: log1p(dt / tau_0), a bounded form for the model
    """
    ts_name = "datetime" if "datetime" in df.columns else "timestamp"
    if ts_name not in df.columns:
        raise ValueError(
            "time_since_update requires a 'datetime' (or 'timestamp') column"
        )

    cols = _book_cols(df, levels)
    ts = pl.col(ts_name)
    day = ts.dt.date().alias("_upd_day")
    df = df.with_columns(day)

    # A row is an update if ANY tracked book field differs from the previous row
    # of the same day. The first row of a day has no predecessor, so it counts as
    # an update and starts the day at dt = 0 rather than inheriting yesterday.
    changed = pl.any_horizontal(
        [
            (pl.col(c) != pl.col(c).shift(1).over("_upd_day"))
            .fill_null(True)
            for c in cols
        ]
    )

    # Timestamp of the last update at or before this row: stamp it on update
    # rows, leave null elsewhere, then forward-fill within the day. Purely
    # backward-looking -- nothing after row t participates.
    last_update_ts = (
        pl.when(changed)
        .then(ts)
        .otherwise(None)
        .forward_fill()
        .over("_upd_day")
    )

    dt_s = (
        (ts - last_update_ts)
        .dt.total_microseconds()
        .cast(pl.Float64)
        / 1e6
    ).clip(0.0, None)

    df = df.with_columns(dt_s.alias("f_dt_since_update"))

    return df.with_columns(
        (pl.col("f_dt_since_update") / tau_0).log1p().alias("f_dt_since_update_log")
    ).drop("_upd_day")


def get_feature_names() -> list[str]:
    """Column names produced by :func:`time_since_update`."""
    return ["f_dt_since_update", "f_dt_since_update_log"]


__all__ = ["time_since_update", "get_feature_names", "TAU_0"]
