"""Datetime helpers shared across feature and data pipelines."""

from __future__ import annotations

import polars as pl


def to_datetime_expr(ts_col: str) -> pl.Expr:
    """Convert numeric or string timestamp column to timezone-aware datetime (Europe/Istanbul).

    Heuristics mirror the previous inline helpers: pick ns/us/ms/s based on value range.
    """

    return (
        pl.when(pl.col(ts_col) > 10**16)
        .then(pl.from_epoch(pl.col(ts_col), time_unit="ns"))
        .when(pl.col(ts_col) > 10**13)
        .then(pl.from_epoch(pl.col(ts_col), time_unit="us"))
        .when(pl.col(ts_col) > 10**10)
        .then(pl.from_epoch(pl.col(ts_col), time_unit="ms"))
        .otherwise(pl.from_epoch(pl.col(ts_col), time_unit="s"))
        .dt.replace_time_zone("UTC")
        .dt.convert_time_zone("Europe/Istanbul")
    )


__all__ = ["to_datetime_expr"]
