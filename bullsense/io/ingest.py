"""
Clean ingestion module for Bullsense pipelines.
This version removes all legacy dependencies and expects the caller
(e.g., prepare_from_clickhouse.py) to provide a fetcher function.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Optional, Tuple

import polars as pl

# Fetcher signature:
# fetcher(obid: int, db: str) -> (messages_df, orderbook_df)
FetchFn = Callable[[int, str], Tuple[pl.DataFrame, pl.DataFrame]]


# ----------------------------------------------------------------------
# Data container (same as before, but clean)
# ----------------------------------------------------------------------
@dataclass(slots=True)
class IngestionResult:
    """
    Container for message and order book frames fetched from ClickHouse.
    """
    messages: pl.DataFrame
    orderbook: pl.DataFrame

    def copy(self) -> "IngestionResult":
        """Return a deep copy to avoid downstream mutation issues."""
        return IngestionResult(self.messages.clone(), self.orderbook.clone())


# ----------------------------------------------------------------------
# Main ingestion API
# ----------------------------------------------------------------------
def ingest_clickhouse(
    obid: int,
    *,
    db: str = "nazar",
    fetcher: Optional[FetchFn] = None,
) -> IngestionResult:
    """
    Fetch LOB + message data for a given order book ID.
    EXPECTS a custom fetcher to be provided.
    """

    if fetcher is None:
        raise RuntimeError(
            "No fetcher provided to ingest_clickhouse().\n"
            "You must pass a fetcher via prepare_from_clickhouse.py, e.g.\n"
            "    ingest_clickhouse(obid, fetcher=my_fetcher)\n"
        )

    messages, orderbook = fetcher(obid, db=db)
    return IngestionResult(messages=messages, orderbook=orderbook)


__all__ = ["IngestionResult", "ingest_clickhouse"]
