"""Databento XNAS.ITCH LOBSTER-parquet ingestion for Bullsense.

Source layouts supported (both produced by TLOB/scripts/dbn_to_lobster.py):

Layout A (canonical, self-contained day dirs)::

    root/{YYYY-MM-DD}/orderbook_{date}.parquet   40 cols: sell1,vsell1,buy1,vbuy1,... (lvl 10)
    root/{YYYY-MM-DD}/message_{date}.parquet     time,event_type,order_id,size,price,direction

Layout B (flat, pre-renamed to Bullsense schema)::

    lob_root/{YYYY-MM-DD}.parquet                timestamp + pa1,qa1,pb1,qb1,... (lvl 10)
    msg_root/{YYYY-MM-DD}/message_{date}.parquet (optional message pairing)

Alignment contract (guaranteed by the converter): the message and orderbook
files for a day are written in lockstep -- row i of the message file is the
event that produced snapshot row i. Messages are therefore attached BY ROW
INDEX (hstack) and never by timestamp join; two events sharing a nanosecond
can never be misaligned. A height mismatch is a contract violation and raises.

Message columns emitted on the combined frame (event grid):

    msgf_is_add / msgf_is_del / msgf_is_exec / msgf_is_modify /
    msgf_is_trade / msgf_is_flush              one-hot event type (Int32)
    msgf_side_sign                             +1 bid / -1 ask side of the event's order
    msgf_aggr_sign                             aggressor sign for executions
                                               (converter stores the RESTING side on 'E',
                                               so aggressor = -side_sign), else 0
    msgf_add_vol / msgf_del_vol / msgf_exec_vol      unsigned size contributions
    msgf_sadd_vol / msgf_sdel_vol / msgf_sexec_vol   side/aggressor-signed contributions
    msg_size, msg_price, order_id              raw per-event fields (metadata, not features)

When ``resample_ms`` is set, book columns take the last observation per bucket,
``msgf_*`` columns are summed (counts/volumes per bucket) and raw per-event
fields are dropped. The emitted timestamp is the LAST EVENT's timestamp inside
the bucket -- not the bucket's left edge -- so downstream as-of logic never
attributes end-of-bucket state to the start of the bucket.

When ``batch_collapse`` is set, rows are collapsed to one SNAPSHOT per unique
timestamp instead of a fixed ms grid: events sharing a nanosecond (a sweep, a
batched update) become a single row carrying the book state AFTER the whole
batch (last row of the group, per the converter's row-order contract) with
``msgf_*`` columns summed across the batch. This is the event/snapshot grid for
event-count horizons: h rows = h book updates, never a fragment of one.
Intra-batch states are not individually tradeable, so no information a strategy
could act on is lost. Mutually exclusive with ``resample_ms``.

Touch-conditioned columns (require ``load_messages=True``): each event is
compared against the PRE-event best (previous row's pb1/pa1 -- row i-1 is the
book after event i-1, hence the state event i arrived into):

    msgf_add_bid_touch_vol / msgf_add_ask_touch_vol   adds at-or-inside the touch
    msgf_del_bid_touch_vol / msgf_del_ask_touch_vol   deletes exactly at the touch

They carry the msgf_ prefix, so batch collapse and ms-resample aggregate them
like any other message column.
"""

from __future__ import annotations

import glob
import re
from collections.abc import Callable, Iterable
from pathlib import Path

import polars as pl

from bullsense.io.ingest import IngestionResult

_PRICE_COL_RE = re.compile(r"^p[ab]\d+$", re.IGNORECASE)
_OB_RAW_RE = re.compile(r"^(sell|vsell|buy|vbuy)(\d+)$", re.IGNORECASE)

_EVENT_TYPE_TO_MSG = {
    "AddOrder": "A",
    "OrderDelete": "D",
    "OrderExecuted": "E",
    "OrderModify": "M",
    "Trade": "T",
    "OrderBookFlush": "R",
    "Unknown": "U",
}

_DIRECTION_TO_SIDE = {
    "Buy": "B",
    "Sell": "S",
    " ": None,
}

#: per-event message columns that are safe to SUM when resampling to a grid
_MSG_SUM_PREFIX = "msgf_"
#: raw per-event fields kept on the event grid but dropped on resample
_MSG_RAW_COLS = ("order_id", "msg_size", "msg_price")


def _ob_rename_map(columns: list[str]) -> dict[str, str]:
    """Map converter naming (sellN/vsellN/buyN/vbuyN) to Bullsense (paN/qaN/pbN/qbN)."""
    mapping: dict[str, str] = {}
    for col in columns:
        m = _OB_RAW_RE.match(col)
        if not m:
            continue
        kind, level = m.group(1).lower(), m.group(2)
        target = {"sell": "pa", "vsell": "qa", "buy": "pb", "vbuy": "qb"}[kind]
        mapping[col] = f"{target}{level}"
    return mapping


def _discover_day_pairs(
    lob_path: str | Path,
    msg_path: str | Path | None,
) -> list[tuple[str, Path, Path | None]]:
    """Resolve per-day (date, orderbook_file, message_file|None) pairs, sorted by date.

    Accepts a directory (layout A or B), a single parquet file, or a glob
    pattern for the orderbook side. Message files are located by date key.
    """
    ob_by_date: dict[str, Path] = {}

    lob_str = str(lob_path)
    if any(ch in lob_str for ch in "*?[]"):
        candidates = [Path(p) for p in sorted(glob.glob(lob_str, recursive=True))]
        for f in candidates:
            if f.suffix == ".parquet" and not f.name.startswith("message_"):
                ob_by_date[f.stem.replace("orderbook_", "")] = f
    else:
        root = Path(lob_path)
        if root.is_file():
            ob_by_date[root.stem.replace("orderbook_", "")] = root
        elif root.is_dir():
            for sub in sorted(root.iterdir()):
                if sub.is_dir():
                    obs = sorted(sub.glob("orderbook_*.parquet"))
                    if obs:
                        ob_by_date[sub.name] = obs[0]
                elif sub.suffix == ".parquet" and not sub.name.startswith("message_"):
                    ob_by_date[sub.stem.replace("orderbook_", "")] = sub
        else:
            raise FileNotFoundError(f"LOB parquet source not found: {lob_path}")

    if not ob_by_date:
        raise FileNotFoundError(f"No orderbook parquet files found under: {lob_path}")

    msg_by_date: dict[str, Path] = {}
    if msg_path is not None:
        msg_root = Path(msg_path)
        if msg_root.is_file():
            msg_by_date[msg_root.stem.replace("message_", "")] = msg_root
        elif msg_root.is_dir():
            for sub in sorted(msg_root.iterdir()):
                if sub.is_dir():
                    msgs = sorted(sub.glob("message_*.parquet"))
                    if msgs:
                        msg_by_date[sub.name] = msgs[0]
                elif sub.suffix == ".parquet" and sub.name.startswith("message_"):
                    msg_by_date[sub.stem.replace("message_", "")] = sub
        else:
            raise FileNotFoundError(f"Message parquet source not found: {msg_path}")

    pairs = [
        (date, ob_file, msg_by_date.get(date))
        for date, ob_file in sorted(ob_by_date.items())
    ]
    return pairs


def _message_event_columns(msg: pl.DataFrame) -> pl.DataFrame:
    """Derive per-event numeric message columns from the raw converter schema."""
    event = pl.col("event_type")
    side_sign = (
        pl.when(pl.col("direction") == "Buy")
        .then(1)
        .when(pl.col("direction") == "Sell")
        .then(-1)
        .otherwise(0)
        .cast(pl.Int32)
    )
    is_add = (event == "AddOrder").cast(pl.Int32)
    is_del = (event == "OrderDelete").cast(pl.Int32)
    is_exec = (event == "OrderExecuted").cast(pl.Int32)
    is_modify = (event == "OrderModify").cast(pl.Int32)
    is_trade = (event == "Trade").cast(pl.Int32)
    is_flush = (event == "OrderBookFlush").cast(pl.Int32)
    size = pl.col("size").cast(pl.Float64)

    # The converter stores an execution's RESTING side in `direction`;
    # the aggressor is the opposite side.
    aggr_sign = (-side_sign * is_exec).cast(pl.Int32)

    return msg.select(
        [
            pl.from_epoch(pl.col("time"), time_unit="ns").alias("timestamp"),
            pl.col("order_id").cast(pl.Int64),
            size.alias("msg_size"),
            pl.col("price").cast(pl.Float64).alias("msg_price"),
            is_add.alias("msgf_is_add"),
            is_del.alias("msgf_is_del"),
            is_exec.alias("msgf_is_exec"),
            is_modify.alias("msgf_is_modify"),
            is_trade.alias("msgf_is_trade"),
            is_flush.alias("msgf_is_flush"),
            side_sign.alias("msgf_side_sign"),
            aggr_sign.alias("msgf_aggr_sign"),
            (size * is_add).alias("msgf_add_vol"),
            (size * is_del).alias("msgf_del_vol"),
            (size * is_exec).alias("msgf_exec_vol"),
            (size * is_modify).alias("msgf_modify_vol"),
            (size * is_trade).alias("msgf_trade_vol"),
            (size * is_flush).alias("msgf_flush_vol"),
            (size * is_add * side_sign).alias("msgf_sadd_vol"),
            (size * is_del * side_sign).alias("msgf_sdel_vol"),
            (size * is_exec * aggr_sign).alias("msgf_sexec_vol"),
            (size * is_modify * side_sign).alias("msgf_smodify_vol"),
            (size * is_trade * side_sign).alias("msgf_strade_vol"),
        ]
    )


def _add_touch_columns(df: pl.DataFrame) -> pl.DataFrame:
    """Attach touch-conditioned per-event volumes (see module docstring).

    Must run on the per-event grid AFTER message columns are hstacked and
    BEFORE any collapse/resample: it compares each event's price against the
    previous row's best bid/ask (the book state the event arrived into).
    Prices are still in raw converter units on both sides here, so equality
    comparisons are exact. A ``prev > 0`` guard keeps empty-book rows
    (price 0) from matching adds via the at-or-inside inequality.
    """
    required = {"msg_price", "msgf_is_add", "msgf_is_del", "msgf_side_sign", "pb1", "pa1"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(
            f"_add_touch_columns: missing columns {sorted(missing)}. "
            "Requires load_messages=True and a renamed 10-level book."
        )

    prev_bid = pl.col("pb1").shift(1)
    prev_ask = pl.col("pa1").shift(1)
    size = pl.col("msg_size")
    is_add = pl.col("msgf_is_add") == 1
    is_del = pl.col("msgf_is_del") == 1
    on_bid = pl.col("msgf_side_sign") == 1
    on_ask = pl.col("msgf_side_sign") == -1

    return df.with_columns(
        (pl.when(is_add & on_bid & (prev_bid > 0) & (pl.col("msg_price") >= prev_bid))
         .then(size).otherwise(0.0)).alias("msgf_add_bid_touch_vol"),
        (pl.when(is_add & on_ask & (prev_ask > 0) & (pl.col("msg_price") <= prev_ask))
         .then(size).otherwise(0.0)).alias("msgf_add_ask_touch_vol"),
        (pl.when(is_del & on_bid & (pl.col("msg_price") == prev_bid))
         .then(size).otherwise(0.0)).alias("msgf_del_bid_touch_vol"),
        (pl.when(is_del & on_ask & (pl.col("msg_price") == prev_ask))
         .then(size).otherwise(0.0)).alias("msgf_del_ask_touch_vol"),
    )


def _collapse_batches(df: pl.DataFrame) -> pl.DataFrame:
    """Collapse to one snapshot per unique timestamp (see module docstring).

    Same aggregation contract as ``_resample_day``: book/last, msgf_*/sum,
    raw per-event fields dropped. Rows arrive in converter order (time is
    monotone non-decreasing), so the group's last row is the post-batch book.
    """
    sum_cols = [c for c in df.columns if c.startswith(_MSG_SUM_PREFIX)]
    drop_cols = [c for c in _MSG_RAW_COLS if c in df.columns]
    last_cols = [
        c for c in df.columns
        if c not in {"timestamp", *sum_cols, *drop_cols}
    ]

    aggs = [pl.col(c).last() for c in last_cols]
    aggs += [pl.col(c).sum() for c in sum_cols]
    return df.group_by("timestamp", maintain_order=True).agg(aggs)


def _resample_day(df: pl.DataFrame, resample_ms: int) -> pl.DataFrame:
    """Resample one day's combined frame to a fixed grid.

    Book columns: last observation per bucket. msgf_* columns: sum per bucket.
    The output timestamp is the last event's timestamp in the bucket so the
    row's anchor matches the book state it carries (no left-edge lookahead).
    """
    every = f"{int(resample_ms)}ms"
    df = df.sort("timestamp").with_columns(pl.col("timestamp").alias("_evt_ts"))

    sum_cols = [c for c in df.columns if c.startswith(_MSG_SUM_PREFIX)]
    drop_cols = [c for c in _MSG_RAW_COLS if c in df.columns]
    last_cols = [
        c for c in df.columns
        if c not in {"timestamp", "_evt_ts", *sum_cols, *drop_cols}
    ]

    aggs = [pl.col("_evt_ts").last()]
    aggs += [pl.col(c).last() for c in last_cols]
    aggs += [pl.col(c).sum() for c in sum_cols]

    out = df.group_by_dynamic("timestamp", every=every).agg(aggs)
    return out.drop("timestamp").rename({"_evt_ts": "timestamp"})


def _load_day(
    date: str,
    ob_file: Path,
    msg_file: Path | None,
    *,
    load_messages: bool,
    resample_ms: int | None,
    batch_collapse: bool = False,
) -> pl.DataFrame:
    ob = pl.read_parquet(ob_file)
    rename = _ob_rename_map(ob.columns)
    if rename:
        ob = ob.rename(rename)

    if msg_file is not None and load_messages:
        msg = pl.read_parquet(msg_file)
        if msg.height != ob.height:
            raise ValueError(
                f"[{date}] message/orderbook row mismatch: "
                f"{msg_file} has {msg.height:,} rows, {ob_file} has {ob.height:,}. "
                "The converter guarantees 1:1 alignment; refusing to guess."
            )
        msg_cols = _message_event_columns(msg)
        if "timestamp" in ob.columns:
            msg_cols = msg_cols.drop("timestamp")
        ob = ob.hstack(msg_cols)
        ob = _add_touch_columns(ob)
    elif "timestamp" not in ob.columns:
        if msg_file is None:
            raise ValueError(
                f"[{date}] {ob_file} has no 'timestamp' column and no message file "
                "was found to take timestamps from."
            )
        times = pl.read_parquet(msg_file, columns=["time"])
        if times.height != ob.height:
            raise ValueError(
                f"[{date}] message/orderbook row mismatch: "
                f"{msg_file} has {times.height:,} rows, {ob_file} has {ob.height:,}."
            )
        ob = ob.hstack(
            times.select(pl.from_epoch(pl.col("time"), time_unit="ns").alias("timestamp"))
        )

    if batch_collapse:
        ob = _collapse_batches(ob)
    elif resample_ms is not None:
        ob = _resample_day(ob, resample_ms)
    return ob


def _finalize_frame(df: pl.DataFrame, *, tz: str, price_scale: float, symbol: str) -> pl.DataFrame:
    """Timezone-normalize, scale prices to dollars, stamp symbol."""
    ts = pl.col("timestamp")
    if getattr(df.schema["timestamp"], "time_zone", None) is None:
        ts = ts.dt.replace_time_zone("UTC")
    dt = ts.dt.convert_time_zone(tz)
    df = df.with_columns([dt.alias("datetime"), dt.alias("readable_timestamp")]).drop("timestamp")

    if price_scale != 1.0:
        price_cols = [c for c in df.columns if _PRICE_COL_RE.match(c)]
        if "msg_price" in df.columns:
            price_cols.append("msg_price")
        if price_cols:
            df = df.with_columns([(pl.col(c) / price_scale).alias(c) for c in price_cols])

    return df.with_columns(pl.lit(symbol).alias("symbol"))


def normalize_lobster_message_schema(df: pl.DataFrame) -> pl.DataFrame:
    """Normalize a standalone converter message frame (kept for notebooks/tools)."""
    df = df.with_columns(
        pl.from_epoch(pl.col("time"), time_unit="ns").alias("timestamp")
    )

    df = df.with_columns(
        [
            pl.col("event_type").replace_strict(_EVENT_TYPE_TO_MSG).alias("message_type"),
            pl.col("direction").replace_strict(_DIRECTION_TO_SIDE).alias("side"),
        ]
    )

    return df.with_columns(
        pl.when(pl.col("message_type") == "E")
        .then(
            pl.when(pl.col("side") == "B")
            .then(pl.lit("S"))
            .when(pl.col("side") == "S")
            .then(pl.lit("B"))
            .otherwise(None)
        )
        .otherwise(None)
        .alias("aggressor_side")
    )


def make_lobster_ingest_fn(
    lob_parquet_path: str | Path,
    msg_parquet_path: str | Path | None = None,
    *,
    symbol: str,
    tz: str = "America/New_York",
    price_scale: float = 1000.0,
    resample_ms: int | None = None,
    batch_collapse: bool = False,
    load_messages: bool = True,
    include_dates: Iterable[str] | None = None,
) -> Callable[[int], IngestionResult]:
    """Build an ingest_fn returning ONE combined frame (book + per-event message cols).

    Messages are attached by row index using the converter's 1:1 contract.
    ``load_messages=False`` skips message columns entirely (message files are
    then only touched for timestamps when the book file has none).
    ``batch_collapse=True`` emits one snapshot per unique timestamp (event
    grid for event-count horizons); mutually exclusive with ``resample_ms``.
    """
    if batch_collapse and resample_ms is not None:
        raise ValueError(
            "batch_collapse and resample_ms are mutually exclusive: the batch "
            "grid is already a (variable-width) aggregation; pick one grid."
        )
    pairs = _discover_day_pairs(lob_parquet_path, msg_parquet_path)
    if include_dates is not None:
        date_set = {str(date) for date in include_dates}
        pairs = [pair for pair in pairs if pair[0] in date_set]
        if not pairs:
            raise FileNotFoundError(
                f"No orderbook parquet files under {lob_parquet_path} matched include_dates."
            )

    parts: list[pl.DataFrame] = []
    for date, ob_file, msg_file in pairs:
        parts.append(
            _load_day(
                date,
                ob_file,
                msg_file,
                load_messages=load_messages,
                resample_ms=resample_ms,
                batch_collapse=batch_collapse,
            )
        )

    combined = pl.concat(parts, how="vertical_relaxed")
    combined = _finalize_frame(combined, tz=tz, price_scale=price_scale, symbol=symbol)
    combined = combined.sort("datetime")

    def _ingest(obid: int = 0) -> IngestionResult:
        return IngestionResult(messages=pl.DataFrame(), orderbook=combined)

    return _ingest


__all__ = ["make_lobster_ingest_fn", "normalize_lobster_message_schema"]
