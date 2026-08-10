"""ITCH order-lifecycle event features (paper Sec. ITCH-Derived Event Features).

Reconstructs the five native order-lifecycle categories

    E = {add, cancel, delete, replace, execute}

from the normalized MBO action flags produced by
``bullsense.io.lobster_parquet`` and aggregates them over causal trailing
windows. For every prediction row t only events at or before t participate, and
every rolling window is scoped to the calendar day so nothing spans the
overnight gap.

Per category e and window W this module emits:

    S^e     signed quantity   Q^{e,bid} - Q^{e,ask}      (paper: bid-heavy > 0)
    s^e     signed RATIO      S^e / Q^e in [-1, 1]       (scale-free form)
    n^e     event count
    f^e     relative frequency  n^e / n^all              (composition of flow)
    CTR     (Q^cancel + Q^delete) / Q^execute            (withdrawal vs execution)
    rate    (Q^cancel + Q^delete) / (Q^cancel + Q^delete + Q^execute)  in [0, 1]

Both an unbounded paper-form ratio and a bounded form are emitted for the
withdrawal/execution comparison: the raw CTR is what the paper defines, the
bounded rate is what a network can actually consume when Q^execute is 0.

CATEGORY RECONSTRUCTION
-----------------------
``execute`` and ``add`` map straight onto ``msgf_is_exec`` / ``msgf_is_add``.

``replace`` uses the feed's native ``msgf_is_modify`` (OrderModify) by default.
The paper reconstructs replaces by pairing delete-and-add sequences because its
normalized feed had no native replace action; ours does, so the native flag is
both cheaper and strictly causal. The pairing heuristic is available via
``pair_delete_add`` -- see the flag's docstring for why it is off by default.

``cancel`` vs ``delete`` splits withdrawals into partial size reductions and
complete order removals. Note that the obvious test -- "does this order_id show
up again later?" -- is a LOOK-AHEAD and is not used. Instead we carry a causal
per-order remaining-size ledger: an order's remaining size before event t is
(everything added to it so far) - (everything removed from it so far), both
strictly at or before t. A withdrawal that takes the remaining size to zero is a
complete ``delete``; one that leaves size behind is a ``cancel``. Orders whose
opening add happened before the ingest window have no known added size and are
therefore treated as complete removals.

GRID REQUIREMENT
----------------
``order_id`` is required for the ledger, and ``lobster_parquet`` drops the raw
per-event fields on resample. These features must therefore be computed on the
EVENT grid, before any resampling or batch collapse.
"""

from __future__ import annotations

import polars as pl

from hft_features.core.base import feature

EPS = 1e-9

#: the paper's five order-lifecycle categories
CATEGORIES = ("add", "cancel", "delete", "replace", "execute")

#: trailing event-count windows
DEFAULT_WINDOWS = (100,)

#: clip for the unbounded paper-form CTR
CTR_CLIP = 20.0

#: size tolerance when deciding whether a withdrawal empties the order
SIZE_TOL = 1e-6

_REQUIRED = (
    "msgf_is_add",
    "msgf_is_del",
    "msgf_is_exec",
    "msgf_is_modify",
    "msgf_side_sign",
    "msg_size",
    "order_id",
)

_TMP_PREFIX = "_evt_"


def _validate(df: pl.DataFrame) -> str:
    missing = [c for c in _REQUIRED if c not in df.columns]
    if missing:
        raise ValueError(
            f"event_lifecycle_features: missing columns {missing}. These come from "
            "the per-event message join -- ingest with load_messages=True and run "
            "this feature on the EVENT grid (order_id is dropped on resample)."
        )
    ts = "datetime" if "datetime" in df.columns else "timestamp"
    if ts not in df.columns:
        raise ValueError(
            "event_lifecycle_features requires a 'datetime' (or 'timestamp') column"
        )
    return ts


def _classify(df: pl.DataFrame, ts: str, pair_delete_add: bool) -> pl.DataFrame:
    """Attach one 0/1 indicator per lifecycle category plus per-side quantities."""
    size = pl.col("msg_size").cast(pl.Float64)
    is_add = pl.col("msgf_is_add") == 1
    is_del = pl.col("msgf_is_del") == 1
    is_exec = pl.col("msgf_is_exec") == 1
    is_modify = pl.col("msgf_is_modify") == 1

    df = df.with_columns(pl.col(ts).dt.date().alias(f"{_TMP_PREFIX}day"))

    # --- causal per-order remaining-size ledger -------------------------------
    # cum_added: everything ever added to this order up to and including now.
    # cum_removed_excl: everything removed BEFORE this row (exclusive), so a
    # withdrawal row sees the size that was still resting when it arrived.
    added = pl.when(is_add).then(size).otherwise(0.0)
    removed = pl.when(is_del | is_exec).then(size).otherwise(0.0)

    cum_added = added.cum_sum().over("order_id")
    cum_removed_excl = removed.cum_sum().over("order_id") - removed
    remaining_before = cum_added - cum_removed_excl

    # A withdrawal empties the order when it takes at least what is left.
    # Orders with no observed add have remaining_before <= 0 and fall here too,
    # which is the conservative reading: we cannot prove a partial reduction.
    empties_order = size >= (remaining_before - SIZE_TOL)

    # --- replace --------------------------------------------------------------
    if pair_delete_add:
        # Paper heuristic: a delete immediately followed, at the SAME timestamp
        # and on the SAME side, by an add is one logical replace. Labelling the
        # delete leg needs the next row, so this path reads one event into the
        # future and is off by default. The add leg carries the replace so the
        # collapsed event is stamped at the time both legs are known.
        nxt_add = is_add.shift(-1).over(f"{_TMP_PREFIX}day").fill_null(False)
        nxt_side = pl.col("msgf_side_sign").shift(-1).over(f"{_TMP_PREFIX}day")
        nxt_ts = pl.col(ts).shift(-1).over(f"{_TMP_PREFIX}day")
        del_leg = (
            is_del
            & nxt_add
            & (nxt_side == pl.col("msgf_side_sign"))
            & (nxt_ts == pl.col(ts))
        )
        add_leg = del_leg.shift(1).over(f"{_TMP_PREFIX}day").fill_null(False)

        is_replace = is_modify | add_leg
        # Both legs of a paired replace leave the add/withdrawal categories.
        eff_add = is_add & ~add_leg
        eff_withdrawal = is_del & ~del_leg
    else:
        is_replace = is_modify
        eff_add = is_add
        eff_withdrawal = is_del

    flags = {
        "add": eff_add,
        "cancel": eff_withdrawal & ~empties_order,
        "delete": eff_withdrawal & empties_order,
        "replace": is_replace,
        "execute": is_exec,
    }

    # Side: +1 bid, -1 ask. Executions are signed by their resting side here so
    # that "bid-side execute quantity" means quantity lifted from the bid queue.
    on_bid = pl.col("msgf_side_sign") == 1
    on_ask = pl.col("msgf_side_sign") == -1

    out: list[pl.Expr] = []
    for cat, flag in flags.items():
        out.extend(
            [
                flag.cast(pl.Float64).alias(f"{_TMP_PREFIX}n_{cat}"),
                pl.when(flag & on_bid).then(size).otherwise(0.0).alias(f"{_TMP_PREFIX}qb_{cat}"),
                pl.when(flag & on_ask).then(size).otherwise(0.0).alias(f"{_TMP_PREFIX}qa_{cat}"),
            ]
        )
    return df.with_columns(out)


@feature
def event_lifecycle_features(
    df: pl.DataFrame,
    windows: tuple[int, ...] | list[int] | None = None,
    pair_delete_add: bool = False,
    ctr_clip: float = CTR_CLIP,
) -> pl.DataFrame:
    """Signed quantities, counts, relative frequencies and CTR per lifecycle category.

    Args:
        df: event-grid frame carrying the msgf_* columns and order_id
        windows: trailing event-count windows (rows), day-scoped
        pair_delete_add: reconstruct replaces from delete+add pairs. Reads one
            event ahead -- leave False unless reproducing the paper's heuristic
            on a feed with no native modify action.
        ctr_clip: bound for the unbounded paper-form CTR
    """
    if windows is None:
        windows = DEFAULT_WINDOWS

    ts = _validate(df)
    df = _classify(df, ts, pair_delete_add)
    day = f"{_TMP_PREFIX}day"

    def roll(col: str, w: int) -> pl.Expr:
        return pl.col(col).rolling_sum(window_size=int(w), min_samples=1).over(day)

    out: list[pl.Expr] = []
    for w in windows:
        w = int(w)
        counts = {c: roll(f"{_TMP_PREFIX}n_{c}", w) for c in CATEGORIES}
        qty_bid = {c: roll(f"{_TMP_PREFIX}qb_{c}", w) for c in CATEGORIES}
        qty_ask = {c: roll(f"{_TMP_PREFIX}qa_{c}", w) for c in CATEGORIES}

        n_all = pl.sum_horizontal([counts[c] for c in CATEGORIES])
        out.append(n_all.alias(f"f_evt_cnt_all_{w}"))

        for c in CATEGORIES:
            signed = qty_bid[c] - qty_ask[c]
            unsigned = qty_bid[c] + qty_ask[c]
            out.extend(
                [
                    signed.alias(f"f_evt_sqty_{c}_{w}"),
                    (signed / (unsigned + EPS)).clip(-1.0, 1.0).alias(f"f_evt_sratio_{c}_{w}"),
                    counts[c].alias(f"f_evt_cnt_{c}_{w}"),
                    (counts[c] / (n_all + EPS)).clip(0.0, 1.0).alias(f"f_evt_freq_{c}_{w}"),
                ]
            )

        # Withdrawal vs execution, on UNSIGNED quantities per the paper.
        withdrawn = (
            qty_bid["cancel"] + qty_ask["cancel"] + qty_bid["delete"] + qty_ask["delete"]
        )
        executed = qty_bid["execute"] + qty_ask["execute"]
        out.extend(
            [
                (withdrawn / (executed + EPS)).clip(0.0, ctr_clip).alias(f"f_evt_ctr_{w}"),
                (withdrawn / (withdrawn + executed + EPS))
                .clip(0.0, 1.0)
                .alias(f"f_evt_cancel_rate_{w}"),
            ]
        )

    tmp = [c for c in df.columns if c.startswith(_TMP_PREFIX)]
    return df.with_columns(out).drop(tmp)


def get_feature_names(windows: tuple[int, ...] | list[int] | None = None) -> list[str]:
    """Column names produced by :func:`event_lifecycle_features`."""
    if windows is None:
        windows = DEFAULT_WINDOWS
    names: list[str] = []
    for w in windows:
        w = int(w)
        names.append(f"f_evt_cnt_all_{w}")
        for c in CATEGORIES:
            names += [
                f"f_evt_sqty_{c}_{w}",
                f"f_evt_sratio_{c}_{w}",
                f"f_evt_cnt_{c}_{w}",
                f"f_evt_freq_{c}_{w}",
            ]
        names += [f"f_evt_ctr_{w}", f"f_evt_cancel_rate_{w}"]
    return names


__all__ = [
    "event_lifecycle_features",
    "get_feature_names",
    "CATEGORIES",
    "DEFAULT_WINDOWS",
]
