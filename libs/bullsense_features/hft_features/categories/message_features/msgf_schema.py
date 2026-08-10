"""Adapter from the BullSense event-grid schema to the message-feature schema.

``queue_features`` and ``order_activity_features`` were written against a
standalone message frame that carries string ``message_type`` / ``side`` columns
plus bare ``timestamp`` / ``price`` / ``quantity``. The frame that
``bullsense.io.lobster_parquet`` actually hands the pipeline is the FUSED event
grid: book columns plus one-hot ``msgf_is_*`` flags, ``msgf_side_sign``, and the
raw fields under their ``msg_*`` names.

Without a bridge those two modules register into the pipeline but raise on any
real frame, which is why they were dead in practice. This step maps

    datetime      -> timestamp
    msg_price     -> price
    msg_size      -> quantity
    msgf_is_*     -> message_type   ("A" add, "D" delete, "E" execute,
                                     "M" modify, "T" trade, "R" flush, "U" other)
    msgf_side_sign-> side           ("B" bid, "S" ask, null otherwise)

Insert it BEFORE any message-feature step in a pipeline config. Existing columns
are never overwritten, so it is safe to run on a frame that already conforms.
Only aliases are added -- nothing is dropped, because later steps still read the
``msgf_*`` columns.
"""

from __future__ import annotations

import polars as pl

from hft_features.core.base import feature

#: msgf_* one-hot flag -> single-letter message_type code
_FLAG_TO_CODE: tuple[tuple[str, str], ...] = (
    ("msgf_is_add", "A"),
    ("msgf_is_del", "D"),
    ("msgf_is_exec", "E"),
    ("msgf_is_modify", "M"),
    ("msgf_is_trade", "T"),
    ("msgf_is_flush", "R"),
)

_ALIASES: tuple[tuple[str, str], ...] = (
    ("timestamp", "datetime"),
    ("price", "msg_price"),
    ("quantity", "msg_size"),
)


@feature
def msgf_to_message_schema(df: pl.DataFrame) -> pl.DataFrame:
    """Add message_type / side / timestamp / price / quantity aliases in place.

    Outputs (only those not already present):
        timestamp, price, quantity, message_type, side
    """
    out: list[pl.Expr] = []

    for target, source in _ALIASES:
        if target not in df.columns:
            if source not in df.columns:
                raise ValueError(
                    f"msgf_to_message_schema: cannot build '{target}' -- neither "
                    f"'{target}' nor '{source}' is present. Ingest with "
                    "load_messages=True and run on the event grid."
                )
            out.append(pl.col(source).alias(target))

    if "message_type" not in df.columns:
        present = [(flag, code) for flag, code in _FLAG_TO_CODE if flag in df.columns]
        if not present:
            raise ValueError(
                "msgf_to_message_schema: no msgf_is_* flags found to derive "
                "'message_type' from."
            )
        chain = None
        for flag, code in present:
            cond = pl.col(flag) == 1
            chain = pl.when(cond).then(pl.lit(code)) if chain is None else chain.when(cond).then(pl.lit(code))
        out.append(chain.otherwise(pl.lit("U")).alias("message_type"))

    if "side" not in df.columns:
        if "msgf_side_sign" not in df.columns:
            raise ValueError(
                "msgf_to_message_schema: 'msgf_side_sign' is required to derive 'side'."
            )
        sign = pl.col("msgf_side_sign")
        out.append(
            pl.when(sign == 1)
            .then(pl.lit("B"))
            .when(sign == -1)
            .then(pl.lit("S"))
            .otherwise(None)
            .alias("side")
        )

    return df.with_columns(out) if out else df


__all__ = ["msgf_to_message_schema"]
