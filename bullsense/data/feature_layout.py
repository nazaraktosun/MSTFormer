"""Feature layout helpers for prepared LOB tensors."""

from __future__ import annotations

import re
from typing import Any, Sequence

LOBSTER_LAYOUT = "lobster_ask_bid_interleaved"
_RAW_LOB_RE = re.compile(r"^(p|q)(a|b)(\d+)$", flags=re.IGNORECASE)


def is_raw_lob_column(name: str) -> bool:
    return bool(_RAW_LOB_RE.match(name))


def canonical_lobster_order(feature_names: Sequence[str]) -> list[str]:
    """Return raw LOB columns in academic LOBSTER order.

    LOBSTER order per depth level is:
    ``ask_price, ask_size, bid_price, bid_size``.

    For Bullsense column names this is:
    ``paN, qaN, pbN, qbN``.
    """

    by_lower = {name.lower(): name for name in feature_names}
    levels: set[int] = set()
    for name in feature_names:
        match = _RAW_LOB_RE.match(name)
        if match:
            levels.add(int(match.group(3)))

    ordered: list[str] = []
    for level in sorted(levels):
        for canonical in (f"pa{level}", f"qa{level}", f"pb{level}", f"qb{level}"):
            original = by_lower.get(canonical)
            if original is not None:
                ordered.append(original)
    return ordered


def canonicalize_feature_columns(feature_names: Sequence[str]) -> list[str]:
    """Move raw LOB columns to LOBSTER order and keep derived features stable."""

    raw_ordered = canonical_lobster_order(feature_names)
    raw_seen = {name.lower() for name in raw_ordered}
    derived = [name for name in feature_names if name.lower() not in raw_seen]
    return raw_ordered + derived


def build_lob_feature_map(feature_names: Sequence[str]) -> dict[str, Any]:
    """Build a metadata map from semantic LOB roles to tensor indices."""

    index = {name.lower(): i for i, name in enumerate(feature_names)}
    levels: dict[int, dict[str, int]] = {}
    for name in feature_names:
        match = _RAW_LOB_RE.match(name)
        if not match:
            continue
        kind, side, level_raw = match.groups()
        level = int(level_raw)
        key = {
            ("p", "a"): "ask_price",
            ("q", "a"): "ask_size",
            ("p", "b"): "bid_price",
            ("q", "b"): "bid_size",
        }[(kind.lower(), side.lower())]
        levels.setdefault(level, {})[key] = index[name.lower()]

    return {
        "layout": LOBSTER_LAYOUT if levels else "non_lob_or_derived",
        "levels": {
            str(level): values
            for level, values in sorted(levels.items(), key=lambda item: item[0])
        },
    }
