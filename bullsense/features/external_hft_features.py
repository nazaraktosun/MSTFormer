"""Integration helpers for the in-repo ``hft_features`` package.

The library lives at ``libs/bullsense_features/hft_features``. Feature functions
register themselves into ``hft_features.core.base._REGISTRY`` by name when their
module is imported, so this list is what decides which features a config file is
allowed to reference.

The list is explicit on purpose. Importing the tree wholesale would also
register experimental and non-causal features, so anything not named here is
deliberately unavailable to the pipeline.
"""

from __future__ import annotations

import importlib
from typing import Sequence

# Canonical list of feature modules to import and register once.
_HFT_MODULES: Sequence[str] = (
    # Core / basic
    "hft_features.categories.book_features.basic.price_features",
    "hft_features.categories.book_features.basic.volume_features",
    "hft_features.categories.book_features.basic.time_features",
    # Paper: microprice + causal microprice log returns (bps)
    "hft_features.categories.book_features.basic.microprice_features",
    # Paper: time since last genuine book update
    "hft_features.categories.book_features.basic.update_time_features",
    # Advanced
    "hft_features.categories.book_features.advanced.advanced_book_features",
    # Microstructure
    "hft_features.categories.book_features.microstructure.order_flow_features",
    "hft_features.categories.book_features.microstructure.orderbook_imbalance_features",
    # Statistical
    "hft_features.categories.book_features.statistical.momentum",
    "hft_features.categories.book_features.statistical.volatility",
    "hft_features.categories.book_features.statistical.statistics",
    # Return-based
    "hft_features.categories.book_features.spread_duration",
    # Message: schema bridge must be registered before the message blocks that
    # depend on message_type / side / quantity
    "hft_features.categories.message_features.msgf_schema",
    # Paper: five-category order-lifecycle event features
    "hft_features.categories.message_features.event_lifecycle_features",
    # Message / queue
    "hft_features.categories.message_features.queue_features",
    "hft_features.categories.message_features.order_activity_features",
)


def register_hft_features() -> dict:
    """
    Import ``hft_features`` modules and return the shared registry.

    The registry is populated by the decorators inside the imported modules.
    """
    base = importlib.import_module("hft_features.core.base")
    for module in _HFT_MODULES:
        importlib.import_module(module)
    return base._REGISTRY


__all__ = ["register_hft_features"]
