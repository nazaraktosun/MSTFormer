"""Lightweight HFT feature registry loader without side-effect hoops."""

from __future__ import annotations

import importlib
import pkgutil
from functools import lru_cache
from typing import Dict


@lru_cache(maxsize=1)
def load_all_hft_modules() -> Dict:
    """Import all hft_features book feature modules once and return the shared registry."""
    base = importlib.import_module("hft_features.core.base")

    pkg = importlib.import_module("hft_features.categories.book_features")
    for module in pkgutil.walk_packages(pkg.__path__, pkg.__name__ + "."):
        importlib.import_module(module.name)

    return base._REGISTRY


__all__ = ["load_all_hft_modules"]
