"""
Typed metadata containers for prepared datasets.
This dataset is 
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional


@dataclass
class LabelMetadata:
    """Description of the labelling strategy used during preparation."""

    strategy: str
    params: Dict[str, Any]


@dataclass
class SplitRatios:
    """Ratio of samples routed to each dataset split."""

    train: float
    val: float
    test: float


@dataclass
class SplitShapes:
    """Tensor shapes written for each split."""

    X_train: List[int]
    X_val: List[int]
    X_test: List[int]


@dataclass
class SplitClassCounts:
    """Class histogram for each split."""

    train: List[int]
    val: List[int]
    test: List[int]


@dataclass
class DatasetMetadata:
    """Structured metadata persisted alongside prepared datasets."""

    symbol: str
    obid: int
    seq_len: int
    feature_dim: int
    feature_names: List[str]
    setting: str
    label: LabelMetadata
    split_ratios: SplitRatios
    shapes: SplitShapes
    class_counts: SplitClassCounts
    feature_layout: Optional[str] = None
    feature_index_map: Optional[Dict[str, Any]] = None

    def to_dict(self) -> Dict[str, Any]:
        """Return a JSON-serialisable representation."""
        return asdict(self)

    def save_json(self, path: Path) -> None:
        """Persist metadata as JSON."""
        target = Path(path)
        target.parent.mkdir(parents=True, exist_ok=True)
        with target.open("w", encoding="utf-8") as handle:
            json.dump(self.to_dict(), handle, ensure_ascii=True, indent=2)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "DatasetMetadata":
        label = LabelMetadata(
            strategy=str(data["label"]["strategy"]),
            params=dict(data["label"].get("params", {})),
        )
        split_ratios = SplitRatios(
            train=float(data["split_ratios"]["train"]),
            val=float(data["split_ratios"]["val"]),
            test=float(data["split_ratios"]["test"]),
        )
        shapes = SplitShapes(
            X_train=[int(x) for x in data["shapes"]["X_train"]],
            X_val=[int(x) for x in data["shapes"]["X_val"]],
            X_test=[int(x) for x in data["shapes"]["X_test"]],
        )
        class_counts = SplitClassCounts(
            train=[int(x) for x in data["class_counts"]["train"]],
            val=[int(x) for x in data["class_counts"]["val"]],
            test=[int(x) for x in data["class_counts"]["test"]],
        )
        return cls(
            symbol=str(data["symbol"]),
            obid=int(data["obid"]),
            seq_len=int(data["seq_len"]),
            feature_dim=int(data["feature_dim"]),
            feature_names=list(data["feature_names"]),
            setting=str(data["setting"]),
            label=label,
            split_ratios=split_ratios,
            shapes=shapes,
            class_counts=class_counts,
            feature_layout=data.get("feature_layout"),
            feature_index_map=data.get("feature_index_map"),
        )

    @classmethod
    def load_json(cls, path: Path) -> "DatasetMetadata":
        """Load metadata from a JSON file."""
        with Path(path).open("r", encoding="utf-8") as handle:
            payload = json.load(handle)
        return cls.from_dict(payload)
