"""Mid-markout analysis for trained Bullsense classification checkpoints.

This is the Bullsense-native counterpart of the exploratory TLOB_copy markout
scripts. It differs in two important ways:

* Bullsense class order is STABLE=0, UP=1, DOWN=2.
* Markout is mid-focused by default:
    BUY  = future_mid - current_mid
    SELL = current_mid - future_mid

The script can also run a leakage-safe cell-selection protocol:
discover candidate cells on an early date split, freeze the best cells, and
evaluate those cells on later unseen dates.
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace
from typing import Iterable

import numpy as np
import pandas as pd
import polars as pl
import torch

from bullsense.data.dataset import _detect_feature_indices
from bullsense.features.pipeline import prepare_basic_columns
from bullsense.io.lobster_parquet import make_lobster_ingest_fn
from bullsense.model.registry import build_model


STABLE = 0
UP = 1
DOWN = 2


@dataclass(frozen=True)
class Grid:
    hours: list[int]
    edge: list[float]
    strength: list[float]
    buy_obi: list[float]
    sell_obi: list[float]


def _csv_numbers(value: str, cast=float) -> list:
    return [cast(x.strip()) for x in str(value).split(",") if x.strip()]


def _load_json(path: Path) -> dict:
    with Path(path).open("r", encoding="utf-8") as fh:
        return json.load(fh)


def _namespace_from_dict(payload: dict) -> SimpleNamespace:
    return SimpleNamespace(**payload)


def _metadata_from_checkpoint_or_data(ckpt: dict, data_dir: Path) -> dict:
    if isinstance(ckpt.get("metadata"), dict):
        return ckpt["metadata"]
    meta_path = data_dir / "metadata.json"
    if not meta_path.exists():
        raise FileNotFoundError(f"metadata.json not found in {data_dir}")
    return _load_json(meta_path)


def _checkpoint_config(ckpt: dict) -> SimpleNamespace:
    cfg = dict(ckpt.get("config") or {})
    if not cfg:
        raise ValueError("Checkpoint does not contain a config snapshot.")

    # Older checkpoints predate these fields. Keep the same defaults as the
    # model/config layers so archived models still load.
    cfg.setdefault("input_shift_norm", "none")
    cfg.setdefault("dish_init", "standard")
    cfg.setdefault("dish_activate", True)
    cfg.setdefault("revin_affine", True)
    cfg.setdefault("revin_detach_stats", True)
    cfg.setdefault("dain_mode", "adaptive_scale")
    cfg.setdefault("dain_mean_lr", 1e-5)
    cfg.setdefault("dain_gate_lr", 1e-3)
    cfg.setdefault("dain_scale_lr", 1e-5)
    cfg.setdefault("grouped_dain_groups", None)
    cfg.setdefault("attn_dropout", 0.0)
    cfg.setdefault("temporal_causal", False)
    cfg.setdefault("gradient_checkpointing", False)
    cfg.setdefault("use_bilinear_norm", True)
    cfg.setdefault("bilinear_norm_type", "bin")
    cfg.setdefault("temporal_readout", "last")
    cfg.setdefault("is_sin_emb", True)
    cfg.setdefault("order_type_idx", 41)
    return _namespace_from_dict(cfg)


def _load_model(model_path: Path, data_dir: Path, device: torch.device):
    ckpt = torch.load(model_path, map_location=device, weights_only=False)
    metadata = _metadata_from_checkpoint_or_data(ckpt, data_dir)
    cfg = _checkpoint_config(ckpt)
    if (
        str(getattr(cfg, "input_shift_norm", "")).lower() == "grouped_dain"
        and not getattr(cfg, "grouped_dain_groups", None)
        and str(getattr(cfg, "lob_normalizer_style", "")).lower() == "features_only"
    ):
        feature_dim = int(metadata["feature_dim"])
        cfg.grouped_dain_groups = {
            "lob_raw": {
                "indices": list(range(min(40, feature_dim))),
                "mode": str(getattr(cfg, "dain_mode", "adaptive_scale")),
            }
        }

    seq_len = int(metadata["seq_len"])
    input_dim = int(metadata["feature_dim"])
    num_classes = 3
    model = build_model(
        model_type=str(getattr(cfg, "model_type", "tlob")),
        input_dim=input_dim,
        seq_len=seq_len,
        num_classes=num_classes,
        cfg=cfg,
    )
    model.load_state_dict(ckpt["state_dict"])
    model.to(device)
    model.eval()
    return model, cfg, metadata


def _predict_split(
    *,
    model,
    cfg: SimpleNamespace,
    data_dir: Path,
    metadata: dict,
    batch_size: int,
    device: torch.device,
    max_rows: int | None,
    split: str = "test",
    norm_stats: dict | None = None,
) -> pd.DataFrame:
    X_test = np.load(data_dir / f"X_{split}.npy", mmap_mode="r")
    y_test = np.load(data_dir / f"y_{split}.npy", mmap_mode="r")
    ts = np.load(data_dir / f"ts_{split}.npy", mmap_mode="r")
    n_rows = len(y_test) if max_rows is None else min(int(max_rows), len(y_test))

    probs_chunks: list[np.ndarray] = []
    pred_chunks: list[np.ndarray] = []
    true_chunks: list[np.ndarray] = []
    with torch.no_grad():
        for start in range(0, n_rows, batch_size):
            end = min(start + batch_size, n_rows)
            xb_np = np.asarray(X_test[start:end], dtype=np.float32).copy()
            if xb_np.ndim == 2:
                xb_np = xb_np[:, None, :]
            if norm_stats is not None:
                _normalize_lob_batch_inplace(xb_np, norm_stats)
            xb = torch.from_numpy(xb_np).to(device)
            logits = model(xb)
            probs = torch.softmax(logits, dim=1).detach().cpu().numpy()
            probs_chunks.append(probs)
            pred_chunks.append(probs.argmax(axis=1))
            true_chunks.append(np.asarray(y_test[start:end]))

    probs = np.concatenate(probs_chunks, axis=0)
    preds = np.concatenate(pred_chunks, axis=0)
    y_true = np.concatenate(true_chunks, axis=0)
    ts = ts[: len(preds)]

    top2 = np.sort(probs, axis=1)[:, -2:]
    confidence = top2[:, 1]
    margin = top2[:, 1] - top2[:, 0]
    top_class = probs.argmax(axis=1)
    up_competitor = np.maximum(probs[:, STABLE], probs[:, DOWN])
    down_competitor = np.maximum(probs[:, STABLE], probs[:, UP])
    top_direction_gap_score = np.select(
        [top_class == UP, top_class == DOWN],
        [probs[:, UP] - up_competitor, -(probs[:, DOWN] - down_competitor)],
        default=0.0,
    )

    return pd.DataFrame(
        {
            "timestamp_utc": pd.to_datetime(ts),
            "target": y_true,
            "Preds": preds,
            "prob_stable": probs[:, STABLE],
            "prob_up": probs[:, UP],
            "prob_down": probs[:, DOWN],
            "confidence": confidence,
            "margin": margin,
            "score": probs[:, UP] - probs[:, DOWN],
            "score_up_down": probs[:, UP] - probs[:, DOWN],
            "score_top_direction_gap": top_direction_gap_score,
        }
    )


def _maybe_norm_stats(cfg: SimpleNamespace, data_dir: Path, metadata: dict) -> dict | None:
    if not bool(getattr(cfg, "use_lob_normalizer", True)):
        return None
    style = str(getattr(cfg, "lob_normalizer_style", "bullsense"))
    return _fit_lob_normalization(data_dir, metadata, style)


def _fit_lob_normalization(data_dir: Path, metadata: dict, style: str) -> dict:
    X_train = np.load(data_dir / "X_train.npy", mmap_mode="r")
    feature_names = metadata.get("feature_names")
    if feature_names:
        price_idx, vol_idx, feature_idx = _detect_feature_indices(feature_names)
    else:
        feature_dim = X_train.shape[2] if X_train.ndim == 3 else X_train.shape[1]
        price_idx = list(range(0, feature_dim, 2))
        vol_idx = list(range(1, feature_dim, 2))
        feature_idx = []

    eps = 1e-8
    stats: dict[str, object] = {
        "price_idx": price_idx,
        "vol_idx": vol_idx,
        "feature_idx": feature_idx,
        "style": style,
        "price_mean": 0.0,
        "price_std": 1.0,
        "vol_mean": 0.0,
        "vol_std": 1.0,
        "feature_means": None,
        "feature_stds": None,
    }

    if price_idx:
        prices = X_train[:, :, price_idx] if X_train.ndim == 3 else X_train[:, price_idx]
        stats["price_mean"] = float(np.mean(prices))
        stats["price_std"] = float(np.std(prices) + eps)

    if vol_idx:
        vols = X_train[:, :, vol_idx] if X_train.ndim == 3 else X_train[:, vol_idx]
        vols = np.log1p(vols) if style == "bullsense" else vols
        stats["vol_mean"] = float(np.mean(vols))
        stats["vol_std"] = float(np.std(vols) + eps)

    if feature_idx:
        feats = X_train[:, :, feature_idx] if X_train.ndim == 3 else X_train[:, feature_idx]
        feats = np.asarray(feats, dtype=np.float32)
        if feats.ndim == 3:
            stats["feature_means"] = feats.mean(axis=(0, 1))
            stats["feature_stds"] = feats.std(axis=(0, 1)) + eps
        else:
            stats["feature_means"] = feats.mean(axis=0)
            stats["feature_stds"] = feats.std(axis=0) + eps

    print(
        f"Stats -> price mean: {stats['price_mean']:.4f}, "
        f"vol mean ({'log1p' if style == 'bullsense' else 'raw'}): {stats['vol_mean']:.4f}, "
        f"feature cols z-scored: {len(feature_idx)}",
        flush=True,
    )
    return stats


def _normalize_lob_batch_inplace(batch: np.ndarray, stats: dict) -> None:
    price_idx = stats["price_idx"]
    vol_idx = stats["vol_idx"]
    feature_idx = stats["feature_idx"]

    if stats["style"] != "features_only" and price_idx:
        batch[:, :, price_idx] = (
            batch[:, :, price_idx] - float(stats["price_mean"])
        ) / float(stats["price_std"])

    if stats["style"] != "features_only" and vol_idx:
        vols = batch[:, :, vol_idx]
        if stats["style"] == "bullsense":
            vols = np.log1p(vols)
        batch[:, :, vol_idx] = (vols - float(stats["vol_mean"])) / float(stats["vol_std"])

    if feature_idx and stats["feature_means"] is not None:
        means = np.asarray(stats["feature_means"], dtype=np.float32).reshape(1, 1, -1)
        stds = np.asarray(stats["feature_stds"], dtype=np.float32).reshape(1, 1, -1)
        batch[:, :, feature_idx] = (batch[:, :, feature_idx] - means) / stds


def _load_raw_lob(
    *,
    raw_lob_parquet: Path,
    symbol: str,
    include_dates: Iterable[str],
    session_tz: str,
    session_start: str,
    session_end: str,
    price_scale: float,
    ingest_resample_ms: int,
) -> pd.DataFrame:
    ingest_fn = make_lobster_ingest_fn(
        raw_lob_parquet,
        raw_lob_parquet,
        symbol=symbol,
        tz=session_tz,
        price_scale=price_scale,
        resample_ms=ingest_resample_ms,
        load_messages=False,
        include_dates=include_dates,
    )
    ingestion = ingest_fn(0)
    frame = prepare_basic_columns(
        ingestion.orderbook,
        session_tz=session_tz,
        session_start=session_start,
        session_end=session_end,
        weekday_only=True,
    )
    required = ["datetime", "pa1", "qa1", "pb1", "qb1", "mid_price", "mid_micro"]
    missing = [c for c in required if c not in frame.columns]
    if missing:
        raise ValueError(f"Raw LOB frame missing required columns: {missing}")

    if ingest_resample_ms:
        ts_expr = pl.col("datetime").dt.truncate(f"{int(ingest_resample_ms)}ms")
    else:
        ts_expr = pl.col("datetime")
    frame = frame.with_columns(
        [
            ts_expr.dt.convert_time_zone("UTC")
            .dt.replace_time_zone(None)
            .alias("timestamp_utc")
        ]
    )
    df = pd.DataFrame(frame.select(required + ["timestamp_utc"]).to_dicts())
    df["datetime"] = pd.to_datetime(df["datetime"])
    df["timestamp_utc"] = pd.to_datetime(df["timestamp_utc"])
    df = df.sort_values("timestamp_utc").drop_duplicates("timestamp_utc", keep="last")
    df["mid"] = (df["pa1"] + df["pb1"]) / 2.0
    df["spread"] = df["pa1"] - df["pb1"]
    df["obi"] = (df["qb1"] - df["qa1"]) / (df["qb1"] + df["qa1"] + 1e-9)
    df["hour"] = df["datetime"].dt.hour
    df["date"] = df["datetime"].dt.date
    return df


def _attach_lob(preds: pd.DataFrame, lob: pd.DataFrame) -> pd.DataFrame:
    out = preds.merge(lob, on="timestamp_utc", how="left", validate="one_to_one")
    miss = out["mid_price"].isna().sum()
    if miss:
        raise ValueError(
            f"Could not align {miss:,}/{len(out):,} predictions to raw LOB timestamps. "
            "Check raw path, timezone, and ingest_resample_ms."
        )
    return out.sort_values("timestamp_utc").reset_index(drop=True)


def _prediction_session_dates(preds: pd.DataFrame, session_tz: str) -> list[str]:
    ts = pd.to_datetime(preds["timestamp_utc"])
    if ts.dt.tz is None:
        ts = ts.dt.tz_localize("UTC")
    else:
        ts = ts.dt.tz_convert("UTC")
    local_dates = ts.dt.tz_convert(session_tz).dt.strftime("%Y-%m-%d")
    return sorted(local_dates.dropna().unique().tolist())


def _add_mid_markouts(df: pd.DataFrame, steps: Iterable[int], price_col: str) -> pd.DataFrame:
    out = df.sort_values("timestamp_utc").reset_index(drop=True).copy()
    out["markout_price"] = out[price_col].astype(float)
    for step in steps:
        future = out.groupby("date", sort=False)["markout_price"].shift(-int(step))
        buy = future - out["markout_price"]
        sell = out["markout_price"] - future
        out[f"future_mid_{step}"] = future
        out[f"buy_markout_{step}"] = buy
        out[f"sell_markout_{step}"] = sell
        out[f"buy_markout_bps_{step}"] = buy / out["markout_price"] * 1e4
        out[f"sell_markout_bps_{step}"] = sell / out["markout_price"] * 1e4
        out[f"signal_markout_{step}"] = np.select(
            [out["Preds"] == UP, out["Preds"] == DOWN],
            [buy, sell],
            default=np.nan,
        )
        out[f"signal_markout_bps_{step}"] = (
            out[f"signal_markout_{step}"] / out["markout_price"] * 1e4
        )
        out[f"passive_markout_bps_{step}"] = np.select(
            [out["Preds"] == UP, out["Preds"] == DOWN],
            [(future - out["pb1"]), (out["pa1"] - future)],
            default=np.nan,
        ) / out["markout_price"] * 1e4
        out[f"aggressive_markout_bps_{step}"] = np.select(
            [out["Preds"] == UP, out["Preds"] == DOWN],
            [(future - out["pa1"]), (out["pb1"] - future)],
            default=np.nan,
        ) / out["markout_price"] * 1e4
    out["half_spread_bps"] = (out["spread"] / 2.0) / out["markout_price"] * 1e4
    return out


def _balanced_edge(df: pd.DataFrame, col: str) -> float:
    buy = df.loc[df["Preds"] == UP, col].dropna()
    sell = df.loc[df["Preds"] == DOWN, col].dropna()
    if buy.empty or sell.empty:
        return np.nan
    return float(0.5 * (buy.mean() + sell.mean()))


def _block_bootstrap_ci(
    df: pd.DataFrame, col: str, *, reps: int, seed: int = 0
) -> tuple[float, float]:
    days = sorted(df["date"].dropna().unique())
    if len(days) < 2:
        return (np.nan, np.nan)
    rng = np.random.default_rng(seed)
    by_day = {key: frame for key, frame in df.groupby("date", sort=False)}
    draws = []
    for _ in range(int(reps)):
        picked = rng.choice(len(days), len(days), replace=True)
        sample = pd.concat([by_day[days[i]] for i in picked], ignore_index=True)
        draws.append(_balanced_edge(sample, col))
    finite = np.asarray([x for x in draws if np.isfinite(x)])
    if finite.size == 0:
        return (np.nan, np.nan)
    return (float(np.percentile(finite, 2.5)), float(np.percentile(finite, 97.5)))


def _directional(df: pd.DataFrame) -> pd.DataFrame:
    return df[df["Preds"].isin([UP, DOWN])]


def _decile_table(df: pd.DataFrame, *, axis: str, step: int, bins: int) -> pd.DataFrame:
    col = f"signal_markout_bps_{step}"
    sub = _directional(df).dropna(subset=[col, axis]).copy()
    if sub.empty:
        return pd.DataFrame()
    sub["bucket"] = pd.qcut(sub[axis], bins, labels=False, duplicates="drop")
    rows = []
    for bucket, group in sub.groupby("bucket", sort=True):
        rows.append(
            {
                "axis": axis,
                "markout_steps": int(step),
                "bucket": int(bucket) + 1,
                "n": int(len(group)),
                "axis_lo": float(group[axis].min()),
                "axis_hi": float(group[axis].max()),
                "balanced_bps": _balanced_edge(group, col),
                "passive_bps": _balanced_edge(group, f"passive_markout_bps_{step}"),
                "aggressive_bps": _balanced_edge(group, f"aggressive_markout_bps_{step}"),
                "win_rate": float((group[col] > 0).mean()),
            }
        )
    return pd.DataFrame(rows)


def _grid_seconds(df: pd.DataFrame) -> float:
    gaps = df.sort_values("timestamp_utc")["timestamp_utc"].diff().dt.total_seconds()
    positive = gaps[gaps > 0]
    return float(positive.median()) if not positive.empty else 0.0


def _non_overlapping(df: pd.DataFrame, *, step: int, grid_seconds: float) -> pd.DataFrame:
    if df.empty:
        return df
    ordered = df.sort_values("timestamp_utc")
    hold = pd.Timedelta(seconds=grid_seconds * int(step))
    keep = np.zeros(len(ordered), dtype=bool)
    last: dict[object, pd.Timestamp] = {}
    for i, (day, ts) in enumerate(zip(ordered["date"].to_numpy(), ordered["timestamp_utc"])):
        if day not in last or (ts - last[day]) >= hold:
            keep[i] = True
            last[day] = ts
    return ordered[keep]


def _summarize_markouts(df: pd.DataFrame, steps: Iterable[int]) -> pd.DataFrame:
    rows = []
    for step in steps:
        for side, pred_value, col, passive_col, aggressive_col in [
            (
                "BUY",
                UP,
                f"buy_markout_bps_{step}",
                f"passive_markout_bps_{step}",
                f"aggressive_markout_bps_{step}",
            ),
            (
                "SELL",
                DOWN,
                f"sell_markout_bps_{step}",
                f"passive_markout_bps_{step}",
                f"aggressive_markout_bps_{step}",
            ),
            (
                "ALL",
                None,
                f"signal_markout_bps_{step}",
                f"passive_markout_bps_{step}",
                f"aggressive_markout_bps_{step}",
            ),
        ]:
            sub = df[df["Preds"] == pred_value] if pred_value is not None else df[df["Preds"].isin([UP, DOWN])]
            x = sub[col].dropna()
            passive = sub.loc[x.index, passive_col] if len(x) else pd.Series(dtype=float)
            aggressive = sub.loc[x.index, aggressive_col] if len(x) else pd.Series(dtype=float)
            rows.append(
                {
                    "markout_steps": int(step),
                    "side": side,
                    "n": int(len(x)),
                    "mean_bps": float(x.mean()) if len(x) else np.nan,
                    "passive_mean_bps": float(passive.mean()) if len(passive) else np.nan,
                    "aggressive_mean_bps": float(aggressive.mean()) if len(aggressive) else np.nan,
                    "passive_pass_rate": float((passive > 0).mean()) if len(passive) else np.nan,
                    "aggressive_pass_rate": float((aggressive > 0).mean()) if len(aggressive) else np.nan,
                    "median_bps": float(x.median()) if len(x) else np.nan,
                    "win_rate": float((x > 0).mean()) if len(x) else np.nan,
                    "mean_margin": float(sub.loc[x.index, "margin"].mean()) if len(x) else np.nan,
                    "mean_confidence": float(sub.loc[x.index, "confidence"].mean()) if len(x) else np.nan,
                }
            )
        col = f"signal_markout_bps_{step}"
        directional = _directional(df)
        drift = df[f"buy_markout_bps_{step}"].dropna()
        rows.append(
            {
                "markout_steps": int(step),
                "side": "BALANCED",
                "n": int(directional[col].notna().sum()),
                "mean_bps": _balanced_edge(directional, col),
                "passive_mean_bps": _balanced_edge(directional, f"passive_markout_bps_{step}"),
                "aggressive_mean_bps": _balanced_edge(directional, f"aggressive_markout_bps_{step}"),
                "passive_pass_rate": float((directional[f"passive_markout_bps_{step}"].dropna() > 0).mean()),
                "aggressive_pass_rate": float((directional[f"aggressive_markout_bps_{step}"].dropna() > 0).mean()),
                "median_bps": np.nan,
                "win_rate": np.nan,
                "mean_margin": np.nan,
                "mean_confidence": np.nan,
            }
        )
        rows.append(
            {
                "markout_steps": int(step),
                "side": "DRIFT",
                "n": int(len(drift)),
                "mean_bps": float(drift.mean()) if len(drift) else np.nan,
                "passive_mean_bps": np.nan,
                "aggressive_mean_bps": np.nan,
                "passive_pass_rate": np.nan,
                "aggressive_pass_rate": np.nan,
                "median_bps": float(drift.median()) if len(drift) else np.nan,
                "win_rate": np.nan,
                "mean_margin": np.nan,
                "mean_confidence": np.nan,
            }
        )
    return pd.DataFrame(rows)


def _signal_filter_report(
    *,
    test: pd.DataFrame,
    val: pd.DataFrame | None,
    step: int,
    keep_fracs: list[float],
    bootstrap_reps: int,
    grid_seconds: float,
) -> pd.DataFrame:
    col = f"signal_markout_bps_{step}"
    universe = _directional(test).dropna(subset=[col])
    if universe.empty:
        return pd.DataFrame()
    selector = _directional(val).dropna(subset=[col]) if val is not None and not val.empty else None

    rows = []
    for frac in [1.0, *keep_fracs]:
        if frac >= 1.0:
            threshold = float(universe["confidence"].min())
            source = "none"
        elif selector is not None:
            threshold = float(selector["confidence"].quantile(1.0 - frac))
            source = "val"
        else:
            threshold = float(universe["confidence"].quantile(1.0 - frac))
            source = "test"
        kept = universe[universe["confidence"] >= threshold]
        if kept.empty:
            continue
        thinned = _non_overlapping(kept, step=step, grid_seconds=grid_seconds)
        lo, hi = _block_bootstrap_ci(thinned, col, reps=bootstrap_reps)
        half = float(thinned["half_spread_bps"].mean())
        balanced = _balanced_edge(thinned, col)
        passive_col = f"passive_markout_bps_{step}"
        aggressive_col = f"aggressive_markout_bps_{step}"
        passive = thinned[passive_col].dropna()
        aggressive = thinned[aggressive_col].dropna()
        rows.append(
            {
                "markout_steps": int(step),
                "keep_frac": float(frac),
                "threshold_source": source,
                "confidence_threshold": threshold,
                "n_overlapping": int(len(kept)),
                "n_trades": int(len(thinned)),
                "days": int(thinned["date"].nunique()),
                "half_spread_bps": half,
                "mid_to_mid_bps": balanced,
                "mid_ci_lo": lo,
                "mid_ci_hi": hi,
                "passive_bps": _balanced_edge(thinned, passive_col),
                "aggressive_bps": _balanced_edge(thinned, aggressive_col),
                "passive_pass_rate": float((passive > 0).mean()) if len(passive) else np.nan,
                "aggressive_pass_rate": float((aggressive > 0).mean()) if len(aggressive) else np.nan,
                "cross_both_bps": balanced - 2.0 * half if np.isfinite(balanced) else np.nan,
                "win_rate": float((thinned[col] > 0).mean()),
            }
        )
    return pd.DataFrame(rows)


def _side_signal_filter_report(
    *,
    test: pd.DataFrame,
    val: pd.DataFrame | None,
    step: int,
    keep_fracs: list[float],
    grid_seconds: float,
) -> pd.DataFrame:
    rows = []
    for side, pred_value, markout_col in [
        ("BUY", UP, f"buy_markout_bps_{step}"),
        ("SELL", DOWN, f"sell_markout_bps_{step}"),
    ]:
        universe = test[test["Preds"] == pred_value].dropna(subset=[markout_col])
        if universe.empty:
            continue
        selector = (
            val[val["Preds"] == pred_value].dropna(subset=[markout_col])
            if val is not None and not val.empty
            else None
        )
        passive_col = f"passive_markout_bps_{step}"
        aggressive_col = f"aggressive_markout_bps_{step}"

        for frac in [1.0, *keep_fracs]:
            if frac >= 1.0:
                threshold = float(universe["confidence"].min())
                source = "none"
            elif selector is not None and not selector.empty:
                threshold = float(selector["confidence"].quantile(1.0 - frac))
                source = "val_side"
            else:
                threshold = float(universe["confidence"].quantile(1.0 - frac))
                source = "test_side"

            kept = universe[universe["confidence"] >= threshold]
            if kept.empty:
                continue
            thinned = _non_overlapping(kept, step=step, grid_seconds=grid_seconds)
            x = thinned[markout_col].dropna()
            passive = thinned.loc[x.index, passive_col] if len(x) else pd.Series(dtype=float)
            aggressive = thinned.loc[x.index, aggressive_col] if len(x) else pd.Series(dtype=float)
            half = float(thinned.loc[x.index, "half_spread_bps"].mean()) if len(x) else np.nan
            mid = float(x.mean()) if len(x) else np.nan
            rows.append(
                {
                    "markout_steps": int(step),
                    "side": side,
                    "keep_frac": float(frac),
                    "threshold_source": source,
                    "confidence_threshold": threshold,
                    "n_overlapping": int(len(kept)),
                    "n_trades": int(len(x)),
                    "days": int(thinned.loc[x.index, "date"].nunique()) if len(x) else 0,
                    "half_spread_bps": half,
                    "mid_to_mid_bps": mid,
                    "passive_bps": float(passive.mean()) if len(passive) else np.nan,
                    "aggressive_bps": float(aggressive.mean()) if len(aggressive) else np.nan,
                    "passive_pass_rate": float((passive > 0).mean()) if len(passive) else np.nan,
                    "aggressive_pass_rate": float((aggressive > 0).mean()) if len(aggressive) else np.nan,
                    "cross_both_bps": mid - 2.0 * half if np.isfinite(mid) and np.isfinite(half) else np.nan,
                    "win_rate": float((x > 0).mean()) if len(x) else np.nan,
                }
            )
    return pd.DataFrame(rows)


def _score_filter_report(
    *,
    test: pd.DataFrame,
    val: pd.DataFrame | None,
    step: int,
    keep_fracs: list[float],
    grid_seconds: float,
    score_col: str = "score",
    score_mode: str = "up_down",
) -> pd.DataFrame:
    """Markout report from score sign, not argmax class.

    This is useful when STABLE is the argmax but p_up - p_down still ranks
    directional pressure.
    """
    universe = test.dropna(subset=[f"buy_markout_bps_{step}", f"sell_markout_bps_{step}", score_col]).copy()
    universe = universe[universe[score_col].abs() > 0]
    if universe.empty:
        return pd.DataFrame()
    selector = (
        val.dropna(subset=[score_col]).copy()
        if val is not None and not val.empty
        else None
    )

    rows = []
    for frac in [1.0, *keep_fracs]:
        if frac >= 1.0:
            threshold = 0.0
            source = f"{score_mode}_sign"
        elif selector is not None and not selector.empty:
            threshold = float(selector[score_col].abs().quantile(1.0 - frac))
            source = f"val_abs_{score_mode}"
        else:
            threshold = float(universe[score_col].abs().quantile(1.0 - frac))
            source = f"test_abs_{score_mode}"

        kept = universe[universe[score_col].abs() >= threshold].copy()
        if kept.empty:
            continue
        thinned = _non_overlapping(kept, step=step, grid_seconds=grid_seconds).copy()
        if thinned.empty:
            continue

        is_buy = thinned[score_col] > 0
        price = thinned["markout_price"].to_numpy(dtype=float)
        future = thinned[f"future_mid_{step}"].to_numpy(dtype=float)
        bid = thinned["pb1"].to_numpy(dtype=float)
        ask = thinned["pa1"].to_numpy(dtype=float)
        mid = np.where(is_buy, future - price, price - future) / price * 1e4
        passive = np.where(is_buy, future - bid, ask - future) / price * 1e4
        aggressive = np.where(is_buy, future - ask, bid - future) / price * 1e4
        finite = np.isfinite(mid)
        if not finite.any():
            continue
        mid = mid[finite]
        passive = passive[finite]
        aggressive = aggressive[finite]
        sides = is_buy.to_numpy()[finite]
        half = float(thinned.loc[finite, "half_spread_bps"].mean()) if len(thinned) == len(finite) else float(thinned["half_spread_bps"].mean())
        rows.append(
            {
                "markout_steps": int(step),
                "score_mode": score_mode,
                "keep_frac": float(frac),
                "threshold_source": source,
                "abs_score_threshold": threshold,
                "n_overlapping": int(len(kept)),
                "n_trades": int(len(mid)),
                "days": int(thinned.loc[finite, "date"].nunique()) if len(thinned) == len(finite) else int(thinned["date"].nunique()),
                "buy_frac": float(np.mean(sides)),
                "half_spread_bps": half,
                "mid_to_mid_bps": float(np.mean(mid)),
                "passive_bps": float(np.mean(passive)),
                "aggressive_bps": float(np.mean(aggressive)),
                "passive_pass_rate": float(np.mean(passive > 0)),
                "aggressive_pass_rate": float(np.mean(aggressive > 0)),
                "cross_both_bps": float(np.mean(mid) - 2.0 * half),
                "win_rate": float(np.mean(mid > 0)),
            }
        )
    return pd.DataFrame(rows)


def _split_discovery_validation(df: pd.DataFrame, discovery_frac: float) -> tuple[pd.DataFrame, pd.DataFrame]:
    dates = sorted(df["date"].dropna().unique())
    if len(dates) < 2:
        raise ValueError("Need at least two dates for leakage-safe discovery/validation split.")
    cut = max(1, min(len(dates) - 1, int(round(len(dates) * discovery_frac))))
    disc_dates = set(dates[:cut])
    return df[df["date"].isin(disc_dates)].copy(), df[~df["date"].isin(disc_dates)].copy()


def _wilson_lower_bound(wins: int, n: int, z: float = 1.96) -> float:
    if n <= 0:
        return np.nan
    p = wins / n
    denom = 1.0 + (z * z) / n
    centre = p + (z * z) / (2.0 * n)
    radius = z * np.sqrt((p * (1.0 - p) + (z * z) / (4.0 * n)) / n)
    return float((centre - radius) / denom)


def _candidate_rows(df: pd.DataFrame, *, side: str, step: int, grid: Grid, spread_cutoff: float) -> list[dict]:
    pred_value = UP if side == "BUY" else DOWN
    markout_col = f"{'buy' if side == 'BUY' else 'sell'}_markout_{step}"
    bps_col = f"{'buy' if side == 'BUY' else 'sell'}_markout_bps_{step}"
    rows = []
    obi_thresholds = grid.buy_obi if side == "BUY" else grid.sell_obi

    for hour in grid.hours:
        hour_df = df[(df["Preds"] == pred_value) & (df["hour"] == hour)]
        if hour_df.empty:
            continue
        for edge_t in grid.edge:
            for strength_t in grid.strength:
                for obi_t in obi_thresholds:
                    mask = (
                        (hour_df["margin"] >= edge_t)
                        & (hour_df["confidence"] >= strength_t)
                        & (hour_df["spread"] <= spread_cutoff)
                    )
                    if side == "BUY":
                        mask = mask & (hour_df["obi"] < obi_t)
                    else:
                        mask = mask & (hour_df["obi"] > obi_t)

                    sub = hour_df[mask]
                    x = sub[markout_col].dropna()
                    if len(x) == 0:
                        continue
                    bps = sub.loc[x.index, bps_col].astype(float)
                    n = int(len(x))
                    wins = int((x > 0).sum())
                    mean_bps = float(bps.mean())
                    std_bps = float(bps.std(ddof=1)) if n > 1 else 0.0
                    mean_bps_lcb = float(mean_bps - 1.96 * std_bps / np.sqrt(n)) if n > 1 else mean_bps
                    rows.append(
                        {
                            "side": side,
                            "markout_steps": int(step),
                            "hour": int(hour),
                            "edge_threshold": float(edge_t),
                            "strength_threshold": float(strength_t),
                            "obi_threshold": float(obi_t),
                            "spread_cutoff": float(spread_cutoff),
                            "n": n,
                            "win_rate": float(wins / n),
                            "win_rate_lcb": _wilson_lower_bound(wins, n),
                            "mean_markout": float(x.mean()),
                            "mean_bps": mean_bps,
                            "std_bps": std_bps,
                            "mean_bps_lcb": mean_bps_lcb,
                            "net_bps": float(bps.sum()),
                            "t_stat_bps": float(mean_bps / (std_bps / np.sqrt(n))) if std_bps > 0 and n > 1 else np.nan,
                            "mean_margin": float(sub.loc[x.index, "margin"].mean()),
                            "mean_confidence": float(sub.loc[x.index, "confidence"].mean()),
                            "mean_obi": float(sub.loc[x.index, "obi"].mean()),
                            "mean_spread": float(sub.loc[x.index, "spread"].mean()),
                        }
                    )
    return rows


def _select_cells(
    candidates: pd.DataFrame,
    min_n: int,
    min_wr: float,
    min_wr_lcb: float,
    min_mean_bps_lcb: float,
    min_net_bps: float,
    max_cells: int,
) -> pd.DataFrame:
    if candidates.empty:
        return candidates
    valid = candidates[
        (candidates["n"] >= min_n)
        & (candidates["win_rate"] >= min_wr)
        & (candidates["win_rate_lcb"] >= min_wr_lcb)
        & (candidates["mean_bps"] > 0)
        & (candidates["mean_bps_lcb"] >= min_mean_bps_lcb)
        & (candidates["net_bps"] >= min_net_bps)
    ].copy()
    if valid.empty:
        return valid
    valid["selection_score"] = valid["mean_bps_lcb"] * np.sqrt(valid["n"])
    valid = valid.sort_values(
        ["side", "markout_steps", "hour", "selection_score", "mean_bps_lcb", "win_rate_lcb", "n"],
        ascending=[True, True, True, False, False, False, False],
    )
    selected = valid.groupby(["side", "markout_steps", "hour"], as_index=False).head(1)
    selected = selected.sort_values(
        ["selection_score", "mean_bps_lcb", "net_bps"],
        ascending=[False, False, False],
    )
    if max_cells > 0:
        selected = selected.head(int(max_cells))
    return selected.reset_index(drop=True)


def _apply_selected_cells(df: pd.DataFrame, selected: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for _, cell in selected.iterrows():
        side = str(cell["side"])
        step = int(cell["markout_steps"])
        pred_value = UP if side == "BUY" else DOWN
        markout_col = f"{'buy' if side == 'BUY' else 'sell'}_markout_{step}"
        bps_col = f"{'buy' if side == 'BUY' else 'sell'}_markout_bps_{step}"
        sub = df[
            (df["Preds"] == pred_value)
            & (df["hour"] == int(cell["hour"]))
            & (df["margin"] >= float(cell["edge_threshold"]))
            & (df["confidence"] >= float(cell["strength_threshold"]))
            & (df["spread"] <= float(cell["spread_cutoff"]))
        ]
        if side == "BUY":
            sub = sub[sub["obi"] < float(cell["obi_threshold"])]
        else:
            sub = sub[sub["obi"] > float(cell["obi_threshold"])]

        x = sub[markout_col].dropna()
        rows.append(
            {
                "side": side,
                "markout_steps": step,
                "hour": int(cell["hour"]),
                "edge_threshold": float(cell["edge_threshold"]),
                "strength_threshold": float(cell["strength_threshold"]),
                "obi_threshold": float(cell["obi_threshold"]),
                "spread_cutoff": float(cell["spread_cutoff"]),
                "n": int(len(x)),
                "win_rate": float((x > 0).mean()) if len(x) else np.nan,
                "mean_markout": float(x.mean()) if len(x) else np.nan,
                "mean_bps": float(sub.loc[x.index, bps_col].mean()) if len(x) else np.nan,
                "mean_margin": float(sub.loc[x.index, "margin"].mean()) if len(x) else np.nan,
                "mean_confidence": float(sub.loc[x.index, "confidence"].mean()) if len(x) else np.nan,
            }
        )
    return pd.DataFrame(rows)


def _selected_cell_trades(df: pd.DataFrame, selected: pd.DataFrame) -> pd.DataFrame:
    rows = []
    if selected.empty:
        return pd.DataFrame()
    for _, cell in selected.iterrows():
        side = str(cell["side"])
        step = int(cell["markout_steps"])
        pred_value = UP if side == "BUY" else DOWN
        bps_col = f"{'buy' if side == 'BUY' else 'sell'}_markout_bps_{step}"
        sub = df[
            (df["Preds"] == pred_value)
            & (df["hour"] == int(cell["hour"]))
            & (df["margin"] >= float(cell["edge_threshold"]))
            & (df["confidence"] >= float(cell["strength_threshold"]))
            & (df["spread"] <= float(cell["spread_cutoff"]))
        ].copy()
        if side == "BUY":
            sub = sub[sub["obi"] < float(cell["obi_threshold"])]
        else:
            sub = sub[sub["obi"] > float(cell["obi_threshold"])]
        sub = sub.dropna(subset=[bps_col]).copy()
        if sub.empty:
            continue
        sub["side"] = side
        sub["markout_steps"] = step
        sub["trade_bps"] = sub[bps_col].astype(float)
        sub["cell_hour"] = int(cell["hour"])
        rows.append(
            sub[
                [
                    "timestamp_utc",
                    "side",
                    "markout_steps",
                    "cell_hour",
                    "trade_bps",
                    "margin",
                    "confidence",
                    "spread",
                    "obi",
                ]
            ]
        )
    if not rows:
        return pd.DataFrame()
    return pd.concat(rows, ignore_index=True).sort_values("timestamp_utc").reset_index(drop=True)


def _summarize_trade_filter(name: str, trades: pd.DataFrame) -> dict:
    if trades.empty:
        return {
            "filter": name,
            "trades": 0,
            "net_pnl_bps": np.nan,
            "win_rate": np.nan,
            "avg_bps": np.nan,
            "max_dd_bps": np.nan,
        }
    pnl = trades["trade_bps"].astype(float).to_numpy()
    equity = np.cumsum(pnl)
    running_max = np.maximum.accumulate(np.r_[0.0, equity])[:-1]
    drawdown = equity - running_max
    return {
        "filter": name,
        "trades": int(len(pnl)),
        "net_pnl_bps": float(np.sum(pnl)),
        "win_rate": float(np.mean(pnl > 0)),
        "avg_bps": float(np.mean(pnl)),
        "max_dd_bps": float(np.min(drawdown)) if len(drawdown) else 0.0,
    }


def _all_directional_trades(df: pd.DataFrame, step: int) -> pd.DataFrame:
    rows = []
    for side, pred_value in [("BUY", UP), ("SELL", DOWN)]:
        bps_col = f"{'buy' if side == 'BUY' else 'sell'}_markout_bps_{int(step)}"
        sub = df[df["Preds"] == pred_value].dropna(subset=[bps_col]).copy()
        if sub.empty:
            continue
        sub["side"] = side
        sub["markout_steps"] = int(step)
        sub["trade_bps"] = sub[bps_col].astype(float)
        rows.append(
            sub[
                [
                    "timestamp_utc",
                    "side",
                    "markout_steps",
                    "trade_bps",
                    "margin",
                    "confidence",
                    "spread",
                    "obi",
                ]
            ]
        )
    if not rows:
        return pd.DataFrame()
    return pd.concat(rows, ignore_index=True).sort_values("timestamp_utc").reset_index(drop=True)


def _run_cell_protocol(
    df: pd.DataFrame,
    *,
    steps: list[int],
    grid: Grid,
    discovery_frac: float,
    min_n: int,
    min_wr: float,
    min_wr_lcb: float,
    min_mean_bps_lcb: float,
    min_net_bps: float,
    max_cells: int,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    discovery, validation = _split_discovery_validation(df, discovery_frac)
    spread_cutoff = float(discovery["spread"].median())

    rows = []
    for step in steps:
        rows.extend(_candidate_rows(discovery, side="BUY", step=step, grid=grid, spread_cutoff=spread_cutoff))
        rows.extend(_candidate_rows(discovery, side="SELL", step=step, grid=grid, spread_cutoff=spread_cutoff))

    candidates = pd.DataFrame(rows)
    selected = _select_cells(
        candidates,
        min_n=min_n,
        min_wr=min_wr,
        min_wr_lcb=min_wr_lcb,
        min_mean_bps_lcb=min_mean_bps_lcb,
        min_net_bps=min_net_bps,
        max_cells=max_cells,
    )
    validation_stats = _apply_selected_cells(validation, selected) if not selected.empty else pd.DataFrame()
    return candidates, selected, validation_stats


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Mid-markout backtest for Bullsense checkpoints.")
    parser.add_argument("--model", type=Path, required=True, help="Path to trained model_*.pt")
    parser.add_argument("--data-dir", type=Path, required=True, help="Prepared data dir with X_test/y_test/ts_test")
    parser.add_argument("--raw-lob-parquet", type=Path, default=None, help="Raw LOBSTER parquet dir. Defaults to DATA_ROOT/SYMBOL")
    parser.add_argument("--data-root", type=Path, default=Path("/data/ai/lobster_parquet"))
    parser.add_argument("--symbol", type=str, default=None)
    parser.add_argument("--out-dir", type=Path, default=None)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--max-test-rows", type=int, default=None, help="Optional smoke-test limit.")
    parser.add_argument("--session-tz", type=str, default="America/New_York")
    parser.add_argument("--session-start", type=str, default="09:30")
    parser.add_argument("--session-end", type=str, default="16:00")
    parser.add_argument("--price-scale", type=float, default=1000.0)
    parser.add_argument("--ingest-resample-ms", type=int, default=1000)
    parser.add_argument("--markout-price-col", choices=["mid_price", "mid_micro"], default="mid_price")
    parser.add_argument("--markout-steps", default="10,30,60")
    parser.add_argument("--hours", default="7,8,9,10,11,12,13,14")
    parser.add_argument("--edge-grid", default="0.00,0.03,0.05,0.08,0.10,0.12")
    parser.add_argument("--strength-grid", default="0.35,0.40,0.45,0.50")
    parser.add_argument("--buy-obi-grid", default="-0.30,-0.50,-0.70")
    parser.add_argument("--sell-obi-grid", default="0.30,0.50,0.70")
    parser.add_argument("--discovery-frac", type=float, default=0.5)
    parser.add_argument("--min-n", type=int, default=20)
    parser.add_argument("--min-win-rate", type=float, default=0.60)
    parser.add_argument("--min-win-rate-lcb", type=float, default=0.50)
    parser.add_argument("--min-mean-bps-lcb", type=float, default=0.0)
    parser.add_argument("--min-net-bps", type=float, default=0.0)
    parser.add_argument("--max-selected-cells", type=int, default=8, help="0 means no cap.")
    parser.add_argument("--keep-fracs", default="0.50,0.25,0.10", help="Confidence-filter keep fractions.")
    parser.add_argument(
        "--score-mode",
        choices=["up_down", "top_direction_gap"],
        default="up_down",
        help=(
            "Score used by score_filter_summary. up_down uses P(up)-P(down). "
            "top_direction_gap uses the top directional class margin against "
            "its strongest competitor, including STABLE; STABLE argmax yields no score trade."
        ),
    )
    parser.add_argument("--bootstrap-reps", type=int, default=500)
    parser.add_argument("--skip-cell-protocol", action="store_true", help="Only write raw markout summary.")
    parser.add_argument("--trades-out", action="store_true", help="Write full aligned per-signal table.")
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    if args.device == "cuda" and not torch.cuda.is_available():
        device = torch.device("cpu")
    else:
        device = torch.device(args.device)

    model, cfg, metadata = _load_model(args.model, args.data_dir, device)
    label_params = metadata.get("label", {}).get("params", {})
    label_price_col = label_params.get("price_col")
    symbol = args.symbol or metadata.get("symbol")
    if not symbol:
        raise ValueError("Could not infer symbol; pass --symbol.")

    raw_lob = args.raw_lob_parquet or (args.data_root / symbol)
    out_dir = args.out_dir or (args.model.parent / "mid_markout")
    out_dir.mkdir(parents=True, exist_ok=True)

    steps = _csv_numbers(args.markout_steps, int)
    grid = Grid(
        hours=_csv_numbers(args.hours, int),
        edge=_csv_numbers(args.edge_grid, float),
        strength=_csv_numbers(args.strength_grid, float),
        buy_obi=_csv_numbers(args.buy_obi_grid, float),
        sell_obi=_csv_numbers(args.sell_obi_grid, float),
    )

    print(f"model: {args.model}", flush=True)
    print(f"data_dir: {args.data_dir}", flush=True)
    print(f"symbol: {symbol}", flush=True)
    print(f"label price_col: {label_price_col}", flush=True)
    print(f"markout price_col: {args.markout_price_col}", flush=True)
    print(f"score mode: {args.score_mode}", flush=True)
    print("class mapping: STABLE=0, UP/BUY=1, DOWN/SELL=2", flush=True)

    print("Predicting test set...", flush=True)
    preds = _predict_split(
        model=model,
        cfg=cfg,
        data_dir=args.data_dir,
        metadata=metadata,
        batch_size=args.batch_size,
        device=device,
        max_rows=args.max_test_rows,
        split="test",
    )
    include_dates = _prediction_session_dates(preds, args.session_tz)
    print(
        f"Loading raw LOB for {len(include_dates)} test dates "
        f"({include_dates[0]} to {include_dates[-1]})...",
        flush=True,
    )
    lob = _load_raw_lob(
        raw_lob_parquet=raw_lob,
        symbol=symbol,
        include_dates=include_dates,
        session_tz=args.session_tz,
        session_start=args.session_start,
        session_end=args.session_end,
        price_scale=args.price_scale,
        ingest_resample_ms=args.ingest_resample_ms,
    )
    print("Aligning predictions and computing markouts...", flush=True)
    aligned = _attach_lob(preds, lob)
    aligned = _add_mid_markouts(aligned, steps, price_col=args.markout_price_col)

    summary = _summarize_markouts(aligned, steps)
    discovery_df, validation_df = _split_discovery_validation(aligned, args.discovery_frac)
    grid_seconds = _grid_seconds(aligned)
    confidence_filter = pd.concat(
        [
            _signal_filter_report(
                test=validation_df,
                val=discovery_df,
                step=int(step),
                keep_fracs=_csv_numbers(args.keep_fracs, float),
                bootstrap_reps=args.bootstrap_reps,
                grid_seconds=grid_seconds,
            )
            for step in steps
        ],
        ignore_index=True,
    )
    side_confidence_filter = pd.concat(
        [
            _side_signal_filter_report(
                test=validation_df,
                val=discovery_df,
                step=int(step),
                keep_fracs=_csv_numbers(args.keep_fracs, float),
                grid_seconds=grid_seconds,
            )
            for step in steps
        ],
        ignore_index=True,
    )
    score_filter = pd.concat(
        [
            _score_filter_report(
                test=validation_df,
                val=discovery_df,
                step=int(step),
                keep_fracs=_csv_numbers(args.keep_fracs, float),
                grid_seconds=grid_seconds,
                score_col="score_top_direction_gap" if args.score_mode == "top_direction_gap" else "score_up_down",
                score_mode=args.score_mode,
            )
            for step in steps
        ],
        ignore_index=True,
    )
    if args.skip_cell_protocol:
        candidates = pd.DataFrame()
        selected = pd.DataFrame()
        validation = pd.DataFrame()
        selected_trades = pd.DataFrame()
        filter_effect = pd.DataFrame()
    else:
        print("Selecting discovery cells and validating frozen rules...", flush=True)
        candidates, selected, validation = _run_cell_protocol(
            aligned,
            steps=steps,
            grid=grid,
            discovery_frac=args.discovery_frac,
            min_n=args.min_n,
            min_wr=args.min_win_rate,
            min_wr_lcb=args.min_win_rate_lcb,
            min_mean_bps_lcb=args.min_mean_bps_lcb,
            min_net_bps=args.min_net_bps,
            max_cells=args.max_selected_cells,
        )
        reference_step = int(steps[0])
        before_trades = _all_directional_trades(validation_df, reference_step)
        selected_trades = _selected_cell_trades(validation_df, selected)
        filter_effect = pd.DataFrame(
            [
                _summarize_trade_filter(
                    f"Before markout filtering ({reference_step}-step all signals)",
                    before_trades,
                ),
                _summarize_trade_filter("After markout filtering (selected cells)", selected_trades),
            ]
        )

    summary.to_csv(out_dir / "markout_summary_by_horizon.csv", index=False)
    candidates.to_csv(out_dir / "candidate_cells_discovery.csv", index=False)
    selected.to_csv(out_dir / "selected_cells.csv", index=False)
    validation.to_csv(out_dir / "selected_cells_validation.csv", index=False)
    selected_trades.to_csv(out_dir / "selected_cell_trades_validation.csv", index=False)
    filter_effect.to_csv(out_dir / "filter_effect_summary.csv", index=False)
    confidence_filter.to_csv(out_dir / "confidence_filter_summary.csv", index=False)
    side_confidence_filter.to_csv(out_dir / "side_confidence_filter_summary.csv", index=False)
    score_filter.to_csv(out_dir / "score_filter_summary.csv", index=False)
    if args.trades_out:
        aligned.to_csv(out_dir / "aligned_predictions_markout.csv", index=False)

    print("\n=== Raw mid-markout by horizon ===")
    print(summary.to_string(index=False))
    print("\n=== Confidence-filter validation ===")
    print(confidence_filter.to_string(index=False) if not confidence_filter.empty else "(none)")
    print("\n=== Side confidence-filter validation ===")
    print(side_confidence_filter.to_string(index=False) if not side_confidence_filter.empty else "(none)")
    print("\n=== Score-filter validation ===")
    print(score_filter.to_string(index=False) if not score_filter.empty else "(none)")
    print("\n=== Selected discovery cells ===")
    print(selected.to_string(index=False) if not selected.empty else "(none)")
    print("\n=== Frozen-cell validation ===")
    print(validation.to_string(index=False) if not validation.empty else "(none)")
    print("\n=== Filter effect summary ===")
    print(filter_effect.to_string(index=False) if not filter_effect.empty else "(none)")
    print(f"\nSaved markout outputs to: {out_dir}", flush=True)


if __name__ == "__main__":
    main()
