"""Model construction registry for Bullsense experiments."""

from __future__ import annotations

from typing import Any

import torch.nn as nn

from bullsense.model.book_side.mlplob import MLPLOB as BullsenseMLPLOB
from bullsense.model.book_side.tlob import TLOB_DeepLOB
from bullsense.model.tlob_original import BiN_CTABL, DeepLOB, OriginalMLPLOB, OriginalTLOB
from bullsense.model.wrappers import InputShiftNormWrapper


BULLSENSE_TLOB_ALIASES = {"tlob", "tlob_spatiotemporal", "bullsense_tlob"}
SUPPORTED_MODEL_TYPES = (
    "tlob",
    "tlob_spatiotemporal",
    "bullsense_tlob",
    "mlplob",
    "tlob_original",
    "original_tlob",
    "mlplob_original",
    "original_mlplob",
    "deeplob",
    "binctabl",
    "binctabl_original",
)


def build_model(
    model_type: str,
    input_dim: int,
    seq_len: int,
    num_classes: int,
    cfg: Any,
) -> nn.Module:
    """Build a configured model by name."""

    model_key = model_type.lower()

    if model_key in BULLSENSE_TLOB_ALIASES:
        d_model = int(getattr(cfg, "d_model", 64))
        n_heads = int(getattr(cfg, "n_heads", 4))
        if d_model % n_heads != 0:
            raise ValueError(
                f"d_model ({d_model}) must be divisible by n_heads ({n_heads}) "
                "for Bullsense TLOB."
            )
        model = TLOB_DeepLOB(
            num_features=input_dim,
            seq_len=seq_len,
            d_model=d_model,
            num_layers=int(getattr(cfg, "t_layers", 2)),
            num_heads=n_heads,
            num_classes=num_classes,
            dropout=float(getattr(cfg, "dropout", 0.25)),
            use_bilinear_norm=bool(getattr(cfg, "use_bilinear_norm", True)),
            bilinear_norm_type=str(getattr(cfg, "bilinear_norm_type", "bin")),
            temporal_readout=str(getattr(cfg, "temporal_readout", "last")),
            attn_dropout=float(getattr(cfg, "attn_dropout", 0.0)),
            temporal_causal=bool(getattr(cfg, "temporal_causal", False)),
            gradient_checkpointing=bool(getattr(cfg, "gradient_checkpointing", False)),
        )
        return _maybe_wrap_shift_norm(model, input_dim, seq_len, cfg)

    if model_key == "mlplob":
        model = BullsenseMLPLOB(
            num_features=input_dim,
            seq_len=seq_len,
            hidden_dim=int(getattr(cfg, "hidden_dim", 256)),
            num_layers=int(getattr(cfg, "num_layers", 3)),
            num_classes=num_classes,
            dropout=float(getattr(cfg, "dropout", 0.2)),
        )
        return _maybe_wrap_shift_norm(model, input_dim, seq_len, cfg)

    if model_key in {"tlob_original", "original_tlob"}:
        model = OriginalTLOB(
            hidden_dim=int(getattr(cfg, "hidden_dim", getattr(cfg, "d_model", 256))),
            num_layers=int(getattr(cfg, "num_layers", getattr(cfg, "t_layers", 3))),
            seq_size=seq_len,
            num_features=input_dim,
            num_heads=int(getattr(cfg, "n_heads", 1)),
            is_sin_emb=bool(getattr(cfg, "is_sin_emb", True)),
            dataset_type=str(getattr(cfg, "dataset_type", "BULLSENSE")),
            output_dim=num_classes,
            order_type_idx=getattr(cfg, "order_type_idx", 41),
        )
        return _maybe_wrap_shift_norm(model, input_dim, seq_len, cfg)

    if model_key in {"mlplob_original", "original_mlplob"}:
        model = OriginalMLPLOB(
            hidden_dim=int(getattr(cfg, "hidden_dim", 256)),
            num_layers=int(getattr(cfg, "num_layers", 3)),
            seq_size=seq_len,
            num_features=input_dim,
            dataset_type=str(getattr(cfg, "dataset_type", "BULLSENSE")),
            output_dim=num_classes,
            order_type_idx=getattr(cfg, "order_type_idx", 41),
        )
        return _maybe_wrap_shift_norm(model, input_dim, seq_len, cfg)

    if model_key == "deeplob":
        model = DeepLOB(num_classes=num_classes)
        return _maybe_wrap_shift_norm(model, input_dim, seq_len, cfg)

    if model_key in {"binctabl", "binctabl_original"}:
        model = BiN_CTABL(
            d2=int(getattr(cfg, "binctabl_d2", 60)),
            d1=input_dim,
            t1=seq_len,
            t2=seq_len,
            d3=int(getattr(cfg, "binctabl_d3", 120)),
            t3=int(getattr(cfg, "binctabl_t3", 5)),
            d4=num_classes,
            t4=1,
            dropout=float(getattr(cfg, "dropout", 0.1)),
        )
        return _maybe_wrap_shift_norm(model, input_dim, seq_len, cfg)

    supported = ", ".join(SUPPORTED_MODEL_TYPES)
    raise ValueError(f"Unknown model_type={model_type!r}. Supported values: {supported}")


def _maybe_wrap_shift_norm(
    model: nn.Module,
    input_dim: int,
    seq_len: int,
    cfg: Any,
) -> nn.Module:
    norm_type = str(getattr(cfg, "input_shift_norm", "none")).lower()
    if norm_type in {"", "none", "off", "false"}:
        return model
    return InputShiftNormWrapper(
        model,
        norm_type=norm_type,
        num_features=input_dim,
        seq_len=seq_len,
        dish_init=str(getattr(cfg, "dish_init", "standard")),
        dish_activate=bool(getattr(cfg, "dish_activate", True)),
        revin_affine=bool(getattr(cfg, "revin_affine", True)),
        revin_detach_stats=bool(getattr(cfg, "revin_detach_stats", True)),
        dain_mode=str(getattr(cfg, "dain_mode", "adaptive_scale")),
        dain_mean_lr=float(getattr(cfg, "dain_mean_lr", 1e-5)),
        dain_gate_lr=float(getattr(cfg, "dain_gate_lr", 1e-3)),
        dain_scale_lr=float(getattr(cfg, "dain_scale_lr", 1e-5)),
        grouped_dain_groups=getattr(cfg, "grouped_dain_groups", None),
    )
