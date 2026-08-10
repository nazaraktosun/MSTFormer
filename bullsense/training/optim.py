"""Optimizer construction helpers."""

from __future__ import annotations

import torch
import torch.nn as nn


def build_optimizer(
    model: nn.Module,
    lr: float,
    weight_decay: float,
) -> torch.optim.AdamW:
    """AdamW with per-sublayer param groups when the model's shift norm needs them.

    DAIN's mean/scale/gate layers diverge under the base LR — they must get their
    own (much smaller) LRs, and no weight decay: decay pulls the identity-initialized
    shift/scale matrices toward zero, fighting the normalization itself.
    """
    extra_groups = (
        model.shift_norm_param_groups()
        if hasattr(model, "shift_norm_param_groups")
        else []
    )
    if not extra_groups:
        return torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)

    special_ids = {id(p) for g in extra_groups for p in g["params"]}
    base_params = [p for p in model.parameters() if id(p) not in special_ids]
    groups = [{"params": base_params, "lr": lr, "weight_decay": weight_decay}]
    groups += [{**g, "weight_decay": 0.0} for g in extra_groups]
    return torch.optim.AdamW(groups, lr=lr, weight_decay=weight_decay)
