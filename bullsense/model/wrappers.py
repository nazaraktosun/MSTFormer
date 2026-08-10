"""Model wrappers used by the registry."""

from __future__ import annotations

import torch
import torch.nn as nn

from bullsense.model.layers.shift_norm import DAIN, DishTS, GroupedDAIN, RevIN


class InputShiftNormWrapper(nn.Module):
    """Apply an optional sequence shift normalizer before a base model."""

    def __init__(
        self,
        model: nn.Module,
        *,
        norm_type: str,
        num_features: int,
        seq_len: int,
        dish_init: str = "standard",
        dish_activate: bool = True,
        revin_affine: bool = True,
        revin_detach_stats: bool = True,
        dain_mode: str = "adaptive_scale",
        dain_mean_lr: float = 1.0e-5,
        dain_gate_lr: float = 1.0e-3,
        dain_scale_lr: float = 1.0e-5,
        grouped_dain_groups: dict | None = None,
    ) -> None:
        super().__init__()
        self.model = model
        self.norm_type = norm_type.lower()
        if self.norm_type == "dish":
            self.shift_norm: nn.Module = DishTS(
                n_series=num_features,
                seq_len=seq_len,
                dish_init=dish_init,
                activate=dish_activate,
            )
        elif self.norm_type == "revin":
            self.shift_norm = RevIN(
                n_series=num_features,
                affine=revin_affine,
                detach_stats=revin_detach_stats,
            )
        elif self.norm_type == "dain":
            self.shift_norm = DAIN(
                n_series=num_features,
                mode=dain_mode,
                mean_lr=dain_mean_lr,
                gate_lr=dain_gate_lr,
                scale_lr=dain_scale_lr,
            )
        elif self.norm_type == "grouped_dain":
            if not grouped_dain_groups:
                raise ValueError(
                    "input_shift_norm='grouped_dain' requires grouped_dain_groups"
                )
            self.shift_norm = GroupedDAIN(
                n_series=num_features,
                groups=grouped_dain_groups,
                mean_lr=dain_mean_lr,
                gate_lr=dain_gate_lr,
                scale_lr=dain_scale_lr,
            )
        else:
            raise ValueError(f"Unknown input shift norm: {norm_type}")

    def forward(self, x: torch.Tensor, *args, **kwargs):
        x = self.shift_norm(x, mode="forward")
        return self.model(x, *args, **kwargs)

    def shift_norm_param_groups(self) -> list[dict]:
        """Per-sublayer optimizer param groups (DAIN needs asymmetric LRs)."""
        if hasattr(self.shift_norm, "param_groups"):
            return self.shift_norm.param_groups()
        return []

    def info(self) -> dict:
        if hasattr(self.model, "info"):
            info = dict(self.model.info())
        else:
            total = sum(p.numel() for p in self.parameters())
            trainable = sum(p.numel() for p in self.parameters() if p.requires_grad)
            info = {
                "name": self.model.__class__.__name__,
                "total_params": total,
                "trainable_params": trainable,
            }
        info["name"] = f"{info.get('name', self.model.__class__.__name__)}+{self.norm_type.upper()}"
        info["input_shift_norm"] = self.norm_type
        return info
