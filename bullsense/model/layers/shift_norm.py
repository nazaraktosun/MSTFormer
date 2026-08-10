"""Sequence shift normalizers adapted from the TLOB regression path."""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class DishTS(nn.Module):
    """DISH-TS normalization for tensors shaped ``[batch, time, features]``."""

    def __init__(
        self,
        n_series: int,
        seq_len: int,
        dish_init: str = "standard",
        activate: bool = True,
    ) -> None:
        super().__init__()
        self.activate = bool(activate)
        self.n_series = int(n_series)
        self.seq_len = int(seq_len)
        self.gamma = nn.Parameter(torch.ones(n_series))
        self.beta = nn.Parameter(torch.zeros(n_series))

        if dish_init == "standard":
            weights = torch.rand(n_series, seq_len, 2) / seq_len
        elif dish_init == "avg":
            weights = torch.ones(n_series, seq_len, 2) / seq_len
        elif dish_init == "uniform":
            weights = (torch.ones(n_series, seq_len, 2) + torch.rand(n_series, seq_len, 2)) / seq_len
        else:
            raise ValueError(f"Unknown dish_init: {dish_init}")
        self.reduce_mlayer = nn.Parameter(weights)

        self.phil: torch.Tensor | None = None
        self.phih: torch.Tensor | None = None
        self.xil: torch.Tensor | None = None
        self.xih: torch.Tensor | None = None

    def forward(self, x: torch.Tensor, mode: str = "forward") -> torch.Tensor:
        if mode == "forward":
            self._precompute_stats(x)
            return self._forward_process(x)
        if mode == "inverse":
            return self._inverse_process(x)
        raise ValueError(f"Unknown mode: {mode}")

    def _precompute_stats(self, x: torch.Tensor) -> None:
        x_t = x.permute(2, 0, 1)
        theta = torch.bmm(x_t, self.reduce_mlayer.to(dtype=x.dtype)).permute(1, 2, 0)
        if self.activate:
            theta = F.gelu(theta)
        self.phil = theta[:, :1, :]
        self.phih = theta[:, 1:, :]
        denom = max(1, x.shape[1] - 1)
        self.xil = torch.sum((x - self.phil) ** 2, dim=1, keepdim=True) / denom
        self.xih = torch.sum((x - self.phih) ** 2, dim=1, keepdim=True) / denom

    def _forward_process(self, x: torch.Tensor) -> torch.Tensor:
        if self.phil is None or self.xil is None:
            raise RuntimeError("DishTS.forward stats are not ready.")
        y = (x - self.phil) / torch.sqrt(self.xil + 1e-8)
        return y * self.gamma.to(dtype=x.dtype) + self.beta.to(dtype=x.dtype)

    def _inverse_process(self, x: torch.Tensor) -> torch.Tensor:
        if self.phih is None or self.xih is None:
            raise RuntimeError("DishTS.inverse called before forward.")
        phih = self.phih.squeeze(1)
        xih = self.xih.squeeze(1)
        return (
            (x - self.beta.to(dtype=x.dtype)) / self.gamma.to(dtype=x.dtype)
        ) * torch.sqrt(xih + 1e-8) + phih


class ScalarDishTS(nn.Module):
    """Scalar target DISH-TS used for raw price regression."""

    def __init__(
        self,
        seq_len: int,
        dish_init: str = "standard",
        activate: bool = True,
    ) -> None:
        super().__init__()
        self.activate = bool(activate)
        self.gamma = nn.Parameter(torch.ones(1))
        self.beta = nn.Parameter(torch.zeros(1))

        if dish_init == "standard":
            weights = torch.rand(1, seq_len, 2) / seq_len
        elif dish_init == "avg":
            weights = torch.ones(1, seq_len, 2) / seq_len
        elif dish_init == "uniform":
            weights = (torch.ones(1, seq_len, 2) + torch.rand(1, seq_len, 2)) / seq_len
        else:
            raise ValueError(f"Unknown dish_init: {dish_init}")
        self.reduce_mlayer = nn.Parameter(weights)
        self.phih: torch.Tensor | None = None
        self.xih: torch.Tensor | None = None

    def precompute(self, history: torch.Tensor) -> None:
        if history.ndim != 3 or history.shape[-1] != 1:
            raise ValueError(f"Expected history shape (B, L, 1), got {tuple(history.shape)}")
        x_t = history.permute(2, 0, 1)
        theta = torch.bmm(x_t, self.reduce_mlayer.to(dtype=history.dtype)).permute(1, 2, 0)
        if self.activate:
            theta = F.gelu(theta)
        self.phih = theta[:, 1:, :]
        denom = max(1, history.shape[1] - 1)
        self.xih = torch.sum((history - self.phih) ** 2, dim=1, keepdim=True) / denom

    def _ensure_ready(self) -> None:
        if self.phih is None or self.xih is None:
            raise RuntimeError("ScalarDishTS stats are not ready. Call precompute(history) first.")

    def _ensure_target_shape(self, target: torch.Tensor) -> torch.Tensor:
        if target.ndim == 1:
            return target.unsqueeze(-1)
        if target.ndim == 2 and target.shape[-1] == 1:
            return target
        raise ValueError(f"Expected target shape (B,) or (B, 1), got {tuple(target.shape)}")

    def normalize_target(self, target: torch.Tensor) -> torch.Tensor:
        self._ensure_ready()
        target = self._ensure_target_shape(target)
        phih = self.phih.squeeze(1)
        xih = self.xih.squeeze(1)
        y = ((target - phih) / torch.sqrt(xih + 1e-8)) * self.gamma.to(dtype=target.dtype)
        y = y + self.beta.to(dtype=target.dtype)
        return y.squeeze(-1)

    def inverse_target(self, target: torch.Tensor) -> torch.Tensor:
        self._ensure_ready()
        target = self._ensure_target_shape(target)
        phih = self.phih.squeeze(1)
        xih = self.xih.squeeze(1)
        y = (target - self.beta.to(dtype=target.dtype)) / self.gamma.to(dtype=target.dtype)
        return (y * torch.sqrt(xih + 1e-8) + phih).squeeze(-1)


class _PaperDAIN(nn.Module):
    """Re-port of the passalis/dain DAIN_Layer (arXiv:1902.07892).

    Forward expects channels-first ``(B, D, L)`` and returns the same shape.
    Identity init on the mean/scaling layers reproduces a per-window z-score
    at step 0; training departs from that baseline.

    Modes map to the paper's ablation: ``adaptive_avg`` = DAIN(1),
    ``adaptive_scale`` = DAIN(1+2), ``full`` = DAIN(1+2+3); ``avg`` is the
    non-adaptive sample-average baseline.
    """

    MODES = (None, "avg", "adaptive_avg", "adaptive_scale", "full")

    def __init__(self, input_dim: int, mode: str = "adaptive_scale") -> None:
        super().__init__()
        if mode not in self.MODES:
            raise ValueError(f"Unknown DAIN mode: {mode}")
        self.mode = mode

        self.mean_layer = nn.Linear(input_dim, input_dim, bias=False)
        nn.init.eye_(self.mean_layer.weight)

        self.scaling_layer = nn.Linear(input_dim, input_dim, bias=False)
        nn.init.eye_(self.scaling_layer.weight)

        self.gating_layer = nn.Linear(input_dim, input_dim)
        nn.init.xavier_uniform_(self.gating_layer.weight)
        nn.init.zeros_(self.gating_layer.bias)

        self.eps = 1e-8

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, D, L)
        if self.mode is None:
            return x

        if self.mode == "avg":
            return x - x.mean(dim=2, keepdim=True)

        # Step 1: adaptive mean shift
        adaptive_avg = self.mean_layer(x.mean(dim=2)).unsqueeze(-1)
        x = x - adaptive_avg
        if self.mode == "adaptive_avg":
            return x

        # Step 2: adaptive scale
        std = torch.sqrt(x.pow(2).mean(dim=2) + self.eps)
        adaptive_std = self.scaling_layer(std)
        # Floor near-zero scales to 1.0 to avoid division blow-up
        adaptive_std = torch.where(
            adaptive_std <= self.eps,
            torch.ones_like(adaptive_std),
            adaptive_std,
        )
        x = x / adaptive_std.unsqueeze(-1)
        if self.mode == "adaptive_scale":
            return x

        # Step 3: gating (mode == "full")
        gate = torch.sigmoid(self.gating_layer(x.mean(dim=2))).unsqueeze(-1)
        return x * gate


class DAIN(nn.Module):
    """Deep Adaptive Input Normalization with the DishTS/RevIN shift-norm contract.

        out = layer(x, mode="forward")   # x: (B, T, F)
        out = layer(x, mode="inverse")   # identity (DAIN has no closed-form inverse)

    The mean/scaling/gating sub-layers need their own (smaller) learning rates;
    expose them to the optimizer via :meth:`param_groups`.
    """

    def __init__(
        self,
        n_series: int,
        mode: str = "adaptive_scale",
        mean_lr: float = 1.0e-5,
        gate_lr: float = 1.0e-3,
        scale_lr: float = 1.0e-5,
    ) -> None:
        super().__init__()
        self.n_series = int(n_series)
        self.dain = _PaperDAIN(input_dim=self.n_series, mode=mode)
        self.mean_lr = float(mean_lr)
        self.gate_lr = float(gate_lr)
        self.scale_lr = float(scale_lr)

    def forward(self, x: torch.Tensor, mode: str = "forward") -> torch.Tensor:
        if mode == "inverse":
            return x
        if mode != "forward":
            raise ValueError(f"Unknown mode: {mode}")
        if x.ndim != 3:
            raise ValueError(f"DAIN expects (B, T, F), got {tuple(x.shape)}")
        y = self.dain(x.permute(0, 2, 1).contiguous())
        return y.permute(0, 2, 1).contiguous()

    def param_groups(self) -> list[dict]:
        return [
            {"params": list(self.dain.mean_layer.parameters()), "lr": self.mean_lr, "name": "dain_mean"},
            {"params": list(self.dain.scaling_layer.parameters()), "lr": self.scale_lr, "name": "dain_scale"},
            {"params": list(self.dain.gating_layer.parameters()), "lr": self.gate_lr, "name": "dain_gate"},
        ]


def _normalize_groups_spec(groups: dict | None) -> dict[str, dict]:
    """Canonicalize a grouped-DAIN groups spec to {name: {'indices', 'mode'}}.

    Accepts ``{'price': {'indices': [...], 'mode': '...'}}``,
    ``{'price': ([...], '...')}`` or ``{'price': [[...], '...']}``.
    """
    if not groups:
        raise ValueError("GroupedDAIN requires a non-empty groups spec")
    canon: dict[str, dict] = {}
    for name, spec in groups.items():
        if isinstance(spec, dict):
            idx = list(spec.get("indices") or spec.get("cols") or [])
            mode = spec.get("mode", "adaptive_scale")
        elif isinstance(spec, (list, tuple)) and len(spec) == 2:
            idx, mode = spec
            idx = list(idx)
        else:
            raise ValueError(f"Bad group spec for '{name}': {spec!r}")
        if not idx:
            raise ValueError(f"Group '{name}' has no column indices")
        canon[str(name)] = {"indices": [int(i) for i in idx], "mode": str(mode)}
    return canon


class GroupedDAIN(nn.Module):
    """One DAIN per named group of feature columns (e.g. prices vs sizes).

    Columns not covered by any group pass through unchanged. Same call
    contract as :class:`DAIN`.
    """

    def __init__(
        self,
        n_series: int,
        groups: dict,
        mean_lr: float = 1.0e-5,
        gate_lr: float = 1.0e-3,
        scale_lr: float = 1.0e-5,
    ) -> None:
        super().__init__()
        self.n_series = int(n_series)
        canon = _normalize_groups_spec(groups)
        self.group_names: list[str] = list(canon.keys())

        all_idx: list[int] = []
        for name, spec in canon.items():
            for i in spec["indices"]:
                if i < 0 or i >= self.n_series:
                    raise ValueError(f"Group '{name}' has out-of-range col {i}")
            all_idx.extend(spec["indices"])
        if len(set(all_idx)) != len(all_idx):
            raise ValueError("GroupedDAIN groups overlap on at least one column")

        self.layers = nn.ModuleDict()
        for name, spec in canon.items():
            idx_tensor = torch.tensor(spec["indices"], dtype=torch.long)
            self.register_buffer(f"_idx_{name}", idx_tensor, persistent=False)
            self.layers[name] = _PaperDAIN(input_dim=len(spec["indices"]), mode=spec["mode"])

        self.mean_lr = float(mean_lr)
        self.gate_lr = float(gate_lr)
        self.scale_lr = float(scale_lr)

    def _idx(self, name: str) -> torch.Tensor:
        return getattr(self, f"_idx_{name}")

    def forward(self, x: torch.Tensor, mode: str = "forward") -> torch.Tensor:
        if mode == "inverse":
            return x
        if mode != "forward":
            raise ValueError(f"Unknown mode: {mode}")
        if x.ndim != 3:
            raise ValueError(f"GroupedDAIN expects (B, T, F), got {tuple(x.shape)}")

        out = x.clone()
        for name in self.group_names:
            idx = self._idx(name).to(x.device)
            sub = x.index_select(-1, idx)
            sub_n = self.layers[name](sub.permute(0, 2, 1).contiguous())
            out.index_copy_(-1, idx, sub_n.permute(0, 2, 1).contiguous())
        return out

    def param_groups(self) -> list[dict]:
        pgs: list[dict] = []
        for name in self.group_names:
            layer = self.layers[name]
            pgs += [
                {"params": list(layer.mean_layer.parameters()), "lr": self.mean_lr, "name": f"gdain_{name}_mean"},
                {"params": list(layer.scaling_layer.parameters()), "lr": self.scale_lr, "name": f"gdain_{name}_scale"},
                {"params": list(layer.gating_layer.parameters()), "lr": self.gate_lr, "name": f"gdain_{name}_gate"},
            ]
        return pgs


class RevIN(nn.Module):
    """Reversible instance normalization for ``[batch, time, features]`` tensors."""

    def __init__(
        self,
        n_series: int,
        eps: float = 1e-5,
        affine: bool = True,
        detach_stats: bool = True,
    ) -> None:
        super().__init__()
        self.eps = float(eps)
        self.affine = bool(affine)
        self.detach_stats = bool(detach_stats)
        if self.affine:
            self.gamma = nn.Parameter(torch.ones(n_series))
            self.beta = nn.Parameter(torch.zeros(n_series))
        else:
            self.register_parameter("gamma", None)
            self.register_parameter("beta", None)
        self._mean: torch.Tensor | None = None
        self._std: torch.Tensor | None = None

    def forward(self, x: torch.Tensor, mode: str = "forward") -> torch.Tensor:
        if mode == "forward":
            mean = x.mean(dim=1, keepdim=True)
            var = x.var(dim=1, keepdim=True, unbiased=False)
            std = torch.sqrt(var + self.eps)
            if self.detach_stats:
                mean = mean.detach()
                std = std.detach()
            self._mean = mean
            self._std = std
            y = (x - mean) / std
            if self.affine:
                y = y * self.gamma.to(dtype=x.dtype) + self.beta.to(dtype=x.dtype)
            return y
        if mode == "inverse":
            if self._mean is None or self._std is None:
                raise RuntimeError("RevIN.inverse called before forward.")
            y = x
            if self.affine:
                y = (y - self.beta.to(dtype=x.dtype)) / (
                    self.gamma.to(dtype=x.dtype) + self.eps
                )
            return y * self._std + self._mean
        raise ValueError(f"Unknown mode: {mode}")


class ScalarRevIN(nn.Module):
    """Scalar target RevIN used for raw price regression."""

    def __init__(
        self,
        eps: float = 1e-5,
        affine: bool = True,
        detach_stats: bool = True,
    ) -> None:
        super().__init__()
        self.eps = float(eps)
        self.affine = bool(affine)
        self.detach_stats = bool(detach_stats)
        if self.affine:
            self.gamma = nn.Parameter(torch.ones(1))
            self.beta = nn.Parameter(torch.zeros(1))
        else:
            self.register_parameter("gamma", None)
            self.register_parameter("beta", None)
        self._mean: torch.Tensor | None = None
        self._std: torch.Tensor | None = None

    def precompute(self, history: torch.Tensor) -> None:
        if history.ndim != 3 or history.shape[-1] != 1:
            raise ValueError(f"Expected history shape (B, L, 1), got {tuple(history.shape)}")
        mean = history.mean(dim=1, keepdim=True)
        var = history.var(dim=1, keepdim=True, unbiased=False)
        std = torch.sqrt(var + self.eps)
        if self.detach_stats:
            mean = mean.detach()
            std = std.detach()
        self._mean = mean
        self._std = std

    def _ensure_ready(self) -> None:
        if self._mean is None or self._std is None:
            raise RuntimeError("ScalarRevIN stats are not ready. Call precompute(history) first.")

    def _ensure_target_shape(self, target: torch.Tensor) -> torch.Tensor:
        if target.ndim == 1:
            return target.unsqueeze(-1)
        if target.ndim == 2 and target.shape[-1] == 1:
            return target
        raise ValueError(f"Expected target shape (B,) or (B, 1), got {tuple(target.shape)}")

    def normalize_target(self, target: torch.Tensor) -> torch.Tensor:
        self._ensure_ready()
        target = self._ensure_target_shape(target)
        mean = self._mean.squeeze(1)
        std = self._std.squeeze(1)
        y = (target - mean) / std
        if self.affine:
            y = y * self.gamma.to(dtype=target.dtype) + self.beta.to(dtype=target.dtype)
        return y.squeeze(-1)

    def inverse_target(self, target: torch.Tensor) -> torch.Tensor:
        self._ensure_ready()
        target = self._ensure_target_shape(target)
        mean = self._mean.squeeze(1)
        std = self._std.squeeze(1)
        y = target
        if self.affine:
            y = (y - self.beta.to(dtype=target.dtype)) / (
                self.gamma.to(dtype=target.dtype) + self.eps
            )
        return (y * std + mean).squeeze(-1)
