"""Bilinear input normalization from the reference TLOB models."""

from __future__ import annotations

import torch
from torch import nn


class BiN(nn.Module):
    """Bilinear normalization over feature and temporal dimensions.

    The input shape is expected to be ``[batch, features, time]``.
    """

    def __init__(self, d1: int, t1: int, eps: float = 1e-4) -> None:
        super().__init__()
        self.t1 = t1
        self.d1 = d1
        self.eps = eps

        self.B1 = nn.Parameter(torch.zeros(t1, 1))
        self.l1 = nn.Parameter(torch.empty(t1, 1))
        nn.init.xavier_normal_(self.l1)

        self.B2 = nn.Parameter(torch.zeros(d1, 1))
        self.l2 = nn.Parameter(torch.empty(d1, 1))
        nn.init.xavier_normal_(self.l2)

        self.y1 = nn.Parameter(torch.tensor([0.5]))
        self.y2 = nn.Parameter(torch.tensor([0.5]))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        with torch.no_grad():
            self.y1.clamp_(min=0.01)
            self.y2.clamp_(min=0.01)

        dtype = x.dtype
        device = x.device

        ones_time = torch.ones((self.t1, 1), device=device, dtype=dtype)
        mean_time = x.mean(dim=2, keepdim=True)
        std_time = x.std(dim=2, keepdim=True).clamp(min=self.eps)
        z_time = (x - mean_time @ ones_time.T) / (std_time @ ones_time.T)
        x_time = (self.l2.to(dtype=dtype) @ ones_time.T) * z_time
        x_time = x_time + self.B2.to(dtype=dtype) @ ones_time.T

        ones_feature = torch.ones((self.d1, 1), device=device, dtype=dtype)
        mean_feature = x.mean(dim=1, keepdim=False).unsqueeze(-1)
        std_feature = x.std(dim=1, keepdim=False).unsqueeze(-1).clamp(min=self.eps)
        mean_feature = (mean_feature @ ones_feature.T).permute(0, 2, 1)
        std_feature = (std_feature @ ones_feature.T).permute(0, 2, 1)
        z_feature = (x - mean_feature) / std_feature
        x_feature = (ones_feature @ self.l1.to(dtype=dtype).T) * z_feature
        x_feature = x_feature + ones_feature @ self.B1.to(dtype=dtype).T

        return self.y1.to(dtype=dtype) * x_feature + self.y2.to(dtype=dtype) * x_time
