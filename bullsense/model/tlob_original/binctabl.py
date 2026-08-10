"""Reference BiN-CTABL adapted for Bullsense."""

from __future__ import annotations

import torch
from torch import nn

from bullsense.model.tlob_original.bin import BiN


class TABLLayer(nn.Module):
    def __init__(self, d2: int, d1: int, t1: int, t2: int) -> None:
        super().__init__()
        self.t1 = t1

        self.W1 = nn.Parameter(torch.empty(d2, d1))
        nn.init.kaiming_uniform_(self.W1, nonlinearity="relu")

        self.W = nn.Parameter(torch.full((t1, t1), 1 / t1))

        self.W2 = nn.Parameter(torch.empty(t1, t2))
        nn.init.kaiming_uniform_(self.W2, nonlinearity="relu")

        self.B = nn.Parameter(torch.zeros(d2, t2))
        self.l = nn.Parameter(torch.tensor([0.5]))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        with torch.no_grad():
            self.l.clamp_(min=0.0, max=1.0)

        x = self.W1 @ x
        eye = torch.eye(self.t1, dtype=x.dtype, device=x.device)
        w = self.W.to(dtype=x.dtype) - self.W.to(dtype=x.dtype) * eye + eye / self.t1
        e = x @ w
        attention = torch.softmax(e, dim=-1)
        x = self.l.to(dtype=x.dtype)[0] * x + (1.0 - self.l.to(dtype=x.dtype)[0]) * x * attention
        return x @ self.W2.to(dtype=x.dtype) + self.B.to(dtype=x.dtype)


class BLLayer(nn.Module):
    def __init__(self, d2: int, d1: int, t1: int, t2: int) -> None:
        super().__init__()
        self.W1 = nn.Parameter(torch.empty(d2, d1))
        nn.init.kaiming_uniform_(self.W1, nonlinearity="relu")

        self.W2 = nn.Parameter(torch.empty(t1, t2))
        nn.init.kaiming_uniform_(self.W2, nonlinearity="relu")

        self.B = nn.Parameter(torch.zeros(d2, t2))
        self.activation = nn.ReLU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.activation(
            self.W1.to(dtype=x.dtype) @ x @ self.W2.to(dtype=x.dtype)
            + self.B.to(dtype=x.dtype)
        )


class BiN_CTABL(nn.Module):
    def __init__(
        self,
        d2: int,
        d1: int,
        t1: int,
        t2: int,
        d3: int,
        t3: int,
        d4: int,
        t4: int,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.d1 = d1
        self.t1 = t1
        self.output_dim = d4
        self.BiN = BiN(d1, t1)
        self.BL = BLLayer(d2, d1, t1, t2)
        self.BL2 = BLLayer(d3, d2, t2, t3)
        self.TABL = TABLLayer(d4, d3, t3, t4)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x.permute(0, 2, 1)
        x = self.BiN(x)

        self.max_norm_(self.BL.W1.data)
        self.max_norm_(self.BL.W2.data)
        x = self.dropout(self.BL(x))

        self.max_norm_(self.BL2.W1.data)
        self.max_norm_(self.BL2.W2.data)
        x = self.dropout(self.BL2(x))

        self.max_norm_(self.TABL.W1.data)
        self.max_norm_(self.TABL.W.data)
        self.max_norm_(self.TABL.W2.data)
        x = self.TABL(x)
        return x.squeeze(-1)

    def max_norm_(self, w: torch.Tensor) -> None:
        with torch.no_grad():
            norm = torch.linalg.matrix_norm(w)
            if norm > 10.0:
                desired = torch.clamp(norm, min=0.0, max=10.0)
                w *= desired / (1e-8 + norm)

    def info(self) -> dict[str, int | str]:
        total = sum(p.numel() for p in self.parameters())
        trainable = sum(p.numel() for p in self.parameters() if p.requires_grad)
        return {
            "name": "BiN_CTABL",
            "num_features": self.d1,
            "seq_len": self.t1,
            "num_classes": self.output_dim,
            "total_params": total,
            "trainable_params": trainable,
        }
