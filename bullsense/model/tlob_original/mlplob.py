"""Reference TLOB MLP-LOB adapted for Bullsense."""

from __future__ import annotations

import torch
from torch import nn

from bullsense.model.tlob_original.bin import BiN


class MLP(nn.Module):
    def __init__(self, start_dim: int, hidden_dim: int, final_dim: int) -> None:
        super().__init__()
        self.layer_norm = nn.LayerNorm(final_dim)
        self.fc = nn.Linear(start_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, final_dim)
        self.gelu = nn.GELU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x
        x = self.fc(x)
        x = self.gelu(x)
        x = self.fc2(x)
        if x.shape[-1] == residual.shape[-1]:
            x = x + residual
        x = self.layer_norm(x)
        return self.gelu(x)


class MLPLOB(nn.Module):
    def __init__(
        self,
        hidden_dim: int,
        num_layers: int,
        seq_size: int,
        num_features: int,
        dataset_type: str = "BULLSENSE",
        output_dim: int = 3,
        order_type_idx: int | None = 41,
    ) -> None:
        super().__init__()
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.seq_size = seq_size
        self.num_features = num_features
        self.dataset_type = dataset_type
        self.output_dim = output_dim
        self.order_type_idx = order_type_idx

        last_hidden = max(1, hidden_dim // 4)
        last_seq = max(1, seq_size // 4)

        self.layers = nn.ModuleList()
        self.order_type_embedder = nn.Embedding(5, 1)
        self.norm_layer = BiN(num_features, seq_size)
        self.layers.append(nn.Linear(num_features, hidden_dim))
        self.layers.append(nn.GELU())
        for i in range(num_layers):
            if i != num_layers - 1:
                self.layers.append(MLP(hidden_dim, hidden_dim * 4, hidden_dim))
                self.layers.append(MLP(seq_size, seq_size * 4, seq_size))
            else:
                self.layers.append(MLP(hidden_dim, max(1, hidden_dim * 2), last_hidden))
                self.layers.append(MLP(seq_size, max(1, seq_size * 2), last_seq))

        total_dim = last_hidden * last_seq
        self.final_layers = nn.ModuleList()
        while total_dim > 128:
            self.final_layers.append(nn.Linear(total_dim, total_dim // 4))
            self.final_layers.append(nn.GELU())
            total_dim = total_dim // 4
        self.final_layers.append(nn.Linear(total_dim, output_dim))

    def _prepare_input(self, x: torch.Tensor) -> torch.Tensor:
        if self.dataset_type != "LOBSTER":
            return x
        idx = int(self.order_type_idx) if self.order_type_idx is not None else 41
        if x.shape[-1] <= idx:
            return x
        continuous = torch.cat([x[:, :, :idx], x[:, :, idx + 1 :]], dim=2)
        order_type = x[:, :, idx].long().clamp(0, self.order_type_embedder.num_embeddings - 1)
        order_type_emb = self.order_type_embedder(order_type).detach()
        return torch.cat([continuous, order_type_emb], dim=2)

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        x = self._prepare_input(input)
        x = x.permute(0, 2, 1)
        x = self.norm_layer(x)
        x = x.permute(0, 2, 1)
        for layer in self.layers:
            x = layer(x)
            x = x.permute(0, 2, 1)
        x = x.reshape(x.shape[0], -1)
        for layer in self.final_layers:
            x = layer(x)
        return x

    def info(self) -> dict[str, int | str | bool]:
        total = sum(p.numel() for p in self.parameters())
        trainable = sum(p.numel() for p in self.parameters() if p.requires_grad)
        return {
            "name": "OriginalMLPLOB",
            "num_features": self.num_features,
            "seq_len": self.seq_size,
            "hidden_dim": self.hidden_dim,
            "num_layers": self.num_layers,
            "num_classes": self.output_dim,
            "total_params": total,
            "trainable_params": trainable,
        }
