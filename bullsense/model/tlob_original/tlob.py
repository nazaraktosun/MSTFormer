"""Reference TLOB transformer adapted for Bullsense."""

from __future__ import annotations

import numpy as np
import torch
from einops import rearrange
from torch import nn

from bullsense.model.tlob_original.bin import BiN
from bullsense.model.tlob_original.mlplob import MLP


class ComputeQKV(nn.Module):
    def __init__(self, hidden_dim: int, num_heads: int) -> None:
        super().__init__()
        self.hidden_dim = hidden_dim
        self.num_heads = num_heads
        self.q = nn.Linear(hidden_dim, hidden_dim * num_heads)
        self.k = nn.Linear(hidden_dim, hidden_dim * num_heads)
        self.v = nn.Linear(hidden_dim, hidden_dim * num_heads)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        return self.q(x), self.k(x), self.v(x)


class TransformerLayer(nn.Module):
    def __init__(self, hidden_dim: int, num_heads: int, final_dim: int) -> None:
        super().__init__()
        self.hidden_dim = hidden_dim
        self.num_heads = num_heads
        self.norm = nn.LayerNorm(hidden_dim)
        self.qkv = ComputeQKV(hidden_dim, num_heads)
        self.attention = nn.MultiheadAttention(
            hidden_dim * num_heads,
            num_heads,
            batch_first=True,
        )
        self.mlp = MLP(hidden_dim, hidden_dim * 4, final_dim)
        self.w0 = nn.Linear(hidden_dim * num_heads, hidden_dim)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        residual = x
        q, k, v = self.qkv(x)
        x, att = self.attention(q, k, v, average_attn_weights=False, need_weights=True)
        x = self.w0(x)
        x = self.norm(x + residual)
        x = self.mlp(x)
        if x.shape[-1] == residual.shape[-1]:
            x = x + residual
        return x, att


class TLOB(nn.Module):
    def __init__(
        self,
        hidden_dim: int,
        num_layers: int,
        seq_size: int,
        num_features: int,
        num_heads: int,
        is_sin_emb: bool = True,
        dataset_type: str = "BULLSENSE",
        output_dim: int = 3,
        order_type_idx: int | None = 41,
    ) -> None:
        super().__init__()
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.is_sin_emb = is_sin_emb
        self.seq_size = seq_size
        self.num_heads = num_heads
        self.dataset_type = dataset_type
        self.output_dim = output_dim
        self.num_features = num_features
        self.order_type_idx = order_type_idx

        final_hidden = max(1, hidden_dim // 4)
        self.layers = nn.ModuleList()
        self.order_type_embedder = nn.Embedding(5, 1)
        self.norm_layer = BiN(num_features, seq_size)
        self.emb_layer = nn.Linear(num_features, hidden_dim)
        if is_sin_emb:
            self.register_buffer(
                "pos_encoder",
                sinusoidal_positional_embedding(seq_size, hidden_dim),
                persistent=False,
            )
        else:
            self.pos_encoder = nn.Parameter(torch.randn(1, seq_size, hidden_dim))

        for i in range(num_layers):
            if i != num_layers - 1:
                self.layers.append(TransformerLayer(hidden_dim, num_heads, hidden_dim))
                self.layers.append(TransformerLayer(seq_size, num_heads, seq_size))
            else:
                self.layers.append(TransformerLayer(hidden_dim, num_heads, final_hidden))
                self.layers.append(TransformerLayer(seq_size, num_heads, seq_size))

        self.att_temporal: list[torch.Tensor] = []
        self.att_feature: list[torch.Tensor] = []
        self.mean_att_distance_temporal: list[np.ndarray] = []
        total_dim = final_hidden * seq_size
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

    def _forward_backbone(self, x: torch.Tensor) -> torch.Tensor:
        x = self.emb_layer(x)
        pos = self.pos_encoder
        if pos.ndim == 2:
            pos = pos.unsqueeze(0)
        x = x + pos[:, : x.shape[1], :].to(device=x.device, dtype=x.dtype)
        for layer in self.layers:
            x, _ = layer(x)
            x = x.permute(0, 2, 1)
        x = rearrange(x, "b s f -> b (f s) 1").reshape(x.shape[0], -1)
        for layer in self.final_layers:
            x = layer(x)
        return x

    def forward(self, input: torch.Tensor, store_att: bool = False) -> torch.Tensor:
        x = self._prepare_input(input)
        x = rearrange(x, "b s f -> b f s")
        x = self.norm_layer(x)
        x = rearrange(x, "b f s -> b s f")
        return self._forward_backbone(x)

    def info(self) -> dict[str, int | str | bool]:
        total = sum(p.numel() for p in self.parameters())
        trainable = sum(p.numel() for p in self.parameters() if p.requires_grad)
        return {
            "name": "OriginalTLOB",
            "num_features": self.num_features,
            "seq_len": self.seq_size,
            "hidden_dim": self.hidden_dim,
            "num_layers": self.num_layers,
            "num_heads": self.num_heads,
            "is_sin_emb": self.is_sin_emb,
            "num_classes": self.output_dim,
            "total_params": total,
            "trainable_params": trainable,
        }


def sinusoidal_positional_embedding(
    token_sequence_size: int,
    token_embedding_dim: int,
    n: float = 10000.0,
) -> torch.Tensor:
    if token_embedding_dim % 2 != 0:
        raise ValueError(
            "Sinusoidal positional embedding cannot apply to odd token embedding dim "
            f"(got dim={token_embedding_dim:d})"
        )

    positions = torch.arange(0, token_sequence_size).unsqueeze_(1)
    embeddings = torch.zeros(token_sequence_size, token_embedding_dim)
    denominators = torch.pow(
        n,
        2 * torch.arange(0, token_embedding_dim // 2) / token_embedding_dim,
    )
    embeddings[:, 0::2] = torch.sin(positions / denominators)
    embeddings[:, 1::2] = torch.cos(positions / denominators)
    return embeddings


def compute_mean_att_distance(att: torch.Tensor) -> np.ndarray:
    att_distances = np.zeros((att.shape[0], att.shape[1]))
    for h in range(att.shape[0]):
        for key in range(att.shape[2]):
            for query in range(att.shape[1]):
                distance = abs(query - key)
                att_distances[h, key] += (
                    torch.abs(att[h, query, key]).cpu().item() * distance
                )
    return att_distances.mean(axis=1)
