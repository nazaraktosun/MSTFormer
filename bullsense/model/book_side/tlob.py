import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint
from typing import Tuple, Optional, List

# Proje yapına göre burayı düzenle
from bullsense.model.layers.bilinear_norm import BilinearNorm
from bullsense.model.tlob_original.bin import BiN

# ==========================================
# 1. BLOK: MLP Mixer
# ==========================================
class MLPAlongdim(nn.Module):
    """
    Belirli bir boyut (dim) boyunca MLP uygular.
    MLPLOB ve TLOB'un FFN kısmında kullanılır.
    Yapı: Linear -> GELU -> Dropout -> Linear -> (Residual) -> Norm -> GELU
    """
    def __init__(self, start_dim: int, hidden_dim: int, final_dim: int, dropout: float = 0.0):
        super().__init__()
        self.fc1 = nn.Linear(start_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, final_dim)
        self.norm = nn.LayerNorm(final_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [..., start_dim]
        residual = x

        z = self.fc1(x)
        z = F.gelu(z)
        z = self.dropout(z)
        z = self.fc2(z)

        # Boyut değişmiyorsa Residual ekle
        if z.shape[-1] == residual.shape[-1]:
            z = z + residual

        z = self.norm(z)
        z = F.gelu(z)
        return z


class FeedForward(nn.Module):
    """Plain pre-norm-friendly FFN over the last dim: Linear -> GELU -> drop -> Linear.

    No internal norm, no internal residual, no trailing activation on the stream —
    the enclosing block owns the pre-norm + residual. Used by the TLOB blocks.
    """
    def __init__(self, dim: int, hidden_dim: int, dropout: float = 0.0):
        super().__init__()
        self.fc1 = nn.Linear(dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, dim)
        self.drop = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.fc2(self.drop(F.gelu(self.fc1(x))))

# ==========================================
# 2. POSITIONAL ENCODING (4D)
# ==========================================
class SeparablePositionalEncoding(nn.Module):
    """
    4D PE: [B,T,F,d]
    """
    def __init__(self, d_model: int, max_len_t: int, num_features: int):
        super().__init__()
        self.d_model = d_model
        pe_t = torch.zeros(max_len_t, d_model)
        pe_f = torch.zeros(num_features, d_model)

        pos_t = torch.arange(0, max_len_t, dtype=torch.float32).unsqueeze(1)
        pos_f = torch.arange(0, num_features, dtype=torch.float32).unsqueeze(1)

        div = torch.exp(torch.arange(0, d_model, 2, dtype=torch.float32) *
                        (-torch.log(torch.tensor(10000.0)) / d_model))

        pe_t[:, 0::2] = torch.sin(pos_t * div)
        pe_t[:, 1::2] = torch.cos(pos_t * div)
        pe_f[:, 0::2] = torch.sin(pos_f * div)
        pe_f[:, 1::2] = torch.cos(pos_f * div)

        self.register_buffer("pe_t", pe_t)
        self.register_buffer("pe_f", pe_f)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, T, F, d = x.shape
        x = x + self.pe_t[:T].unsqueeze(1) + self.pe_f[:F].unsqueeze(0)
        return x

# ==========================================
# 3. ATTENTION MODULES (pre-norm)
# ==========================================
class SpatialAttention(nn.Module):
    """Attention across FEATURES per timestep (pre-norm residual)."""
    def __init__(self, d_model: int, num_heads: int, dropout: float = 0.1, attn_dropout: float = 0.0):
        super().__init__()
        if d_model % num_heads != 0:
            raise ValueError(f"d_model={d_model} must be divisible by num_heads={num_heads}")
        self.num_heads = num_heads
        self.head_dim = d_model // num_heads
        self.qkv = nn.Linear(d_model, 3 * d_model)
        self.out_proj = nn.Linear(d_model, d_model)
        self.norm = nn.LayerNorm(d_model)
        self.drop = nn.Dropout(dropout)
        self.attn_dropout = float(attn_dropout)

    def forward(self, x: torch.Tensor, return_weights: bool = False) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        # x: [B, T, F, d]
        B, T, n_feat, d = x.shape
        # [B, T, F, d] -> [F, B*T, d]
        z = x.permute(2, 0, 1, 3).reshape(n_feat, B * T, d)
        # pre-norm: normalize the input to attention, add the raw residual back.
        out, w = self._attention(self.norm(z), return_weights=return_weights, causal=False)
        z = z + self.drop(out)
        # [F, B*T, d] -> [B, T, F, d]
        y = z.reshape(n_feat, B, T, d).permute(1, 2, 0, 3)
        return y, w if return_weights else None

    def _attention(
        self,
        z: torch.Tensor,
        *,
        return_weights: bool,
        causal: bool,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        seq_len, batch_axis, d = z.shape
        qkv = self.qkv(z).view(seq_len, batch_axis, 3, self.num_heads, self.head_dim)
        q, k, v = qkv.unbind(dim=2)
        q = q.permute(1, 2, 0, 3)
        k = k.permute(1, 2, 0, 3)
        v = v.permute(1, 2, 0, 3)

        if return_weights:
            scale = self.head_dim ** -0.5
            scores = torch.matmul(q, k.transpose(-2, -1)) * scale
            if causal:
                mask = torch.ones(seq_len, seq_len, device=z.device, dtype=torch.bool).triu(1)
                scores = scores.masked_fill(mask, float("-inf"))
            weights = torch.softmax(scores, dim=-1)
            out = torch.matmul(weights, v)
        else:
            weights = None
            out = F.scaled_dot_product_attention(
                q,
                k,
                v,
                dropout_p=self.attn_dropout if self.training else 0.0,
                is_causal=causal,
            )

        out = out.permute(2, 0, 1, 3).reshape(seq_len, batch_axis, d)
        return self.out_proj(out), weights

class TemporalAttention(nn.Module):
    """Attention across TIME per feature (pre-norm residual).

    Non-causal by default: the whole window is observed history (the label is a
    future return *after* the window), so bidirectional attention leaks nothing
    and uses the full context, matching the original TLOB.
    """
    def __init__(self, d_model: int, num_heads: int, dropout: float = 0.1, attn_dropout: float = 0.0, causal: bool = False):
        super().__init__()
        if d_model % num_heads != 0:
            raise ValueError(f"d_model={d_model} must be divisible by num_heads={num_heads}")
        self.num_heads = num_heads
        self.head_dim = d_model // num_heads
        self.qkv = nn.Linear(d_model, 3 * d_model)
        self.out_proj = nn.Linear(d_model, d_model)
        self.norm = nn.LayerNorm(d_model)
        self.drop = nn.Dropout(dropout)
        self.attn_dropout = float(attn_dropout)
        self.causal = causal

    def forward(self, x: torch.Tensor, return_weights: bool = False) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        # x: [B, T, F, d]
        B, T, n_feat, d = x.shape
        # [B, T, F, d] -> [T, B*F, d]
        z = x.permute(1, 0, 2, 3).reshape(T, B * n_feat, d)
        out, w = self._attention(self.norm(z), return_weights=return_weights, causal=self.causal)
        z = z + self.drop(out)
        # [T, B*F, d] -> [B, T, F, d]
        y = z.reshape(T, B, n_feat, d).permute(1, 0, 2, 3)
        return y, w if return_weights else None

    def _attention(
        self,
        z: torch.Tensor,
        *,
        return_weights: bool,
        causal: bool,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        seq_len, batch_axis, d = z.shape
        qkv = self.qkv(z).view(seq_len, batch_axis, 3, self.num_heads, self.head_dim)
        q, k, v = qkv.unbind(dim=2)
        q = q.permute(1, 2, 0, 3)
        k = k.permute(1, 2, 0, 3)
        v = v.permute(1, 2, 0, 3)

        if return_weights:
            scale = self.head_dim ** -0.5
            scores = torch.matmul(q, k.transpose(-2, -1)) * scale
            if causal:
                mask = torch.ones(seq_len, seq_len, device=z.device, dtype=torch.bool).triu(1)
                scores = scores.masked_fill(mask, float("-inf"))
            weights = torch.softmax(scores, dim=-1)
            out = torch.matmul(weights, v)
        else:
            weights = None
            out = F.scaled_dot_product_attention(
                q,
                k,
                v,
                dropout_p=self.attn_dropout if self.training else 0.0,
                is_causal=causal,
            )

        out = out.permute(2, 0, 1, 3).reshape(seq_len, batch_axis, d)
        return self.out_proj(out), weights

# ==========================================
# 4. SPATIO-TEMPORAL BLOCK (pre-norm)
# ==========================================
class SpatioTemporalBlock(nn.Module):
    def __init__(self, d_model: int, num_heads: int, seq_len: int, num_features: int,
                 dropout: float = 0.1, attn_dropout: float = 0.0, temporal_causal: bool = False):
        super().__init__()
        d_ff = 4 * d_model

        self.t_attn = TemporalAttention(d_model, num_heads, dropout, attn_dropout, causal=temporal_causal)
        self.s_attn = SpatialAttention(d_model, num_heads, dropout, attn_dropout)

        # Channel (feature-embedding) FFN over d_model, pre-norm residual.
        self.norm_ff = nn.LayerNorm(d_model)
        self.ff = FeedForward(d_model, d_ff, dropout=dropout)

        # Time mixing over the T axis, pre-norm residual.
        self.norm_tm = nn.LayerNorm(seq_len)
        self.tm = FeedForward(seq_len, max(4, seq_len * 2), dropout=dropout)

        self.drop = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor, return_weights: bool = False) -> Tuple[torch.Tensor, Optional[dict]]:
        # Attention (each is pre-norm + residual internally)
        x, w_t = self.t_attn(x, return_weights=return_weights)
        x, w_s = self.s_attn(x, return_weights=return_weights)

        # Channel FFN: [B,T,F,d] -> pre-norm over d -> FFN -> residual
        x = x + self.drop(self.ff(self.norm_ff(x)))

        # Time mixing: bring T to the last axis, pre-norm over T -> FFN -> residual
        xt = x.permute(0, 2, 3, 1)              # [B, F, d, T]
        xt = xt + self.drop(self.tm(self.norm_tm(xt)))
        x = xt.permute(0, 3, 1, 2)              # [B, T, F, d]

        if return_weights and w_t is not None:
            return x, {"temporal": w_t.detach(), "spatial": w_s.detach()}
        return x, None

# ==========================================
# 5. FULL MODEL: TLOB (Transformer)
# ==========================================
class TLOB_DeepLOB(nn.Module):
    def __init__(self, num_features:int, seq_len:int,
                 d_model:int=64, num_layers:int=4, num_heads:int=8,
                 num_classes:int=3, dropout:float=0.1, use_bilinear_norm: bool = True,
                 bilinear_norm_type: str = "bin",
                 temporal_readout: str = "last",
                 attn_dropout: float = 0.0,
                 temporal_causal: bool = False,
                 gradient_checkpointing: bool = False):
        super().__init__()
        self.num_features = num_features
        self.seq_len = seq_len
        self.d_model = d_model
        self.num_layers = num_layers
        self.num_heads = num_heads
        self.num_classes = num_classes
        self.use_bilinear_norm = use_bilinear_norm
        self.bilinear_norm_type = bilinear_norm_type.lower()
        self.temporal_readout = temporal_readout.lower()
        self.attn_dropout = float(attn_dropout)
        self.temporal_causal = bool(temporal_causal)
        self.gradient_checkpointing = bool(gradient_checkpointing)

        if not use_bilinear_norm:
            self.bilinear_norm = None
        elif self.bilinear_norm_type == "simple":
            self.bilinear_norm = BilinearNorm(eps=1e-6)
        elif self.bilinear_norm_type == "bin":
            self.bilinear_norm = BiN(num_features, seq_len)
        else:
            raise ValueError(f"Unknown bilinear_norm_type: {bilinear_norm_type}")

        # Per-feature scalar embedding, vectorized: emb[b,t,f,:] = x[b,t,f] * w[f,:] + b[f,:]
        # (equivalent to a separate Linear(1 -> d) per feature, but one op instead of F.)
        self.feat_w = nn.Parameter(torch.empty(num_features, d_model))
        self.feat_b = nn.Parameter(torch.empty(num_features, d_model))
        nn.init.uniform_(self.feat_w, -1.0, 1.0)   # matches Linear(1, d) default (fan_in=1)
        nn.init.uniform_(self.feat_b, -1.0, 1.0)

        # 4D PE
        self.pos_encoder = SeparablePositionalEncoding(
            d_model, max_len_t=seq_len, num_features=num_features
        )

        self.st_blocks = nn.ModuleList([
            SpatioTemporalBlock(d_model, num_heads, seq_len=seq_len, num_features=num_features,
                                dropout=dropout, attn_dropout=attn_dropout, temporal_causal=temporal_causal)
            for _ in range(num_layers)
        ])

        # Final pre-norm before readout (standard for pre-norm stacks).
        self.final_norm = nn.LayerNorm(d_model)

        self.temporal_pool = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Linear(d_model, 1),
        )

        if self.temporal_readout not in {"attention", "mean", "last"}:
            raise ValueError(f"Unknown temporal_readout: {temporal_readout}")
        readout_dim = d_model

        self.feature_fusion = nn.Sequential(
            nn.Linear(readout_dim, d_model // 2),
            nn.GELU(),
            nn.Dropout(dropout),
        )
        fusion_dim = num_features * (d_model // 2)
        self.classifier = nn.Sequential(
            nn.Linear(fusion_dim, fusion_dim // 4),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(fusion_dim // 4, fusion_dim // 16),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(fusion_dim // 16, num_classes),
        )

        self.attention_weights = []

    def forward(self, x: torch.Tensor, return_attention: bool = False) -> torch.Tensor:
        B, T, F = x.shape
        if self.bilinear_norm is not None:
            if self.bilinear_norm_type == "bin":
                x = self.bilinear_norm(x.transpose(1, 2)).transpose(1, 2)
            else:
                x = self.bilinear_norm(x)

        # Vectorized per-feature embedding: [B,T,F] -> [B,T,F,d]
        x = x.unsqueeze(-1) * self.feat_w + self.feat_b

        x = self.pos_encoder(x)

        if return_attention:
            self.attention_weights = []

        for blk in self.st_blocks:
            if self.gradient_checkpointing and self.training and x.requires_grad and not return_attention:
                x = checkpoint(lambda inp, block=blk: block(inp, return_weights=False)[0], x, use_reentrant=False)
                w = None
            else:
                x, w = blk(x, return_weights=return_attention)
            if return_attention and w is not None:
                self.attention_weights.append(w)

        x = self.final_norm(x)
        x = self._readout_time(x)
        x = self.feature_fusion(x)
        x = x.reshape(B, -1)
        logits = self.classifier(x)
        return logits

    def _readout_time(self, x: torch.Tensor) -> torch.Tensor:
        # x: [B, T, F, d] -> [B, F, d]
        if self.temporal_readout == "last":
            return x[:, -1]
        if self.temporal_readout == "mean":
            return x.mean(dim=1)
        # attention pool over time
        time_scores = self.temporal_pool(x).squeeze(-1)  # [B, T, F]
        time_weights = torch.softmax(time_scores, dim=1).unsqueeze(-1)
        return (x * time_weights).sum(dim=1)

    def get_attention_weights(self):
        return self.attention_weights

    def info(self):
        total = sum(p.numel() for p in self.parameters())
        trainable = sum(p.numel() for p in self.parameters() if p.requires_grad)
        return dict(
            name="TLOB",
            num_features=self.num_features,
            seq_len=self.seq_len,
            d_model=self.d_model,
            num_layers=self.num_layers,
            num_heads=self.num_heads,
            num_classes=self.num_classes,
            bilinear_norm_type=self.bilinear_norm_type,
            temporal_readout=self.temporal_readout,
            attn_dropout=self.attn_dropout,
            temporal_causal=self.temporal_causal,
            gradient_checkpointing=self.gradient_checkpointing,
            total_params=total,
            trainable_params=trainable,
        )

# ==========================================
# 6. FULL MODEL: MLPLOB (Standalone MLP Mixer)
# ==========================================
class MLPLOB(nn.Module):
    """
    MLP-LOB: Sadece MLP Mixer blokları kullanan (Attention yok) hafif model.
    Giriş: X [B, T, F] -> Embedding Yok, direkt ham veri üzerinde çalışır veya basit projeksiyon yapılır.
    Ancak orijinal makale embedding sonrası karıştırma öneriyor.
    Burada Bullsense yapısına uygun olarak:
    1. Giriş [B, T, F] -> Permute [B, F, T] -> Linear(T -> H) -> Permute [B, T, H] (Feature Embedding gibi)
    2. Sonra MLPLOB blokları.
    """
    def __init__(self, num_features: int, seq_len: int, hidden_dim: int = 256, num_layers: int = 4, num_classes: int = 3, dropout: float = 0.1):
        super().__init__()
        self.num_features = num_features
        self.seq_len = seq_len
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.num_classes = num_classes

        self.bilinear = BilinearNorm(eps=1e-6)

        # Giriş Projeksiyonu: Feature (F) boyutunu Hidden (H) boyutuna çek
        self.input_proj = nn.Linear(num_features, hidden_dim)

        self.blocks = nn.ModuleList()
        for i in range(num_layers):
            # Her blokta: Feature Mix + Time Mix
            # Blok içi hidden dimension'ı biraz genişletip daraltabiliriz (Expansion factor)
            # Burada basitlik için hidden_dim koruyoruz.

            # Feature Mix (Hidden Dim üzerinde)
            f_mix = MLPAlongdim(start_dim=hidden_dim, hidden_dim=hidden_dim*4, final_dim=hidden_dim, dropout=dropout)

            # Time Mix (Seq Len üzerinde)
            t_mix = MLPAlongdim(start_dim=seq_len, hidden_dim=seq_len*2, final_dim=seq_len, dropout=dropout)

            self.blocks.append(nn.ModuleDict({'f_mix': f_mix, 't_mix': t_mix}))

        self.head = nn.Sequential(
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, num_classes)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [B, T, F]
        B, T, F = x.shape

        # 1. Bilinear Norm
        x = self.bilinear(x)

        # 2. Input Projection: [B, T, F] -> [B, T, H]
        # Linear son boyuta (F) uygulanır, H çıkar.
        x = self.input_proj(x)

        # 3. MLPLOB Blocks
        for block in self.blocks:
            # a) Feature Mixing (H ekseninde) -> [B, T, H]
            x = x + block['f_mix'](x) # Residual

            # b) Time Mixing (T ekseninde)
            # T'yi sona at: [B, H, T]
            x_perm = x.transpose(1, 2)
            x_perm = block['t_mix'](x_perm)
            x = x + x_perm.transpose(1, 2) # Geri al ve Residual ekle

        # 4. Global Average Pooling over Time
        x = x.mean(dim=1) # [B, H]

        # 5. Classification
        logits = self.head(x)
        return logits

    def info(self):
        total = sum(p.numel() for p in self.parameters())
        trainable = sum(p.numel() for p in self.parameters() if p.requires_grad)
        return dict(
            name="MLPLOB",
            num_features=self.num_features,
            seq_len=self.seq_len,
            hidden_dim=self.hidden_dim,
            num_layers=self.num_layers,
            num_classes=self.num_classes,
            total_params=total,
            trainable_params=trainable,
        )
