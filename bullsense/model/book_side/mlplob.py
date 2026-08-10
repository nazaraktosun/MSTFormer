"""
MLP-LOB Model adapted for BullSense orderbook data
Based on: https://github.com/LeonardoBerti00/TLOB
"""

from typing import List
import torch
import torch.nn as nn
import torch.nn.functional as F
from bullsense.model.layers.bilinear_norm import BilinearNorm 

class MLPAlongdim(nn.Module):

    def __init__(self, start_dim: int, hidden_dim: int, final_dim: int, dropout: float = 0.0):
        super().__init__()
        self.fc1 = nn.Linear(start_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, final_dim)
        self.norm = nn.LayerNorm(final_dim)
        self.dropout = nn.Dropout(dropout)

    
    def forward(self,x):
        residual = x #sadece referansı kopyalıyor
        z = F.gelu(self.fc1(x)) 
        z = self.dropout(z)
        z = self.fc2(z)

        if z.shape[-1] == residual.shape[-1]:
            z = z+residual

        z = self.norm(z)
        z = F.gelu(z)
        return z
    
class MLPLOB(nn.Module):
    """
    MLP-LOB (feature-mix + time-mix blokları)
    Giriş:  X [B, T, F]
    Çıkış:  logits [B, C]
    """

    def __init__(self, 
                num_features:int, 
                seq_len: int, 
                hidden_dim : int = 256, 
                num_layers : int = 4, 
                num_classes: int = 3, 
                dropout : float = 0.2):
        
        super().__init__()
        self.num_features = num_features
        self.seq_len = seq_len
        self.hidden_dim= hidden_dim
        self.num_layers = num_layers
        self.num_classes = num_classes

        # bilinear normalization
        self.bilinear = BilinearNorm(eps = 1e-6)

        # ilk projeksiyon
        self.input_proj = nn.Linear(num_features,hidden_dim)


        # her blokta feature ve time mix
        self.feature_mlps = nn.ModuleList()
        self.time_mlps = nn.ModuleList()
        for i in range (num_layers):
            # feature mix son   H    üzerine mlp
            self.feature_mlps.append(
                    MLPAlongdim(
                        start_dim= hidden_dim,
                        hidden_dim= hidden_dim*4 if i < num_layers -1 else hidden_dim*2,
                        final_dim= hidden_dim if i < num_layers -1 else hidden_dim,
                        dropout=dropout,
                    )
            )

            # time mix son T boyut üstünde  MLP 
            self.time_mlps.append(MLPAlongdim(start_dim= seq_len,
                                              hidden_dim=seq_len*4 if i < num_layers-1 else max(4,seq_len*2),
                                              final_dim= seq_len,
                                              dropout=dropout
                                              ))
            

            # 3) head: GAP(T,F) -> d -> sınıflar

            self.head = nn.Sequential(nn.LayerNorm(hidden_dim),
                                      nn.LayerNorm(hidden_dim),
                                      nn.Linear(hidden_dim,hidden_dim//2),
                                      nn.GELU(),
                                    nn.Dropout(dropout),
                                    nn.Linear(hidden_dim//2,num_classes))
            

    def forward(self, x):  # x: [B, T, F]
            B, T, F = x.shape

            # BilinearNorm: [B,T,F] -> [B,F,T] (norm) -> [B,T,F]
            x = x.transpose(1, 2)         # [B, F, T]
            x = self.bilinear(x)          # normalize over (F,T) per sample
            x = x.transpose(1, 2)         # [B, T, F]

            # İlk projeksiyon: F -> H
            x = self.input_proj(x)        # [B, T, H]

            # Bloklar: feature-mix -> time-mix
            for f_mlp, t_mlp in zip(self.feature_mlps, self.time_mlps):
                # feature-mix H yönünde
                x = f_mlp(x)              # [B, T, H]
                # time-mix T yönünde (son boyut T olacak şekilde permute et)
                x = x.transpose(1, 2)     # [B, H, T]
                x = t_mlp(x)              # [B, H, T]
                x = x.transpose(1, 2)     # [B, T, H]

            # GAP(T,F) yerine artık F=H (projeksiyon sonrası), T üzerinde ve feature üzerinde ortalama:
            x = x.mean(dim=1)             # mean over T  -> [B, H]
            logits = self.head(x)         # [B, C]
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
            
