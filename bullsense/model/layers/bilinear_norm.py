import torch
import torch.nn as nn

class BilinearNorm(nn.Module):
    """ Per sample normalization for [B, T,F] tensors"""

    def __init__(self, eps = 1e-6):
        super().__init__()
        self.eps = eps


    def forward(self, x):
        # x: [B, T, F]

        mean = x.mean(dim = (1,2), keepdim = True)
        std = x.std(dim = (1,2), keepdim = True)

        return (x-mean) / (std + self.eps)
    