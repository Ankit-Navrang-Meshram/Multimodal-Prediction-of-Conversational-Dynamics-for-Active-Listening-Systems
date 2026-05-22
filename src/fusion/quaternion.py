"""
Quaternion fusion.

Represents the three modalities as the three imaginary components
(i, j, k) of a quaternion and a learnable global-context vector as the
real component (r). The Hamilton product then couples all four components
non-commutatively, which means the *ordering* of the modalities is part
of the learned representation — unlike Hadamard or dot-product fusion.

This module serves as an algebraic-family comparison point in the
benchmark (~395K parameters).
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.parameter import Parameter
from torch.nn.init import xavier_normal_


class QuaternionLinear(nn.Module):
    """Standard quaternion-valued linear layer with the Hamilton product.

    For input ``x`` of shape ``(B, 4 * in_features)`` split into
    quaternion components ``(r, i, j, k)``, computes::

        out_r = W_r r - W_i i - W_j j - W_k k
        out_i = W_i r + W_r i + W_k j - W_j k
        out_j = W_j r - W_k i + W_r j + W_i k
        out_k = W_k r + W_j i - W_i j + W_r k

    The non-commutative coupling preserves the role of each modality in
    a way that element-wise products cannot.
    """

    def __init__(self, in_features: int, out_features: int):
        super().__init__()
        self.in_features = in_features // 4
        self.out_features = out_features // 4

        # 4 distinct weight blocks form the Hamilton product matrix.
        self.r_weight = Parameter(torch.Tensor(self.out_features, self.in_features))
        self.i_weight = Parameter(torch.Tensor(self.out_features, self.in_features))
        self.j_weight = Parameter(torch.Tensor(self.out_features, self.in_features))
        self.k_weight = Parameter(torch.Tensor(self.out_features, self.in_features))
        self.bias = Parameter(torch.Tensor(out_features))

        for w in [self.r_weight, self.i_weight, self.j_weight, self.k_weight]:
            xavier_normal_(w)
        self.bias.data.fill_(0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        r, i, j, k = torch.chunk(x, 4, dim=1)

        cat_r = F.linear(r, self.r_weight) - F.linear(i, self.i_weight) - F.linear(j, self.j_weight) - F.linear(k, self.k_weight)
        cat_i = F.linear(r, self.i_weight) + F.linear(i, self.r_weight) + F.linear(j, self.k_weight) - F.linear(k, self.j_weight)
        cat_j = F.linear(r, self.j_weight) - F.linear(i, self.k_weight) + F.linear(j, self.r_weight) + F.linear(k, self.i_weight)
        cat_k = F.linear(r, self.k_weight) + F.linear(i, self.j_weight) - F.linear(j, self.i_weight) + F.linear(k, self.r_weight)

        return torch.cat([cat_r, cat_i, cat_j, cat_k], dim=-1) + self.bias


class QuaternionFusion(nn.Module):
    """Quaternion fusion with a learned global-context real component.

    The global-context vector ``r`` (initialised with small Gaussian noise)
    plays the role of the real part, while the three modality embeddings
    fill the imaginary parts.
    """

    def __init__(self, feature_dim: int = 256, output_dim: int = 3,
                 dropout: float = 0.2):
        super().__init__()
        self.dim = feature_dim

        # Trainable global context — the real (r) quaternion component.
        self.global_context = Parameter(torch.zeros(1, self.dim))
        nn.init.normal_(self.global_context, std=0.02)

        # 4 quaternion components of size `dim` → total 4*dim.
        self.quat_linear = QuaternionLinear(self.dim * 4, self.dim * 4)

        self.classifier = nn.Sequential(
            nn.Linear(self.dim * 4, 128),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(128, output_dim),
        )

    def forward(self, text_x, audio_x, video_x):
        ref = text_x if text_x is not None else \
              audio_x if audio_x is not None else video_x
        batch_size, device = ref.size(0), ref.device

        if text_x is None:  text_x  = torch.zeros(batch_size, self.dim, device=device)
        if audio_x is None: audio_x = torch.zeros(batch_size, self.dim, device=device)
        if video_x is None: video_x = torch.zeros(batch_size, self.dim, device=device)

        r = self.global_context.expand(batch_size, -1)
        q_in = torch.cat([r, text_x, audio_x, video_x], dim=1)

        q_out = self.quat_linear(q_in)
        return self.classifier(q_out)
