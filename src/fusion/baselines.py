"""
Baseline fusion mechanisms.

Implements the nine baseline architectures benchmarked in the thesis:

    * :class:`LMF`                       — Low-rank Multimodal Fusion (Liu et al., 2018)
    * :class:`EarlyFusion`               — Concatenation + MLP
    * :class:`LateFusion`                — Per-modality classifier + learned weights
    * :class:`TensorFusionNetwork`       — Outer product of all modalities (Zadeh et al., 2017)
    * :class:`MultimodalFactorizedBilinear` — Pairwise bilinear pooling (Yu et al., 2017)
    * :class:`CrossModalAttention`       — Pairwise MHA between modalities
    * :class:`GatedMultimodalUnit`       — Context-conditioned gating (Arevalo et al., 2017)
    * :class:`MultimodalTransformer`     — Transformer over modality tokens (Tsai et al., 2019)
    * :class:`TuckerFusion`              — Tucker-decomposed tensor fusion

All modules share the same interface::

    forward(text_x, audio_x, video_x) -> logits (B, output_dim)

Each modality is a tensor of shape ``(batch, hidden_dim)`` or ``None``.
Missing modalities are replaced with zero vectors unless stated otherwise.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from torch.nn.parameter import Parameter
from torch.nn.init import xavier_normal_


# ---------------------------------------------------------------------------
# Small utility: pick batch size and device from whichever modality exists.
# ---------------------------------------------------------------------------

def _batch_and_device(text_x, audio_x, video_x):
    ref = text_x if text_x is not None else \
          audio_x if audio_x is not None else video_x
    return ref.shape[0], ref.device


# =============================================================================
#  LMF — Low-rank Multimodal Fusion (Liu et al., 2018)
# =============================================================================

class LMF(nn.Module):
    """Rank-decomposed tensor fusion.

    Decomposes the full TFN weight tensor into modality-specific rank-r
    factors and combines them via element-wise (Hadamard) product. Very
    parameter-efficient (~37K params with rank 16, d=256), but the
    Hadamard product means that when any modality is zero the fused
    representation collapses to zero — a critical failure mode under
    sensor dropout, documented in the thesis ablation.
    """

    def __init__(self, hidden_dim: int = 256, output_dim: int = 3,
                 rank: int = 16, use_softmax: bool = False,
                 dropout: float = 0.1):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        self.rank = rank
        self.use_softmax = use_softmax

        self.post_fusion_dropout = nn.Dropout(p=dropout)
        self.factor_text  = Parameter(torch.Tensor(rank, hidden_dim + 1, output_dim))
        self.factor_audio = Parameter(torch.Tensor(rank, hidden_dim + 1, output_dim))
        self.factor_video = Parameter(torch.Tensor(rank, hidden_dim + 1, output_dim))
        self.fusion_weights = Parameter(torch.Tensor(1, rank))
        self.fusion_bias = Parameter(torch.Tensor(1, output_dim))

        for p in [self.factor_text, self.factor_audio, self.factor_video,
                  self.fusion_weights]:
            xavier_normal_(p)
        self.fusion_bias.data.fill_(0)

    def forward(self, text_x, audio_x, video_x):
        batch_size, device = _batch_and_device(text_x, audio_x, video_x)
        DTYPE = torch.cuda.FloatTensor if device.type == "cuda" else torch.FloatTensor

        fusion_text = fusion_audio = fusion_video = None

        if text_x is not None:
            _h = torch.cat([Variable(torch.ones(batch_size, 1).type(DTYPE),
                                     requires_grad=False), text_x], dim=1)
            fusion_text = torch.matmul(_h, self.factor_text)
        if audio_x is not None:
            _h = torch.cat([Variable(torch.ones(batch_size, 1).type(DTYPE),
                                     requires_grad=False), audio_x], dim=1)
            fusion_audio = torch.matmul(_h, self.factor_audio)
        if video_x is not None:
            _h = torch.cat([Variable(torch.ones(batch_size, 1).type(DTYPE),
                                     requires_grad=False), video_x], dim=1)
            fusion_video = torch.matmul(_h, self.factor_video)

        available = [f for f in [fusion_text, fusion_audio, fusion_video]
                     if f is not None]
        if len(available) == 0:
            raise ValueError("At least one modality must be provided")
        if len(available) == 1:
            fusion_zy = available[0]
        elif len(available) == 2:
            fusion_zy = available[0] * available[1]
        else:
            fusion_zy = fusion_text * fusion_audio * fusion_video

        output = torch.matmul(self.fusion_weights,
                              fusion_zy.permute(1, 0, 2)).squeeze() + self.fusion_bias
        output = output.view(-1, self.output_dim)
        if self.use_softmax:
            output = F.softmax(output, dim=-1)
        return output


# =============================================================================
#  Early Fusion
# =============================================================================

class EarlyFusion(nn.Module):
    """Concatenation followed by a 3-layer MLP with LayerNorm and dropout."""

    def __init__(self, hidden_dim: int = 256, output_dim: int = 3,
                 dropout: float = 0.1):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.fc1 = nn.Linear(hidden_dim * 3, hidden_dim * 2)
        self.fc2 = nn.Linear(hidden_dim * 2, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, output_dim)
        self.layer_norm = nn.LayerNorm(hidden_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, text_x, audio_x, video_x):
        batch_size, device = _batch_and_device(text_x, audio_x, video_x)
        if text_x is None:  text_x  = torch.zeros(batch_size, self.hidden_dim, device=device)
        if audio_x is None: audio_x = torch.zeros(batch_size, self.hidden_dim, device=device)
        if video_x is None: video_x = torch.zeros(batch_size, self.hidden_dim, device=device)

        fused = torch.cat([text_x, audio_x, video_x], dim=1)
        x = F.relu(self.fc1(fused))
        x = self.dropout(x)
        x = F.relu(self.fc2(x))
        x = self.layer_norm(x)
        x = self.dropout(x)
        return self.fc3(x)


# =============================================================================
#  Late Fusion
# =============================================================================

class LateFusion(nn.Module):
    """Independent per-modality classifiers combined by softmax-weighted sum."""

    def __init__(self, hidden_dim: int = 256, output_dim: int = 3,
                 dropout: float = 0.1):
        super().__init__()
        def _head():
            return nn.Sequential(
                nn.Linear(hidden_dim, hidden_dim // 2),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(hidden_dim // 2, output_dim),
            )
        self.text_classifier  = _head()
        self.audio_classifier = _head()
        self.video_classifier = _head()
        # Learnable scalar weights — combined via softmax over present modalities.
        self.weights = nn.Parameter(torch.ones(3))

    def forward(self, text_x, audio_x, video_x):
        outputs, active = [], []
        if text_x is not None:
            outputs.append(self.text_classifier(text_x));  active.append(self.weights[0])
        if audio_x is not None:
            outputs.append(self.audio_classifier(audio_x)); active.append(self.weights[1])
        if video_x is not None:
            outputs.append(self.video_classifier(video_x)); active.append(self.weights[2])

        w = F.softmax(torch.stack(active), dim=0)
        return sum(wi * oi for wi, oi in zip(w, outputs))


# =============================================================================
#  Tensor Fusion Network (Zadeh et al., 2017)
# =============================================================================

class TensorFusionNetwork(nn.Module):
    """Outer product of bias-augmented modality vectors → flatten → MLP.

    The full (d+1)^3 representation grows cubically with d, so TFN is
    only practical for small d.
    """

    def __init__(self, hidden_dim: int = 256, output_dim: int = 3,
                 dropout: float = 0.1):
        super().__init__()
        self.hidden_dim = hidden_dim
        post_dim = (hidden_dim + 1) ** 3
        self.post_dropout = nn.Dropout(dropout)
        self.fc1 = nn.Linear(post_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim // 2)
        self.fc3 = nn.Linear(hidden_dim // 2, output_dim)

    def forward(self, text_x, audio_x, video_x):
        batch_size, device = _batch_and_device(text_x, audio_x, video_x)
        if text_x is None:  text_x  = torch.zeros(batch_size, self.hidden_dim, device=device)
        if audio_x is None: audio_x = torch.zeros(batch_size, self.hidden_dim, device=device)
        if video_x is None: video_x = torch.zeros(batch_size, self.hidden_dim, device=device)

        # Add bias term so that unimodal/bimodal interactions are included.
        ones = torch.ones(batch_size, 1, device=device)
        text_x  = torch.cat([ones, text_x],  dim=1)
        audio_x = torch.cat([ones, audio_x], dim=1)
        video_x = torch.cat([ones, video_x], dim=1)

        # 3-way outer product, flattened.
        fusion = torch.bmm(text_x.unsqueeze(2), audio_x.unsqueeze(1))
        fusion = fusion.view(batch_size, -1, 1)
        fusion = torch.bmm(fusion, video_x.unsqueeze(1))
        fusion = fusion.view(batch_size, -1)

        x = self.post_dropout(fusion)
        x = F.relu(self.fc1(x))
        x = self.post_dropout(x)
        x = F.relu(self.fc2(x))
        return self.fc3(x)


# =============================================================================
#  Multimodal Factorized Bilinear Pooling (Yu et al., 2017)
# =============================================================================

class MultimodalFactorizedBilinear(nn.Module):
    """Pairwise MFB pooling with signed sqrt and L2 normalisation."""

    def __init__(self, hidden_dim: int = 256, output_dim: int = 3,
                 mfb_factor: int = 5, dropout: float = 0.1):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.mfb_factor = mfb_factor
        self.mfb_out_dim = hidden_dim

        def _pair():
            return (nn.Linear(hidden_dim, self.mfb_out_dim * mfb_factor),
                    nn.Linear(hidden_dim, self.mfb_out_dim * mfb_factor))

        self.text_audio_proj1,  self.text_audio_proj2  = _pair()
        self.audio_video_proj1, self.audio_video_proj2 = _pair()
        self.text_video_proj1,  self.text_video_proj2  = _pair()

        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(self.mfb_out_dim * 3, output_dim)

    def _pool(self, x1, x2, p1, p2):
        z1 = p1(x1).view(-1, self.mfb_factor, self.mfb_out_dim)
        z2 = p2(x2).view(-1, self.mfb_factor, self.mfb_out_dim)
        z = (z1 * z2).sum(1)
        z = torch.sqrt(F.relu(z)) - torch.sqrt(F.relu(-z))
        return F.normalize(z, p=2, dim=1)

    def forward(self, text_x, audio_x, video_x):
        batch_size, device = _batch_and_device(text_x, audio_x, video_x)
        if text_x is None:  text_x  = torch.zeros(batch_size, self.hidden_dim, device=device)
        if audio_x is None: audio_x = torch.zeros(batch_size, self.hidden_dim, device=device)
        if video_x is None: video_x = torch.zeros(batch_size, self.hidden_dim, device=device)

        ta = self._pool(text_x, audio_x, self.text_audio_proj1, self.text_audio_proj2)
        av = self._pool(audio_x, video_x, self.audio_video_proj1, self.audio_video_proj2)
        tv = self._pool(text_x, video_x, self.text_video_proj1, self.text_video_proj2)

        fused = torch.cat([ta, av, tv], dim=1)
        return self.fc(self.dropout(fused))


# =============================================================================
#  Cross-Modal Attention
# =============================================================================

class CrossModalAttention(nn.Module):
    """Pairwise multi-head attention between every modality pair."""

    def __init__(self, hidden_dim: int = 256, output_dim: int = 3,
                 num_heads: int = 4, dropout: float = 0.1):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.text_audio_attn  = nn.MultiheadAttention(hidden_dim, num_heads, dropout=dropout, batch_first=True)
        self.text_video_attn  = nn.MultiheadAttention(hidden_dim, num_heads, dropout=dropout, batch_first=True)
        self.audio_video_attn = nn.MultiheadAttention(hidden_dim, num_heads, dropout=dropout, batch_first=True)
        self.norm1, self.norm2, self.norm3 = nn.LayerNorm(hidden_dim), nn.LayerNorm(hidden_dim), nn.LayerNorm(hidden_dim)
        self.fusion = nn.Sequential(
            nn.Linear(hidden_dim * 3, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, output_dim),
        )

    def forward(self, text_x, audio_x, video_x):
        batch_size, device = _batch_and_device(text_x, audio_x, video_x)
        if text_x is None:  text_x  = torch.zeros(batch_size, self.hidden_dim, device=device)
        if audio_x is None: audio_x = torch.zeros(batch_size, self.hidden_dim, device=device)
        if video_x is None: video_x = torch.zeros(batch_size, self.hidden_dim, device=device)

        text_x, audio_x, video_x = (x.unsqueeze(1) for x in (text_x, audio_x, video_x))

        ta, _ = self.text_audio_attn(text_x, audio_x, audio_x)
        ta = self.norm1(ta + text_x)
        tv, _ = self.text_video_attn(text_x, video_x, video_x)
        tv = self.norm2(tv + text_x)
        av, _ = self.audio_video_attn(audio_x, video_x, video_x)
        av = self.norm3(av + audio_x)

        fused = torch.cat([ta, tv, av], dim=-1).squeeze(1)
        return self.fusion(fused)


# =============================================================================
#  Gated Multimodal Unit (Arevalo et al., 2017)
# =============================================================================

class GatedMultimodalUnit(nn.Module):
    """Context-conditioned gating — the strongest baseline on MM-F2F.

    Gates are computed from the full multimodal context so each modality's
    contribution is conditioned on what the others are saying.
    """

    def __init__(self, hidden_dim: int = 256, output_dim: int = 3,
                 dropout: float = 0.1):
        super().__init__()
        self.hidden_dim = hidden_dim
        def _gate():
            return nn.Sequential(nn.Linear(hidden_dim * 3, hidden_dim), nn.Sigmoid())
        self.text_gate, self.audio_gate, self.video_gate = _gate(), _gate(), _gate()
        self.text_transform  = nn.Linear(hidden_dim, hidden_dim)
        self.audio_transform = nn.Linear(hidden_dim, hidden_dim)
        self.video_transform = nn.Linear(hidden_dim, hidden_dim)
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, output_dim),
        )

    def forward(self, text_x, audio_x, video_x):
        batch_size, device = _batch_and_device(text_x, audio_x, video_x)
        if text_x is None:  text_x  = torch.zeros(batch_size, self.hidden_dim, device=device)
        if audio_x is None: audio_x = torch.zeros(batch_size, self.hidden_dim, device=device)
        if video_x is None: video_x = torch.zeros(batch_size, self.hidden_dim, device=device)

        ctx = torch.cat([text_x, audio_x, video_x], dim=1)
        t_h = self.text_gate(ctx)  * torch.tanh(self.text_transform(text_x))
        a_h = self.audio_gate(ctx) * torch.tanh(self.audio_transform(audio_x))
        v_h = self.video_gate(ctx) * torch.tanh(self.video_transform(video_x))
        fused = self.dropout(t_h + a_h + v_h)
        return self.fc(fused)


# =============================================================================
#  Multimodal Transformer (Tsai et al., 2019)
# =============================================================================

class MultimodalTransformer(nn.Module):
    """Transformer encoder over a sequence of modality tokens."""

    def __init__(self, hidden_dim: int = 256, output_dim: int = 3,
                 num_heads: int = 4, num_layers: int = 2, dropout: float = 0.1):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.text_embed  = nn.Linear(hidden_dim, hidden_dim)
        self.audio_embed = nn.Linear(hidden_dim, hidden_dim)
        self.video_embed = nn.Linear(hidden_dim, hidden_dim)
        self.modality_embedding = nn.Embedding(3, hidden_dim)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim, nhead=num_heads,
            dim_feedforward=hidden_dim * 4, dropout=dropout, batch_first=True,
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(hidden_dim, output_dim)

    def forward(self, text_x, audio_x, video_x):
        batch_size, device = _batch_and_device(text_x, audio_x, video_x)

        modalities, ids = [], []
        if text_x is not None:
            modalities.append(self.text_embed(text_x).unsqueeze(1));  ids.append(0)
        if audio_x is not None:
            modalities.append(self.audio_embed(audio_x).unsqueeze(1)); ids.append(1)
        if video_x is not None:
            modalities.append(self.video_embed(video_x).unsqueeze(1)); ids.append(2)

        x = torch.cat(modalities, dim=1)
        ids = torch.tensor(ids, device=device)
        mod_embs = self.modality_embedding(ids).unsqueeze(0).expand(batch_size, -1, -1)
        x = x + mod_embs

        x = self.transformer(x).mean(dim=1)
        return self.fc(self.dropout(x))


# =============================================================================
#  Tucker Fusion
# =============================================================================

class TuckerFusion(nn.Module):
    """Tucker-decomposed tensor fusion (generalises LMF with a core tensor)."""

    def __init__(self, hidden_dim: int = 256, output_dim: int = 3,
                 rank=(16, 16, 16), dropout: float = 0.1):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.text_factor  = Parameter(torch.Tensor(rank[0], hidden_dim + 1))
        self.audio_factor = Parameter(torch.Tensor(rank[1], hidden_dim + 1))
        self.video_factor = Parameter(torch.Tensor(rank[2], hidden_dim + 1))
        self.core_tensor  = Parameter(torch.Tensor(rank[0], rank[1], rank[2]))

        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Sequential(
            nn.Linear(rank[0] * rank[1] * rank[2], hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, output_dim),
        )

        for p in [self.text_factor, self.audio_factor, self.video_factor,
                  self.core_tensor]:
            xavier_normal_(p)

    def forward(self, text_x, audio_x, video_x):
        batch_size, device = _batch_and_device(text_x, audio_x, video_x)
        if text_x is None:  text_x  = torch.zeros(batch_size, self.hidden_dim, device=device)
        if audio_x is None: audio_x = torch.zeros(batch_size, self.hidden_dim, device=device)
        if video_x is None: video_x = torch.zeros(batch_size, self.hidden_dim, device=device)

        ones = torch.ones(batch_size, 1, device=device)
        text_x  = torch.cat([ones, text_x],  dim=1)
        audio_x = torch.cat([ones, audio_x], dim=1)
        video_x = torch.cat([ones, video_x], dim=1)

        t_p = torch.matmul(text_x,  self.text_factor.t())
        a_p = torch.matmul(audio_x, self.audio_factor.t())
        v_p = torch.matmul(video_x, self.video_factor.t())

        fusion = torch.einsum("bi,bj,bk->bijk", t_p, a_p, v_p).view(batch_size, -1)
        return self.fc(self.dropout(fusion))
