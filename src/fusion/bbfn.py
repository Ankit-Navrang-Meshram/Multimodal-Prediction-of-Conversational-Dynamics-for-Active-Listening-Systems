"""
Bi-Bimodal Fusion Network (BBFN, Han et al., 2021).

Two bimodal complementation modules (Text-Acoustic, Text-Visual), each
stacking ``num_layers`` complementation layers with gated cross-attention,
feature separators, and a final prediction head over the concatenation of
four CLS-style head representations.

This is by far the largest fusion module benchmarked (~11.6M parameters)
and serves primarily as a high-capacity reference point for the
parameter-efficiency analysis.

During training, the per-layer feature-separator losses are returned alongside
the logits and aggregated with a fixed weight ``lambda_sep = 0.1`` in the
total loss (see ``scripts/train_fusion.py``).
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


# ---------------------------------------------------------------------------
# Sub-module 1 : Single complementation layer (BiGRU + gated cross-attention)
# ---------------------------------------------------------------------------

class ModalityComplementationLayer(nn.Module):
    """A single complementation layer following the BBFN paper.

    For each modality, runs a BiGRU, then cross-attends to the partner
    modality, then applies retain / compound gates from the mean-pooled
    BiGRU states, then a feed-forward block — all with residual connections
    and layer norms.
    """

    def __init__(self, hidden_dim: int = 256, num_heads: int = 4,
                 dropout: float = 0.1):
        super().__init__()
        self.hidden_dim = hidden_dim

        # BiGRU for sequence encoding before attention.
        self.bigru_m1 = nn.GRU(hidden_dim, hidden_dim // 2, num_layers=1,
                               batch_first=True, bidirectional=True)
        self.bigru_m2 = nn.GRU(hidden_dim, hidden_dim // 2, num_layers=1,
                               batch_first=True, bidirectional=True)

        # Cross-modal multi-head attention (Q from main, K=V from partner).
        self.mha_m1 = nn.MultiheadAttention(hidden_dim, num_heads, dropout=dropout, batch_first=True)
        self.mha_m2 = nn.MultiheadAttention(hidden_dim, num_heads, dropout=dropout, batch_first=True)

        # Feed-forward networks.
        def _ffn():
            return nn.Sequential(
                nn.Linear(hidden_dim, hidden_dim * 4),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(hidden_dim * 4, hidden_dim),
            )
        self.ffn_m1, self.ffn_m2 = _ffn(), _ffn()

        # Layer norms (two per pipeline).
        self.norm1_m1, self.norm2_m1 = nn.LayerNorm(hidden_dim), nn.LayerNorm(hidden_dim)
        self.norm1_m2, self.norm2_m2 = nn.LayerNorm(hidden_dim), nn.LayerNorm(hidden_dim)

        # Retain (W_r) and Compound (W_c) gate projections.
        self.W_r_m1 = nn.Linear(hidden_dim * 2, hidden_dim)
        self.W_c_m1 = nn.Linear(hidden_dim * 2, hidden_dim)
        self.W_r_m2 = nn.Linear(hidden_dim * 2, hidden_dim)
        self.W_c_m2 = nn.Linear(hidden_dim * 2, hidden_dim)

        self.dropout = nn.Dropout(dropout)

    def forward(self, X_m1, X_m2):
        """
        Parameters
        ----------
        X_m1, X_m2 : (batch, seq_len, hidden_dim) or (batch, hidden_dim)

        Returns
        -------
        X_m1_out, X_m2_out, h_bar_m1, h_bar_m2 :
            Updated sequence representations + their mean-pooled summaries
            (needed by the feature separator).
        """
        # Promote (B, D) → (B, 1, D) so seq-aware modules work uniformly.
        if X_m1.dim() == 2: X_m1 = X_m1.unsqueeze(1)
        if X_m2.dim() == 2: X_m2 = X_m2.unsqueeze(1)

        # BiGRU encoding + mean-pool for gate context.
        h_m1, _ = self.bigru_m1(X_m1);  h_bar_m1 = torch.mean(h_m1, dim=1)
        h_m2, _ = self.bigru_m2(X_m2);  h_bar_m2 = torch.mean(h_m2, dim=1)

        # Gates from the concatenated mean-pooled summaries.
        concat_m1 = torch.cat([h_bar_m1, h_bar_m2], dim=1)
        g_r_m1 = torch.sigmoid(self.W_r_m1(concat_m1))
        g_c_m1 = torch.sigmoid(self.W_c_m1(concat_m1))

        concat_m2 = torch.cat([h_bar_m2, h_bar_m1], dim=1)
        g_r_m2 = torch.sigmoid(self.W_r_m2(concat_m2))
        g_c_m2 = torch.sigmoid(self.W_c_m2(concat_m2))

        # Cross-modal attention.
        m_m1, _ = self.mha_m1(X_m1, X_m2, X_m2)
        m_m2, _ = self.mha_m2(X_m2, X_m1, X_m1)

        # Gated residual + layer norm.
        g_r_m1_e, g_c_m1_e = g_r_m1.unsqueeze(1), g_c_m1.unsqueeze(1)
        g_r_m2_e, g_c_m2_e = g_r_m2.unsqueeze(1), g_c_m2.unsqueeze(1)

        X_tilde_m1 = self.norm1_m1(g_c_m1_e * m_m1 + g_r_m1_e * X_m1)
        X_tilde_m2 = self.norm1_m2(g_c_m2_e * m_m2 + g_r_m2_e * X_m2)

        X_m1_out = self.norm2_m1(X_tilde_m1 + self.ffn_m1(X_tilde_m1))
        X_m2_out = self.norm2_m2(X_tilde_m2 + self.ffn_m2(X_tilde_m2))

        return X_m1_out, X_m2_out, h_bar_m1, h_bar_m2


# ---------------------------------------------------------------------------
# Sub-module 2 : Discriminator-based feature separator
# ---------------------------------------------------------------------------

class ModalitySpecificFeatureSeparator(nn.Module):
    """Adversarial-style discriminator that enforces modality discriminability.

    Optionally groups consecutive samples to reduce noise on small batches.
    Returns the BCE loss between the discriminator predictions and pseudo
    labels (0 for modality 1, 1 for modality 2).
    """

    def __init__(self, hidden_dim: int = 256, group_size: int = 4):
        super().__init__()
        self.group_size = group_size
        self.discriminator = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim // 2, 1),
            nn.Sigmoid(),
        )

    def forward(self, h_bar_m1, h_bar_m2):
        batch_size = h_bar_m1.shape[0]
        num_groups = batch_size // self.group_size
        if num_groups == 0:
            h_tilde_m1, h_tilde_m2 = h_bar_m1, h_bar_m2
        else:
            h_tilde_m1 = torch.mean(
                h_bar_m1[:num_groups * self.group_size].view(num_groups, self.group_size, -1),
                dim=1,
            )
            h_tilde_m2 = torch.mean(
                h_bar_m2[:num_groups * self.group_size].view(num_groups, self.group_size, -1),
                dim=1,
            )

        combined = torch.cat([h_tilde_m1, h_tilde_m2], dim=0)
        labels = torch.cat([
            torch.zeros(h_tilde_m1.shape[0], 1, device=h_tilde_m1.device),
            torch.ones(h_tilde_m2.shape[0], 1, device=h_tilde_m2.device),
        ], dim=0)

        preds = self.discriminator(combined)
        loss = F.binary_cross_entropy(preds, labels)
        return loss, preds


# ---------------------------------------------------------------------------
# Full BBFN module
# ---------------------------------------------------------------------------

class BiBimodalFusionNetwork(nn.Module):
    """Two bimodal complementation modules + feature separators + MLP head.

    Set ``return_losses=True`` during training to obtain the per-layer
    separator losses, which are added to the cross-entropy loss with a
    weight of ``lambda_sep`` (default 0.1).
    """

    def __init__(self, hidden_dim: int = 256, output_dim: int = 3,
                 num_layers: int = 2, num_heads: int = 4,
                 dropout: float = 0.1, lambda_sep: float = 0.1,
                 group_size: int = 4):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.lambda_sep = lambda_sep

        # Text-Acoustic stack.
        self.ta_layers = nn.ModuleList([
            ModalityComplementationLayer(hidden_dim, num_heads, dropout)
            for _ in range(num_layers)
        ])
        self.ta_separators = nn.ModuleList([
            ModalitySpecificFeatureSeparator(hidden_dim, group_size)
            for _ in range(num_layers)
        ])

        # Text-Visual stack.
        self.tv_layers = nn.ModuleList([
            ModalityComplementationLayer(hidden_dim, num_heads, dropout)
            for _ in range(num_layers)
        ])
        self.tv_separators = nn.ModuleList([
            ModalitySpecificFeatureSeparator(hidden_dim, group_size)
            for _ in range(num_layers)
        ])

        # Final classifier over concatenated head representations.
        self.prediction_layer = nn.Sequential(
            nn.Linear(hidden_dim * 4, hidden_dim * 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, output_dim),
        )

    def forward(self, text_x, audio_x, video_x, return_losses: bool = False):
        ref = text_x if text_x is not None else \
              audio_x if audio_x is not None else video_x
        batch_size, device = ref.shape[0], ref.device

        if text_x is None:  text_x  = torch.zeros(batch_size, self.hidden_dim, device=device)
        if audio_x is None: audio_x = torch.zeros(batch_size, self.hidden_dim, device=device)
        if video_x is None: video_x = torch.zeros(batch_size, self.hidden_dim, device=device)

        if text_x.dim() == 2:  text_x  = text_x.unsqueeze(1)
        if audio_x.dim() == 2: audio_x = audio_x.unsqueeze(1)
        if video_x.dim() == 2: video_x = video_x.unsqueeze(1)

        sep_losses = []

        # Text-Acoustic complementation.
        x_t, x_a = text_x, audio_x
        for layer, sep in zip(self.ta_layers, self.ta_separators):
            x_t, x_a, hb_t, hb_a = layer(x_t, x_a)
            loss, _ = sep(hb_t, hb_a)
            sep_losses.append(loss)
        h_ta_text, h_ta_audio = x_t[:, 0, :], x_a[:, 0, :]

        # Text-Visual complementation.
        x_t2, x_v = text_x, video_x
        for layer, sep in zip(self.tv_layers, self.tv_separators):
            x_t2, x_v, hb_t, hb_v = layer(x_t2, x_v)
            loss, _ = sep(hb_t, hb_v)
            sep_losses.append(loss)
        h_tv_text, h_tv_video = x_t2[:, 0, :], x_v[:, 0, :]

        # Concatenate four heads and predict.
        final = torch.cat([h_ta_audio, h_ta_text, h_tv_text, h_tv_video], dim=1)
        output = self.prediction_layer(final)

        if return_losses:
            return output, sep_losses
        return output
