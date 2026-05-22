"""
Proposed fusion architectures
=============================

Two novel fusion modules that explicitly model **cross-modal conflict**,
addressing the *positive-correlation ceiling* identified across every
baseline benchmarked in this work:

    1. :class:`AntiCorrelationGatedFusion` (ACGF, ~788K parameters)
    2. :class:`TinyAntiCorrelator`         (TAC,  ~5K parameters)

Both architectures introduce **signed difference vectors** as a first-class
signal alongside the standard agreement pathway used by LMF, GMU, TFN,
MFB, MulT, etc.

The positive-correlation ceiling
--------------------------------
Existing fusion mechanisms — Hadamard-product methods (LMF, TFN, Tucker),
dot-product attention methods (Cross-Attn, MulT), and gated/additive
methods (GMU, Early, Late) — are all designed to **amplify signal when
modalities agree** and to suppress or average signal when they conflict.

This is problematic for conversational dynamics prediction because
cross-modal conflict is *itself informative*. A syntactically complete
utterance (text → TURN) combined with a rising pitch contour and sustained
gaze (audio + video → KEEP) is the textbook ambiguous case — exactly the
moment where conflict carries the decision signal. Averaging that away is
a strict loss of information.

Signed difference vectors
-------------------------
Given embeddings :math:`z_T, z_A, z_V \\in \\mathbb{R}^{256}`, the proposed
modules compute pairwise signed differences::

    d_TA = z_T - z_A      # who weights this feature more, and by how much
    d_TV = z_T - z_V
    d_AV = z_A - z_V

The **sign** of each coordinate encodes *which* modality is dominant for
that feature dimension; the **magnitude** encodes *how strongly* they
disagree. Hadamard products and dot products discard exactly this
directional information.

Missing-modality behaviour (graceful, not catastrophic)
-------------------------------------------------------
When a modality is absent we replace its embedding with the zero vector.
The difference vectors involving the missing modality then become
:math:`\\pm z_{\\text{other}}` — a maximum-discrepancy signal proportional
to the remaining modalities. This is the opposite of LMF, whose Hadamard
product collapses the entire fused representation to zero whenever any
modality is missing (Mac-F1 → 0.09–0.27, below random chance).
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.init import xavier_normal_


# =============================================================================
#  Anti-Correlation Gated Fusion (ACGF)
# =============================================================================

class AntiCorrelationGatedFusion(nn.Module):
    r"""Anti-Correlation Gated Fusion (~788K parameters).

    Architecture
    ------------
    Given embeddings :math:`z_T, z_A, z_V \in \mathbb{R}^{d}` (d = hidden_dim):

    **Step 1** — Signed difference vectors::

        d_TA = z_T - z_A
        d_TV = z_T - z_V
        d_AV = z_A - z_V

    **Step 2** — Positive (agreement) stream::

        h_pos = ReLU( W_pos · [z_T ; z_A ; z_V] + b_pos )
        W_pos : R^{d x 3d}

    **Step 3** — Negative (conflict) stream::

        h_neg = ReLU( W_neg · [d_TA ; d_TV ; d_AV] + b_neg )
        W_neg : R^{d x 3d}

    **Step 4** — Anti-correlation gate (per-dimension routing)::

        gamma = sigmoid( W_gate · [z_T;z_A;z_V; d_TA;d_TV;d_AV] + b_gate )
        W_gate : R^{d x 6d}

        gamma_i ~ 1  ->  trust agreement (positive stream)
        gamma_i ~ 0  ->  trust conflict   (negative stream)

    **Step 5** — Gated fusion + classification::

        h     = gamma * h_pos + (1 - gamma) * h_neg
        y_hat = W_out * LayerNorm(Dropout(h))

    Parameter budget (d = 256)
    --------------------------
    =====================  ============  ===========
    Component              Shape         Params
    =====================  ============  ===========
    Positive projection    256 x 768     196 608
    Negative projection    256 x 768     196 608
    Gate projection        256 x 1536    393 216
    Output classifier      3 x 256           768
    Biases + LayerNorm     —             ~1 024
    **Total**                            **~788 224 (~0.79 M)**
    =====================  ============  ===========
    """

    def __init__(self, hidden_dim: int = 256, output_dim: int = 3,
                 dropout: float = 0.1):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim

        # Positive stream — summarises agreement from raw embeddings.
        self.W_pos = nn.Linear(hidden_dim * 3, hidden_dim, bias=True)

        # Negative stream — summarises conflict from signed differences.
        self.W_neg = nn.Linear(hidden_dim * 3, hidden_dim, bias=True)

        # Gate — conditions on BOTH raw and difference signals so it can
        # distinguish meaningful conflict (e.g. rising pitch opposing a
        # syntactically complete sentence) from random noise.
        self.W_gate = nn.Linear(hidden_dim * 6, hidden_dim, bias=True)

        # Output head.
        self.layer_norm = nn.LayerNorm(hidden_dim)
        self.dropout = nn.Dropout(dropout)
        self.classifier = nn.Linear(hidden_dim, output_dim, bias=True)

        self._init_weights()

    def _init_weights(self):
        for module in [self.W_pos, self.W_neg, self.W_gate, self.classifier]:
            xavier_normal_(module.weight)
            if module.bias is not None:
                nn.init.zeros_(module.bias)

    def forward(self, text_x, audio_x, video_x):
        """
        Parameters
        ----------
        text_x, audio_x, video_x : torch.Tensor or None
            Shape ``(batch, hidden_dim)`` each. ``None`` indicates the modality
            is absent and is replaced with a zero vector.

        Returns
        -------
        logits : torch.Tensor
            Shape ``(batch, output_dim)``.
        """
        # Determine batch size and device from whichever modality is present.
        ref = text_x if text_x is not None else \
              audio_x if audio_x is not None else video_x
        batch_size, device = ref.shape[0], ref.device

        # Replace missing modalities with zeros. The neutral element of
        # subtraction; difference vectors then encode max discrepancy.
        z_T = text_x  if text_x  is not None else torch.zeros(batch_size, self.hidden_dim, device=device)
        z_A = audio_x if audio_x is not None else torch.zeros(batch_size, self.hidden_dim, device=device)
        z_V = video_x if video_x is not None else torch.zeros(batch_size, self.hidden_dim, device=device)

        # Signed pairwise differences — direction matters: d_ij = -d_ji.
        d_TA = z_T - z_A
        d_TV = z_T - z_V
        d_AV = z_A - z_V

        # Positive stream — captures inter-modal agreement.
        cat_raw = torch.cat([z_T, z_A, z_V], dim=1)
        h_pos = F.relu(self.W_pos(cat_raw))

        # Negative stream — captures inter-modal conflict.
        cat_diff = torch.cat([d_TA, d_TV, d_AV], dim=1)
        h_neg = F.relu(self.W_neg(cat_diff))

        # Anti-correlation gate — per-dimension routing.
        cat_all = torch.cat([cat_raw, cat_diff], dim=1)
        gamma = torch.sigmoid(self.W_gate(cat_all))

        # Fuse the two streams.
        h_fused = gamma * h_pos + (1.0 - gamma) * h_neg

        # Classification head.
        h_fused = self.dropout(self.layer_norm(h_fused))
        return self.classifier(h_fused)


# =============================================================================
#  Tiny Anti-Correlator (TAC)
# =============================================================================

class TinyAntiCorrelator(nn.Module):
    r"""Ultra-lightweight fusion (~5,173 parameters total).

    A radical distillation of the ACGF principle. All learned linear
    projections are replaced with **parameter-free element-wise operations**,
    concentrating learnable parameters in (a) a tiny gating MLP and
    (b) the final classifier.

    Architecture
    ------------
    **Step 1** — Agreement signal (parameter-free Hadamard product)::

        c_pos = z_T ⊙ z_A ⊙ z_V

    Large in dimension *k* when all three modalities agree in sign and
    magnitude on feature *k*.

    **Step 2** — Conflict signal (parameter-free absolute differences)::

        c_neg = |z_T - z_A| + |z_T - z_V|

    Large in dimension *k* when modalities disagree strongly.

    **Step 3** — Global context (parameter-free mean pooling)::

        g = (z_T + z_A + z_V) / 3

    **Step 4** — Tiny gating network with 16-unit bottleneck::

        [omega_pos, omega_neg] = softmax( MLP(g) )    # MLP: 256 -> 16 -> 2

    **Step 5** — Gated mixture and learnable diagonal scaling::

        h = (omega_pos * c_pos + omega_neg * c_neg) ⊙ lambda
        lambda in R^{256}  (initialised to all-ones)

    **Step 6** — Linear classifier to 3 logits.

    Parameter budget (d = 256)
    --------------------------
    =====================  ============  ===========
    Component              Shape         Params
    =====================  ============  ===========
    Gate layer 1 (256→16)  16 x 256+16        4 112
    Gate layer 2 (16→2)    2 x 16+2              34
    Feature importance λ   256                  256
    Classifier (256→3)     3 x 256+3            771
    **Total**                            **5 173**
    =====================  ============  ===========

    That is **~159× smaller than GMU** (821K params) for only 0.84% drop in
    Macro-F1 on MM-F2F.

    Missing modality behaviour
    --------------------------
    If text is absent (``z_T = 0``)::

        c_pos = 0                              (agreement collapses)
        c_neg = |z_A| + |z_V|                  (conflict retains magnitudes)

    The gate then raises ``omega_neg``, automatically shifting TAC into a
    conflict-only mode that **preserves information** from the remaining
    modalities. This is the opposite of LMF, whose Hadamard structure
    collapses the entire representation to zero under the same condition.
    """

    def __init__(self, hidden_dim: int = 256, output_dim: int = 3):
        super().__init__()
        self.hidden_dim = hidden_dim

        # Learnable diagonal scaling — costs 256 params instead of 256x256.
        self.feature_importance = nn.Parameter(torch.ones(1, hidden_dim))

        # Tiny MLP that decides agreement vs. conflict balance.
        self.correlation_gate = nn.Sequential(
            nn.Linear(hidden_dim, 16),
            nn.ReLU(),
            nn.Linear(16, 2),       # [weight_pos, weight_neg]
            nn.Softmax(dim=-1),
        )

        self.classifier = nn.Linear(hidden_dim, output_dim)

    def forward(self, text_x, audio_x, video_x):
        # Zero-fill missing modalities.
        ref = text_x if text_x is not None else \
              audio_x if audio_x is not None else video_x
        z_t = text_x  if text_x  is not None else torch.zeros_like(ref)
        z_a = audio_x if audio_x is not None else torch.zeros_like(ref)
        z_v = video_x if video_x is not None else torch.zeros_like(ref)

        # Parameter-free agreement: triple Hadamard product.
        pos_corr = z_t * z_a * z_v

        # Parameter-free conflict: sum of pairwise absolute differences.
        neg_corr = torch.abs(z_t - z_a) + torch.abs(z_t - z_v)

        # Global context for the tiny gate.
        global_state = (z_t + z_a + z_v) / 3.0
        weights = self.correlation_gate(global_state)        # (B, 2)

        # Gated mixture.
        fused = weights[:, 0:1] * pos_corr + weights[:, 1:2] * neg_corr

        # Learnable per-dimension scaling (diagonal "attention").
        fused = fused * self.feature_importance

        return self.classifier(fused)
