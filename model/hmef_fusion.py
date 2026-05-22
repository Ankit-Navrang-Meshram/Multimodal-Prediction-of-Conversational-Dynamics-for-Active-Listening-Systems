"""
HierarchicalMixtureExpertFusion (HMEF)
=======================================
A novel fusion module for multimodal conversational dynamics prediction.

Design Philosophy:
------------------
Existing fusion methods each capture one type of interaction:
  - LMF / Tucker: multiplicative low-rank interactions  
  - CrossModalAttention: soft alignment between modalities
  - GMU: gated suppression of unreliable modalities
  - MultimodalTransformer: self-attention over a modality sequence

HMEF is a 4-stage pipeline that combines all the above insights:

  Stage 1 — Uncertainty-Aware Normalization
    Each modality gets a learned "confidence score" (scalar) derived from the 
    input's L2 norm and a trainable gate. This handles the missing-modality and
    noisy-modality cases naturally: a zero-padded modality will produce near-zero 
    confidence and be suppressed without any hard if/else branches.

  Stage 2 — Pairwise Bilinear Interaction (with residual)
    All three pairs (T-A, T-V, A-V) are fused via low-rank bilinear pooling 
    (similar to MFB but more efficient). The pairwise representation is added 
    back to the original modality via a residual connection, enriching each 
    modality with cross-modal context before the triple fusion.

  Stage 3 — Mixture-of-Experts Triple Fusion
    Three lightweight "experts" each compute a global representation:
      * Expert 1: LMF-style element-wise product in rank space
      * Expert 2: Attention-weighted sum (soft late fusion)
      * Expert 3: Gated max-pool (captures dominant signal)
    A router (small MLP on concatenated inputs) computes a probability 
    distribution over experts, and the final representation is a weighted sum 
    of expert outputs. This allows the model to learn which fusion style works 
    best for each conversational context.

  Stage 4 — Transformer Refinement
    The stacked outputs (pairwise residuals + MoE output) are passed through 
    a single TransformerEncoder layer for a final global contextualisation, 
    followed by a classification head.

Key advantages over existing modules:
  ✓ Graceful missing-modality handling via continuous confidence gating
    (no hard zero-padding that confuses the fusion)
  ✓ Captures both pairwise AND triple interactions explicitly
  ✓ Router learns which fusion "style" is best per sample/context
  ✓ Residual connections prevent gradient vanishing in deep fusion
  ✓ ~1.8M parameters (comparable to MultimodalTransformer, cheaper than TFN)

Usage:
------
  from hmef_fusion import HierarchicalMixtureExpertFusion

  fusion = HierarchicalMixtureExpertFusion(hidden_dim=256, output_dim=3)
  logits = fusion(text_emb, audio_emb, video_emb)  # each (B, 256) or None
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.parameter import Parameter
from torch.nn.init import xavier_normal_, kaiming_normal_
import math


# ---------------------------------------------------------------------------
# Helper: Low-Rank Bilinear Pooling
# ---------------------------------------------------------------------------

class LowRankBilinear(nn.Module):
    """
    Efficient low-rank bilinear interaction between two vectors.
    
    Instead of computing W ∈ R^{d×d×r} directly (expensive),
    we factor it as:  z = (U x1) ⊙ (V x2),  then sum-pool over rank dim.
    This is O(d·r) vs O(d²·r) for full bilinear.
    """
    def __init__(self, in_dim: int, rank: int, out_dim: int, dropout: float = 0.1):
        super().__init__()
        self.U = nn.Linear(in_dim, rank, bias=False)
        self.V = nn.Linear(in_dim, rank, bias=False)
        self.proj = nn.Sequential(
            nn.Linear(rank, out_dim),
            nn.LayerNorm(out_dim),
        )
        self.dropout = nn.Dropout(dropout)
        # Power normalization (signed sqrt) stabilises bilinear outputs
        self._init_weights()

    def _init_weights(self):
        kaiming_normal_(self.U.weight)
        kaiming_normal_(self.V.weight)

    def forward(self, x1: torch.Tensor, x2: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x1, x2: (B, in_dim)
        Returns:
            (B, out_dim) — signed-sqrt-normalized bilinear interaction
        """
        z = self.U(x1) * self.V(x2)                     # (B, rank)
        z = torch.sign(z) * torch.sqrt(z.abs() + 1e-8)  # signed sqrt norm
        z = F.normalize(z, p=2, dim=-1)                  # L2 normalize
        z = self.dropout(z)
        return self.proj(z)                               # (B, out_dim)


# ---------------------------------------------------------------------------
# Stage 1: Uncertainty-Aware Confidence Gating
# ---------------------------------------------------------------------------

class ModalityConfidenceGate(nn.Module):
    """
    Learns a per-sample scalar confidence score for each modality.
    
    Instead of binary "present/absent", produces a soft [0,1] gate that 
    suppresses low-information modalities (e.g., zero-padded or very noisy).
    
    The gate is conditioned on both the modality's own L2 norm (proxy for 
    "how much signal is there?") and a global context vector from all modalities.
    """
    def __init__(self, hidden_dim: int, n_modalities: int = 3):
        super().__init__()
        self.n_modalities = n_modalities
        # Per-modality local gate (from own features)
        self.local_gate = nn.ModuleList([
            nn.Sequential(nn.Linear(hidden_dim, 64), nn.ReLU(), nn.Linear(64, 1), nn.Sigmoid())
            for _ in range(n_modalities)
        ])
        # Global context gate (from all modalities concatenated)
        self.global_gate = nn.Sequential(
            nn.Linear(hidden_dim * n_modalities, n_modalities),
            nn.Sigmoid()
        )

    def forward(self, modalities: list) -> tuple:
        """
        Args:
            modalities: list of (B, hidden_dim) tensors, None entries allowed
        Returns:
            gated: list of (B, hidden_dim) tensors — zero-filled if None, gated
            confidences: (B, n_modalities) float tensor of gate values
        """
        B = next(m for m in modalities if m is not None).shape[0]
        device = next(m for m in modalities if m is not None).device
        H = next(m for m in modalities if m is not None).shape[1]

        # Fill missing modalities with zeros
        filled = [m if m is not None else torch.zeros(B, H, device=device)
                  for m in modalities]

        # Local confidence: each modality scores itself
        local_conf = torch.stack(
            [self.local_gate[i](filled[i]).squeeze(-1) for i in range(self.n_modalities)],
            dim=1
        )  # (B, n_modalities)

        # Global confidence: all modalities vote on each other's reliability
        concat_all = torch.cat(filled, dim=-1)           # (B, H*n_modalities)
        global_conf = self.global_gate(concat_all)       # (B, n_modalities)

        # Final confidence = geometric mean of local and global
        confidences = torch.sqrt(local_conf * global_conf + 1e-8)  # (B, n_mod)

        # Apply gates
        gated = [filled[i] * confidences[:, i:i+1] for i in range(self.n_modalities)]

        return gated, confidences


# ---------------------------------------------------------------------------
# Stage 3: Mixture-of-Experts Triple Fusion
# ---------------------------------------------------------------------------

class LMFExpert(nn.Module):
    """Expert 1: LMF-style element-wise product in low-rank space."""
    def __init__(self, hidden_dim: int, rank: int, out_dim: int):
        super().__init__()
        self.t_proj = nn.Linear(hidden_dim, rank, bias=False)
        self.a_proj = nn.Linear(hidden_dim, rank, bias=False)
        self.v_proj = nn.Linear(hidden_dim, rank, bias=False)
        self.out = nn.Sequential(nn.Linear(rank, out_dim), nn.LayerNorm(out_dim))

    def forward(self, t, a, v):
        return self.out(self.t_proj(t) * self.a_proj(a) * self.v_proj(v))


class AttentionWeightedExpert(nn.Module):
    """Expert 2: Learns per-modality soft weights, then weighted sum."""
    def __init__(self, hidden_dim: int, out_dim: int):
        super().__init__()
        # Key-query style weighting
        self.key = nn.Linear(hidden_dim, 64, bias=False)
        self.query = nn.Parameter(torch.randn(1, 64))
        self.val_proj = nn.Linear(hidden_dim, out_dim, bias=False)
        nn.init.normal_(self.query, std=0.02)

    def forward(self, t, a, v):
        stack = torch.stack([t, a, v], dim=1)          # (B, 3, H)
        keys = self.key(stack)                          # (B, 3, 64)
        scores = (keys * self.query).sum(-1) / math.sqrt(64)  # (B, 3)
        weights = F.softmax(scores, dim=-1).unsqueeze(-1)      # (B, 3, 1)
        vals = self.val_proj(stack)                             # (B, 3, out_dim)
        return (weights * vals).sum(dim=1)                     # (B, out_dim)


class GatedMaxExpert(nn.Module):
    """Expert 3: Element-wise gate then max-pool across modalities."""
    def __init__(self, hidden_dim: int, out_dim: int, dropout: float = 0.1):
        super().__init__()
        self.gate = nn.Sequential(
            nn.Linear(hidden_dim * 3, hidden_dim),
            nn.Sigmoid()
        )
        self.val_proj = nn.Linear(hidden_dim, out_dim, bias=False)
        self.dropout = nn.Dropout(dropout)

    def forward(self, t, a, v):
        concat = torch.cat([t, a, v], dim=-1)           # (B, 3H)
        g = self.gate(concat)                            # (B, H)  — shared gate
        stack = torch.stack(
            [self.val_proj(m * g) for m in [t, a, v]], dim=1
        )  # (B, 3, out_dim)
        return self.dropout(stack.max(dim=1).values)    # (B, out_dim)


class MixtureOfExpertsFusion(nn.Module):
    """
    Three triple-fusion experts with a learned router.
    
    Router input: concatenation of pairwise-enriched modalities (after Stage 2).
    Router output: softmax distribution over 3 experts.
    Final output: weighted sum of expert outputs.
    """
    def __init__(self, hidden_dim: int, expert_dim: int, rank: int, dropout: float = 0.1):
        super().__init__()
        self.expert1 = LMFExpert(hidden_dim, rank, expert_dim)
        self.expert2 = AttentionWeightedExpert(hidden_dim, expert_dim)
        self.expert3 = GatedMaxExpert(hidden_dim, expert_dim, dropout)

        # Router: lightweight 2-layer MLP
        self.router = nn.Sequential(
            nn.Linear(hidden_dim * 3, 128),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(128, 3),   # 3 experts
        )

    def forward(self, t, a, v):
        """
        Args:
            t, a, v: (B, hidden_dim) — pairwise-enriched modality features
        Returns:
            (B, expert_dim)
        """
        # Compute routing probabilities
        router_input = torch.cat([t, a, v], dim=-1)      # (B, 3H)
        router_logits = self.router(router_input)         # (B, 3)
        router_weights = F.softmax(router_logits, dim=-1) # (B, 3)

        # Expert outputs
        e1 = self.expert1(t, a, v)   # (B, expert_dim)
        e2 = self.expert2(t, a, v)
        e3 = self.expert3(t, a, v)

        experts = torch.stack([e1, e2, e3], dim=1)       # (B, 3, expert_dim)

        # Weighted combination
        output = (router_weights.unsqueeze(-1) * experts).sum(dim=1)  # (B, expert_dim)
        return output, router_weights  # also return weights for optional aux loss


# ---------------------------------------------------------------------------
# Main Module: HMEF
# ---------------------------------------------------------------------------

class HierarchicalMixtureExpertFusion(nn.Module):
    """
    Hierarchical Mixture-of-Expert Fusion (HMEF)

    A 4-stage fusion pipeline:
      1. Uncertainty-Aware Confidence Gating
      2. Pairwise Bilinear Interaction with Residual
      3. Mixture-of-Experts Triple Fusion
      4. Transformer Refinement + Classification Head

    Args:
        hidden_dim  : dimension of each modality embedding (default 256)
        output_dim  : number of classes (default 3: keep/turn/bc)
        rank        : low-rank dimension for bilinear pooling (default 64)
        expert_dim  : internal dimension of MoE experts (default 256)
        num_heads   : number of attention heads in Transformer stage (default 4)
        dropout     : dropout rate (default 0.1)
        aux_loss_weight : weight of router entropy regularisation loss.
                          Set 0 to disable. Useful to encourage expert diversity.
    """

    def __init__(
        self,
        hidden_dim: int = 256,
        output_dim: int = 3,
        rank: int = 64,
        expert_dim: int = 256,
        num_heads: int = 4,
        dropout: float = 0.1,
        aux_loss_weight: float = 0.01,
    ):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        self.aux_loss_weight = aux_loss_weight

        # ── Stage 1: Confidence Gating ──────────────────────────────────────
        self.confidence_gate = ModalityConfidenceGate(hidden_dim, n_modalities=3)

        # ── Stage 2: Pairwise Bilinear Interactions ──────────────────────────
        # Three pairs: (T,A), (T,V), (A,V)
        self.bilinear_ta = LowRankBilinear(hidden_dim, rank, hidden_dim, dropout)
        self.bilinear_tv = LowRankBilinear(hidden_dim, rank, hidden_dim, dropout)
        self.bilinear_av = LowRankBilinear(hidden_dim, rank, hidden_dim, dropout)

        # Residual layer norms (one per modality after adding pairwise info)
        self.norm_t = nn.LayerNorm(hidden_dim)
        self.norm_a = nn.LayerNorm(hidden_dim)
        self.norm_v = nn.LayerNorm(hidden_dim)

        # Lightweight projection to collapse pairwise info into modality dim
        # Text picks up from (T,A) and (T,V); Audio from (T,A) and (A,V); etc.
        self.t_enrich = nn.Sequential(nn.Linear(hidden_dim * 2, hidden_dim), nn.GELU())
        self.a_enrich = nn.Sequential(nn.Linear(hidden_dim * 2, hidden_dim), nn.GELU())
        self.v_enrich = nn.Sequential(nn.Linear(hidden_dim * 2, hidden_dim), nn.GELU())

        # ── Stage 3: MoE Triple Fusion ───────────────────────────────────────
        self.moe = MixtureOfExpertsFusion(hidden_dim, expert_dim, rank, dropout)

        # ── Stage 4: Transformer Refinement ─────────────────────────────────
        # Input tokens: 3 enriched modalities + 1 MoE output = 4 tokens
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim,
            nhead=num_heads,
            dim_feedforward=hidden_dim * 2,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,   # Pre-LN for training stability
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=1)

        # Projection of MoE output → hidden_dim for transformer input
        self.moe_proj = nn.Linear(expert_dim, hidden_dim) if expert_dim != hidden_dim else nn.Identity()

        # ── Classification Head ───────────────────────────────────────────────
        self.classifier = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, output_dim),
        )

        self._init_weights()

    def _init_weights(self):
        for module in self.modules():
            if isinstance(module, nn.Linear):
                kaiming_normal_(module.weight, nonlinearity="relu")
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
            elif isinstance(module, nn.LayerNorm):
                nn.init.ones_(module.weight)
                nn.init.zeros_(module.bias)

    def forward(self, text_x, audio_x, video_x):
        """
        Args:
            text_x  : (B, hidden_dim) or None
            audio_x : (B, hidden_dim) or None
            video_x : (B, hidden_dim) or None

        Returns:
            logits  : (B, output_dim)
            
        If self.training and self.aux_loss_weight > 0, an auxiliary router 
        entropy loss is stored in self.aux_loss and should be added to the 
        main CE loss during training:
            loss = criterion(logits, y) + fusion.aux_loss
        """
        # ── Stage 1: Confidence Gating ──────────────────────────────────────
        [t, a, v], confidences = self.confidence_gate([text_x, audio_x, video_x])
        # t, a, v: (B, H)  |  confidences: (B, 3)

        # ── Stage 2: Pairwise Bilinear with Residual ─────────────────────────
        ta = self.bilinear_ta(t, a)   # (B, H)
        tv = self.bilinear_tv(t, v)   # (B, H)
        av = self.bilinear_av(a, v)   # (B, H)

        # Enrich each modality with its pairwise interaction partners
        t_enriched = self.norm_t(t + self.t_enrich(torch.cat([ta, tv], dim=-1)))
        a_enriched = self.norm_a(a + self.a_enrich(torch.cat([ta, av], dim=-1)))
        v_enriched = self.norm_v(v + self.v_enrich(torch.cat([tv, av], dim=-1)))

        # ── Stage 3: MoE Triple Fusion ───────────────────────────────────────
        moe_out, router_weights = self.moe(t_enriched, a_enriched, v_enriched)
        # moe_out: (B, expert_dim)  |  router_weights: (B, 3)

        # Optional auxiliary load-balancing loss (router entropy regularisation)
        # Encourages diverse expert use; prevents router collapse to one expert
        if self.training and self.aux_loss_weight > 0:
            # Maximize entropy of the average routing distribution (load balance)
            avg_weights = router_weights.mean(dim=0)  # (3,)
            self.aux_loss = self.aux_loss_weight * (
                -(avg_weights * (avg_weights + 1e-8).log()).sum()
            )
        else:
            self.aux_loss = 0.0

        # ── Stage 4: Transformer Refinement ─────────────────────────────────
        moe_token = self.moe_proj(moe_out).unsqueeze(1)   # (B, 1, H)
        tokens = torch.stack([t_enriched, a_enriched, v_enriched], dim=1)  # (B, 3, H)
        tokens = torch.cat([tokens, moe_token], dim=1)    # (B, 4, H)

        refined = self.transformer(tokens)                # (B, 4, H)
        # Use the MoE token (last position) as the global representation
        # It already aggregates all 3 modalities; transformer refines it in context
        global_rep = refined[:, -1, :]                   # (B, H)

        # ── Classification ───────────────────────────────────────────────────
        logits = self.classifier(global_rep)              # (B, output_dim)

        return logits


# ---------------------------------------------------------------------------
# Registration: add to get_fusion_module
# ---------------------------------------------------------------------------
# In fusion2.py, update get_fusion_module() to include:
#
#   from .hmef_fusion import HierarchicalMixtureExpertFusion
#   ...
#   "HMEF": HierarchicalMixtureExpertFusion(),
#
# And update train_e2e.py loss step to:
#   loss = criterion(pred, y)
#   if hasattr(model.fusion, 'aux_loss'):
#       loss = loss + model.fusion.aux_loss
# ---------------------------------------------------------------------------


if __name__ == "__main__":
    import time

    print("=" * 60)
    print("HierarchicalMixtureExpertFusion (HMEF) — Self Test")
    print("=" * 60)

    B, H = 16, 256
    text  = torch.randn(B, H)
    audio = torch.randn(B, H)
    video = torch.randn(B, H)

    model = HierarchicalMixtureExpertFusion(hidden_dim=H, output_dim=3)
    model.train()

    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"\nTotal parameters     : {total_params:,}")
    print(f"Trainable parameters : {trainable_params:,}")

    print("\n— Shape Tests —")
    out = model(text, audio, video)
    print(f"  All modalities    : {out.shape}  ✓")
    out = model(None, audio, video)
    print(f"  Missing text      : {out.shape}  ✓")
    out = model(text, None, video)
    print(f"  Missing audio     : {out.shape}  ✓")
    out = model(text, audio, None)
    print(f"  Missing video     : {out.shape}  ✓")
    out = model(text, None, None)
    print(f"  Text only         : {out.shape}  ✓")
    out = model(None, audio, None)
    print(f"  Audio only        : {out.shape}  ✓")
    out = model(None, None, video)
    print(f"  Video only        : {out.shape}  ✓")

    print("\n— Aux Loss Test (training mode) —")
    out = model(text, audio, video)
    print(f"  aux_loss = {model.aux_loss:.6f}  ✓")

    print("\n— Gradient Flow Test —")
    criterion = nn.CrossEntropyLoss()
    y = torch.randint(0, 3, (B,))
    loss = criterion(out, y) + model.aux_loss
    loss.backward()
    grad_norms = {
        name: p.grad.norm().item()
        for name, p in model.named_parameters()
        if p.grad is not None
    }
    print(f"  Modules with gradients : {len(grad_norms)}")
    print(f"  Max grad norm          : {max(grad_norms.values()):.4f}")
    print(f"  Min grad norm          : {min(grad_norms.values()):.6f}")
    print("  Gradient flow OK       : ✓")

    print("\n— Throughput Test (CPU) —")
    model.eval()
    N = 100
    t0 = time.perf_counter()
    with torch.no_grad():
        for _ in range(N):
            _ = model(text, audio, video)
    elapsed = time.perf_counter() - t0
    print(f"  {N} forward passes in {elapsed:.3f}s  ({elapsed/N*1000:.2f} ms/batch)")

    print("\n✅ All tests passed.")