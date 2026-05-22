# Architecture Walkthrough

This document gives a brief, code-oriented tour of the model. For the full
theoretical treatment see the thesis (`docs/thesis.pdf`).

---

## Two-stage training

```
Stage 1 — uni-modal encoder training (scripts/train_unimodal.py)
    Text  encoder + temp 3-class head  →  Adam, lr=1e-5, ~10 epochs
    Audio encoder + temp 3-class head  →  Adam, lr=1e-5, ~10 epochs
    Video encoder + temp 3-class head  →  Adam, lr=1e-5, ~10 epochs
    ⇒ Save 3 checkpoints. Discard the temporary heads.

Stage 2 — fusion training         (scripts/train_fusion.py)
    Freeze all 3 encoders. Train ONLY the selected fusion module.
    Cross-entropy loss (+ BBFN auxiliary loss if applicable).
    5% random modality dropout regularisation.
```

This separation is deliberate: by holding the encoders fixed across all
fusion experiments, any performance difference between fusion modules is
attributable to the fusion strategy itself rather than to representation
quality.

---

## Encoders (`src/encoders.py`)

All three encoders share the same shape — a pretrained transformer
backbone, an optional pooling step, and a 768 → 256 linear projection:

| Modality | Backbone (HuggingFace ID)          | Pooling             |
| -------- | ---------------------------------- | ------------------- |
| Text     | `openai-community/gpt2`            | Last token          |
| Audio    | `facebook/hubert-base-ls960`       | Adaptive mean pool  |
| Video    | `MCG-NJU/videomae-base`            | Adaptive mean pool  |

Each encoder has a `return_embeddings` flag: `True` for Stage 2 (returns
the 256-d projection), `False` for Stage 1 (attaches a 256 → 3 head).

---

## Fusion package (`src/fusion/`)

```
src/fusion/
├── __init__.py        — FUSION_REGISTRY + get_fusion_module(name)
├── baselines.py       — LMF, EarlyFusion, LateFusion, TFN, MFB,
│                        CrossModalAttention, GMU, MulT, TuckerFusion
├── bbfn.py            — Bi-Bimodal Fusion Network (Han et al., 2021)
├── quaternion.py      — Hamilton-product fusion (algebraic baseline)
└── proposed.py        — ACGF (~788K) and TAC (~5K)  ← novel contributions
```

All fusion modules expose the same interface:

```python
logits = fusion_module(text_x, audio_x, video_x)
# text_x, audio_x, video_x: (batch, 256) or None
# logits:                   (batch, 3)
```

Missing modalities (`None`) are replaced with zero vectors at the call
site. The two **proposed** modules turn that zero into a useful signal —
the signed difference vector `z_other - 0 = z_other` becomes a
maximum-discrepancy marker that the conflict pathway can exploit.

---

## End-to-end model (`src/model.py`)

`LanguageAudioVisionModel` wires the three frozen encoders to the chosen
fusion module:

```python
from src.model import LanguageAudioVisionModel

model = LanguageAudioVisionModel(
    text_ckpt_path  = "log/text_epoch_9.pt",
    audio_ckpt_path = "log/audio_epoch_9.pt",
    vision_ckpt_path= "log/video_epoch_9.pt",
    fusion_module   = "ACGF",   # any key from FUSION_REGISTRY
)
model.freeze_encoders()         # disables grads on the three encoders
```

The forward pass accepts `None` for any modality, propagating it through
to the fusion module — which is how the missing-modality ablations work.

---

## Key idea: signed difference vectors

The thesis identifies a *positive-correlation ceiling* shared by every
prior fusion mechanism: they all amplify modality agreement and discard
modality conflict. The two proposed modules introduce signed differences
as a first-class feature:

```
d_TA = z_T - z_A
d_TV = z_T - z_V
d_AV = z_A - z_V
```

* **ACGF** (`AntiCorrelationGatedFusion`) — two MLP streams (agreement
  from raw embeddings, conflict from differences) merged by a learned
  sigmoid gate. ~788K parameters.
* **TAC** (`TinyAntiCorrelator`) — replaces both MLP streams with
  parameter-free element-wise operations and concentrates learnables in a
  16-unit gate. 5,173 parameters total, within 0.84% of GMU's Macro-F1.

Both modules degrade gracefully under modality dropout: an absent
modality becomes a maximum-discrepancy signal that keeps the conflict
pathway informative, rather than zeroing the entire representation as
LMF does.
