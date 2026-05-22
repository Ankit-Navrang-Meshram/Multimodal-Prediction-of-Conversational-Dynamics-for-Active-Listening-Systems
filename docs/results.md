# Full Benchmark Results

All experiments run on the MM-F2F test split (Lin et al., 2025). Macro-F1
is the primary metric — it weights all three classes (KEEP/TURN/BC)
equally regardless of frequency, penalising models that achieve high
accuracy by ignoring the minority BC class.

Raw per-method text dumps live in `results/tri_modal/` (full tri-modal
inference) and `results/ablation/` (modality-dropout conditions). The
consolidated spreadsheet is `results/all_results.xlsx`. All ten figures
are in `figures/`.

---

## Table 1 — Tri-modal benchmark

Ranked by Macro-F1. **Bold** = best in column. † = proposed.

| Rank | Method                | Fusion Params | Accuracy | **Mac-F1** | Wt-F1  | KEEP-F1 | TURN-F1 | BC-F1  |
| ---- | --------------------- | ------------: | -------: | ---------: | -----: | ------: | ------: | -----: |
| 1    | **GMU**               |       821,251 | **0.8445** | **0.8607** | **0.8441** | 0.8145  | 0.8390  | **0.9286** |
| 2    | MFB                   |     1,976,067 |   0.8434 |     0.8588 | 0.8430 | 0.8139  | **0.8390** | 0.9235 |
| 3    | Early Fusion          |       526,339 |   0.8421 |     0.8580 | 0.8419 | **0.8156** | 0.8348  | 0.9234 |
| 4    | Cross-Attn            |       988,675 |   0.8426 |     0.8577 | 0.8418 | 0.8104  | 0.8395  | 0.9233 |
| 5    | **ACGF†**             |       788,483 |   0.8409 |     0.8573 | 0.8411 | 0.8155  | 0.8332  | 0.9231 |
| 6    | BBFN                  |    11,579,911 |   0.8407 |     0.8565 | 0.8399 | 0.8065  | 0.8380  | 0.9249 |
| 7    | LMF                   |        37,027 |   0.8404 |     0.8558 | 0.8402 | 0.8134  | 0.8346  | 0.9193 |
| 8    | Late Fusion           |        99,852 |   0.8397 |     0.8558 | 0.8397 | 0.8141  | 0.8321  | 0.9211 |
| 9    | MulT                  |     1,778,435 |   0.8364 |     0.8529 | 0.8364 | 0.8132  | 0.8259  | 0.9195 |
| 10   | **TAC†**              |     **5,173** |   0.8378 |     0.8523 | 0.8379 | 0.8135  | 0.8324  | 0.9111 |
| 11   | Quaternion Fusion     |       395,011 |   0.8364 |     0.8482 | 0.8355 | 0.8074  | 0.8362  | 0.9010 |

**Headline observations**

- The top 10 methods cluster within **0.0085** Macro-F1 of each other
  despite covering four very different architectural families. This
  tight clustering is the empirical signature of the
  *positive-correlation ceiling*.
- **TAC achieves competitive Macro-F1 with 159× fewer parameters than
  GMU** — a 5K-parameter fusion module within 0.84% of the best baseline.
- BC-F1 is the most discriminating per-class metric (range 0.0276 vs.
  0.0091 for KEEP-F1). Methods that integrate audio more flexibly
  achieve higher BC-F1.

---

## Table 2 — Uni-modal (Stage 1) baselines

| Modality | Backbone  | Mac-F1   | KEEP-F1 | TURN-F1 | BC-F1   |
| -------- | --------- | -------: | ------: | ------: | ------: |
| Text     | GPT-2     | **0.751** | **0.767** | 0.707   | 0.740   |
| Audio    | HuBERT    | **0.751** | 0.735   | **0.805** | **0.759** |
| Video    | VideoMAE  | 0.559    | 0.536   | 0.513   | 0.549   |

Text and audio are roughly tied on Macro-F1 but lead on different
classes; video is a meaningful but weaker supporting modality at the
sentence level.

---

## Table 3 — Parameter efficiency

| Method            | Type            | Fusion Params | Mac-F1 | × fewer than GMU |
| ----------------- | --------------- | ------------: | -----: | ---------------: |
| **TAC†**          | Anti-correl.    |     **5,173** | 0.8523 |       **158.8×** |
| LMF               | Tensor          |        37,027 | 0.8558 |            22.2× |
| Tucker            | Tensor          |        99,852 | —      |             8.2× |
| Late Fusion       | Decision-level  |       100,352 | 0.8558 |             8.2× |
| Quaternion        | Algebraic       |       395,011 | 0.8482 |             2.1× |
| Early Fusion      | Feature-level   |       526,339 | 0.8580 |             1.6× |
| TFN               | Tensor          |       526,339 | —      |             1.6× |
| **ACGF†**         | Anti-correl.    |       788,483 | 0.8573 |             1.0× |
| GMU (baseline)    | Gating          |       821,251 | **0.8607** | —              |
| Cross-Attn        | Attention       |       988,675 | 0.8577 |             0.8× |
| MulT              | Attention       |     1,778,435 | 0.8529 |             0.46×|
| MFB               | Bilinear        |     1,976,067 | 0.8588 |             0.42×|
| BBFN              | Gating          |    11,579,911 | 0.8565 |             0.07×|

See `figures/plot7_size_vs_accuracy.png` — no positive correlation
between fusion parameter count and accuracy when encoders are frozen.

---

## Table 4 — Robustness under missing modalities

Macro-F1 per dropout condition. **Red** = below random chance (0.33).
↓ shows absolute drop from the full tri-modal baseline.

|                                | GMU            | LMF                 | ACGF†          | TAC†           |
| ------------------------------ | --------------: | ------------------: | --------------: | --------------: |
| Full tri-modal                 | 0.8607          | 0.8558              | 0.8573          | 0.8523          |
| Text + Audio (−V)              | 0.8474 (↓0.013) | 0.8427 (↓0.013)     | 0.8428 (↓0.015) | 0.8414 (↓0.011) |
| Audio + Video (−T)             | 0.7943 (↓0.066) | 0.7985 (↓0.057)     | 0.8003 (↓0.057) | 0.7835 (↓0.069) |
| Text + Video (−A)              | 0.7710 (↓0.090) | 0.7733 (↓0.082)     | 0.7654 (↓0.092) | 0.7475 (↓0.105) |
| **Text only** (−A−V)           | 0.7249 (↓0.136) | **🔴 0.0909** (↓0.765) | 0.7191 (↓0.138) | 0.6904 (↓0.162) |
| **Audio only** (−T−V)          | 0.7810 (↓0.080) | **🔴 0.1770** (↓0.679) | 0.7846 (↓0.073) | 0.7606 (↓0.092) |
| **Video only** (−T−A)          | 0.4668 (↓0.394) | **🔴 0.2660** (↓0.590) | 0.4518 (↓0.405) | 0.4417 (↓0.411) |

**Headline observations**

- **LMF collapses below random chance under any uni-modal condition.**
  The Hadamard product structure means that when two of three modalities
  are zero, the fused representation collapses to ~0 and the classifier
  produces effectively random predictions. *LMF should not be deployed
  where sensor dropout is possible.*
- Both ACGF and TAC degrade gracefully — under the audio-only condition
  ACGF (**0.7846**) even marginally outperforms GMU (0.7810), suggesting
  the explicit conflict stream provides a useful inductive bias when only
  one modality is available.
- Modality importance ranks **Audio > Text > Video** at the sentence
  level. Dropping audio causes the largest Macro-F1 drop for GMU
  (−0.090); dropping video is nearly negligible (−0.013).

---

## Figures

All ten figures (PNG) are in `figures/`. The most illustrative:

| Figure                                  | Story                                              |
| --------------------------------------- | -------------------------------------------------- |
| `plot1_accuracy_comparison.png`         | Bar chart of accuracy per model (tri-modal)        |
| `plot6_ablation_macro_f1_lines.png`     | Macro-F1 trajectories across ablation conditions   |
| `plot7_size_vs_accuracy.png`            | Log-scale params vs. accuracy — **no correlation** |
| `plot9_accuracy_distribution.png`       | Histogram showing the LMF collapse tail            |
| `plot10_ablation_vs_full_delta.png`     | Per-model drop from full tri-modal per condition   |
