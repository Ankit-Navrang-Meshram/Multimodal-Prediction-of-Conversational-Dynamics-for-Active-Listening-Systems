"""
Generate all plots used in the thesis / presentation.

Reads ``results/all_results.xlsx`` (the consolidated benchmark spreadsheet)
and writes ten PNG figures to the current directory::

    plot1_accuracy_comparison.png       — bar chart of accuracy per model
    plot2_grouped_metrics.png           — accuracy/macro-F1/weighted-F1
    plot3_perclass_f1_heatmap.png       — per-class F1 heatmap
    plot4_precision_recall_scatter.png  — P-R scatter per class
    plot5_ablation_accuracy_drop.png    — drop in accuracy under modality dropout
    plot6_ablation_macro_f1_lines.png   — Mac-F1 line chart across ablation conditions
    plot7_size_vs_accuracy.png          — log-scale params vs. accuracy
    plot8_radar_top3.png                — radar chart of top-3 models
    plot9_accuracy_distribution.png     — histogram of accuracy across runs
    plot10_ablation_vs_full_delta.png   — delta from full tri-modal per condition

Run from the repo root::

    python -m scripts.make_plots
"""

import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np
import warnings
warnings.filterwarnings("ignore")

# ── Load data ──────────────────────────────────────────────────────────────────
df = pd.read_excel("results/all_results.xlsx")

# Separate full-modality rows (no ablation) vs ablation rows
df_full = df[df["ablation"].isna()].copy()
df_abl  = df[df["ablation"].notna()].copy()

PALETTE = plt.cm.tab20.colors
MODEL_COLORS = {m: PALETTE[i] for i, m in enumerate(df["model"].unique())}

# ── Helper ─────────────────────────────────────────────────────────────────────
def savefig(name):
    plt.tight_layout()
    plt.savefig(name, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved: {name}")


# ══════════════════════════════════════════════════════════════════════════════
# Plot 1 – Accuracy comparison across models (full modality)
# ══════════════════════════════════════════════════════════════════════════════
fig, ax = plt.subplots(figsize=(12, 5))
df_full_sorted = df_full.sort_values("accuracy", ascending=False)
bars = ax.bar(df_full_sorted["model"], df_full_sorted["accuracy"],
              color=[MODEL_COLORS[m] for m in df_full_sorted["model"]], edgecolor="white")
ax.bar_label(bars, fmt="%.4f", padding=3, fontsize=8)
ax.set_ylim(0, 1.0)
ax.set_xlabel("Model")
ax.set_ylabel("Accuracy")
ax.set_title("Model Accuracy Comparison (Full Modality)")
ax.axhline(df_full["accuracy"].mean(), color="red", linestyle="--", linewidth=1.2, label="Mean")
ax.legend()
savefig("plot1_accuracy_comparison.png")


# ══════════════════════════════════════════════════════════════════════════════
# Plot 2 – Macro F1 / Weighted F1 / Accuracy grouped bar
# ══════════════════════════════════════════════════════════════════════════════
metrics = ["accuracy", "macro_f1", "weighted_f1"]
x = np.arange(len(df_full_sorted))
width = 0.25

fig, ax = plt.subplots(figsize=(14, 5))
for i, (metric, color) in enumerate(zip(metrics, ["steelblue", "darkorange", "seagreen"])):
    ax.bar(x + i * width, df_full_sorted[metric], width, label=metric.replace("_", " ").title(),
           color=color, alpha=0.85)
ax.set_xticks(x + width)
ax.set_xticklabels(df_full_sorted["model"], rotation=20, ha="right")
ax.set_ylim(0, 1.1)
ax.set_ylabel("Score")
ax.set_title("Accuracy, Macro F1, Weighted F1 per Model (Full Modality)")
ax.legend()
savefig("plot2_grouped_metrics.png")


# ══════════════════════════════════════════════════════════════════════════════
# Plot 3 – Per-class F1 heatmap (full modality)
# ══════════════════════════════════════════════════════════════════════════════
classes = ["keep_f1", "turn_f1", "bc_f1"]
heat_df = df_full.set_index("model")[classes].rename(
    columns={"keep_f1": "Keep F1", "turn_f1": "Turn F1", "bc_f1": "BC F1"})

fig, ax = plt.subplots(figsize=(7, 6))
im = ax.imshow(heat_df.values, aspect="auto", cmap="YlOrRd", vmin=0, vmax=1)
ax.set_xticks(range(len(heat_df.columns)))
ax.set_xticklabels(heat_df.columns)
ax.set_yticks(range(len(heat_df.index)))
ax.set_yticklabels(heat_df.index)
for r in range(heat_df.shape[0]):
    for c in range(heat_df.shape[1]):
        ax.text(c, r, f"{heat_df.values[r, c]:.3f}", ha="center", va="center", fontsize=9)
plt.colorbar(im, ax=ax, label="F1 Score")
ax.set_title("Per-Class F1 Heatmap (Full Modality)")
savefig("plot3_perclass_f1_heatmap.png")


# ══════════════════════════════════════════════════════════════════════════════
# Plot 4 – Precision vs Recall scatter (macro) – full modality
# ══════════════════════════════════════════════════════════════════════════════
fig, ax = plt.subplots(figsize=(8, 6))
for _, row in df_full.iterrows():
    ax.scatter(row["macro_recall"], row["macro_precision"],
               color=MODEL_COLORS[row["model"]], s=120, zorder=3)
    ax.annotate(row["model"], (row["macro_recall"], row["macro_precision"]),
                textcoords="offset points", xytext=(6, 4), fontsize=8)
ax.plot([0, 1], [0, 1], "k--", linewidth=0.8, alpha=0.4, label="Precision = Recall")
ax.set_xlim(0.5, 1.0); ax.set_ylim(0.5, 1.0)
ax.set_xlabel("Macro Recall"); ax.set_ylabel("Macro Precision")
ax.set_title("Macro Precision vs Recall (Full Modality)")
ax.legend(); ax.grid(True, alpha=0.3)
savefig("plot4_precision_recall_scatter.png")


# ══════════════════════════════════════════════════════════════════════════════
# Plot 5 – Ablation impact on accuracy (drop per model, per modality dropped)
# ══════════════════════════════════════════════════════════════════════════════
# Only single-modality drops
single_drops = df_abl[df_abl["ablation"].str.count(",") == 0].copy()
single_drops["dropped"] = single_drops["ablation"].str.replace("Dropped modalities: ", "")

# Merge with full to get baseline accuracy
merged = single_drops.merge(df_full[["model", "accuracy"]], on="model", suffixes=("_abl", "_full"))
merged["accuracy_drop"] = merged["accuracy_full"] - merged["accuracy_abl"]

fig, ax = plt.subplots(figsize=(12, 5))
pivot = merged.pivot(index="model", columns="dropped", values="accuracy_drop")
pivot.plot(kind="bar", ax=ax, colormap="Set2", edgecolor="white", width=0.7)
ax.axhline(0, color="black", linewidth=0.8)
ax.set_ylabel("Accuracy Drop (Full − Ablation)")
ax.set_xlabel("Model")
ax.set_title("Accuracy Drop When Single Modality is Removed")
ax.legend(title="Dropped Modality")
ax.tick_params(axis="x", rotation=25)
savefig("plot5_ablation_accuracy_drop.png")


# ══════════════════════════════════════════════════════════════════════════════
# Plot 6 – Macro F1 across all ablation conditions (line per model)
# ══════════════════════════════════════════════════════════════════════════════
all_abl_labels = df_abl["ablation"].unique()
fig, ax = plt.subplots(figsize=(14, 6))
for model, grp in df_abl.groupby("model"):
    grp = grp.set_index("ablation").reindex(all_abl_labels)
    ax.plot(all_abl_labels, grp["macro_f1"], marker="o", label=model,
            color=MODEL_COLORS[model], linewidth=1.5)
ax.set_xticks(range(len(all_abl_labels)))
ax.set_xticklabels(all_abl_labels, rotation=30, ha="right", fontsize=8)
ax.set_ylabel("Macro F1")
ax.set_title("Macro F1 Across Ablation Conditions (per Model)")
ax.legend(bbox_to_anchor=(1.01, 1), loc="upper left", fontsize=8)
ax.grid(True, alpha=0.3)
savefig("plot6_ablation_macro_f1_lines.png")


# ══════════════════════════════════════════════════════════════════════════════
# Plot 7 – Model size (parameters) vs Accuracy bubble chart
# ══════════════════════════════════════════════════════════════════════════════
fig, ax = plt.subplots(figsize=(10, 6))
for _, row in df_full.iterrows():
    ax.scatter(row["No. of Parameters"], row["accuracy"],
               s=200, color=MODEL_COLORS[row["model"]], zorder=3, alpha=0.85)
    ax.annotate(row["model"], (row["No. of Parameters"], row["accuracy"]),
                textcoords="offset points", xytext=(6, 4), fontsize=8)
ax.set_xscale("log")
ax.xaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"{x:,.0f}"))
ax.set_xlabel("No. of Parameters (log scale)")
ax.set_ylabel("Accuracy")
ax.set_title("Model Size vs Accuracy (Full Modality)")
ax.grid(True, which="both", alpha=0.25)
savefig("plot7_size_vs_accuracy.png")


# ══════════════════════════════════════════════════════════════════════════════
# Plot 8 – Radar chart – top-3 models (macro metrics)
# ══════════════════════════════════════════════════════════════════════════════
radar_metrics = ["accuracy", "macro_precision", "macro_recall", "macro_f1",
                 "keep_f1", "turn_f1", "bc_f1"]
top3 = df_full.nlargest(3, "accuracy")["model"].tolist()
angles = np.linspace(0, 2 * np.pi, len(radar_metrics), endpoint=False).tolist()
angles += angles[:1]

fig, ax = plt.subplots(figsize=(7, 7), subplot_kw=dict(polar=True))
for model in top3:
    vals = df_full[df_full["model"] == model][radar_metrics].values.flatten().tolist()
    vals += vals[:1]
    ax.plot(angles, vals, "o-", linewidth=1.8, label=model, color=MODEL_COLORS[model])
    ax.fill(angles, vals, alpha=0.08, color=MODEL_COLORS[model])
ax.set_xticks(angles[:-1])
ax.set_xticklabels([m.replace("_", "\n") for m in radar_metrics], fontsize=9)
ax.set_ylim(0, 1)
ax.set_title("Radar Chart – Top 3 Models (Full Modality)", pad=20)
ax.legend(loc="upper right", bbox_to_anchor=(1.3, 1.1))
savefig("plot8_radar_top3.png")


# ══════════════════════════════════════════════════════════════════════════════
# Plot 9 – Distribution of accuracy across all runs (histogram + KDE)
# ══════════════════════════════════════════════════════════════════════════════
fig, ax = plt.subplots(figsize=(9, 5))
ax.hist(df["accuracy"], bins=20, color="steelblue", edgecolor="white", alpha=0.7, density=True)
df["accuracy"].plot.kde(ax=ax, color="darkblue", linewidth=2)
ax.axvline(df["accuracy"].mean(), color="red", linestyle="--", label=f"Mean={df['accuracy'].mean():.3f}")
ax.axvline(df["accuracy"].median(), color="green", linestyle="--", label=f"Median={df['accuracy'].median():.3f}")
ax.set_xlabel("Accuracy"); ax.set_ylabel("Density")
ax.set_title("Distribution of Accuracy Across All Runs (Full + Ablation)")
ax.legend(); ax.grid(True, alpha=0.3)
savefig("plot9_accuracy_distribution.png")


# ══════════════════════════════════════════════════════════════════════════════
# Plot 10 – Best ablation vs Full modality: Macro F1 delta bar
# ══════════════════════════════════════════════════════════════════════════════
best_abl = df_abl.groupby("model")["macro_f1"].max().reset_index().rename(
    columns={"macro_f1": "best_abl_f1"})
compare = df_full[["model", "macro_f1"]].merge(best_abl, on="model")
compare["delta"] = compare["best_abl_f1"] - compare["macro_f1"]
compare = compare.sort_values("delta")

fig, ax = plt.subplots(figsize=(10, 5))
colors = ["tomato" if d < 0 else "seagreen" for d in compare["delta"]]
bars = ax.barh(compare["model"], compare["delta"], color=colors, edgecolor="white")
ax.axvline(0, color="black", linewidth=0.8)
ax.bar_label(bars, fmt="%.4f", padding=3, fontsize=8)
ax.set_xlabel("Best Ablation Macro F1 − Full Modality Macro F1")
ax.set_title("Gain / Loss in Macro F1: Best Ablation vs Full Modality")
ax.grid(True, axis="x", alpha=0.3)
savefig("plot10_ablation_vs_full_delta.png")

print("\nAll 10 plots saved successfully.")