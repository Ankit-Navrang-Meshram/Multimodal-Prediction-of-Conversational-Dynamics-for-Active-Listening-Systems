import os
import json
import argparse
import numpy as np
from tqdm import tqdm
from sklearn.metrics import (
    accuracy_score, f1_score, precision_score, recall_score,
    classification_report, confusion_matrix,
)

import torch
from torch.utils.data import DataLoader
from transformers import AutoTokenizer, AutoFeatureExtractor, AutoImageProcessor

from model import DNCCEnsembleModel
from dataloader import MultiModalDataset, collate_fn

# ── Argument parsing ──────────────────────────────────────────────────────────

parser = argparse.ArgumentParser(
    description="Evaluate a trained DNCCEnsembleModel on the test split."
)

# Model
parser.add_argument("--t_ckpt_path",   type=str, default=None,
                    help="Pre-trained text encoder checkpoint (.pt)")
parser.add_argument("--a_ckpt_path",   type=str, default=None,
                    help="Pre-trained audio encoder checkpoint (.pt)")
parser.add_argument("--v_ckpt_path",   type=str, default=None,
                    help="Pre-trained vision encoder checkpoint (.pt)")
parser.add_argument("--model_weights", type=str, required=True,
                    help="Path to trained DNCC model weights (.pt)")
parser.add_argument("--device",        type=str, default="cuda",
                    help="Device to run inference on (cuda / cpu)")

# Data
parser.add_argument("--data_root",  type=str, default="dataset/",
                    help="Root directory containing train/val/test CSV + media")
parser.add_argument("--batch_size", type=int, default=1,
                    help="Inference batch size")
parser.add_argument("--n_workers",  type=int, default=4,
                    help="DataLoader worker processes")

# Output
parser.add_argument("--output_dir",       type=str, default="results/",
                    help="Directory to write result files")
parser.add_argument("--save_predictions", action="store_true",
                    help="Write per-sample predictions to a JSON file")

# Ablation — drop one or more modalities to assess individual contributions
parser.add_argument("--drop_text",   action="store_true",
                    help="Zero-out text modality (ablation)")
parser.add_argument("--drop_audio",  action="store_true",
                    help="Zero-out audio modality (ablation)")
parser.add_argument("--drop_vision", action="store_true",
                    help="Zero-out vision modality (ablation)")

# Member-level analysis
parser.add_argument("--per_member_metrics", action="store_true",
                    help="Also report per-member accuracy (requires training mode forward)")

args = parser.parse_args()

LABELS = ["keep", "turn", "bc"]


# ── Helpers ───────────────────────────────────────────────────────────────────

def load_processors():
    tokenizer = AutoTokenizer.from_pretrained("openai-community/gpt2")
    tokenizer.pad_token = tokenizer.eos_token
    audio_processor = AutoFeatureExtractor.from_pretrained("facebook/hubert-base-ls960")
    video_processor = AutoImageProcessor.from_pretrained("MCG-NJU/videomae-base")
    return tokenizer, audio_processor, video_processor


def idx2label(idx):
    return LABELS[idx]


def get_ablation_tag(drop_text, drop_audio, drop_vision):
    dropped = []
    if drop_text:   dropped.append("Text")
    if drop_audio:  dropped.append("Audio")
    if drop_vision: dropped.append("Vision")
    return f"Dropped: {', '.join(dropped)}" if dropped else "All modalities"


def cal_metrics(all_labels, all_preds):
    """Return a nested dict of per-class, macro, and weighted metrics."""
    return {
        "accuracy": accuracy_score(all_labels, all_preds),
        "per_class": {
            "precision": precision_score(all_labels, all_preds, average=None,       zero_division=0),
            "recall":    recall_score(   all_labels, all_preds, average=None,       zero_division=0),
            "f1":        f1_score(       all_labels, all_preds, average=None,       zero_division=0),
        },
        "macro": {
            "precision": precision_score(all_labels, all_preds, average="macro",    zero_division=0),
            "recall":    recall_score(   all_labels, all_preds, average="macro",    zero_division=0),
            "f1":        f1_score(       all_labels, all_preds, average="macro",    zero_division=0),
        },
        "weighted": {
            "precision": precision_score(all_labels, all_preds, average="weighted", zero_division=0),
            "recall":    recall_score(   all_labels, all_preds, average="weighted", zero_division=0),
            "f1":        f1_score(       all_labels, all_preds, average="weighted", zero_division=0),
        },
        "confusion_matrix": confusion_matrix(all_labels, all_preds).tolist(),
        "classification_report": classification_report(
            all_labels, all_preds,
            target_names=LABELS, zero_division=0,
        ),
    }


def _metric_block(metrics, ablation_tag="", member_id=None):
    """Return a formatted string block for one set of metrics."""
    lines = []
    w = 60
    header = f"Member G{member_id}" if member_id is not None else "Ensemble"
    lines.append("=" * w)
    lines.append(f"EVALUATION RESULTS  [{header}]")
    if ablation_tag:
        lines.append(f"Ablation : {ablation_tag}")
    lines.append("=" * w)
    lines.append(f"\nOverall Accuracy : {metrics['accuracy']:.4f}")

    lines.append("\n" + "-" * w)
    lines.append("Per-Class Metrics:")
    lines.append("-" * w)
    lines.append(f"{'Class':<10} {'Precision':<12} {'Recall':<12} {'F1-Score':<12}")
    lines.append("-" * w)
    for idx, lbl in enumerate(LABELS):
        lines.append(
            f"{lbl:<10} "
            f"{metrics['per_class']['precision'][idx]:<12.4f} "
            f"{metrics['per_class']['recall'][idx]:<12.4f} "
            f"{metrics['per_class']['f1'][idx]:<12.4f}"
        )

    for avg in ("macro", "weighted"):
        lines.append("\n" + "-" * w)
        lines.append(f"{avg.capitalize()}-Averaged Metrics:")
        lines.append("-" * w)
        lines.append(f"Precision : {metrics[avg]['precision']:.4f}")
        lines.append(f"Recall    : {metrics[avg]['recall']:.4f}")
        lines.append(f"F1-Score  : {metrics[avg]['f1']:.4f}")

    lines.append("\n" + "-" * w)
    lines.append("Confusion Matrix  (rows=true, cols=pred):")
    lines.append("-" * w)
    lines.append(f"{'':>10}" + "".join(f"{l:>10}" for l in LABELS))
    for row_label, row in zip(LABELS, metrics["confusion_matrix"]):
        lines.append(f"{row_label:>10}" + "".join(f"{v:>10}" for v in row))

    lines.append("\n" + "-" * w)
    lines.append("Classification Report:")
    lines.append("-" * w)
    lines.append(metrics["classification_report"])
    lines.append("=" * w)
    return "\n".join(lines)


def print_metrics(metrics, ablation_tag="", member_id=None):
    print(_metric_block(metrics, ablation_tag, member_id))


def save_metrics(metrics, filepath, ablation_tag="", member_id=None):
    os.makedirs(os.path.dirname(filepath) or ".", exist_ok=True)
    with open(filepath, "w") as f:
        f.write(_metric_block(metrics, ablation_tag, member_id))
    print(f"  Metrics saved → {filepath}")


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    os.makedirs(args.output_dir, exist_ok=True)
    ablation_tag = get_ablation_tag(args.drop_text, args.drop_audio, args.drop_vision)
    print(f"\nAblation : {ablation_tag}\n")

    # ── Processors & dataset ──────────────────────────────────────────────
    print("Loading processors …")
    tokenizer, audio_processor, video_processor = load_processors()

    print(f"Loading test dataset from {args.data_root} …")
    test_set = MultiModalDataset(
        data_root=args.data_root, split="test",
        tokenizer=tokenizer,
        audio_processor=audio_processor,
        video_processor=video_processor,
    )
    test_loader = DataLoader(
        test_set, batch_size=args.batch_size, shuffle=False,
        collate_fn=collate_fn, num_workers=args.n_workers,
    )
    print(f"Test set size : {len(test_set)} samples")

    # ── Model ─────────────────────────────────────────────────────────────
    print("Building DNCCEnsembleModel …")
    model = DNCCEnsembleModel(
        text_ckpt=args.t_ckpt_path,
        audio_ckpt=args.a_ckpt_path,
        vision_ckpt=args.v_ckpt_path,
    ).to(args.device)

    print(f"Loading weights from {args.model_weights} …")
    model.load_state_dict(
        torch.load(args.model_weights, map_location=args.device), strict=False
    )
    model.eval()
    print(f"Model ready on {args.device}\n")

    # ── Inference ─────────────────────────────────────────────────────────
    all_labels      = []
    all_ens_preds   = []
    all_predictions = []                         # for --save_predictions
    # per-member collectors: list of 9 lists
    member_preds    = [[] for _ in range(9)]

    with torch.no_grad():
        for i, (text, audio, vision, y) in enumerate(
            tqdm(test_loader, desc="Evaluating")
        ):
            if audio is None:
                print(f"  [WARN] Skipping batch {i} — audio is None")
                continue

            text   = text["input_ids"].to(args.device)
            audio  = audio.to(args.device)
            vision = vision["pixel_values"].to(args.device)
            y      = y.to(args.device)

            # Ablation: zero-out selected modalities
            # Use explicit float32 zeros to avoid dtype issues under autocast
            if args.drop_text:
                text   = torch.zeros(text.shape,   dtype=torch.long,    device=args.device)
            if args.drop_audio:
                audio  = torch.zeros(audio.shape,  dtype=torch.float32, device=args.device)
            if args.drop_vision:
                vision = torch.zeros(vision.shape, dtype=torch.float32, device=args.device)

            # ── Ensemble prediction (eval mode → averaged probs (B, 3)) ──
            pred_probs  = model(text, audio, vision)        # (B, 3)
            pred_labels = pred_probs.argmax(dim=1)          # (B,)

            all_labels.extend(y.cpu().numpy())
            all_ens_preds.extend(pred_labels.cpu().numpy())

            # ── Per-member predictions (requires a second forward in train mode) ──
            if args.per_member_metrics:
                model.train()
                logits_all = model(text, audio, vision)     # (9, B, 3)
                model.eval()
                for m in range(9):
                    m_pred = logits_all[m].argmax(dim=1).cpu().numpy()
                    member_preds[m].extend(m_pred)

            # ── Store per-sample predictions ──────────────────────────────
            if args.save_predictions:
                for j in range(len(y)):
                    all_predictions.append({
                        "true_label":            int(y[j].cpu().numpy()),
                        "true_label_name":       idx2label(int(y[j].cpu().numpy())),
                        "predicted_label":       int(pred_labels[j].cpu().numpy()),
                        "predicted_label_name":  idx2label(int(pred_labels[j].cpu().numpy())),
                        "ensemble_probabilities": pred_probs[j].cpu().numpy().tolist(),
                    })

    all_labels    = np.array(all_labels)
    all_ens_preds = np.array(all_ens_preds)

    # ── Build output filename prefix ──────────────────────────────────────
    dropped_parts = []
    if args.drop_text:   dropped_parts.append("no_text")
    if args.drop_audio:  dropped_parts.append("no_audio")
    if args.drop_vision: dropped_parts.append("no_vision")
    prefix = "DNCC_" + ("_".join(dropped_parts) if dropped_parts else "all_modalities")

    # ── Ensemble metrics ──────────────────────────────────────────────────
    print("\nComputing ensemble metrics …")
    ens_metrics = cal_metrics(all_labels, all_ens_preds)
    print_metrics(ens_metrics, ablation_tag)

    txt_path = os.path.join(args.output_dir, f"{prefix}_results.txt")
    save_metrics(ens_metrics, txt_path, ablation_tag)

    # ── Per-member metrics ────────────────────────────────────────────────
    if args.per_member_metrics:
        print("\nComputing per-member metrics …")
        member_names = [
            "G1_Text", "G2_Audio", "G3_Vision",
            "G4_Text→Audio", "G5_Audio→Text",
            "G6_Text→Vision", "G7_Vision→Text",
            "G8_Audio→Vision", "G9_Vision→Audio",
        ]
        member_summary = {}
        for m in range(9):
            m_preds   = np.array(member_preds[m])
            m_metrics = cal_metrics(all_labels, m_preds)
            print_metrics(m_metrics, ablation_tag, member_id=m + 1)

            m_path = os.path.join(
                args.output_dir, f"{prefix}_member_{m+1}_{member_names[m]}_results.txt"
            )
            save_metrics(m_metrics, m_path, ablation_tag, member_id=m + 1)

            member_summary[member_names[m]] = {
                "accuracy":        round(m_metrics["accuracy"], 4),
                "macro_f1":        round(m_metrics["macro"]["f1"], 4),
                "macro_precision": round(m_metrics["macro"]["precision"], 4),
                "macro_recall":    round(m_metrics["macro"]["recall"], 4),
            }

        # Print summary comparison table
        print("\n" + "=" * 70)
        print("  MEMBER vs ENSEMBLE ACCURACY SUMMARY")
        print("=" * 70)
        print(f"  {'Member':<25} {'Accuracy':>10} {'Macro-F1':>10}")
        print("  " + "-" * 48)
        for name, s in member_summary.items():
            print(f"  {name:<25} {s['accuracy']:>10.4f} {s['macro_f1']:>10.4f}")
        print("  " + "-" * 48)
        print(f"  {'ENSEMBLE':<25} {ens_metrics['accuracy']:>10.4f} "
              f"{ens_metrics['macro']['f1']:>10.4f}")
        print("=" * 70)

        # Save summary as JSON
        summary_path = os.path.join(args.output_dir, f"{prefix}_member_summary.json")
        member_summary["ensemble"] = {
            "accuracy":        round(ens_metrics["accuracy"], 4),
            "macro_f1":        round(ens_metrics["macro"]["f1"], 4),
            "macro_precision": round(ens_metrics["macro"]["precision"], 4),
            "macro_recall":    round(ens_metrics["macro"]["recall"], 4),
        }
        with open(summary_path, "w") as f:
            json.dump(member_summary, f, indent=4)
        print(f"  Member summary saved → {summary_path}")

    # ── Save per-sample predictions ───────────────────────────────────────
    if args.save_predictions:
        json_path = os.path.join(args.output_dir, f"{prefix}_predictions.json")
        with open(json_path, "w") as f:
            json.dump(all_predictions, f, indent=4)
        print(f"  Per-sample predictions saved → {json_path}")

    print("\nEvaluation complete.")


if __name__ == "__main__":
    main()


# ── Usage examples ────────────────────────────────────────────────────────────
"""
# Standard evaluation (all modalities):
python evaluate.py \
    --t_ckpt_path  weights/text_model.pt \
    --a_ckpt_path  weights/audio_model.pt \
    --v_ckpt_path  weights/vision_model.pt \
    --model_weights log/best_model.pt \
    --data_root    dataset/ \
    --output_dir   results/ \
    --save_predictions

# Ablation — text only (drop audio and vision):
python evaluate.py \
    --model_weights log/best_model.pt \
    --data_root     dataset/ \
    --output_dir    results/ \
    --drop_audio --drop_vision

# Per-member breakdown (compares all 9 members against the ensemble):
python evaluate.py \
    --model_weights     log/best_model.pt \
    --data_root         dataset/ \
    --output_dir        results/ \
    --per_member_metrics
"""