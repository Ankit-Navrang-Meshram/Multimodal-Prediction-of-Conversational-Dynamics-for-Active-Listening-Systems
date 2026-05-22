"""
Evaluate a trained tri-modal model on the test split (all modalities present).

Prints per-class, macro-averaged and weighted-averaged precision / recall /
F1, plus overall accuracy. Saves the same to a text file under ``--output_dir``.

For missing-modality ablations, use :file:`scripts/evaluate_ablation.py`.

Example
-------
::

    python -m scripts.evaluate \\
        --model_weights log/<run>/epoch_9.pt \\
        --fusion_module ACGF \\
        --data_root dataset/
"""

from __future__ import annotations

import argparse
import json
import os
import sys

import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.encoders import load_processors
from src.dataloader import MultiModalDataset, collate_fn
from src.model import LanguageAudioVisionModel
from src.fusion import FUSION_REGISTRY
from src.utils import compute_metrics, format_metrics, idx2label


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--t_ckpt_path", default=None, help="(Optional) Stage-1 text ckpt for encoder init")
    p.add_argument("--a_ckpt_path", default=None, help="(Optional) Stage-1 audio ckpt")
    p.add_argument("--v_ckpt_path", default=None, help="(Optional) Stage-1 video ckpt")
    p.add_argument("--model_weights", required=True,
                   help="Stage-2 .pt checkpoint to load (overrides encoder ckpts)")
    p.add_argument("--fusion_module", required=True, choices=sorted(FUSION_REGISTRY.keys()))
    p.add_argument("--device", default="cuda")
    p.add_argument("--data_root", default="dataset/")
    p.add_argument("--batch_size", type=int, default=1)
    p.add_argument("--n_workers", type=int, default=4)
    p.add_argument("--output_dir", default="results/")
    p.add_argument("--save_predictions", action="store_true",
                   help="Dump per-sample predictions as JSON")
    return p.parse_args()


def main():
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)

    print("[eval] Loading processors")
    tokenizer, _, audio_proc, video_proc = load_processors()

    print(f"[eval] Loading test dataset from {args.data_root}")
    test_set = MultiModalDataset(
        data_root=args.data_root, split="test",
        tokenizer=tokenizer, audio_processor=audio_proc, video_processor=video_proc,
    )
    print(f"[eval] {len(test_set)} test samples")

    loader = DataLoader(test_set, batch_size=args.batch_size, shuffle=False,
                        collate_fn=collate_fn, num_workers=args.n_workers)

    print(f"[eval] Building model with fusion='{args.fusion_module}'")
    model = LanguageAudioVisionModel(
        text_ckpt_path=args.t_ckpt_path,
        audio_ckpt_path=args.a_ckpt_path,
        vision_ckpt_path=args.v_ckpt_path,
        fusion_module=args.fusion_module,
    ).to(args.device)
    model.load_state_dict(torch.load(args.model_weights, map_location=args.device))
    model.eval()
    print(f"[eval] Loaded weights from {args.model_weights}")

    labels_all, preds_all, predictions_log = [], [], []
    with torch.no_grad():
        for batch in tqdm(loader, desc="test"):
            text, audio, vision, labels = batch
            if audio is None:
                continue
            text   = text["input_ids"].to(args.device)
            audio  = audio.to(args.device)
            vision = vision["pixel_values"].to(args.device)
            labels = labels.to(args.device)

            logits = model(text, audio, vision)
            preds = logits.argmax(dim=1)

            labels_all.extend(labels.cpu().numpy())
            preds_all.extend(preds.cpu().numpy())

            if args.save_predictions:
                for j in range(len(labels)):
                    predictions_log.append({
                        "true_label":      int(labels[j].cpu().numpy()),
                        "true_label_name": idx2label(int(labels[j].cpu().numpy())),
                        "predicted_label": int(preds[j].cpu().numpy()),
                        "predicted_label_name": idx2label(int(preds[j].cpu().numpy())),
                        "probabilities":   logits[j].cpu().numpy().tolist(),
                    })

    metrics = compute_metrics(labels_all, preds_all)
    formatted = format_metrics(metrics)
    print(formatted)

    out_txt = os.path.join(args.output_dir, f"{args.fusion_module}_results.txt")
    with open(out_txt, "w") as f:
        f.write(formatted + "\n")
    print(f"[eval] Wrote {out_txt}")

    if args.save_predictions:
        out_json = os.path.join(args.output_dir, f"{args.fusion_module}_predictions.json")
        with open(out_json, "w") as f:
            json.dump(predictions_log, f, indent=2)
        print(f"[eval] Wrote {out_json}")


if __name__ == "__main__":
    main()
