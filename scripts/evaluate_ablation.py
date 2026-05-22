"""
Missing-modality ablation evaluation.

Evaluates a trained model under sensor-dropout conditions by zeroing out
one or two modalities at inference time. The six ablation conditions used
in the thesis are::

    bi-modal:   text+audio (-V) ;  audio+video (-T) ;  text+video (-A)
    uni-modal:  text only   (-A-V) ; audio only (-T-V) ; video only (-T-A)

Run this script once per dropout condition (and once with no flags for the
tri-modal baseline). Results land in ``--output_dir`` as text files like::

    <fusion_module>_no_audio_results.txt
    <fusion_module>_no_text_no_vision_results.txt
    <fusion_module>_all_modalities_results.txt

Examples
--------
::

    # full tri-modal baseline
    python -m scripts.evaluate_ablation --model_weights ckpt.pt --fusion_module GMU

    # text+audio only (drop vision)
    python -m scripts.evaluate_ablation --model_weights ckpt.pt --fusion_module GMU --drop_vision

    # audio only (drop text and vision)
    python -m scripts.evaluate_ablation --model_weights ckpt.pt --fusion_module GMU --drop_text --drop_vision
"""

from __future__ import annotations

import argparse
import os
import sys

import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.encoders import load_processors
from src.dataloader import MultiModalDataset, collate_fn
from src.model import LanguageAudioVisionModel
from src.fusion import FUSION_REGISTRY
from src.utils import compute_metrics, format_metrics


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--t_ckpt_path", default=None)
    p.add_argument("--a_ckpt_path", default=None)
    p.add_argument("--v_ckpt_path", default=None)
    p.add_argument("--model_weights", required=True)
    p.add_argument("--fusion_module", required=True, choices=sorted(FUSION_REGISTRY.keys()))
    p.add_argument("--device", default="cuda")
    p.add_argument("--data_root", default="dataset/")
    p.add_argument("--batch_size", type=int, default=1)
    p.add_argument("--n_workers", type=int, default=4)
    p.add_argument("--output_dir", default="results/")
    # Ablation flags.
    p.add_argument("--drop_text",   action="store_true")
    p.add_argument("--drop_audio",  action="store_true")
    p.add_argument("--drop_vision", action="store_true")
    return p.parse_args()


def ablation_label(args) -> str:
    dropped = []
    if args.drop_text:   dropped.append("Text")
    if args.drop_audio:  dropped.append("Audio")
    if args.drop_vision: dropped.append("Vision")
    return "All modalities enabled" if not dropped else f"Dropped: {', '.join(dropped)}"


def ablation_suffix(args) -> str:
    dropped = []
    if args.drop_text:   dropped.append("no_text")
    if args.drop_audio:  dropped.append("no_audio")
    if args.drop_vision: dropped.append("no_vision")
    return "_".join(dropped) if dropped else "all_modalities"


def main():
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)

    print(f"[ablation] {ablation_label(args)}")
    tokenizer, _, audio_proc, video_proc = load_processors()

    test_set = MultiModalDataset(
        data_root=args.data_root, split="test",
        tokenizer=tokenizer, audio_processor=audio_proc, video_processor=video_proc,
    )
    loader = DataLoader(test_set, batch_size=args.batch_size, shuffle=False,
                        collate_fn=collate_fn, num_workers=args.n_workers)

    model = LanguageAudioVisionModel(
        text_ckpt_path=args.t_ckpt_path,
        audio_ckpt_path=args.a_ckpt_path,
        vision_ckpt_path=args.v_ckpt_path,
        fusion_module=args.fusion_module,
    ).to(args.device)
    model.load_state_dict(torch.load(args.model_weights, map_location=args.device))
    model.eval()

    labels_all, preds_all = [], []
    with torch.no_grad():
        for batch in tqdm(loader, desc="ablation"):
            text, audio, vision, labels = batch
            if audio is None:
                continue
            text   = text["input_ids"].to(args.device)
            audio  = audio.to(args.device)
            vision = vision["pixel_values"].to(args.device)
            labels = labels.to(args.device)

            text_in   = None if args.drop_text   else text
            audio_in  = None if args.drop_audio  else audio
            vision_in = None if args.drop_vision else vision

            logits = model(text_in, audio_in, vision_in)
            labels_all.extend(labels.cpu().numpy())
            preds_all.extend(logits.argmax(dim=1).cpu().numpy())

    metrics = compute_metrics(labels_all, preds_all)
    header = f"EVALUATION RESULTS — {ablation_label(args)}"
    formatted = format_metrics(metrics, header=header)
    print(formatted)

    out = os.path.join(args.output_dir,
                       f"{args.fusion_module}_{ablation_suffix(args)}_results.txt")
    with open(out, "w") as f:
        f.write(formatted + "\n")
    print(f"[ablation] Wrote {out}")


if __name__ == "__main__":
    main()
