"""
Stage 2 — Fusion module training with frozen encoders.

Loads pretrained uni-modal encoders from Stage 1, freezes them, then trains
the selected fusion module on the tri-modal TURN/BC/KEEP classification task.

Random modality dropout (default 5%) is applied during training: with
probability ``--random_drop_modal_rate``, one modality is replaced with
``None`` for that batch. This regularises the fusion module against
sensor dropout at inference time.

Example
-------
::

    python -m scripts.train_fusion \\
        --t_ckpt_path  log/text_epoch_9.pt \\
        --a_ckpt_path  log/audio_epoch_9.pt \\
        --v_ckpt_path  log/video_epoch_9.pt \\
        --fusion_module ACGF \\
        --data_root dataset/ \\
        --batch_size 8 --n_epoch 10 --lr 1e-5 \\
        --log_dir log/

To benchmark a different fusion mechanism, just change ``--fusion_module``
to any name listed in :data:`src.fusion.FUSION_REGISTRY` — e.g. ``GMU``,
``LMF``, ``TAC``, ``BBFN``, ``MFB``, etc.
"""

from __future__ import annotations

import argparse
import os
import random
import sys
import time

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.encoders import load_processors
from src.dataloader import MultiModalDataset, collate_fn
from src.model import LanguageAudioVisionModel
from src.fusion import FUSION_REGISTRY
from src.utils import compute_metrics, idx2label, count_parameters

os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")


def parse_args():
    p = argparse.ArgumentParser()
    # Stage 1 checkpoints.
    p.add_argument("--t_ckpt_path", required=True, help="Stage-1 text encoder checkpoint")
    p.add_argument("--a_ckpt_path", required=True, help="Stage-1 audio encoder checkpoint")
    p.add_argument("--v_ckpt_path", required=True, help="Stage-1 video encoder checkpoint")
    p.add_argument("--device", default="cuda")
    # Fusion module.
    p.add_argument("--fusion_module", default="LMF",
                   choices=sorted(FUSION_REGISTRY.keys()),
                   help="Name of fusion module (see src.fusion.FUSION_REGISTRY)")
    # Data params.
    p.add_argument("--data_root", default="dataset/")
    p.add_argument("--batch_size", type=int, default=1)
    p.add_argument("--n_workers", type=int, default=4)
    # Training params.
    p.add_argument("--n_epoch", type=int, default=10)
    p.add_argument("--lr", type=float, default=1e-5)
    p.add_argument("--random_drop_modal_rate", type=float, default=0.05,
                   help="Probability of dropping one modality per training step")
    # BBFN-specific.
    p.add_argument("--bbfn_lambda_sep", type=float, default=0.1,
                   help="Weight of BBFN's feature-separator regulariser")
    # Logging.
    p.add_argument("--log_dir", default="log")
    p.add_argument("--model_name", default="",
                   help="Optional suffix appended to the log directory name")
    return p.parse_args()


def init_log_dir(args):
    os.makedirs(args.log_dir, exist_ok=True)
    time_str = time.strftime("%Y-%m-%d_%H-%M-%S")
    name = f"{time_str}_{args.model_name}_{args.fusion_module}"
    task_dir = os.path.join(args.log_dir, name)
    return SummaryWriter(log_dir=task_dir), task_dir


def maybe_drop_modality(text, audio, vision, drop_rate: float):
    """With probability ``drop_rate``, zero out one of the three modalities."""
    if random.random() >= drop_rate:
        return text, audio, vision
    choice = random.choice(("text", "audio", "vision"))
    if choice == "text":   return None,  audio, vision
    if choice == "audio":  return text,  None,  vision
    return text, audio, None


def compute_loss(criterion, logits, labels, model, args):
    """Cross-entropy + (BBFN only) feature-separator regulariser."""
    loss = criterion(logits, labels)
    # BBFN exposes its per-layer separator losses via a ``last_aux_losses``
    # attribute when called with ``return_losses=True`` — but to keep the
    # training step uniform across fusion modules we instead inspect the
    # module's name and recompute the auxiliary term inline. The simplest
    # backwards-compatible scheme: if the fusion module has an ``aux_loss``
    # attribute set during forward, add it.
    if hasattr(model.fusion, "aux_loss"):
        loss = loss + args.bbfn_lambda_sep * model.fusion.aux_loss
    return loss


def forward(model, text, audio, vision, args):
    """Forward pass that handles BBFN's optional auxiliary losses."""
    if args.fusion_module == "BBFN":
        # BBFN forward signature: (text, audio, video, return_losses=True)
        logits, sep_losses = model.fusion(
            model.text_model(text)   if text   is not None else None,
            model.audio_model(audio) if audio  is not None else None,
            model.vision_model(vision) if vision is not None else None,
            return_losses=True,
        )
        model.fusion.aux_loss = torch.stack(sep_losses).mean() if sep_losses else 0.0
        return logits
    return model(text, audio, vision)


def run_epoch(model, loader, optimizer, criterion, args, *, train: bool):
    model.train(mode=train)
    total_loss, n_batches = 0.0, 0
    labels_all, preds_all = [], []

    pbar = tqdm(loader, desc="train" if train else "val")
    for batch in pbar:
        text, audio, vision, labels = batch
        if audio is None:  # corrupt sample — skip
            continue

        text   = text["input_ids"].to(args.device)
        audio  = audio.to(args.device)
        vision = vision["pixel_values"].to(args.device)
        labels = labels.to(args.device)

        if train:
            text, audio, vision = maybe_drop_modality(
                text, audio, vision, args.random_drop_modal_rate
            )

        with torch.set_grad_enabled(train):
            logits = forward(model, text, audio, vision, args)
            loss = compute_loss(criterion, logits, labels, model, args)
            if train:
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

        total_loss += loss.item() * args.batch_size
        n_batches += 1
        labels_all.extend(labels.detach().cpu().numpy())
        preds_all.extend(logits.argmax(dim=1).detach().cpu().numpy())
        pbar.set_postfix(loss=loss.item())

    metrics = compute_metrics(labels_all, preds_all)
    return total_loss / max(n_batches, 1), metrics


def log_metrics(writer, prefix, loss, metrics, epoch):
    writer.add_scalar(f"{prefix}/loss",     loss,                   epoch)
    writer.add_scalar(f"{prefix}/accuracy", metrics["accuracy"],    epoch)
    writer.add_scalar(f"{prefix}/macro_f1", metrics["macro"]["f1"], epoch)
    for idx in range(3):
        writer.add_scalar(f"{prefix}/{idx2label(idx)}_f1",
                          metrics["per_class"]["f1"][idx], epoch)


def main():
    args = parse_args()
    tokenizer, _, audio_proc, video_proc = load_processors()

    common = dict(
        data_root=args.data_root,
        tokenizer=tokenizer,
        audio_processor=audio_proc,
        video_processor=video_proc,
    )
    train_set = MultiModalDataset(split="train", **common)
    val_set   = MultiModalDataset(split="val",   **common)

    train_loader = DataLoader(train_set, batch_size=args.batch_size, shuffle=True,
                              collate_fn=collate_fn, num_workers=args.n_workers)
    val_loader   = DataLoader(val_set,   batch_size=args.batch_size, shuffle=False,
                              collate_fn=collate_fn, num_workers=args.n_workers)

    model = LanguageAudioVisionModel(
        text_ckpt_path=args.t_ckpt_path,
        audio_ckpt_path=args.a_ckpt_path,
        vision_ckpt_path=args.v_ckpt_path,
        fusion_module=args.fusion_module,
    ).to(args.device)
    model.freeze_encoders()

    fusion_params = count_parameters(model.fusion)
    print(f"[train] Fusion module '{args.fusion_module}' "
          f"has {fusion_params:,} trainable parameters")

    optimizer = torch.optim.Adam(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=args.lr,
    )
    criterion = nn.CrossEntropyLoss()

    writer, task_dir = init_log_dir(args)
    print(f"[train] Logging to {task_dir}")

    for epoch in range(args.n_epoch):
        tr_loss, tr_m = run_epoch(model, train_loader, optimizer, criterion,
                                  args, train=True)
        va_loss, va_m = run_epoch(model, val_loader,   optimizer, criterion,
                                  args, train=False)

        log_metrics(writer, "train", tr_loss, tr_m, epoch)
        log_metrics(writer, "val",   va_loss, va_m, epoch)

        print(f"[epoch {epoch}] train_loss={tr_loss:.4f} "
              f"val_loss={va_loss:.4f} val_macro_f1={va_m['macro']['f1']:.4f}")

        torch.save(model.state_dict(), os.path.join(task_dir, f"epoch_{epoch}.pt"))

    writer.close()


if __name__ == "__main__":
    main()
