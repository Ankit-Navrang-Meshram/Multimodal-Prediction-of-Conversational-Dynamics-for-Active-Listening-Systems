"""
Stage 1 — Uni-modal encoder training.

Trains a single modality encoder (text, audio, or video) end-to-end on the
TURN/BC/KEEP classification task. After training converges, the encoder's
weights are saved for use in Stage 2 (fusion training), and the temporary
classification head is discarded.

Example
-------
::

    python -m scripts.train_unimodal \\
        --modal text \\
        --data_root dataset/ \\
        --batch_size 16 \\
        --n_epoch 10 \\
        --lr 1e-5 \\
        --log_dir log/
"""

from __future__ import annotations

import argparse
import os
import time
import sys

import numpy as np
import torch
from torch import nn, optim
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

# Make the project root importable regardless of cwd.
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.encoders import LanguageModel, AudioModel, VisionModel, load_processors
from src.dataloader import MultiModalDataset, collate_fn
from src.utils import compute_metrics, idx2label

os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--modal", choices=["text", "audio", "video"], required=True)
    p.add_argument("--device", default="cuda")
    # Data params
    p.add_argument("--data_root", default="dataset/")
    p.add_argument("--batch_size", type=int, default=1)
    p.add_argument("--n_workers", type=int, default=4)
    # Training params
    p.add_argument("--n_epoch", type=int, default=10)
    p.add_argument("--lr", type=float, default=1e-5)
    # Logging
    p.add_argument("--log_dir", default="log")
    return p.parse_args()


def init_log_dir(args):
    os.makedirs(args.log_dir, exist_ok=True)
    time_str = time.strftime("%Y-%m-%d_%H-%M-%S")
    task_dir = os.path.join(args.log_dir, f"{time_str}_{args.modal}")
    return SummaryWriter(log_dir=task_dir), task_dir


def load_model(modal: str):
    if modal == "text":  return LanguageModel(return_embeddings=False)
    if modal == "audio": return AudioModel(return_embeddings=False)
    if modal == "video": return VisionModel(return_embeddings=False)
    raise ValueError(f"Unknown modality: {modal}")


def unpack_batch(batch, modal: str, device: str):
    """Pull the modality + label out of the collate_fn tuple."""
    # When `modal != 'all'`, collate_fn returns (modality_tensor_or_dict, labels)
    if modal == "text":
        text, labels = batch
        return text["input_ids"].to(device), labels.to(device)
    if modal == "audio":
        audio, labels = batch
        return audio.to(device), labels.to(device)
    if modal == "video":
        video, labels = batch
        return video["pixel_values"].to(device), labels.to(device)
    raise ValueError(modal)


def run_epoch(model, loader, optimizer, criterion, args, *, train: bool):
    """One pass over the data loader. Returns mean loss and metrics."""
    model.train(mode=train)
    total_loss, n_batches = 0.0, 0
    labels_all, preds_all = [], []

    pbar = tqdm(loader, desc="train" if train else "val")
    for batch in pbar:
        x, y = unpack_batch(batch, args.modal, args.device)

        with torch.set_grad_enabled(train):
            logits = model(x)
            loss = criterion(logits, y)
            if train:
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

        total_loss += loss.item() * args.batch_size
        n_batches += 1
        labels_all.extend(y.detach().cpu().numpy())
        preds_all.extend(logits.argmax(dim=1).detach().cpu().numpy())
        pbar.set_postfix(loss=loss.item())

    metrics = compute_metrics(labels_all, preds_all)
    return total_loss / max(n_batches, 1), metrics


def log_metrics(writer, prefix: str, loss, metrics, epoch):
    writer.add_scalar(f"{prefix}/loss",     loss,                   epoch)
    writer.add_scalar(f"{prefix}/accuracy", metrics["accuracy"],    epoch)
    writer.add_scalar(f"{prefix}/macro_f1", metrics["macro"]["f1"], epoch)
    for idx in range(3):
        writer.add_scalar(f"{prefix}/{idx2label(idx)}_f1",
                          metrics["per_class"]["f1"][idx], epoch)


def main():
    args = parse_args()
    tokenizer, _, audio_proc, video_proc = load_processors()

    common_kwargs = dict(
        data_root=args.data_root,
        modal=args.modal,
        tokenizer=tokenizer,
        audio_processor=audio_proc,
        video_processor=video_proc,
    )
    train_set = MultiModalDataset(split="train", **common_kwargs)
    val_set   = MultiModalDataset(split="val",   **common_kwargs)

    train_loader = DataLoader(train_set, batch_size=args.batch_size, shuffle=True,
                              collate_fn=collate_fn, num_workers=args.n_workers)
    val_loader   = DataLoader(val_set,   batch_size=args.batch_size, shuffle=False,
                              collate_fn=collate_fn, num_workers=args.n_workers)

    model = load_model(args.modal).to(args.device)
    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    criterion = nn.CrossEntropyLoss()

    writer, task_dir = init_log_dir(args)
    print(f"[train] Logging to {task_dir}")

    for epoch in range(args.n_epoch):
        train_loss, train_m = run_epoch(model, train_loader, optimizer, criterion,
                                        args, train=True)
        val_loss,   val_m   = run_epoch(model, val_loader,   optimizer, criterion,
                                        args, train=False)

        log_metrics(writer, "train", train_loss, train_m, epoch)
        log_metrics(writer, "val",   val_loss,   val_m,   epoch)

        print(f"[epoch {epoch}] train_loss={train_loss:.4f} "
              f"val_loss={val_loss:.4f} val_macro_f1={val_m['macro']['f1']:.4f}")

        # Save checkpoint after every epoch.
        ckpt_path = os.path.join(task_dir, f"epoch_{epoch}.pt")
        torch.save(model.state_dict(), ckpt_path)

    writer.close()


if __name__ == "__main__":
    main()
