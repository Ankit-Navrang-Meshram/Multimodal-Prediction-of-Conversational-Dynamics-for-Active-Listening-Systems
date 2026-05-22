import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import time
import random
import numpy as np
from tqdm import tqdm

from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score

import torch
from torch import nn
from torch.utils.tensorboard import SummaryWriter
from torch.utils.data import DataLoader
from transformers import AutoTokenizer, AutoFeatureExtractor, AutoImageProcessor

from model import DNCCEnsembleModel
from loss import DNCCLoss
from dataloader import MultiModalDataset, collate_fn

import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--t_ckpt_path",  type=str, required=True)
parser.add_argument("--a_ckpt_path",  type=str, required=True)
parser.add_argument("--v_ckpt_path",  type=str, required=True)
parser.add_argument("--device",       type=str,   default="cuda")
parser.add_argument("--lambda_div",   type=float, default=0.5)
parser.add_argument("--data_root",    type=str,   default="dataset/")
parser.add_argument("--batch_size",   type=int,   default=1)
parser.add_argument("--n_workers",    type=int,   default=4)
parser.add_argument("--n_epoch",      type=int,   default=100)
parser.add_argument("--lr",           type=float, default=1e-5)
parser.add_argument("--random_drop_modal_rate", type=float, default=0.05)
parser.add_argument("--accumulation_steps",     type=int,   default=8)
parser.add_argument("--log_dir",      type=str,   default="log")
parser.add_argument("--model_name",   type=str,   default="DNCC_Ensemble")
args = parser.parse_args()


def init_log_dir():
    os.makedirs(args.log_dir, exist_ok=True)
    time_str  = time.strftime("%Y-%m-%d_%H-%M-%S", time.localtime())
    task_dir  = os.path.join(args.log_dir, f"{time_str}_{args.model_name}_lambda{args.lambda_div}")
    os.makedirs(task_dir, exist_ok=True)
    return SummaryWriter(log_dir=task_dir), task_dir


def cal_metric(all_labels, all_preds):
    accuracy  = accuracy_score(all_labels,  all_preds)
    recall    = recall_score(all_labels,    all_preds, average=None, zero_division=0)
    f1        = f1_score(all_labels,        all_preds, average=None, zero_division=0)
    precision = precision_score(all_labels, all_preds, average=None, zero_division=0)
    return accuracy, recall, f1, precision


def idx2label(idx):
    return ["keep", "turn", "bc"][idx]


def load_processors():
    tokenizer       = AutoTokenizer.from_pretrained("openai-community/gpt2")
    tokenizer.pad_token = tokenizer.eos_token
    audio_processor = AutoFeatureExtractor.from_pretrained("facebook/hubert-base-ls960")
    video_processor = AutoImageProcessor.from_pretrained("MCG-NJU/videomae-base")
    return tokenizer, audio_processor, video_processor


def main():
    # FIX Bug 3 (previous session): disable cuDNN benchmark so variable-length
    # sequences don't cause "no engine" crashes.
    torch.backends.cudnn.benchmark     = False
    torch.backends.cudnn.deterministic = True

    tokenizer, audio_processor, video_processor = load_processors()

    train_set = MultiModalDataset(
        data_root=args.data_root, split="train",
        tokenizer=tokenizer, audio_processor=audio_processor,
        video_processor=video_processor,
    )
    val_set = MultiModalDataset(
        data_root=args.data_root, split="val",
        tokenizer=tokenizer, audio_processor=audio_processor,
        video_processor=video_processor,
    )

    train_loader = DataLoader(train_set, batch_size=args.batch_size, shuffle=True,
                              collate_fn=collate_fn, num_workers=args.n_workers)
    val_loader   = DataLoader(val_set,   batch_size=args.batch_size, shuffle=False,
                              collate_fn=collate_fn, num_workers=args.n_workers)

    model = DNCCEnsembleModel(
        text_ckpt=args.t_ckpt_path,
        audio_ckpt=args.a_ckpt_path,
        vision_ckpt=args.v_ckpt_path,
    ).to(args.device)

    for p in model.text_encoder.parameters():   p.requires_grad = False
    for p in model.audio_encoder.parameters():  p.requires_grad = False
    for p in model.vision_encoder.parameters(): p.requires_grad = False

    optimizer = torch.optim.Adam(
        filter(lambda p: p.requires_grad, model.parameters()), lr=args.lr
    )

    criterion_train = DNCCLoss(lambda_div=args.lambda_div)
    criterion_val   = nn.NLLLoss()

    # GradScaler for the model forward pass (float16 activations).
    # The loss itself runs in float32 — see training loop below.
    scaler = torch.cuda.amp.GradScaler()

    writer, task_dir = init_log_dir()
    best_val_loss    = float("inf")

    t_bar = tqdm(range(args.n_epoch))
    for epoch in t_bar:

        # ── Training ──────────────────────────────────────────────────────
        model.train()
        n_batch    = len(train_loader)
        epoch_loss = 0.0
        n_processed = 0
        all_labels, all_preds = [], []

        optimizer.zero_grad()

        for i, (text, audio, vision, y) in enumerate(train_loader):
            if audio is None:
                continue

            text   = text["input_ids"].to(args.device)
            y      = y.to(args.device)
            vision = vision["pixel_values"].to(args.device)

            # FIX Bug 4: always create zero-replacement tensors in float32,
            # not via zeros_like (which would inherit float16 under autocast).
            audio = audio.to(args.device)
            if random.random() < args.random_drop_modal_rate:
                drop_modal = random.choice(["text", "audio", "vision"])
                if drop_modal == "text":
                    text   = torch.zeros_like(text)
                elif drop_modal == "audio":
                    # zeros_like inherits dtype — use explicit float32 instead
                    audio  = torch.zeros(audio.shape,  dtype=torch.float32, device=args.device)
                elif drop_modal == "vision":
                    vision = torch.zeros(vision.shape, dtype=torch.float32, device=args.device)

            # ── Forward: model runs under autocast (float16 activations) ──
            with torch.cuda.amp.autocast():
                pred_logits = model(text, audio, vision)
                # pred_logits is float32 because model.forward() casts back
                # after the encoder block (see model.py Bug 2 fix).

            # FIX Bug 1: compute DNCCLoss OUTSIDE autocast in float32.
            # Inside autocast, softmax on float16 logits produces exact 0.0
            # for non-maximum classes (float16 has ~3 decimal digits), then
            # log(0) = -inf and -inf * 0 = NaN inside KL divergence.
            # pred_logits.float() is a no-op here (already float32 from the
            # fix in model.py) but is kept as an explicit safety guarantee.
            loss = criterion_train(pred_logits.float(), y)
            loss = loss / args.accumulation_steps

            scaler.scale(loss).backward()

            if (i + 1) % args.accumulation_steps == 0 or (i + 1) == n_batch:
                scaler.step(optimizer)
                scaler.update()
                # FIX Bug 3: always zero_grad after the accumulation window,
                # even if scaler skipped the step due to NaN/Inf gradients.
                # Without this, NaN gradients from a skipped step persist in
                # .grad buffers and poison the next accumulation_steps batches.
                optimizer.zero_grad()

            # Rescale for logging (undo the /accumulation_steps division)
            loss_for_log = loss.item() * args.accumulation_steps

            with torch.no_grad():
                ensemble_pred = pred_logits.mean(dim=0).argmax(dim=1)

            all_labels.extend(y.detach().cpu().numpy())
            all_preds.extend(ensemble_pred.detach().cpu().numpy())

            epoch_loss  += loss_for_log * len(y)
            n_processed += 1
            t_bar.set_description(
                f"Epoch {epoch} | Batch {i}/{n_batch} | Loss {loss_for_log:.4f}"
            )

        epoch_loss /= max(n_processed * args.batch_size, 1)
        writer.add_scalar("train/loss", epoch_loss, epoch)
        accuracy, recall, f1, precision = cal_metric(np.array(all_labels), np.array(all_preds))
        writer.add_scalar("train/accuracy", accuracy, epoch)
        for idx in range(len(recall)):
            writer.add_scalar(f"train/{idx2label(idx)}_recall",    recall[idx],    epoch)
            writer.add_scalar(f"train/{idx2label(idx)}_f1",        f1[idx],        epoch)
            writer.add_scalar(f"train/{idx2label(idx)}_precision", precision[idx], epoch)

        # ── Validation ────────────────────────────────────────────────────
        model.eval()
        val_loss = 0.0
        n_val    = 0
        all_labels, all_preds = [], []

        for i, (text, audio, vision, y) in enumerate(val_loader):
            if audio is None:
                continue

            text   = text["input_ids"].to(args.device)
            audio  = audio.to(args.device)
            vision = vision["pixel_values"].to(args.device)
            y      = y.to(args.device)

            with torch.no_grad():
                pred_probs    = model(text, audio, vision)          # (B, 3)
                loss          = criterion_val(torch.log(pred_probs + 1e-8), y)
                ensemble_pred = pred_probs.argmax(dim=1)

                all_labels.extend(y.cpu().numpy())
                all_preds.extend(ensemble_pred.cpu().numpy())
                val_loss += loss.item() * len(y)
                n_val    += 1

        val_loss /= max(n_val * args.batch_size, 1)
        writer.add_scalar("val/loss", val_loss, epoch)
        accuracy, recall, f1, precision = cal_metric(np.array(all_labels), np.array(all_preds))
        writer.add_scalar("val/accuracy", accuracy, epoch)
        for idx in range(len(recall)):
            writer.add_scalar(f"val/{idx2label(idx)}_recall",    recall[idx],    epoch)
            writer.add_scalar(f"val/{idx2label(idx)}_f1",        f1[idx],        epoch)
            writer.add_scalar(f"val/{idx2label(idx)}_precision", precision[idx], epoch)

        # ── Checkpointing ─────────────────────────────────────────────────
        torch.save(model.state_dict(), os.path.join(task_dir, f"epoch_{epoch}.pt"))
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), os.path.join(task_dir, "best_model.pt"))
            print(f"\n  ✓ Best model saved  (val_loss={val_loss:.4f})")

    writer.close()
    print(f"\nTraining complete. Best val_loss: {best_val_loss:.4f}")


if __name__ == "__main__":
    main()

"""
python train_e2e.py \
    --t_ckpt_path ../model_weights/text_model.pt \
    --a_ckpt_path ../model_weights/audio_model.pt \
    --v_ckpt_path ../model_weights/vision_model.pt \
    --lambda_div 0.1 \
    --data_root ../dataset/ \
    --n_epoch 20
"""