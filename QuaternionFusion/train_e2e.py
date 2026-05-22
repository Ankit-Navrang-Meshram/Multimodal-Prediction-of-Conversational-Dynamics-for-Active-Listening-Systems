# import os
# import time
# import random
# import numpy as np
# from tqdm import tqdm

# from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score

# import torch
# from torch import nn
# import torch.optim as optim
# from torch.utils.tensorboard import SummaryWriter

# from datasets import Dataset
# from torch.utils.data import DataLoader

# from mm import LanguageAudioVisionModel, load_processors
# from dataloader import MultiModalDataset, collate_fn

# import argparse
# parser = argparse.ArgumentParser()
# parser.add_argument("--t_ckpt_path", type=str, required=True)
# parser.add_argument("--a_ckpt_path", type=str, required=True)
# parser.add_argument("--v_ckpt_path", type=str, required=True)
# parser.add_argument("--device", type=str, default="cuda")
# # data params
# parser.add_argument("--data_root", type=str, default="dataset/")
# parser.add_argument("--batch_size", type=int, default=1)
# parser.add_argument("--n_workers", type=int, default=4)
# # train params
# parser.add_argument("--n_epoch", type=int, default=100)
# parser.add_argument("--lr", type=float, default=1e-5)
# parser.add_argument("--random_drop_modal_rate", type=float, default=0.05)
# # training log
# parser.add_argument("--log_dir", type=str, default="log")
# parser.add_argument("--model_name", type=str, default="_")
# args = parser.parse_args()


# def init_log_dir():
#     os.makedirs(args.log_dir, exist_ok=True)
#     time_str = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()).replace(" ", "_")
#     task_dir = os.path.join(args.log_dir, f"{time_str}{args.model_name}")
#     writer = SummaryWriter(log_dir=task_dir)
#     return writer, task_dir


# def cal_metric(all_labels, all_logits):
#     accuracy = accuracy_score(all_labels, all_logits)
#     recall = recall_score(all_labels, all_logits, average=None)
#     f1 = f1_score(all_labels, all_logits, average=None)
#     precision = precision_score(all_labels, all_logits, average=None)
#     return accuracy, recall, f1, precision


# def idx2label(idx):
#     return ["keep", "turn", "bc"][idx]


# def main():
#     tokenizer, _, audio_processor, video_processor = load_processors()
#     train_set = MultiModalDataset(
#         data_root=args.data_root,
#         split="train",
#         tokenizer=tokenizer,
#         audio_processor=audio_processor,
#         video_processor=video_processor,
#     )
#     val_set = MultiModalDataset(
#         data_root=args.data_root,
#         split="val",
#         tokenizer=tokenizer,
#         audio_processor=audio_processor,
#         video_processor=video_processor,
#     )

#     train_loader = DataLoader(train_set, batch_size=args.batch_size, shuffle=True, collate_fn=collate_fn, num_workers=args.n_workers)
#     val_loader = DataLoader(val_set, batch_size=args.batch_size, shuffle=False, collate_fn=collate_fn, num_workers=args.n_workers)

#     # model = LanguageAudioVisionModel().to(args.device)
#     # optimizer = optim.Adam(model.parameters(), lr=args.lr)
#     model = LanguageAudioVisionModel(text_ckpt_path=args.t_ckpt_path,audio_ckpt_path=args.a_ckpt_path,
#                                     vision_ckpt_path=args.v_ckpt_path,fusion_module="quaternion" # pass "quaternion" from CLI
#                                     ).to(args.device)

#     for name, param in model.named_parameters():
#         if "lora" in name or "proj" in name or "fusion" in name:
#             param.requires_grad = True
#         else:
#             param.requires_grad = False

#     # Check trainable parameters to confirm efficiency
#     trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
#     print(f"Trainable Parameters: {trainable_params:,}")

#     optimizer = torch.optim.Adam(
#         filter(lambda p: p.requires_grad, model.parameters()),
#         lr=args.lr
#     )


#     criterion = nn.CrossEntropyLoss()

#     writer, task_dir = init_log_dir()

#     n_epoch = 10
#     random_drop_modal_rate = 0.05
#     t_bar = tqdm(range(n_epoch))
#     best_val_loss = float("inf")
#     for epoch in t_bar:
#         # train
#         model.train()
#         n_batch = len(train_loader)
#         epoch_loss = 0.0
#         all_labels, all_logits = [], []
#         for i, (text, audio, vision, y) in enumerate(train_loader):
#             if audio is None:
#                 continue
#             text = text["input_ids"].to("cuda")
#             audio = audio.to("cuda")
#             vision = vision["pixel_values"].to("cuda")
#             y = y.to("cuda")
#             optimizer.zero_grad()
            
#             if random.random() < random_drop_modal_rate:
#                 drop_modal = random.choice(["text", "audio", "vision"])
#                 if drop_modal == "text":
#                     text = None
#                 elif drop_modal == "audio":
#                     audio = None
#                 elif drop_modal == "vision":
#                     vision = None
#                 # print("drop modal:", drop_modal)
#             pred = model(text, audio, vision)
#             loss = criterion(pred, y)
#             loss.backward()
#             optimizer.step()

#             all_labels.extend(y.detach().cpu().numpy())
#             all_logits.extend(pred.argmax(dim=1).detach().cpu().numpy())

#             t_bar.set_description(f"Epoch {epoch} training | Batch {i}/{n_batch} | Loss {loss.item():.4f}")
#             epoch_loss += (loss.item() * args.batch_size)
#         epoch_loss = epoch_loss / n_batch
#         writer.add_scalar("train/loss", epoch_loss, epoch)
#         accuracy, recall, f1, precision = cal_metric(np.array(all_labels), np.array(all_logits))
#         writer.add_scalar("train/accuracy", accuracy, epoch)
#         for idx, r in enumerate(recall):
#             writer.add_scalar(f"train/{idx2label(idx)}_recall", r, epoch)
#         for idx, f in enumerate(f1):
#             writer.add_scalar(f"train/{idx2label(idx)}_f1", f, epoch)
#         for idx, p in enumerate(precision):
#             writer.add_scalar(f"train/{idx2label(idx)}_precision", p, epoch)

#         # val
#         model.eval()
#         n_batch = len(val_loader)
#         val_loss = 0.0
#         all_labels, all_logits = [], []
#         for i, (text, audio, vision, y) in enumerate(val_loader):
#             if audio is None:
#                 continue
#             text = text["input_ids"].to("cuda")
#             audio = audio.to("cuda")
#             vision = vision["pixel_values"].to("cuda")
#             y = y.to("cuda")
#             with torch.no_grad():
#                 pred = model(text, audio, vision)
#                 loss = criterion(pred, y)
                
#                 all_labels.extend(y.detach().cpu().numpy())
#                 all_logits.extend(pred.argmax(dim=1).detach().cpu().numpy())
#                 val_loss += (loss.item() * args.batch_size)
#         val_loss = val_loss / n_batch
#         writer.add_scalar("val/loss", val_loss, epoch)
#         accuracy, recall, f1, precision = cal_metric(np.array(all_labels), np.array(all_logits))
#         writer.add_scalar("val/accuracy", accuracy, epoch)
#         for idx, r in enumerate(recall):
#             writer.add_scalar(f"val/{idx2label(idx)}_recall", r, epoch)
#         for idx, f in enumerate(f1):
#             writer.add_scalar(f"val/{idx2label(idx)}_f1", f, epoch)
#         for idx, p in enumerate(precision):
#             writer.add_scalar(f"val/{idx2label(idx)}_precision", p, epoch)

#         if val_loss < best_val_loss:
#             best_val_loss = val_loss
#             best_model_path = os.path.join(task_dir, f"best_epoch_{epoch}.pt")
#             torch.save(model.state_dict(), best_model_path)
#             print(f"Saved best model to {best_model_path}")
        
#         save_path = os.path.join(task_dir, f"epoch_{epoch}.pt")
#         torch.save(model.state_dict(), save_path)
#         print(f"Saved model to {save_path}")

# if __name__ == "__main__":
#     main()



import os
import time
import random
import numpy as np
from tqdm import tqdm

from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from sklearn.utils.class_weight import compute_class_weight

import torch
from torch import nn
from torch.utils.tensorboard import SummaryWriter
from transformers import get_cosine_schedule_with_warmup

from torch.utils.data import DataLoader

from mm import LanguageAudioVisionModel, load_processors
from dataloader import MultiModalDataset, collate_fn

import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--t_ckpt_path", type=str, required=True)
parser.add_argument("--a_ckpt_path", type=str, required=True)
parser.add_argument("--v_ckpt_path", type=str, required=True)
parser.add_argument("--device", type=str, default="cuda")
# data params
parser.add_argument("--data_root", type=str, default="dataset/")
parser.add_argument("--batch_size", type=int, default=1)
parser.add_argument("--n_workers", type=int, default=4)
# train params
parser.add_argument("--n_epoch", type=int, default=100)
parser.add_argument("--lr", type=float, default=1e-5)
parser.add_argument("--weight_decay", type=float, default=1e-2)
parser.add_argument("--warmup_ratio", type=float, default=0.1)
parser.add_argument("--random_drop_modal_rate", type=float, default=0.15)
parser.add_argument("--grad_clip", type=float, default=1.0)
parser.add_argument("--patience", type=int, default=8)
# training log
parser.add_argument("--log_dir", type=str, default="log")
parser.add_argument("--model_name", type=str, default="_")
args = parser.parse_args()


# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------

def init_log_dir():
    os.makedirs(args.log_dir, exist_ok=True)
    time_str = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()).replace(" ", "_")
    task_dir = os.path.join(args.log_dir, f"{time_str}{args.model_name}")
    os.makedirs(task_dir, exist_ok=True)
    writer = SummaryWriter(log_dir=task_dir)
    return writer, task_dir


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------

def cal_metric(all_labels, all_logits):
    accuracy  = accuracy_score(all_labels, all_logits)
    recall    = recall_score(all_labels, all_logits, average=None, zero_division=0)
    f1        = f1_score(all_labels, all_logits, average=None, zero_division=0)
    precision = precision_score(all_labels, all_logits, average=None, zero_division=0)
    return accuracy, recall, f1, precision


def idx2label(idx):
    return ["keep", "turn", "bc"][idx]


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    tokenizer, _, audio_processor, video_processor = load_processors()

    train_set = MultiModalDataset(
        data_root=args.data_root,
        split="train",
        tokenizer=tokenizer,
        audio_processor=audio_processor,
        video_processor=video_processor,
    )
    val_set = MultiModalDataset(
        data_root=args.data_root,
        split="val",
        tokenizer=tokenizer,
        audio_processor=audio_processor,
        video_processor=video_processor,
    )

    train_loader = DataLoader(
        train_set, batch_size=args.batch_size, shuffle=True,
        collate_fn=collate_fn, num_workers=args.n_workers,
    )
    val_loader = DataLoader(
        val_set, batch_size=args.batch_size, shuffle=False,
        collate_fn=collate_fn, num_workers=args.n_workers,
    )

    # -----------------------------------------------------------------------
    # Plain CrossEntropyLoss with class weights only (no label smoothing)
    # -----------------------------------------------------------------------
    all_train_labels = [d["label"] for d in train_set.data_list]
    class_weights = compute_class_weight(
        class_weight="balanced",
        classes=np.array([0, 1, 2]),
        y=np.array(all_train_labels),
    )
    class_weights_tensor = torch.tensor(class_weights, dtype=torch.float).to(args.device)
    print(f"Class weights: {dict(zip(['keep','turn','bc'], class_weights.round(3)))}")

    # Same loss function as Run A — plain CrossEntropyLoss — but with class
    # weights to correct for keep/turn/bc imbalance. No label smoothing.
    criterion = nn.CrossEntropyLoss(weight=class_weights_tensor)

    # -----------------------------------------------------------------------
    # Model
    # -----------------------------------------------------------------------
    model = LanguageAudioVisionModel(
        text_ckpt_path=args.t_ckpt_path,
        audio_ckpt_path=args.a_ckpt_path,
        vision_ckpt_path=args.v_ckpt_path,
        fusion_module="quaternion",
    ).to(args.device)

    for name, param in model.named_parameters():
        param.requires_grad = any(k in name for k in ("lora", "proj", "fusion"))

    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total     = sum(p.numel() for p in model.parameters())
    print(f"Trainable parameters: {trainable:,} / {total:,} "
          f"({100 * trainable / total:.1f}%)")

    # -----------------------------------------------------------------------
    # AdamW — fixes overfitting via weight decay (Run A used plain Adam)
    # -----------------------------------------------------------------------
    optimizer = torch.optim.AdamW(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=args.lr,
        weight_decay=args.weight_decay,
    )

    # -----------------------------------------------------------------------
    # Cosine LR schedule with linear warmup — prevents val loss divergence
    # that appeared in Run A from epoch 3 onward
    # -----------------------------------------------------------------------
    total_steps  = args.n_epoch * len(train_loader)
    warmup_steps = int(total_steps * args.warmup_ratio)
    scheduler = get_cosine_schedule_with_warmup(
        optimizer,
        num_warmup_steps=warmup_steps,
        num_training_steps=total_steps,
    )
    print(f"Total steps: {total_steps:,} | Warmup steps: {warmup_steps:,}")

    writer, task_dir = init_log_dir()

    # -----------------------------------------------------------------------
    # Training loop
    # -----------------------------------------------------------------------
    best_val_loss   = float("inf")
    no_improve      = 0
    best_model_path = None

    t_bar = tqdm(range(args.n_epoch), desc="Epochs")

    for epoch in t_bar:

        # -------------------------------------------------------------------
        # Train
        # -------------------------------------------------------------------
        model.train()
        n_batch    = len(train_loader)
        epoch_loss = 0.0
        all_labels, all_logits = [], []

        for i, (text, audio, vision, y) in enumerate(train_loader):
            if audio is None:
                continue

            text   = text["input_ids"].to(args.device)
            audio  = audio.to(args.device)
            vision = vision["pixel_values"].to(args.device)
            y      = y.to(args.device)

            # Modal dropout — raised from 0.05 → 0.15 to reduce overfitting
            # on fixed modality combinations seen during training
            if random.random() < args.random_drop_modal_rate:
                drop_modal = random.choice(["text", "audio", "vision"])
                if drop_modal == "text":
                    text = None
                elif drop_modal == "audio":
                    audio = None
                else:
                    vision = None

            optimizer.zero_grad()
            pred = model(text, audio, vision)
            loss = criterion(pred, y)
            loss.backward()

            # Gradient clipping — stabilises training with AdamW
            if args.grad_clip > 0:
                torch.nn.utils.clip_grad_norm_(
                    filter(lambda p: p.requires_grad, model.parameters()),
                    max_norm=args.grad_clip,
                )

            optimizer.step()
            scheduler.step()

            all_labels.extend(y.detach().cpu().numpy())
            all_logits.extend(pred.argmax(dim=1).detach().cpu().numpy())
            epoch_loss += loss.item()

            t_bar.set_description(
                f"Epoch {epoch} | Batch {i}/{n_batch} | Loss {loss.item():.4f} "
                f"| LR {scheduler.get_last_lr()[0]:.2e}"
            )

        epoch_loss /= n_batch
        writer.add_scalar("train/loss", epoch_loss, epoch)
        writer.add_scalar("train/lr", scheduler.get_last_lr()[0], epoch)

        accuracy, recall, f1, precision = cal_metric(
            np.array(all_labels), np.array(all_logits)
        )
        writer.add_scalar("train/accuracy", accuracy, epoch)
        for idx, (r, f, p) in enumerate(zip(recall, f1, precision)):
            lbl = idx2label(idx)
            writer.add_scalar(f"train/{lbl}_recall",    r, epoch)
            writer.add_scalar(f"train/{lbl}_f1",        f, epoch)
            writer.add_scalar(f"train/{lbl}_precision", p, epoch)

        # -------------------------------------------------------------------
        # Validate
        # -------------------------------------------------------------------
        model.eval()
        val_loss    = 0.0
        n_val_batch = len(val_loader)
        all_labels, all_logits = [], []

        with torch.no_grad():
            for i, (text, audio, vision, y) in enumerate(val_loader):
                if audio is None:
                    continue

                text   = text["input_ids"].to(args.device)
                audio  = audio.to(args.device)
                vision = vision["pixel_values"].to(args.device)
                y      = y.to(args.device)

                pred = model(text, audio, vision)
                loss = criterion(pred, y)

                all_labels.extend(y.detach().cpu().numpy())
                all_logits.extend(pred.argmax(dim=1).detach().cpu().numpy())
                val_loss += loss.item()

        val_loss /= n_val_batch
        writer.add_scalar("val/loss", val_loss, epoch)

        accuracy, recall, f1, precision = cal_metric(
            np.array(all_labels), np.array(all_logits)
        )
        writer.add_scalar("val/accuracy", accuracy, epoch)
        for idx, (r, f, p) in enumerate(zip(recall, f1, precision)):
            lbl = idx2label(idx)
            writer.add_scalar(f"val/{lbl}_recall",    r, epoch)
            writer.add_scalar(f"val/{lbl}_f1",        f, epoch)
            writer.add_scalar(f"val/{lbl}_precision", p, epoch)

        print(
            f"Epoch {epoch:03d} | "
            f"train loss {epoch_loss:.4f} | val loss {val_loss:.4f} | "
            f"val acc {accuracy:.4f}"
        )

        # -------------------------------------------------------------------
        # Checkpoint + early stopping
        # -------------------------------------------------------------------
        save_path = os.path.join(task_dir, f"epoch_{epoch}.pt")
        torch.save(model.state_dict(), save_path)

        if val_loss < best_val_loss:
            best_val_loss   = val_loss
            no_improve      = 0
            best_model_path = os.path.join(task_dir, f"best_epoch_{epoch}.pt")
            torch.save(model.state_dict(), best_model_path)
            print(f"  New best val loss {best_val_loss:.4f} — saved to {best_model_path}")
        else:
            no_improve += 1
            print(f"  No improvement ({no_improve}/{args.patience})")
            if no_improve >= args.patience:
                print(f"Early stopping triggered at epoch {epoch}. "
                      f"Best val loss: {best_val_loss:.4f}")
                break

    writer.close()
    print(f"\nTraining complete. Best model: {best_model_path}")


if __name__ == "__main__":
    main()