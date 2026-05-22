import os
os.environ['TOKENIZERS_PARALLELISM'] = "false"
import time
import random
import numpy as np
from tqdm import tqdm

from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score

import torch
from torch import nn
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter

from datasets import Dataset
from torch.utils.data import DataLoader

from model.mm import LanguageModel, AudioModel, VisionModel, load_processors
from dataloader import MultiModalDataset, collate_fn


import argparse
parser = argparse.ArgumentParser()
parser.add_argument("--modal", type=str, required=True)
parser.add_argument("--device", type=str, default="cuda")
# data params
parser.add_argument("--data_root", type=str, default="dataset/")
parser.add_argument("--batch_size", type=int, default=1)
parser.add_argument("--n_workers", type=int, default=4)
# train params
parser.add_argument("--n_epoch", type=int, default=100)
parser.add_argument("--lr", type=float, default=1e-5)
# training log
parser.add_argument("--log_dir", type=str, default="log")
args = parser.parse_args()

assert args.modal in ["text", "audio", "video"]

def init_log_dir():
    os.makedirs(args.log_dir, exist_ok=True)
    time_str = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()).replace(" ", "_")
    task_dir = os.path.join(args.log_dir, f"{time_str}_{args.modal}")
    writer = SummaryWriter(log_dir=task_dir)
    return writer, task_dir


def load_model():
    if args.modal == "text":
        model = LanguageModel(return_embeddings=False)
    elif args.modal == "audio":
        model = AudioModel(return_embeddings=False)
    elif args.modal == "video":
        model = VisionModel(return_embeddings=False)
    return model


def cal_metric(all_labels, all_logits):
    accuracy = accuracy_score(all_labels, all_logits)
    recall = recall_score(all_labels, all_logits, average=None)
    f1 = f1_score(all_labels, all_logits, average=None)
    precision = precision_score(all_labels, all_logits, average=None)
    return accuracy, recall, f1, precision


def idx2label(idx):
    return ["keep", "turn", "bc"][idx]


def main():
    tokenizer, _, audio_processor, video_processor = load_processors()
    train_set = MultiModalDataset(
        data_root=args.data_root,
        split="train",
        modal=args.modal,
        tokenizer=tokenizer,
        audio_processor=audio_processor,
        video_processor=video_processor,
    )
    val_set = MultiModalDataset(
        data_root=args.data_root,
        split="val",
        modal=args.modal,
        tokenizer=tokenizer,
        audio_processor=audio_processor,
        video_processor=video_processor,
    )

    train_loader = DataLoader(train_set, batch_size=args.batch_size, shuffle=True, collate_fn=collate_fn, num_workers=args.n_workers)
    val_loader = DataLoader(val_set, batch_size=args.batch_size, shuffle=False, collate_fn=collate_fn, num_workers=args.n_workers)

    model = load_model().to(args.device)
    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    criterion = nn.CrossEntropyLoss()

    writer, task_dir = init_log_dir()

    n_epoch = 10
    t_bar = tqdm(range(n_epoch))
    for epoch in t_bar:
        # train
        model.train()
        n_batch = len(train_loader)
        epoch_loss = 0.0
        all_labels, all_logits = [], []
        for i, (X, y) in enumerate(train_loader):
            if X is None:
                continue
            if args.modal == "text":
                X = X["input_ids"].to("cuda")
            elif args.modal == "audio":
                X = X.to("cuda")
            elif args.modal == "video":
                X = X["pixel_values"].to("cuda")
            y = y.to("cuda")
            optimizer.zero_grad()
            
            pred = model(X)
            loss = criterion(pred, y)
            loss.backward()
            optimizer.step()

            all_labels.extend(y.detach().cpu().numpy())
            all_logits.extend(pred.argmax(dim=1).detach().cpu().numpy())

            t_bar.set_description(f"Epoch {epoch} training | Batch {i}/{n_batch} | Loss {loss.item():.4f}")
            epoch_loss += (loss.item() * args.batch_size)
        epoch_loss = epoch_loss / n_batch
        writer.add_scalar("train/loss", epoch_loss, epoch)
        accuracy, recall, f1, precision = cal_metric(np.array(all_labels), np.array(all_logits))
        writer.add_scalar("train/accuracy", accuracy, epoch)
        for idx, r in enumerate(recall):
            writer.add_scalar(f"train/{idx2label(idx)}_recall", r, epoch)
        for idx, f in enumerate(f1):
            writer.add_scalar(f"train/{idx2label(idx)}_f1", f, epoch)
        for idx, p in enumerate(precision):
            writer.add_scalar(f"train/{idx2label(idx)}_precision", p, epoch)

        # val
        model.eval()
        n_batch = len(val_loader)
        val_loss = 0.0
        all_labels, all_logits = [], []
        for i, (X, y) in enumerate(train_loader):
            if X is None:
                continue
            if args.modal == "text":
                X = X["input_ids"].to("cuda")
            elif args.modal == "audio":
                X = X.to("cuda")
            elif args.modal == "video":
                X = X["pixel_values"].to("cuda")
            y = y.to("cuda")
            with torch.no_grad():
                pred = model(X)
                loss = criterion(pred, y)
                
                all_labels.extend(y.detach().cpu().numpy())
                all_logits.extend(pred.argmax(dim=1).detach().cpu().numpy())
                val_loss += (loss.item() * args.batch_size)
        val_loss = val_loss / n_batch
        writer.add_scalar("val/loss", val_loss, epoch)
        accuracy, recall, f1, precision = cal_metric(np.array(all_labels), np.array(all_logits))
        writer.add_scalar("val/accuracy", accuracy, epoch)
        for idx, r in enumerate(recall):
            writer.add_scalar(f"val/{idx2label(idx)}_recall", r, epoch)
        for idx, f in enumerate(f1):
            writer.add_scalar(f"val/{idx2label(idx)}_f1", f, epoch)
        for idx, p in enumerate(precision):
            writer.add_scalar(f"val/{idx2label(idx)}_precision", p, epoch)

        save_path = os.path.join(task_dir, f"epoch_{epoch}.pt")
        torch.save(model.state_dict(), save_path)


if __name__ == "__main__":
    main()

# import os
# os.environ['TOKENIZERS_PARALLELISM'] = "false"
# os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True'
# import time
# import gc
# import numpy as np
# from tqdm import tqdm

# from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score

# import torch
# from torch import nn
# import torch.optim as optim
# from torch.utils.tensorboard import SummaryWriter
# from torch.utils.data import DataLoader

# from model.mm import LanguageModel, AudioModel, VisionModel, load_processors
# from training_dataloader import MultiModalDataset, collate_fn

# import argparse
# parser = argparse.ArgumentParser()
# parser.add_argument("--modal", type=str, required=True)
# parser.add_argument("--device", type=str, default="cuda")
# # data params
# parser.add_argument("--data_root", type=str, default="dataset/")
# parser.add_argument("--batch_size", type=int, default=1)
# parser.add_argument("--n_workers", type=int, default=0)
# # train params
# parser.add_argument("--n_epoch", type=int, default=100)
# parser.add_argument("--lr", type=float, default=1e-5)
# parser.add_argument("--gradient_accumulation_steps", type=int, default=4)
# parser.add_argument("--mixed_precision", action="store_true")
# parser.add_argument("--max_grad_norm", type=float, default=1.0)
# # training log
# parser.add_argument("--log_dir", type=str, default="log")
# args = parser.parse_args()

# assert args.modal in ["text", "audio", "video"]

# def init_log_dir():
#     os.makedirs(args.log_dir, exist_ok=True)
#     time_str = time.strftime("%Y-%m-%d_%H-%M-%S", time.localtime())
#     task_dir = os.path.join(args.log_dir, f"{time_str}_{args.modal}")
#     writer = SummaryWriter(log_dir=task_dir)
#     return writer, task_dir


# def load_model():
#     if args.modal == "text":
#         model = LanguageModel(return_embeddings=False)
#     elif args.modal == "audio":
#         model = AudioModel(return_embeddings=False)
#     elif args.modal == "video":
#         model = VisionModel(return_embeddings=False)
#     return model


# def cal_metric(all_labels, all_logits):
#     accuracy = accuracy_score(all_labels, all_logits)
#     recall = recall_score(all_labels, all_logits, average=None, zero_division=0)
#     f1 = f1_score(all_labels, all_logits, average=None, zero_division=0)
#     precision = precision_score(all_labels, all_logits, average=None, zero_division=0)
#     return accuracy, recall, f1, precision


# def idx2label(idx):
#     return ["keep", "turn", "bc"][idx]


# def clear_memory():
#     """Aggressively clear GPU memory"""
#     gc.collect()
#     if torch.cuda.is_available():
#         torch.cuda.empty_cache()
#         torch.cuda.synchronize()


# def main():
#     # Clear memory at start
#     clear_memory()
    
#     tokenizer, _, audio_processor, video_processor = load_processors()
    
#     train_set = MultiModalDataset(
#         data_root=args.data_root,
#         split="train",
#         modal=args.modal,
#         tokenizer=tokenizer,
#         audio_processor=audio_processor,
#         video_processor=video_processor,
#     )
#     val_set = MultiModalDataset(
#         data_root=args.data_root,
#         split="val",
#         modal=args.modal,
#         tokenizer=tokenizer,
#         audio_processor=audio_processor,
#         video_processor=video_processor,
#     )

#     train_loader = DataLoader(
#         train_set, 
#         batch_size=args.batch_size, 
#         shuffle=True, 
#         collate_fn=collate_fn, 
#         num_workers=args.n_workers,
#         pin_memory=False,  # Disable to save memory
#         prefetch_factor=None if args.n_workers == 0 else 2
#     )
#     val_loader = DataLoader(
#         val_set, 
#         batch_size=args.batch_size, 
#         shuffle=False, 
#         collate_fn=collate_fn, 
#         num_workers=args.n_workers,
#         pin_memory=False,
#         prefetch_factor=None if args.n_workers == 0 else 2
#     )

#     model = load_model().to(args.device)
    
#     # Enable gradient checkpointing to save memory
#     try:
#         if hasattr(model, 'transformer'):  # Text model
#             model.transformer.gradient_checkpointing_enable()
#         elif hasattr(model, 'hubert'):  # Audio model
#             model.hubert.gradient_checkpointing_enable()
#         elif hasattr(model, 'model'):  # Vision model
#             model.model.gradient_checkpointing_enable()
#     except Exception as e:
#         print(f"Could not enable gradient checkpointing: {e}")
    
#     optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.01)
#     criterion = nn.CrossEntropyLoss()
    
#     # Mixed precision training with updated API
#     scaler = torch.amp.GradScaler('cuda') if args.mixed_precision and args.device == "cuda" else None

#     writer, task_dir = init_log_dir()
    
#     print(f"Starting training for {args.modal} model")
#     print(f"Batch size: {args.batch_size}, Gradient accumulation steps: {args.gradient_accumulation_steps}")
#     print(f"Effective batch size: {args.batch_size * args.gradient_accumulation_steps}")
    
#     # Print GPU memory info
#     if torch.cuda.is_available():
#         print(f"GPU: {torch.cuda.get_device_name(0)}")
#         print(f"Initial GPU memory allocated: {torch.cuda.memory_allocated(0)/1024**3:.2f} GB")
#         print(f"Initial GPU memory reserved: {torch.cuda.memory_reserved(0)/1024**3:.2f} GB")

#     t_bar = tqdm(range(args.n_epoch))
#     for epoch in t_bar:
#         # ============ TRAINING ============
#         model.train()
#         n_batch = len(train_loader)
#         epoch_loss = 0.0
#         all_labels, all_logits = [], []
        
#         optimizer.zero_grad()
        
#         for i, batch_data in enumerate(train_loader):
#             try:
#                 # Unpack based on modality
#                 if args.modal == "text":
#                     X, y = batch_data
#                     X = X["input_ids"].to(args.device, non_blocking=True)
#                 elif args.modal == "audio":
#                     X, y = batch_data
#                     X = X.to(args.device, non_blocking=True)
#                 elif args.modal == "video":
#                     X, y = batch_data
#                     X = X["pixel_values"].to(args.device, non_blocking=True)
                
#                 y = y.to(args.device, non_blocking=True)
                
#                 # Mixed precision training with updated API
#                 if scaler is not None:
#                     with torch.amp.autocast('cuda'):
#                         pred = model(X)
#                         loss = criterion(pred, y)
#                         loss = loss / args.gradient_accumulation_steps
                    
#                     scaler.scale(loss).backward()
                    
#                     # Update weights every N steps
#                     if (i + 1) % args.gradient_accumulation_steps == 0:
#                         # Gradient clipping
#                         scaler.unscale_(optimizer)
#                         torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)
                        
#                         scaler.step(optimizer)
#                         scaler.update()
#                         optimizer.zero_grad()
#                 else:
#                     pred = model(X)
#                     loss = criterion(pred, y)
#                     loss = loss / args.gradient_accumulation_steps
#                     loss.backward()
                    
#                     # Update weights every N steps
#                     if (i + 1) % args.gradient_accumulation_steps == 0:
#                         # Gradient clipping
#                         torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)
#                         optimizer.step()
#                         optimizer.zero_grad()

#                 # Collect predictions (detach before moving to CPU)
#                 with torch.no_grad():
#                     all_labels.extend(y.cpu().numpy())
#                     all_logits.extend(pred.argmax(dim=1).cpu().numpy())

#                 # Calculate actual loss value for logging
#                 actual_loss = loss.item() * args.gradient_accumulation_steps
                
#                 t_bar.set_description(
#                     f"Epoch {epoch} training | Batch {i+1}/{n_batch} | Loss {actual_loss:.4f}"
#                 )
#                 epoch_loss += (actual_loss * args.batch_size)
                
#                 # Aggressive memory cleanup every batch
#                 del X, y, pred, loss
                
#                 # More frequent memory clearing for stability
#                 if (i + 1) % 10 == 0:
#                     torch.cuda.empty_cache()
                    
#             except RuntimeError as e:
#                 if "out of memory" in str(e):
#                     print(f"\n[WARNING] OOM at batch {i+1}, clearing cache and skipping batch...")
#                     torch.cuda.empty_cache()
#                     optimizer.zero_grad()
#                     continue
#                 else:
#                     raise e
        
#         # Handle remaining gradients
#         if (len(train_loader)) % args.gradient_accumulation_steps != 0:
#             if scaler is not None:
#                 scaler.unscale_(optimizer)
#                 torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)
#                 scaler.step(optimizer)
#                 scaler.update()
#             else:
#                 torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)
#                 optimizer.step()
#             optimizer.zero_grad()
        
#         epoch_loss = epoch_loss / len(train_set)
#         writer.add_scalar("train/loss", epoch_loss, epoch)
#         accuracy, recall, f1, precision = cal_metric(np.array(all_labels), np.array(all_logits))
#         writer.add_scalar("train/accuracy", accuracy, epoch)
#         for idx, r in enumerate(recall):
#             writer.add_scalar(f"train/{idx2label(idx)}_recall", r, epoch)
#         for idx, f in enumerate(f1):
#             writer.add_scalar(f"train/{idx2label(idx)}_f1", f, epoch)
#         for idx, p in enumerate(precision):
#             writer.add_scalar(f"train/{idx2label(idx)}_precision", p, epoch)

#         # ============ VALIDATION ============
#         model.eval()
#         n_batch = len(val_loader)
#         val_loss = 0.0
#         all_labels, all_logits = [], []
        
#         with torch.no_grad():
#             for i, batch_data in enumerate(val_loader):
#                 try:
#                     # Unpack based on modality
#                     if args.modal == "text":
#                         X, y = batch_data
#                         X = X["input_ids"].to(args.device, non_blocking=True)
#                     elif args.modal == "audio":
#                         X, y = batch_data
#                         X = X.to(args.device, non_blocking=True)
#                     elif args.modal == "video":
#                         X, y = batch_data
#                         X = X["pixel_values"].to(args.device, non_blocking=True)
                    
#                     y = y.to(args.device, non_blocking=True)
                    
#                     if scaler is not None:
#                         with torch.amp.autocast('cuda'):
#                             pred = model(X)
#                             loss = criterion(pred, y)
#                     else:
#                         pred = model(X)
#                         loss = criterion(pred, y)
                    
#                     all_labels.extend(y.cpu().numpy())
#                     all_logits.extend(pred.argmax(dim=1).cpu().numpy())
#                     val_loss += (loss.item() * args.batch_size)
                    
#                     # Free memory
#                     del X, y, pred, loss
                    
#                     if (i + 1) % 10 == 0:
#                         torch.cuda.empty_cache()
                        
#                 except RuntimeError as e:
#                     if "out of memory" in str(e):
#                         print(f"\n[WARNING] OOM at validation batch {i+1}, skipping...")
#                         torch.cuda.empty_cache()
#                         continue
#                     else:
#                         raise e
        
#         val_loss = val_loss / len(val_set)
#         writer.add_scalar("val/loss", val_loss, epoch)
#         accuracy, recall, f1, precision = cal_metric(np.array(all_labels), np.array(all_logits))
#         writer.add_scalar("val/accuracy", accuracy, epoch)
#         for idx, r in enumerate(recall):
#             writer.add_scalar(f"val/{idx2label(idx)}_recall", r, epoch)
#         for idx, f in enumerate(f1):
#             writer.add_scalar(f"val/{idx2label(idx)}_f1", f, epoch)
#         for idx, p in enumerate(precision):
#             writer.add_scalar(f"val/{idx2label(idx)}_precision", p, epoch)

#         # Save checkpoint
#         save_path = os.path.join(task_dir, f"epoch_{epoch}.pt")
#         torch.save(model.state_dict(), save_path)
        
#         # Clear memory after each epoch
#         clear_memory()
        
#         print(f"\nEpoch {epoch} | Train Loss: {epoch_loss:.4f} | Val Loss: {val_loss:.4f} | Val Acc: {accuracy:.4f}")
        
#         if torch.cuda.is_available():
#             print(f"GPU memory allocated: {torch.cuda.memory_allocated(0)/1024**3:.2f} GB")

#     writer.close()
#     print(f"Training complete! Models saved to {task_dir}")


# if __name__ == "__main__":
#     main()