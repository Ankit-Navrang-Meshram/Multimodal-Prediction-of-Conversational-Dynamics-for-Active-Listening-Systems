# ============================================================================
# File: train_token_llm.py
# Training Script for Multimodal Token LLM
# ============================================================================

import os
import time
import argparse
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
from torch.utils.data import DataLoader
from tqdm import tqdm
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from mm_tokenizers import VideoTokenizer, SpeechTokenizer
from transformers import AutoTokenizer
from dataloader_token import MultimodalTokenDataset, collate_fn_token
from model import MultimodalTokenLLM
# Assuming all classes are imported from above modules


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_root", type=str, default="dataset/")
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--n_workers", type=int, default=4)
    parser.add_argument("--n_epochs", type=int, default=20)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--llm_lr", type=float, default=1e-5)
    parser.add_argument("--freeze_llm", action="store_true", default=False)
    parser.add_argument("--log_dir", type=str, default="log_token_llm")
    parser.add_argument("--save_every", type=int, default=5)
    #return parser.parse_args()
    args, unknown = parser.parse_known_args()
    return args

def cal_metrics(all_labels, all_preds):
    accuracy = accuracy_score(all_labels, all_preds)
    recall = recall_score(all_labels, all_preds, average=None, zero_division=0)
    f1 = f1_score(all_labels, all_preds, average=None, zero_division=0)
    precision = precision_score(all_labels, all_preds, average=None, zero_division=0)
    return accuracy, recall, f1, precision


def idx2label(idx):
    return ["keep", "turn", "bc"][idx]


def main():
    args = parse_args()
    
    # Create log directory
    os.makedirs(args.log_dir, exist_ok=True)
    time_str = time.strftime("%Y-%m-%d_%H-%M-%S")
    task_dir = os.path.join(args.log_dir, f"{time_str}_token_llm")
    writer = SummaryWriter(log_dir=task_dir)
    
    # Initialize tokenizers
    print("Initializing tokenizers...")
    video_tokenizer = VideoTokenizer()
    audio_tokenizer = SpeechTokenizer()
    text_tokenizer = AutoTokenizer.from_pretrained("gpt2")
    text_tokenizer.pad_token = text_tokenizer.eos_token
    
    # Load datasets
    print("Loading datasets...")
    train_dataset = MultimodalTokenDataset(
        args.data_root, "train",
        video_tokenizer, audio_tokenizer, text_tokenizer
    )
    val_dataset = MultimodalTokenDataset(
        args.data_root, "val",
        video_tokenizer, audio_tokenizer, text_tokenizer
    )
    
    train_loader = DataLoader(
        train_dataset, 
        batch_size=args.batch_size, 
        shuffle=True,
        collate_fn=collate_fn_token,
        num_workers=args.n_workers
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        collate_fn=collate_fn_token,
        num_workers=args.n_workers
    )
    
    # Initialize model
    print("Initializing model...")
    model = MultimodalTokenLLM(
        freeze_llm=args.freeze_llm,
        use_pretrained_tokenizers=False  # We pass tokenizers separately
    )
    model.video_tokenizer = video_tokenizer
    model.audio_tokenizer = audio_tokenizer
    model = model.to(args.device)
    
    # Optimizer with different learning rates
    llm_params = list(model.llm.parameters())
    other_params = [p for n, p in model.named_parameters() if 'llm' not in n]
    
    optimizer = optim.AdamW([
        {'params': other_params, 'lr': args.lr},
        {'params': llm_params, 'lr': args.llm_lr}
    ])
    
    criterion = nn.CrossEntropyLoss()
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.n_epochs)
    
    # Training loop
    print("Starting training...")
    best_val_acc = 0.0
    
    for epoch in range(args.n_epochs):
        # Train
        model.train()
        train_loss = 0.0
        all_labels, all_preds = [], []
        
        train_bar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{args.n_epochs} [Train]")
        for video_tokens, audio_tokens, text_ids, text_mask, labels in train_bar:
            video_tokens = video_tokens.to(args.device)
            audio_tokens = audio_tokens.to(args.device)
            text_ids = text_ids.to(args.device)
            labels = labels.to(args.device)
            
            optimizer.zero_grad()
            
            logits = model(
                video_tokens=video_tokens,
                audio_tokens=audio_tokens,
                text_ids=text_ids
            )
            
            loss = criterion(logits, labels)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            
            train_loss += loss.item()
            preds = logits.argmax(dim=1)
            all_labels.extend(labels.cpu().numpy())
            all_preds.extend(preds.cpu().numpy())
            
            train_bar.set_postfix({'loss': f'{loss.item():.4f}'})
        
        train_loss /= len(train_loader)
        train_acc, train_recall, train_f1, train_prec = cal_metrics(all_labels, all_preds)
        
        # Log training metrics
        writer.add_scalar("train/loss", train_loss, epoch)
        writer.add_scalar("train/accuracy", train_acc, epoch)
        
        # Validation
        model.eval()
        val_loss = 0.0
        all_labels, all_preds = [], []
        
        val_bar = tqdm(val_loader, desc=f"Epoch {epoch+1}/{args.n_epochs} [Val]")
        with torch.no_grad():
            for video_tokens, audio_tokens, text_ids, text_mask, labels in val_bar:
                video_tokens = video_tokens.to(args.device)
                audio_tokens = audio_tokens.to(args.device)
                text_ids = text_ids.to(args.device)
                labels = labels.to(args.device)
                
                logits = model(
                    video_tokens=video_tokens,
                    audio_tokens=audio_tokens,
                    text_ids=text_ids
                )
                
                loss = criterion(logits, labels)
                val_loss += loss.item()
                
                preds = logits.argmax(dim=1)
                all_labels.extend(labels.cpu().numpy())
                all_preds.extend(preds.cpu().numpy())
        
        val_loss /= len(val_loader)
        val_acc, val_recall, val_f1, val_prec = cal_metrics(all_labels, all_preds)
        
        # Log validation metrics
        writer.add_scalar("val/loss", val_loss, epoch)
        writer.add_scalar("val/accuracy", val_acc, epoch)
        
        print(f"\nEpoch {epoch+1}/{args.n_epochs}")
        print(f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f}")
        print(f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}")
        
        # Save best model
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            save_path = os.path.join(task_dir, "best_model.pt")
            torch.save(model.state_dict(), save_path)
            print(f"Saved best model with val_acc: {val_acc:.4f}")
        
        # Save checkpoint
        if (epoch + 1) % args.save_every == 0:
            save_path = os.path.join(task_dir, f"checkpoint_epoch_{epoch+1}.pt")
            torch.save(model.state_dict(), save_path)
        
        scheduler.step()
    
    writer.close()
    print(f"Training complete! Best val accuracy: {best_val_acc:.4f}")


if __name__ == "__main__":
    main()