"""
Training script for multimodal models with different fusion mechanisms.
"""

import os
os.environ['TOKENIZERS_PARALLELISM'] = "false"
import time
import numpy as np
from tqdm import tqdm
import argparse

from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score

import torch
from torch import nn
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
from torch.utils.data import DataLoader

from transformers import AutoTokenizer, AutoProcessor, AutoImageProcessor

from data.dataset import VideoConversationDataset, collate_fn
from models.multimodal import create_multimodal_model


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Train multimodal models")
    
    # Model config
    parser.add_argument("--fusion_type", type=str, required=True,
                        choices=['early', 'late', 'lmf', 'tensor', 'tfn', 'attention', 'gated', 'gmu', 'transformer'],
                        help="Type of fusion mechanism")
    parser.add_argument("--hidden_dim", type=int, default=256,
                        help="Hidden dimension for fusion")
    parser.add_argument("--freeze_encoders", action="store_true",
                        help="Freeze pretrained encoder weights")
    
    # Pretrained encoder checkpoints (from unimodal training)
    parser.add_argument("--text_ckpt", type=str, default=None,
                        help="Path to pretrained text encoder checkpoint")
    parser.add_argument("--audio_ckpt", type=str, default=None,
                        help="Path to pretrained audio encoder checkpoint")
    parser.add_argument("--video_ckpt", type=str, default=None,
                        help="Path to pretrained video encoder checkpoint")
    
    # Data paths
    parser.add_argument("--train_csv", type=str, default="./splits/train.csv")
    parser.add_argument("--val_csv", type=str, default="./splits/val.csv")
    parser.add_argument("--test_csv", type=str, default="./splits/test.csv")
    parser.add_argument("--transcript_folder", type=str, default="transcripts/")
    parser.add_argument("--audio_folder", type=str, required=True)
    parser.add_argument("--video_folder", type=str, required=True)
    
    # Dataset params
    parser.add_argument("--t", type=float, default=2.0,
                        help="Time window in seconds")
    parser.add_argument("--max_frames", type=int, default=16,
                        help="Number of video frames")
    
    # Training params
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--n_workers", type=int, default=4)
    parser.add_argument("--n_epoch", type=int, default=100)
    parser.add_argument("--lr", type=float, default=1e-5)
    parser.add_argument("--weight_decay", type=float, default=0.01)
    
    # Fusion-specific params
    parser.add_argument("--rank", type=int, default=16,
                        help="Rank for LMF fusion")
    parser.add_argument("--num_heads", type=int, default=4,
                        help="Number of attention heads")
    parser.add_argument("--num_layers", type=int, default=2,
                        help="Number of transformer layers")
    parser.add_argument("--dropout", type=float, default=0.1,
                        help="Dropout rate")
    
    # Device
    parser.add_argument("--device", type=str, default="cuda")
    
    # Logging
    parser.add_argument("--log_dir", type=str, default="log_multimodal")
    parser.add_argument("--save_freq", type=int, default=5)
    
    return parser.parse_args()


def init_log_dir(args):
    """Initialize logging directory."""
    os.makedirs(args.log_dir, exist_ok=True)
    time_str = time.strftime("%Y-%m-%d_%H-%M-%S", time.localtime())
    task_name = f"{time_str}_{args.fusion_type}_t{args.t}_bs{args.batch_size}_lr{args.lr}"
    if args.freeze_encoders:
        task_name += "_frozen"
    task_dir = os.path.join(args.log_dir, task_name)
    os.makedirs(task_dir, exist_ok=True)
    writer = SummaryWriter(log_dir=task_dir)
    
    # Save args
    with open(os.path.join(task_dir, "args.txt"), "w") as f:
        for arg, value in vars(args).items():
            f.write(f"{arg}: {value}\n")
    
    return writer, task_dir


def load_processors():
    """Load all processors."""
    print("Loading processors...")
    tokenizer = AutoTokenizer.from_pretrained("openai-community/gpt2")
    tokenizer.pad_token = tokenizer.eos_token
    audio_processor = AutoProcessor.from_pretrained("facebook/hubert-large-ls960-ft")
    video_processor = AutoImageProcessor.from_pretrained("MCG-NJU/videomae-base")
    return tokenizer, audio_processor, video_processor


def cal_metric(all_labels, all_logits):
    """Calculate classification metrics."""
    accuracy = accuracy_score(all_labels, all_logits)
    recall = recall_score(all_labels, all_logits, average=None, zero_division=0)
    f1 = f1_score(all_labels, all_logits, average=None, zero_division=0)
    precision = precision_score(all_labels, all_logits, average=None, zero_division=0)
    
    macro_recall = recall_score(all_labels, all_logits, average='macro', zero_division=0)
    macro_f1 = f1_score(all_labels, all_logits, average='macro', zero_division=0)
    macro_precision = precision_score(all_labels, all_logits, average='macro', zero_division=0)
    
    return accuracy, recall, f1, precision, macro_recall, macro_f1, macro_precision


def idx2label(idx):
    """Convert index to label name."""
    return ["keep", "turn", "backchannel"][idx]


def train_epoch(model, dataloader, optimizer, criterion, device):
    """Train for one epoch."""
    model.train()
    epoch_loss = 0.0
    all_labels, all_logits = [], []
    
    pbar = tqdm(dataloader, desc="Training")
    for batch in pbar:
        # Prepare inputs
        text_inputs = {
            'input_ids': batch['text']['input_ids'].to(device),
            'attention_mask': batch['text']['attention_mask'].to(device)
        }
        audio_inputs = batch['audio'].to(device)
        video_inputs = {
            'pixel_values': batch['video']['pixel_values'].to(device)
        }
        labels = batch['label'].to(device)
        
        optimizer.zero_grad()
        
        # Forward pass
        pred = model(text_inputs, audio_inputs, video_inputs)
        loss = criterion(pred, labels)
        
        loss.backward()
        optimizer.step()
        
        all_labels.extend(labels.detach().cpu().numpy())
        all_logits.extend(pred.argmax(dim=1).detach().cpu().numpy())
        epoch_loss += loss.item()
        
        pbar.set_postfix({'loss': loss.item()})
    
    epoch_loss /= len(dataloader)
    return epoch_loss, np.array(all_labels), np.array(all_logits)


def validate(model, dataloader, criterion, device):
    """Validate the model."""
    model.eval()
    val_loss = 0.0
    all_labels, all_logits = [], []
    
    with torch.no_grad():
        pbar = tqdm(dataloader, desc="Validation")
        for batch in pbar:
            # Prepare inputs
            text_inputs = {
                'input_ids': batch['text']['input_ids'].to(device),
                'attention_mask': batch['text']['attention_mask'].to(device)
            }
            audio_inputs = batch['audio'].to(device)
            video_inputs = {
                'pixel_values': batch['video']['pixel_values'].to(device)
            }
            labels = batch['label'].to(device)
            
            # Forward pass
            pred = model(text_inputs, audio_inputs, video_inputs)
            loss = criterion(pred, labels)
            
            all_labels.extend(labels.detach().cpu().numpy())
            all_logits.extend(pred.argmax(dim=1).detach().cpu().numpy())
            val_loss += loss.item()
            
            pbar.set_postfix({'loss': loss.item()})
    
    val_loss /= len(dataloader)
    return val_loss, np.array(all_labels), np.array(all_logits)


def main():
    args = parse_args()
    
    print("="*60)
    print(f"Training MULTIMODAL model with {args.fusion_type.upper()} fusion")
    print("="*60)
    print(f"Time window: {args.t} seconds")
    print(f"Batch size: {args.batch_size}")
    print(f"Learning rate: {args.lr}")
    print(f"Epochs: {args.n_epoch}")
    print(f"Freeze encoders: {args.freeze_encoders}")
    print(f"Device: {args.device}")
    print("="*60)
    
    # Load processors
    tokenizer, audio_processor, video_processor = load_processors()
    
    # Create datasets
    print("\nLoading training dataset...")
    train_set = VideoConversationDataset(
        csv_file=args.train_csv,
        transcript_folder=args.transcript_folder,
        audio_folder=args.audio_folder,
        video_folder=args.video_folder,
        t=args.t,
        modal="all",
        max_frames=args.max_frames,
        tokenizer=tokenizer,
        audio_processor=audio_processor,
        video_processor=video_processor
    )
    
    print("\nLoading validation dataset...")
    val_set = VideoConversationDataset(
        csv_file=args.val_csv,
        transcript_folder=args.transcript_folder,
        audio_folder=args.audio_folder,
        video_folder=args.video_folder,
        t=args.t,
        modal="all",
        max_frames=args.max_frames,
        tokenizer=tokenizer,
        audio_processor=audio_processor,
        video_processor=video_processor
    )
    
    # Create dataloaders
    train_loader = DataLoader(
        train_set,
        batch_size=args.batch_size,
        shuffle=True,
        collate_fn=collate_fn,
        num_workers=args.n_workers,
        pin_memory=True
    )
    
    val_loader = DataLoader(
        val_set,
        batch_size=args.batch_size,
        shuffle=False,
        collate_fn=collate_fn,
        num_workers=args.n_workers,
        pin_memory=True
    )
    
    # Build fusion kwargs
    fusion_kwargs = {
        'dropout': args.dropout
    }
    if args.fusion_type == 'lmf':
        fusion_kwargs['rank'] = args.rank
    elif args.fusion_type in ['attention', 'transformer']:
        fusion_kwargs['num_heads'] = args.num_heads
    if args.fusion_type == 'transformer':
        fusion_kwargs['num_layers'] = args.num_layers
    
    # Create model
    print(f"\nCreating multimodal model with {args.fusion_type} fusion...")
    model = create_multimodal_model(
        fusion_type=args.fusion_type,
        hidden_dim=args.hidden_dim,
        output_dim=3,
        freeze_encoders=args.freeze_encoders,
        text_ckpt_path=args.text_ckpt,
        audio_ckpt_path=args.audio_ckpt,
        video_ckpt_path=args.video_ckpt,
        **fusion_kwargs
    ).to(args.device)
    
    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Total parameters: {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")
    
    # Initialize optimizer and criterion
    optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    criterion = nn.CrossEntropyLoss()
    
    # Initialize logging
    writer, task_dir = init_log_dir(args)
    print(f"\nLogging to: {task_dir}")
    print(f"Starting training...\n")
    
    best_val_acc = 0.0
    best_val_f1 = 0.0
    
    # Training loop
    for epoch in range(args.n_epoch):
        print(f"\n{'='*60}")
        print(f"Epoch {epoch+1}/{args.n_epoch}")
        print(f"{'='*60}")
        
        # Train
        train_loss, train_labels, train_logits = train_epoch(
            model, train_loader, optimizer, criterion, args.device
        )
        
        train_acc, train_recall, train_f1, train_precision, train_macro_recall, train_macro_f1, train_macro_precision = cal_metric(
            train_labels, train_logits
        )
        
        # Log training metrics
        writer.add_scalar("train/loss", train_loss, epoch)
        writer.add_scalar("train/accuracy", train_acc, epoch)
        writer.add_scalar("train/macro_f1", train_macro_f1, epoch)
        
        for idx, (r, f, p) in enumerate(zip(train_recall, train_f1, train_precision)):
            writer.add_scalar(f"train/{idx2label(idx)}_recall", r, epoch)
            writer.add_scalar(f"train/{idx2label(idx)}_f1", f, epoch)
            writer.add_scalar(f"train/{idx2label(idx)}_precision", p, epoch)
        
        # Validate
        val_loss, val_labels, val_logits = validate(
            model, val_loader, criterion, args.device
        )
        
        val_acc, val_recall, val_f1, val_precision, val_macro_recall, val_macro_f1, val_macro_precision = cal_metric(
            val_labels, val_logits
        )
        
        # Log validation metrics
        writer.add_scalar("val/loss", val_loss, epoch)
        writer.add_scalar("val/accuracy", val_acc, epoch)
        writer.add_scalar("val/macro_f1", val_macro_f1, epoch)
        
        for idx, (r, f, p) in enumerate(zip(val_recall, val_f1, val_precision)):
            writer.add_scalar(f"val/{idx2label(idx)}_recall", r, epoch)
            writer.add_scalar(f"val/{idx2label(idx)}_f1", f, epoch)
            writer.add_scalar(f"val/{idx2label(idx)}_precision", p, epoch)
        
        # Print summary
        print(f"\n{'Train':<10} Loss: {train_loss:.4f} | Acc: {train_acc:.4f} | Macro-F1: {train_macro_f1:.4f}")
        print(f"{'Val':<10} Loss: {val_loss:.4f} | Acc: {val_acc:.4f} | Macro-F1: {val_macro_f1:.4f}")
        print(f"\nPer-class F1: Keep: {val_f1[0]:.4f} | Turn: {val_f1[1]:.4f} | BC: {val_f1[2]:.4f}")
        
        # Save checkpoints
        if (epoch + 1) % args.save_freq == 0:
            save_path = os.path.join(task_dir, f"epoch_{epoch+1}.pt")
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_acc': val_acc,
                'val_f1': val_macro_f1
            }, save_path)
        
        # Save best models
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_path = os.path.join(task_dir, "best_model_acc.pt")
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_acc': val_acc,
                'val_f1': val_macro_f1
            }, best_path)
            print(f"✓ New best accuracy model saved! Acc: {best_val_acc:.4f}")
        
        if val_macro_f1 > best_val_f1:
            best_val_f1 = val_macro_f1
            best_path = os.path.join(task_dir, "best_model_f1.pt")
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_acc': val_acc,
                'val_f1': val_macro_f1
            }, best_path)
            print(f"✓ New best F1 model saved! F1: {best_val_f1:.4f}")
    
    writer.close()
    print(f"\n{'='*60}")
    print("Training completed!")
    print(f"Best validation accuracy: {best_val_acc:.4f}")
    print(f"Best validation macro-F1: {best_val_f1:.4f}")
    print(f"Models saved to: {task_dir}")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()



"""
# Step 1: Train unimodal models
python train_unimodal.py --modal text --batch_size 16 --n_epoch 50 \
  --transcript_folder transcripts/ \
  --audio_folder /path/to/audio/ \
  --video_folder /path/to/videos/

python train_unimodal.py --modal audio --batch_size 8 --n_epoch 50 \
  --transcript_folder transcripts/ \
  --audio_folder /path/to/audio/ \
  --video_folder /path/to/videos/

python train_unimodal.py --modal video --batch_size 4 --n_epoch 50 \
  --transcript_folder transcripts/ \
  --audio_folder /path/to/audio/ \
  --video_folder /path/to/videos/

# Step 2: Train multimodal with pretrained encoders
python train_multimodal.py \
  --fusion_type lmf \
  --text_ckpt log/2025-01-12_10-30-00_text_t2.0_bs16_lr1e-05/best_model_acc.pt \
  --audio_ckpt log/2025-01-12_11-15-00_audio_t2.0_bs8_lr1e-05/best_model_acc.pt \
  --video_ckpt log/2025-01-12_12-00-00_video_t2.0_bs4_lr1e-05/best_model_acc.pt \
  --freeze_encoders \
  --batch_size 8 \
  --n_epoch 50 \
  --transcript_folder transcripts/ \
  --audio_folder /path/to/audio/ \
  --video_folder /path/to/videos/

"""