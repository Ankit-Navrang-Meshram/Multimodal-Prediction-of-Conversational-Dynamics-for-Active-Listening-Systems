import os
os.environ['TOKENIZERS_PARALLELISM'] = "false"
import time
import numpy as np
from tqdm import tqdm

from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score

import torch
from torch import nn
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
from torch.utils.data import DataLoader

from transformers import AutoTokenizer, AutoProcessor, AutoImageProcessor, AutoModel

from dataset.dataset import VideoConversationDataset, collate_fn

import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--modal", type=str, required=True, choices=["text", "audio", "video"])
parser.add_argument("--device", type=str, default="cuda")

# Data paths
parser.add_argument("--train_csv", type=str, default="./splits/train.csv")
parser.add_argument("--val_csv", type=str, default="./splits/val.csv")
parser.add_argument("--test_csv", type=str, default="./splits/test.csv")
parser.add_argument("--transcript_folder", type=str, default="transcripts/")
parser.add_argument("--audio_folder", type=str, required=True)
parser.add_argument("--video_folder", type=str, required=True)

# Dataset params
parser.add_argument("--t", type=float, default=2.0, help="Time window in seconds")
parser.add_argument("--max_frames", type=int, default=16, help="Number of video frames")
parser.add_argument("--batch_size", type=int, default=8)
parser.add_argument("--n_workers", type=int, default=4)

# Training params
parser.add_argument("--n_epoch", type=int, default=100)
parser.add_argument("--lr", type=float, default=1e-5)

# Logging
parser.add_argument("--log_dir", type=str, default="log")

args = parser.parse_args()


class LanguageModel(nn.Module):
    """GPT-2 based text model for classification."""
    def __init__(self, num_classes=3, return_embeddings=False):
        super().__init__()
        self.encoder = AutoModel.from_pretrained("openai-community/gpt2")
        self.return_embeddings = return_embeddings
        hidden_size = 768
        
        if not return_embeddings:
            self.classifier = nn.Sequential(
                nn.Linear(hidden_size, 256),
                nn.ReLU(),
                nn.Dropout(0.1),
                nn.Linear(256, num_classes)
            )
    
    def forward(self, input_ids, attention_mask=None):
        outputs = self.encoder(input_ids=input_ids, attention_mask=attention_mask)
        # Mean pooling over sequence
        pooled = outputs.last_hidden_state.mean(dim=1)
        
        if self.return_embeddings:
            return pooled
        return self.classifier(pooled)


class AudioModel(nn.Module):
    """HuBERT based audio model for classification."""
    def __init__(self, num_classes=3, return_embeddings=False):
        super().__init__()
        self.encoder = AutoModel.from_pretrained("facebook/hubert-large-ls960-ft")
        self.return_embeddings = return_embeddings
        hidden_size = 1024
        
        if not return_embeddings:
            self.classifier = nn.Sequential(
                nn.Linear(hidden_size, 256),
                nn.ReLU(),
                nn.Dropout(0.1),
                nn.Linear(256, num_classes)
            )
    
    def forward(self, input_values):
        outputs = self.encoder(input_values)
        # Mean pooling over time
        pooled = outputs.last_hidden_state.mean(dim=1)
        
        if self.return_embeddings:
            return pooled
        return self.classifier(pooled)


class VisionModel(nn.Module):
    """VideoMAE based vision model for classification."""
    def __init__(self, num_classes=3, return_embeddings=False):
        super().__init__()
        self.encoder = AutoModel.from_pretrained("MCG-NJU/videomae-base")
        self.return_embeddings = return_embeddings
        hidden_size = 768
        
        if not return_embeddings:
            self.classifier = nn.Sequential(
                nn.Linear(hidden_size, 256),
                nn.ReLU(),
                nn.Dropout(0.1),
                nn.Linear(256, num_classes)
            )
    
    def forward(self, pixel_values):
        outputs = self.encoder(pixel_values)
        # Mean pooling over frames
        pooled = outputs.last_hidden_state.mean(dim=1)
        
        if self.return_embeddings:
            return pooled
        return self.classifier(pooled)


def init_log_dir():
    """Initialize logging directory and tensorboard writer."""
    os.makedirs(args.log_dir, exist_ok=True)
    time_str = time.strftime("%Y-%m-%d_%H-%M-%S", time.localtime())
    task_dir = os.path.join(args.log_dir, f"{time_str}_{args.modal}_t{args.t}")
    os.makedirs(task_dir, exist_ok=True)
    writer = SummaryWriter(log_dir=task_dir)
    return writer, task_dir


def load_model():
    """Load appropriate model based on modality."""
    if args.modal == "text":
        model = LanguageModel(num_classes=3, return_embeddings=False)
    elif args.modal == "audio":
        model = AudioModel(num_classes=3, return_embeddings=False)
    elif args.modal == "video":
        model = VisionModel(num_classes=3, return_embeddings=False)
    return model


def load_processors():
    """Load appropriate processors based on modality."""
    tokenizer = None
    audio_processor = None
    video_processor = None
    
    if args.modal == "text":
        tokenizer = AutoTokenizer.from_pretrained("openai-community/gpt2")
        tokenizer.pad_token = tokenizer.eos_token
    elif args.modal == "audio":
        audio_processor = AutoProcessor.from_pretrained("facebook/hubert-large-ls960-ft")
    elif args.modal == "video":
        video_processor = AutoImageProcessor.from_pretrained("MCG-NJU/videomae-base")
    
    return tokenizer, audio_processor, video_processor


def cal_metric(all_labels, all_logits):
    """Calculate classification metrics."""
    accuracy = accuracy_score(all_labels, all_logits)
    recall = recall_score(all_labels, all_logits, average=None, zero_division=0)
    f1 = f1_score(all_labels, all_logits, average=None, zero_division=0)
    precision = precision_score(all_labels, all_logits, average=None, zero_division=0)
    return accuracy, recall, f1, precision


def idx2label(idx):
    """Convert index to label name."""
    return ["keep", "turn", "backchannel"][idx]


def train_epoch(model, dataloader, optimizer, criterion, device):
    """Train for one epoch."""
    model.train()
    epoch_loss = 0.0
    all_labels, all_logits = [], []
    
    for batch in tqdm(dataloader, desc="Training"):
        # Get data based on modality
        if args.modal == "text":
            X = batch['text']['input_ids'].to(device)
            attention_mask = batch['text']['attention_mask'].to(device)
        elif args.modal == "audio":
            X = batch['audio'].to(device)
            attention_mask = None
        elif args.modal == "video":
            X = batch['video']['pixel_values'].to(device)
            attention_mask = None
        
        y = batch['label'].to(device)
        
        optimizer.zero_grad()
        
        # Forward pass
        if args.modal == "text":
            pred = model(X, attention_mask)
        else:
            pred = model(X)
        
        loss = criterion(pred, y)
        loss.backward()
        optimizer.step()
        
        all_labels.extend(y.detach().cpu().numpy())
        all_logits.extend(pred.argmax(dim=1).detach().cpu().numpy())
        epoch_loss += loss.item()
    
    epoch_loss /= len(dataloader)
    return epoch_loss, np.array(all_labels), np.array(all_logits)


def validate(model, dataloader, criterion, device):
    """Validate the model."""
    model.eval()
    val_loss = 0.0
    all_labels, all_logits = [], []
    
    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Validation"):
            # Get data based on modality
            if args.modal == "text":
                X = batch['text']['input_ids'].to(device)
                attention_mask = batch['text']['attention_mask'].to(device)
            elif args.modal == "audio":
                X = batch['audio'].to(device)
                attention_mask = None
            elif args.modal == "video":
                X = batch['video']['pixel_values'].to(device)
                attention_mask = None
            
            y = batch['label'].to(device)
            
            # Forward pass
            if args.modal == "text":
                pred = model(X, attention_mask)
            else:
                pred = model(X)
            
            loss = criterion(pred, y)
            
            all_labels.extend(y.detach().cpu().numpy())
            all_logits.extend(pred.argmax(dim=1).detach().cpu().numpy())
            val_loss += loss.item()
    
    val_loss /= len(dataloader)
    return val_loss, np.array(all_labels), np.array(all_logits)


def main():
    print("="*60)
    print(f"Training {args.modal.upper()} model")
    print("="*60)
    print(f"Time window: {args.t} seconds")
    print(f"Batch size: {args.batch_size}")
    print(f"Learning rate: {args.lr}")
    print(f"Epochs: {args.n_epoch}")
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
        modal=args.modal,
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
        modal=args.modal,
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
        num_workers=args.n_workers
    )
    
    val_loader = DataLoader(
        val_set,
        batch_size=args.batch_size,
        shuffle=False,
        collate_fn=collate_fn,
        num_workers=args.n_workers
    )
    
    # Initialize model, optimizer, criterion
    model = load_model().to(args.device)
    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    criterion = nn.CrossEntropyLoss()
    
    # Initialize logging
    writer, task_dir = init_log_dir()
    
    print(f"\nLogging to: {task_dir}")
    print(f"Starting training...\n")
    
    best_val_acc = 0.0
    
    # Training loop
    for epoch in range(args.n_epoch):
        print(f"\n{'='*60}")
        print(f"Epoch {epoch+1}/{args.n_epoch}")
        print(f"{'='*60}")
        
        # Train
        train_loss, train_labels, train_logits = train_epoch(
            model, train_loader, optimizer, criterion, args.device
        )
        
        train_acc, train_recall, train_f1, train_precision = cal_metric(
            train_labels, train_logits
        )
        
        # Log training metrics
        writer.add_scalar("train/loss", train_loss, epoch)
        writer.add_scalar("train/accuracy", train_acc, epoch)
        for idx, r in enumerate(train_recall):
            writer.add_scalar(f"train/{idx2label(idx)}_recall", r, epoch)
        for idx, f in enumerate(train_f1):
            writer.add_scalar(f"train/{idx2label(idx)}_f1", f, epoch)
        for idx, p in enumerate(train_precision):
            writer.add_scalar(f"train/{idx2label(idx)}_precision", p, epoch)
        
        # Validate
        val_loss, val_labels, val_logits = validate(
            model, val_loader, criterion, args.device
        )
        
        val_acc, val_recall, val_f1, val_precision = cal_metric(
            val_labels, val_logits
        )
        
        # Log validation metrics
        writer.add_scalar("val/loss", val_loss, epoch)
        writer.add_scalar("val/accuracy", val_acc, epoch)
        for idx, r in enumerate(val_recall):
            writer.add_scalar(f"val/{idx2label(idx)}_recall", r, epoch)
        for idx, f in enumerate(val_f1):
            writer.add_scalar(f"val/{idx2label(idx)}_f1", f, epoch)
        for idx, p in enumerate(val_precision):
            writer.add_scalar(f"val/{idx2label(idx)}_precision", p, epoch)
        
        # Print epoch summary
        print(f"\nTrain Loss: {train_loss:.4f} | Train Acc: {train_acc:.4f}")
        print(f"Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.4f}")
        print(f"Val F1 - Keep: {val_f1[0]:.4f}, Turn: {val_f1[1]:.4f}, BC: {val_f1[2]:.4f}")
        
        # Save checkpoint
        save_path = os.path.join(task_dir, f"epoch_{epoch}.pt")
        torch.save(model.state_dict(), save_path)
        
        # Save best model
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_path = os.path.join(task_dir, "best_model.pt")
            torch.save(model.state_dict(), best_path)
            print(f"✓ New best model saved! Accuracy: {best_val_acc:.4f}")
    
    writer.close()
    print(f"\n{'='*60}")
    print("Training completed!")
    print(f"Best validation accuracy: {best_val_acc:.4f}")
    print(f"Models saved to: {task_dir}")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()



"""
# Train text model
python train_unimodal.py \
  --modal text \
  --train_csv ./splits/train.csv \
  --val_csv ./splits/val.csv \
  --transcript_folder transcripts/ \
  --audio_folder /path/to/audio/ \
  --video_folder /path/to/videos/ \
  --t 2.0 \
  --batch_size 16 \
  --n_epoch 50 \
  --lr 1e-5

# Train audio model
python train_unimodal.py \
  --modal audio \
  --train_csv ./splits/train.csv \
  --val_csv ./splits/val.csv \
  --transcript_folder transcripts/ \
  --audio_folder /path/to/audio/ \
  --video_folder /path/to/videos/ \
  --t 2.0 \
  --batch_size 8 \
  --n_epoch 50

# Train video model
python train_unimodal.py \
  --modal video \
  --train_csv ./splits/train.csv \
  --val_csv ./splits/val.csv \
  --transcript_folder transcripts/ \
  --audio_folder /path/to/audio/ \
  --video_folder /path/to/videos/ \
  --t 2.0 \
  --max_frames 16 \
  --batch_size 4 \
  --n_epoch 50
```

**Output Structure:**
```
log/
  └── 2025-01-12_10-30-00_text_t2.0/
      ├── events.out.tfevents...  (TensorBoard)
      ├── epoch_0.pt
      ├── epoch_1.pt
      ├── ...
      └── best_model.pt

"""