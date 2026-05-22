import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import json
import argparse
import numpy as np
from tqdm import tqdm
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, classification_report, confusion_matrix

import torch
from torch import nn
from torch.utils.data import DataLoader
from transformers import AutoTokenizer, AutoProcessor, AutoImageProcessor

# Import the newly defined DNCC model and dataloader
from model import DNCCEnsembleModel
from dataloader import MultiModalDataset, collate_fn

parser = argparse.ArgumentParser()
# model params
parser.add_argument("--t_ckpt_path", type=str, default=None, help="Text model checkpoint path")
parser.add_argument("--a_ckpt_path", type=str, default=None, help="Audio model checkpoint path")
parser.add_argument("--v_ckpt_path", type=str, default=None, help="Vision model checkpoint path")
parser.add_argument("--model_weights", type=str, required=True, help="Path to trained DNCC model weights (.pt file)")
parser.add_argument("--device", type=str, default="cuda", help="Device to use (cuda/cpu)")

# data params
parser.add_argument("--data_root", type=str, default="dataset/", help="Root directory of dataset")
parser.add_argument("--batch_size", type=int, default=1, help="Batch size for inference")
parser.add_argument("--n_workers", type=int, default=4, help="Number of workers for data loading")

# output params
parser.add_argument("--output_dir", type=str, default="results/", help="Directory to save results")
parser.add_argument("--save_predictions", action="store_true", help="Save individual predictions")

# ablation study params
parser.add_argument("--drop_text", action="store_true", help="Drop text modality for ablation study")
parser.add_argument("--drop_audio", action="store_true", help="Drop audio modality for ablation study")
parser.add_argument("--drop_vision", action="store_true", help="Drop vision modality for ablation study")

args = parser.parse_args()


def load_processors():
    """Helper to initialize processors needed by the dataloader"""
    tokenizer = AutoTokenizer.from_pretrained("openai-community/gpt2")
    tokenizer.pad_token = tokenizer.eos_token
    audio_processor = AutoProcessor.from_pretrained("facebook/hubert-large-ls960-ft")
    video_processor = AutoImageProcessor.from_pretrained("MCG-NJU/videomae-base")
    return tokenizer, None, audio_processor, video_processor


def idx2label(idx):
    """Convert index to label name"""
    return ["keep", "turn", "bc"][idx]


def cal_metrics(all_labels, all_logits):
    """Calculate evaluation metrics"""
    accuracy = accuracy_score(all_labels, all_logits)
    recall = recall_score(all_labels, all_logits, average=None, zero_division=0)
    f1 = f1_score(all_labels, all_logits, average=None, zero_division=0)
    precision = precision_score(all_labels, all_logits, average=None, zero_division=0)
    
    recall_macro = recall_score(all_labels, all_logits, average='macro', zero_division=0)
    f1_macro = f1_score(all_labels, all_logits, average='macro', zero_division=0)
    precision_macro = precision_score(all_labels, all_logits, average='macro', zero_division=0)
    
    recall_weighted = recall_score(all_labels, all_logits, average='weighted', zero_division=0)
    f1_weighted = f1_score(all_labels, all_logits, average='weighted', zero_division=0)
    precision_weighted = precision_score(all_labels, all_logits, average='weighted', zero_division=0)
    
    return {
        'accuracy': accuracy,
        'per_class': {'recall': recall, 'f1': f1, 'precision': precision},
        'macro': {'recall': recall_macro, 'f1': f1_macro, 'precision': precision_macro},
        'weighted': {'recall': recall_weighted, 'f1': f1_weighted, 'precision': precision_weighted}
    }


def print_metrics(metrics, ablation_info=""):
    """Pretty print metrics"""
    print("\n" + "="*60)
    print("EVALUATION RESULTS")
    if ablation_info:
        print(f"Ablation: {ablation_info}")
    print("="*60)
    print(f"\nOverall Accuracy: {metrics['accuracy']:.4f}")
    
    print("\n" + "-"*60)
    print("Per-Class Metrics:")
    print("-"*60)
    print(f"{'Class':<10} {'Precision':<12} {'Recall':<12} {'F1-Score':<12}")
    print("-"*60)
    for idx in range(len(metrics['per_class']['precision'])):
        label = idx2label(idx)
        precision = metrics['per_class']['precision'][idx]
        recall = metrics['per_class']['recall'][idx]
        f1 = metrics['per_class']['f1'][idx]
        print(f"{label:<10} {precision:<12.4f} {recall:<12.4f} {f1:<12.4f}")
    
    print("\n" + "-"*60)
    print("Macro-Averaged Metrics:")
    print("-"*60)
    print(f"Precision: {metrics['macro']['precision']:.4f}")
    print(f"Recall:    {metrics['macro']['recall']:.4f}")
    print(f"F1-Score:  {metrics['macro']['f1']:.4f}")
    
    print("\n" + "-"*60)
    print("Weighted-Averaged Metrics:")
    print("-"*60)
    print(f"Precision: {metrics['weighted']['precision']:.4f}")
    print(f"Recall:    {metrics['weighted']['recall']:.4f}")
    print(f"F1-Score:  {metrics['weighted']['f1']:.4f}")
    print("="*60 + "\n")


def save_metrics(metrics, filepath="evaluation_results.txt", ablation_info=""):
    """Save metrics to a text file"""
    with open(filepath, "w") as f:
        f.write("="*60 + "\n")
        f.write("EVALUATION RESULTS\n")
        if ablation_info:
            f.write(f"Ablation: {ablation_info}\n")
        f.write("="*60 + "\n")
        f.write(f"\nOverall Accuracy: {metrics['accuracy']:.4f}\n")
        
        f.write("\n" + "-"*60 + "\n")
        f.write("Per-Class Metrics:\n")
        f.write("-"*60 + "\n")
        f.write(f"{'Class':<10} {'Precision':<12} {'Recall':<12} {'F1-Score':<12}\n")
        f.write("-"*60 + "\n")
        for idx in range(len(metrics['per_class']['precision'])):
            label = idx2label(idx)
            precision = metrics['per_class']['precision'][idx]
            recall = metrics['per_class']['recall'][idx]
            f1 = metrics['per_class']['f1'][idx]
            f.write(f"{label:<10} {precision:<12.4f} {recall:<12.4f} {f1:<12.4f}\n")
        
        f.write("\n" + "-"*60 + "\n")
        f.write("Macro-Averaged Metrics:\n")
        f.write("-"*60 + "\n")
        f.write(f"Precision: {metrics['macro']['precision']:.4f}\n")
        f.write(f"Recall:    {metrics['macro']['recall']:.4f}\n")
        f.write(f"F1-Score:  {metrics['macro']['f1']:.4f}\n")
        
        f.write("\n" + "-"*60 + "\n")
        f.write("Weighted-Averaged Metrics:\n")
        f.write("-"*60 + "\n")
        f.write(f"Precision: {metrics['weighted']['precision']:.4f}\n")
        f.write(f"Recall:    {metrics['weighted']['recall']:.4f}\n")
        f.write(f"F1-Score:  {metrics['weighted']['f1']:.4f}\n")
        f.write("="*60 + "\n")
    print(f"Metrics saved to {filepath}")


def get_ablation_info(drop_text, drop_audio, drop_vision):
    """Generate ablation information string"""
    dropped = []
    if drop_text: dropped.append("Text")
    if drop_audio: dropped.append("Audio")
    if drop_vision: dropped.append("Vision")
    return f"Dropped modalities: {', '.join(dropped)}" if dropped else "All modalities enabled"


def main():
    os.makedirs(args.output_dir, exist_ok=True)
    ablation_info = get_ablation_info(args.drop_text, args.drop_audio, args.drop_vision)
    print(f"\n{ablation_info}\n")
    
    print("Loading processors...")
    tokenizer, _, audio_processor, video_processor = load_processors()
    
    print(f"Loading test dataset from {args.data_root}...")
    test_set = MultiModalDataset(
        data_root=args.data_root,
        split="test",
        tokenizer=tokenizer,
        audio_processor=audio_processor,
        video_processor=video_processor,
    )
    test_loader = DataLoader(
        test_set, batch_size=args.batch_size, shuffle=False, 
        collate_fn=collate_fn, num_workers=args.n_workers
    )
    print(f"Test dataset size: {len(test_set)}")
    
    print("Loading DNCC model...")
    model = DNCCEnsembleModel(
        text_ckpt=args.t_ckpt_path,
        audio_ckpt=args.a_ckpt_path,
        vision_ckpt=args.v_ckpt_path
    ).to(args.device)
    
    print(f"Loading trained weights from {args.model_weights}...")
    # Load the state dict (assuming it was saved via model.state_dict())
    model.load_state_dict(torch.load(args.model_weights, map_location=args.device), strict=False)
    model.eval()
    print(f"Model loaded successfully on {args.device}")
    
    print("\nRunning inference on test set...")
    all_labels, all_logits, all_predictions = [], [], []
    
    with torch.no_grad():
        for i, (text, audio, vision, y) in enumerate(tqdm(test_loader, desc="Testing")):
            if audio is None:
                print(f"Warning: Skipping batch {i} due to None audio")
                continue
            
            # Move base tensors to device
            text = text["input_ids"].to(args.device)
            audio = audio.to(args.device)
            vision = vision["pixel_values"].to(args.device)
            y = y.to(args.device)
            
            # ============ ABLATION STUDY: DROP MODALITIES ============
            # Rather than passing 'None', we pass tensors of zeros so the cross-attention layers don't fail
            if args.drop_text: text = torch.zeros_like(text)
            if args.drop_audio: audio = torch.zeros_like(audio)
            if args.drop_vision: vision = torch.zeros_like(vision)
            # =========================================================
            
            # In eval mode, DNCC returns averaged ensemble probabilities
            pred_probs = model(text, audio, vision)
            pred_labels = pred_probs.argmax(dim=1)
            
            all_labels.extend(y.cpu().numpy())
            all_logits.extend(pred_labels.cpu().numpy())
            
            if args.save_predictions:
                for j in range(len(y)):
                    all_predictions.append({
                        'true_label': int(y[j].cpu().numpy()),
                        'true_label_name': idx2label(int(y[j].cpu().numpy())),
                        'predicted_label': int(pred_labels[j].cpu().numpy()),
                        'predicted_label_name': idx2label(int(pred_labels[j].cpu().numpy())),
                        'probabilities': pred_probs[j].cpu().numpy().tolist()
                    })
    
    metrics = cal_metrics(np.array(all_labels), np.array(all_logits))
    print_metrics(metrics, ablation_info)
    
    # Save results handling
    dropped_mods = []
    if args.drop_text: dropped_mods.append("no_text")
    if args.drop_audio: dropped_mods.append("no_audio")
    if args.drop_vision: dropped_mods.append("no_vision")
    
    prefix = "DNCC_" + ("_".join(dropped_mods) if dropped_mods else "all_modalities")
    txt_filename = os.path.join(args.output_dir, f"{prefix}_results.txt")
    
    save_metrics(metrics, txt_filename, ablation_info)

    if args.save_predictions:
        json_filename = os.path.join(args.output_dir, f"{prefix}_predictions.json")
        with open(json_filename, 'w') as f:
            json.dump(all_predictions, f, indent=4)
        print(f"Predictions saved to {json_filename}")
    
    print("\nInference completed successfully!")

if __name__ == "__main__":
    main()