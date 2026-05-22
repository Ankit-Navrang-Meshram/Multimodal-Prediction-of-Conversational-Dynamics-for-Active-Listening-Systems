import os
import json
import argparse
import numpy as np
from tqdm import tqdm
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, classification_report, confusion_matrix
import torch
from torch import nn
from torch.utils.data import DataLoader
from model.mm import LanguageAudioVisionModel, load_processors
from dataloader import MultiModalDataset, collate_fn

parser = argparse.ArgumentParser()
# model params
parser.add_argument("--t_ckpt_path", type=str, help="Text model checkpoint path")
parser.add_argument("--a_ckpt_path", type=str, help="Audio model checkpoint path")
parser.add_argument("--v_ckpt_path", type=str, help="Vision model checkpoint path")
parser.add_argument("--model_weights", type=str, required=True, help="Path to trained model weights (.pt file)")
parser.add_argument("--fusion_module", type=str, default=None, help="Fusion module type")
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

def idx2label(idx):
    """Convert index to label name"""
    return ["keep", "turn", "bc"][idx]

def cal_metrics(all_labels, all_logits):
    """Calculate evaluation metrics"""
    accuracy = accuracy_score(all_labels, all_logits)
    recall = recall_score(all_labels, all_logits, average=None, zero_division=0)
    f1 = f1_score(all_labels, all_logits, average=None, zero_division=0)
    precision = precision_score(all_labels, all_logits, average=None, zero_division=0)
    
    # Also calculate macro and weighted averages
    recall_macro = recall_score(all_labels, all_logits, average='macro', zero_division=0)
    f1_macro = f1_score(all_labels, all_logits, average='macro', zero_division=0)
    precision_macro = precision_score(all_labels, all_logits, average='macro', zero_division=0)
    
    recall_weighted = recall_score(all_labels, all_logits, average='weighted', zero_division=0)
    f1_weighted = f1_score(all_labels, all_logits, average='weighted', zero_division=0)
    precision_weighted = precision_score(all_labels, all_logits, average='weighted', zero_division=0)
    
    return {
        'accuracy': accuracy,
        'per_class': {
            'recall': recall,
            'f1': f1,
            'precision': precision
        },
        'macro': {
            'recall': recall_macro,
            'f1': f1_macro,
            'precision': precision_macro
        },
        'weighted': {
            'recall': recall_weighted,
            'f1': f1_weighted,
            'precision': precision_weighted
        }
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
    if drop_text:
        dropped.append("Text")
    if drop_audio:
        dropped.append("Audio")
    if drop_vision:
        dropped.append("Vision")
    
    if not dropped:
        return "All modalities enabled"
    else:
        return f"Dropped modalities: {', '.join(dropped)}"

def main():
    # Create ablation information
    ablation_info = get_ablation_info(args.drop_text, args.drop_audio, args.drop_vision)
    print(f"\n{ablation_info}\n")
    
    # Load processors
    print("Loading processors...")
    tokenizer, _, audio_processor, video_processor = load_processors()
    
    # Load test dataset
    print(f"Loading test dataset from {args.data_root}...")
    test_set = MultiModalDataset(
        data_root=args.data_root,
        split="test",
        tokenizer=tokenizer,
        audio_processor=audio_processor,
        video_processor=video_processor,
    )
    
    test_loader = DataLoader(
        test_set, 
        batch_size=args.batch_size, 
        shuffle=False, 
        collate_fn=collate_fn, 
        num_workers=args.n_workers
    )
    
    print(f"Test dataset size: {len(test_set)}")
    
    # Load model
    print("Loading model...")
    model = LanguageAudioVisionModel(
        text_ckpt_path=args.t_ckpt_path,
        audio_ckpt_path=args.a_ckpt_path,
        vision_ckpt_path=args.v_ckpt_path,
        fusion_module=args.fusion_module
    ).to(args.device)
    
    # Load trained weights
    print(f"Loading model weights from {args.model_weights}...")
    checkpoint = torch.load(args.model_weights, map_location=args.device)
    model.load_state_dict(checkpoint)
    model.eval()
    print(f"Model loaded successfully on {args.device}")
    
    # Inference
    print("\nRunning inference on test set...")
    all_labels = []
    all_logits = []
    all_predictions = []
    
    with torch.no_grad():
        for i, (text, audio, vision, y) in enumerate(tqdm(test_loader, desc="Testing")):
            # Skip if audio is None
            if audio is None:
                print(f"Warning: Skipping batch {i} due to None audio")
                continue
            
            # Move to device
            text = text["input_ids"].to(args.device)
            audio = audio.to(args.device)
            vision = vision["pixel_values"].to(args.device)
            y = y.to(args.device)
            
            # ============ ABLATION STUDY: DROP MODALITIES ============
            # Set modalities to None based on ablation flags
            text_input = None if args.drop_text else text
            audio_input = None if args.drop_audio else audio
            vision_input = None if args.drop_vision else vision
            
            # Forward pass with potentially dropped modalities
            pred = model(text_input, audio_input, vision_input)
            # =========================================================
            
            pred_labels = pred.argmax(dim=1)
            
            # Store results
            all_labels.extend(y.cpu().numpy())
            all_logits.extend(pred_labels.cpu().numpy())
            
            # Store individual predictions if requested
            if args.save_predictions:
                batch_predictions = []
                for j in range(len(y)):
                    batch_predictions.append({
                        'true_label': int(y[j].cpu().numpy()),
                        'true_label_name': idx2label(int(y[j].cpu().numpy())),
                        'predicted_label': int(pred_labels[j].cpu().numpy()),
                        'predicted_label_name': idx2label(int(pred_labels[j].cpu().numpy())),
                        'probabilities': pred[j].cpu().numpy().tolist()
                    })
                all_predictions.extend(batch_predictions)
    
    # Convert to numpy arrays
    all_labels = np.array(all_labels)
    all_logits = np.array(all_logits)
    
    # Calculate metrics
    print("\nCalculating metrics...")
    metrics = cal_metrics(all_labels, all_logits)
    
    # Print metrics
    print_metrics(metrics, ablation_info)
    
    # Create filename based on ablation study
    dropped_modalities = []
    if args.drop_text:
        dropped_modalities.append("no_text")
    if args.drop_audio:
        dropped_modalities.append("no_audio")
    if args.drop_vision:
        dropped_modalities.append("no_vision")
    
    if dropped_modalities:
        filename = f'./results/{args.fusion_module}_{"_".join(dropped_modalities)}_results.txt'
    else:
        filename = f'./results/{args.fusion_module}_all_modalities_results.txt'
    
    # Save results
    os.makedirs('./results', exist_ok=True)
    save_metrics(metrics, filename, ablation_info)
    
    print("\nInference completed successfully!")

if __name__ == "__main__":
    main()




"""

# Evaluate with all modalities
python evaluateMissingModality.py --model_weights model.pt --fusion_module attention

# Drop text modality
python evaluateMissingModality.py --model_weights model.pt --fusion_module attention --drop_text

# Drop audio modality
python evaluateMissingModality.py --model_weights model.pt --fusion_module attention --drop_audio

# Drop vision modality
python evaluateMissingModality.py --model_weights model.pt --fusion_module attention --drop_vision

# Drop multiple modalities (e.g., text and audio)
python evaluateMissingModality.py --model_weights model.pt --fusion_module attention --drop_text --drop_audio

# Drop all modalities (extreme case)
python evaluateMissingModality.py --model_weights model.pt --fusion_module attention --drop_text --drop_audio --drop_vision


"""