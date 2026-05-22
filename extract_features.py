# """
# extract_features.py

# Script to extract and save features from text, audio, and video modalities.
# Saves preprocessed features as .pt files for faster training.

# Usage:
#     python extract_features.py --data_root ./dataset --output_dir ./features --modal all
# """

# import os
# import argparse
# from pathlib import Path
# from typing import Dict, Any, Optional
# import pandas as pd
# import librosa
# import cv2
# from tqdm import tqdm
# import warnings

# import torch
# from transformers import AutoTokenizer, AutoProcessor, AutoImageProcessor


# class FeatureExtractor:
#     """Extract and save features from multimodal data."""
    
#     def __init__(
#         self,
#         data_root: str,
#         output_dir: str,
#         modal: str = "all",
#         n_imgs: int = 16,
#         max_audio_length: Optional[int] = None,
#         max_text_length: int = 512
#     ):
#         """
#         Args:
#             data_root: Root directory containing raw data
#             output_dir: Directory to save extracted features
#             modal: Modality to extract ('text', 'audio', 'video', 'all')
#             n_imgs: Number of frames to extract from video
#             max_audio_length: Maximum audio length in samples
#             max_text_length: Maximum text sequence length
#         """
#         assert modal in ["audio", "video", "text", "all"]
        
#         self.data_root = Path(data_root)
#         self.output_dir = Path(output_dir)
#         self.modal = modal
#         self.n_imgs = n_imgs
#         self.max_audio_length = max_audio_length
#         self.max_text_length = max_text_length
        
#         # Create output directories
#         self.output_dir.mkdir(parents=True, exist_ok=True)
        
#         # Initialize processors based on modal
#         self._init_processors()
        
#     def _init_processors(self):
#         """Initialize processors for each modality."""
#         print("Initializing processors...")
        
#         self.tokenizer = None
#         self.audio_processor = None
#         self.video_processor = None
#         self.sampling_rate = None
        
#         if self.modal in ["text", "all"]:
#             self.tokenizer = AutoTokenizer.from_pretrained("openai-community/gpt2")
#             self.tokenizer.pad_token = self.tokenizer.eos_token
#             print("✓ Text tokenizer loaded")
            
#         if self.modal in ["audio", "all"]:
#             self.audio_processor = AutoProcessor.from_pretrained("facebook/hubert-large-ls960-ft")
#             self.sampling_rate = self.audio_processor.feature_extractor.sampling_rate
#             print("✓ Audio processor loaded")
            
#         if self.modal in ["video", "all"]:
#             self.video_processor = AutoImageProcessor.from_pretrained("MCG-NJU/videomae-base")
#             print("✓ Video processor loaded")
    
#     def extract_text_features(self, text: str) -> Dict[str, torch.Tensor]:
#         """Extract text features using tokenizer."""
#         try:
#             encoded = self.tokenizer(
#                 text,
#                 truncation=True,
#                 max_length=self.max_text_length,
#                 padding='max_length',
#                 return_tensors="pt"
#             )
#             return {
#                 "input_ids": encoded["input_ids"].squeeze(0),
#                 "attention_mask": encoded["attention_mask"].squeeze(0)
#             }
#         except Exception as e:
#             warnings.warn(f"Error extracting text features: {e}")
#             return None
    
#     def extract_audio_features(self, audio_path: str) -> torch.Tensor:
#         """Extract audio features using audio processor."""
#         try:
#             # Load audio
#             audio, _ = librosa.load(audio_path, sr=self.sampling_rate)
            
#             # Limit audio length if specified
#             if self.max_audio_length and len(audio) > self.max_audio_length:
#                 audio = audio[:self.max_audio_length]
            
#             # Process audio
#             processed = self.audio_processor(
#                 audio,
#                 sampling_rate=self.sampling_rate,
#                 return_tensors="pt"
#             ).input_values
            
#             return processed.squeeze(0)
            
#         except Exception as e:
#             warnings.warn(f"Error extracting audio features from {audio_path}: {e}")
#             return None
    
#     def extract_video_features(self, img_paths: list) -> torch.Tensor:
#         """Extract video features from frame paths."""
#         try:
#             imgs = []
#             for path in img_paths:
#                 img = cv2.imread(path)
#                 if img is None:
#                     raise ValueError(f"Failed to load image: {path}")
#                 # Convert BGR to RGB
#                 img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
#                 imgs.append(img)
            
#             # Process images
#             processed = self.video_processor(imgs, return_tensors="pt")
#             return processed["pixel_values"].squeeze(0)
            
#         except Exception as e:
#             warnings.warn(f"Error extracting video features: {e}")
#             return None
    
#     def process_split(self, split: str):
#         """
#         Process and save features for a data split.
        
#         Args:
#             split: One of ['train', 'val', 'test']
#         """
#         print(f"\n{'='*60}")
#         print(f"Processing {split} split...")
#         print(f"{'='*60}")
        
#         # Load CSV
#         csv_path = self.data_root / f"{split}.csv"
#         if not csv_path.exists():
#             raise FileNotFoundError(f"CSV file not found: {csv_path}")
        
#         df = pd.read_csv(csv_path, sep="\t")
        
#         # Create output directory for this split
#         split_dir = self.output_dir / split
#         split_dir.mkdir(parents=True, exist_ok=True)
        
#         # Statistics
#         stats = {
#             "total": len(df),
#             "processed": 0,
#             "skipped": 0,
#             "text_extracted": 0,
#             "audio_extracted": 0,
#             "video_extracted": 0
#         }
        
#         # Process each sample
#         for idx, row in tqdm(df.iterrows(), total=len(df), desc=f"Extracting {split} features"):
#             sid = row["sentence_id"]
#             text = row["text"]
#             label = row["label"]
            
#             features = {
#                 "sentence_id": sid,
#                 "label": label
#             }
            
#             video_id = "_".join(sid.split("_")[:-2])
#             skip_sample = False
            
#             # Extract text features
#             if self.modal in ["text", "all"]:
#                 text_feat = self.extract_text_features(text)
#                 if text_feat is not None:
#                     features["text"] = text_feat
#                     stats["text_extracted"] += 1
#                 else:
#                     skip_sample = True
            
#             # Extract audio features
#             if self.modal in ["audio", "all"]:
#                 audio_path = self.data_root / "audio" / video_id / f"{sid}.mp3"
#                 if audio_path.exists():
#                     audio_feat = self.extract_audio_features(str(audio_path))
#                     if audio_feat is not None:
#                         features["audio"] = audio_feat
#                         stats["audio_extracted"] += 1
#                     else:
#                         if self.modal == "audio":
#                             skip_sample = True
#                 else:
#                     warnings.warn(f"Audio file not found: {audio_path}")
#                     if self.modal == "audio":
#                         skip_sample = True
            
#             # Extract video features
#             if self.modal in ["video", "all"]:
#                 img_paths = []
#                 for i in range(self.n_imgs):
#                     img_path = self.data_root / "video" / video_id / sid / f"{i}.jpg"
#                     if not img_path.exists():
#                         break
#                     img_paths.append(str(img_path))
                
#                 if len(img_paths) == self.n_imgs:
#                     video_feat = self.extract_video_features(img_paths)
#                     if video_feat is not None:
#                         features["video"] = video_feat
#                         stats["video_extracted"] += 1
#                     else:
#                         if self.modal == "video":
#                             skip_sample = True
#                 else:
#                     if self.modal == "video":
#                         skip_sample = True
            
#             # Save features
#             if not skip_sample:
#                 feature_path = split_dir / f"{sid}.pt"
#                 torch.save(features, feature_path)
#                 stats["processed"] += 1
#             else:
#                 stats["skipped"] += 1
        
#         # Print statistics
#         print(f"\n{split.upper()} Split Statistics:")
#         print(f"  Total samples: {stats['total']}")
#         print(f"  Processed: {stats['processed']}")
#         print(f"  Skipped: {stats['skipped']}")
#         if self.modal in ["text", "all"]:
#             print(f"  Text features: {stats['text_extracted']}")
#         if self.modal in ["audio", "all"]:
#             print(f"  Audio features: {stats['audio_extracted']}")
#         if self.modal in ["video", "all"]:
#             print(f"  Video features: {stats['video_extracted']}")
        
#         # Save metadata
#         metadata = {
#             "split": split,
#             "modal": self.modal,
#             "n_imgs": self.n_imgs,
#             "max_audio_length": self.max_audio_length,
#             "max_text_length": self.max_text_length,
#             "stats": stats
#         }
#         torch.save(metadata, split_dir / "metadata.pt")
#         print(f"✓ Saved metadata to {split_dir / 'metadata.pt'}")
        
#         return stats
    
#     def extract_all_splits(self):
#         """Extract features for all splits."""
#         print("\n" + "="*60)
#         print("FEATURE EXTRACTION")
#         print("="*60)
#         print(f"Data root: {self.data_root}")
#         print(f"Output dir: {self.output_dir}")
#         print(f"Modality: {self.modal}")
#         print("="*60)
        
#         all_stats = {}
#         for split in ["train", "val", "test"]:
#             try:
#                 stats = self.process_split(split)
#                 all_stats[split] = stats
#             except Exception as e:
#                 print(f"Error processing {split}: {e}")
        
#         # Save overall statistics
#         torch.save(all_stats, self.output_dir / "extraction_stats.pt")
        
#         print("\n" + "="*60)
#         print("EXTRACTION COMPLETE")
#         print("="*60)
#         print(f"Features saved to: {self.output_dir}")
        
#         # Calculate total size
#         total_size = sum(f.stat().st_size for f in self.output_dir.rglob("*.pt"))
#         print(f"Total size: {total_size / (1024**3):.2f} GB")
        
#         return all_stats


# def main():
#     parser = argparse.ArgumentParser(description="Extract features from multimodal data")
#     parser.add_argument("--data_root", type=str, default="./dataset",
#                        help="Root directory containing raw data")
#     parser.add_argument("--output_dir", type=str, default="./features",
#                        help="Directory to save extracted features")
#     parser.add_argument("--modal", type=str, default="all",
#                        choices=["text", "audio", "video", "all"],
#                        help="Modality to extract")
#     parser.add_argument("--n_imgs", type=int, default=16,
#                        help="Number of frames to extract from video")
#     parser.add_argument("--max_audio_length", type=int, default=None,
#                        help="Maximum audio length in samples")
#     parser.add_argument("--max_text_length", type=int, default=512,
#                        help="Maximum text sequence length")
    
#     args = parser.parse_args()
    
#     # Create extractor and process
#     extractor = FeatureExtractor(
#         data_root=args.data_root,
#         output_dir=args.output_dir,
#         modal=args.modal,
#         n_imgs=args.n_imgs,
#         max_audio_length=args.max_audio_length,
#         max_text_length=args.max_text_length
#     )
    
#     extractor.extract_all_splits()


# if __name__ == "__main__":
#     main()



import os
import pandas as pd
import librosa
import cv2
from tqdm import tqdm
import torch
from transformers import AutoTokenizer, AutoProcessor, AutoImageProcessor

# Initialize processors
tokenizer = AutoTokenizer.from_pretrained("openai-community/gpt2")
tokenizer.pad_token = tokenizer.eos_token
audio_processor = AutoProcessor.from_pretrained("facebook/hubert-large-ls960-ft")
video_processor = AutoImageProcessor.from_pretrained("MCG-NJU/videomae-base")

def extract_and_save_features(data_root, split):
    """Extract features and save them to .pt files"""
    print(f"Processing {split} split...")
    
    # Read CSV
    df = pd.read_csv(os.path.join(data_root, f"{split}.csv"), sep="\t")
    sid = df["sentence_id"].values
    text = df["text"].values
    label = df["label"].values
    
    # Create output directory
    features_dir = os.path.join(data_root, "features", split)
    os.makedirs(features_dir, exist_ok=True)
    
    # Paths
    audio_root = os.path.join(data_root, "audio")
    video_root = os.path.join(data_root, "video")
    sampling_rate = audio_processor.feature_extractor.sampling_rate
    n_imgs = 16
    
    # Store metadata for valid samples
    valid_samples = []
    
    for (_sid, _text, _label) in tqdm(zip(sid, text, label), total=len(sid)):
        video_id = "_".join(_sid.split("_")[:-2])
        
        # Check if video has all required frames
        img_paths = []
        for i in range(n_imgs):
            img_path = os.path.join(video_root, video_id, _sid, f"{i}.jpg")
            if not os.path.exists(img_path):
                break
            img_paths.append(img_path)
        
        if len(img_paths) < n_imgs:
            continue  # Skip samples without complete video frames
        
        # Process text
        text_ids = tokenizer(_text)
        text_features = {
            "input_ids": text_ids["input_ids"],
            "attention_mask": text_ids["attention_mask"]
        }
        
        # Process audio
        audio_path = os.path.join(audio_root, video_id, f"{_sid}.mp3")
        audio, _ = librosa.load(audio_path, sr=sampling_rate)
        audio_features = audio_processor(audio, sampling_rate=sampling_rate, return_tensors="pt").input_values
        audio_features = audio_features.squeeze(0)  # Remove batch dimension
        
        # Process video
        imgs = [cv2.imread(path)[:,:,::-1] for path in img_paths]
        video_features = video_processor(imgs, return_tensors="pt")
        video_features = video_features["pixel_values"].squeeze(0)  # Remove batch dimension
        
        # Save features
        feature_file = os.path.join(features_dir, f"{_sid}.pt")
        torch.save({
            "text": text_features,
            "audio": audio_features,
            "video": video_features,
            "label": _label
        }, feature_file)
        
        valid_samples.append({
            "sentence_id": _sid,
            "label": _label,
            "feature_file": f"{_sid}.pt"
        })
    
    # Save metadata
    metadata_df = pd.DataFrame(valid_samples)
    metadata_df.to_csv(os.path.join(features_dir, "metadata.csv"), index=False)
    print(f"Saved {len(valid_samples)} samples to {features_dir}")

if __name__ == "__main__":
    data_root = "./dataset"
    
    # Extract features for all splits
    for split in ["train", "val", "test"]:
        extract_and_save_features(data_root, split)
    
    print("Feature extraction completed!")