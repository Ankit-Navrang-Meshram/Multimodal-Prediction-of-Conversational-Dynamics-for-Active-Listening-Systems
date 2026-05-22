


# ============================================================================
# File: dataloader_token.py
# DataLoader for Tokenized Multimodal Data
# ============================================================================

import os
import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
import cv2
import librosa
from tqdm import tqdm


class MultimodalTokenDataset(Dataset):
    """
    Dataset that loads and tokenizes video, audio, text on-the-fly
    """
    def __init__(
        self, 
        data_root, 
        split, 
        video_tokenizer,
        audio_tokenizer,
        text_tokenizer,
        n_video_frames=16,
        max_audio_length=80000,
        max_text_length=128
    ):
        assert split in ["train", "val", "test"]
        
        self.data_root = data_root
        self.split = split
        self.video_tokenizer = video_tokenizer
        self.audio_tokenizer = audio_tokenizer
        self.text_tokenizer = text_tokenizer
        self.n_video_frames = n_video_frames
        self.max_audio_length = max_audio_length
        self.max_text_length = max_text_length
        self.sampling_rate = 16000
        
        # Load metadata
        df = pd.read_csv(os.path.join(data_root, f"{split}.csv"), sep="\t")
        self.sentence_ids = df["sentence_id"].values
        self.texts = df["text"].values
        self.labels = df["label"].values
        
        self.audio_root = os.path.join(data_root, "audio")
        self.video_root = os.path.join(data_root, "video")
        
    def __len__(self):
        return len(self.sentence_ids)
    
    def load_video_frames(self, video_id, sentence_id):
        """Load video frames"""
        frames = []
        for i in range(self.n_video_frames):
            img_path = os.path.join(self.video_root, video_id, sentence_id, f"{i}.jpg")
            if not os.path.exists(img_path):
                break
            img = cv2.imread(img_path)
            if img is not None:
                img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                frames.append(img)
        
        # Pad if needed
        while len(frames) < self.n_video_frames:
            if len(frames) > 0:
                frames.append(frames[-1])
            else:
                frames.append(np.zeros((224, 224, 3), dtype=np.uint8))
        
        return frames[:self.n_video_frames]
    
    def load_audio(self, audio_path):
        """Load audio"""
        try:
            audio, _ = librosa.load(audio_path, sr=self.sampling_rate)
            # Truncate or pad
            if len(audio) > self.max_audio_length:
                audio = audio[:self.max_audio_length]
            else:
                audio = np.pad(audio, (0, self.max_audio_length - len(audio)))
            return audio
        except:
            return np.zeros(self.max_audio_length)
    
    def __getitem__(self, idx):
        sentence_id = self.sentence_ids[idx]
        text = self.texts[idx]
        label = self.labels[idx]
        
        video_id = "_".join(sentence_id.split("_")[:-2])
        
        # Load raw data
        frames = self.load_video_frames(video_id, sentence_id)
        audio_path = os.path.join(self.audio_root, video_id, f"{sentence_id}.mp3")
        audio = self.load_audio(audio_path)
        
        # Tokenize
        with torch.no_grad():
            # Video tokens
            video_inputs = self.video_tokenizer.processor(frames, return_tensors="pt")
            video_tokens = self.video_tokenizer(video_inputs['pixel_values'])
            
            # Audio tokens
            audio_inputs = self.audio_tokenizer.processor(
                audio, sampling_rate=self.sampling_rate, return_tensors="pt"
            )
            audio_tokens = self.audio_tokenizer(audio_inputs.input_values)
            
            # Text tokens
            text_inputs = self.text_tokenizer(
                text,
                max_length=self.max_text_length,
                padding='max_length',
                truncation=True,
                return_tensors="pt"
            )
        
        return {
            'video_tokens': video_tokens.squeeze(0),  # (T_v,)
            'audio_tokens': audio_tokens.squeeze(0),  # (T_a,)
            'text_ids': text_inputs['input_ids'].squeeze(0),  # (T_t,)
            'text_attention_mask': text_inputs['attention_mask'].squeeze(0),
            'label': label
        }


def collate_fn_token(batch):
    """Custom collate function for variable-length sequences"""
    video_tokens = torch.stack([x['video_tokens'] for x in batch])
    audio_tokens = torch.nn.utils.rnn.pad_sequence(
        [x['audio_tokens'] for x in batch], 
        batch_first=True, 
        padding_value=0
    )
    text_ids = torch.stack([x['text_ids'] for x in batch])
    text_attention_mask = torch.stack([x['text_attention_mask'] for x in batch])
    labels = torch.tensor([x['label'] for x in batch])
    
    return video_tokens, audio_tokens, text_ids, text_attention_mask, labels