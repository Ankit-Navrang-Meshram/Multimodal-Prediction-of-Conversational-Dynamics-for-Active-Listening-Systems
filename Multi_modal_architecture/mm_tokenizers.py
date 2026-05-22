# ============================================================================
# File: tokenizers.py
# Multimodal Tokenizers for Video, Audio, and Text
# ============================================================================

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoProcessor, AutoImageProcessor, Wav2Vec2FeatureExtractor
import numpy as np
import librosa


class VideoTokenizer(nn.Module):
    """
    Video Tokenizer using VideoMAE + VQ-VAE
    Converts video frames to discrete tokens
    """
    def __init__(self, vocab_size=8192, codebook_dim=768):
        super().__init__()
        from transformers import VideoMAEModel
        
        self.encoder = VideoMAEModel.from_pretrained("MCG-NJU/videomae-base")
        self.processor = AutoImageProcessor.from_pretrained("MCG-NJU/videomae-base")
        
        # Freeze encoder
        for param in self.encoder.parameters():
            param.requires_grad = False
        
        # VQ-VAE codebook
        self.vocab_size = vocab_size
        self.codebook_dim = codebook_dim
        self.codebook = nn.Embedding(vocab_size, codebook_dim)
        
        # Projection to codebook dimension
        self.proj = nn.Linear(768, codebook_dim)
        
        # Initialize codebook
        self.codebook.weight.data.uniform_(-1.0 / vocab_size, 1.0 / vocab_size)
        
    def forward(self, pixel_values):
        """
        Args:
            pixel_values: (B, T, C, H, W) or preprocessed
        Returns:
            tokens: (B, T) discrete token indices
        """
        device = next(self.encoder.parameters()).device
        pixel_values = pixel_values.to(device)
        with torch.no_grad():
            outputs = self.encoder(pixel_values)
            features = outputs.last_hidden_state.mean(dim=1)  # (B, 768)
        
        # Project to codebook dimension
        z = self.proj(features)  # (B, codebook_dim)
        
        # Quantize
        distances = torch.cdist(z, self.codebook.weight)  # (B, vocab_size)
        tokens = torch.argmin(distances, dim=-1)  # (B,)
        
        return tokens.unsqueeze(1)  # (B, 1) for single frame per sample
    
    def encode_frames(self, frames):
        """
        Encode multiple frames
        Args:
            frames: list of (C, H, W) or (H, W, C) numpy arrays
        Returns:
            tokens: (1, n_frames) 
        """
        # Process frames
        inputs = self.processor(frames, return_tensors="pt")
        pixel_values = inputs['pixel_values']
        
        tokens = []
        with torch.no_grad():
            for i in range(0, len(frames), 16):  # Process in chunks of 16
                chunk = pixel_values[i:min(i+16, len(frames))]
                chunk_tokens = self.forward(chunk)
                tokens.append(chunk_tokens)
        
        return torch.cat(tokens, dim=1)  # (1, n_frames)


class SpeechTokenizer(nn.Module):
    """
    Speech Tokenizer using HuBERT + K-means clustering
    Converts audio to discrete tokens
    """
    def __init__(self, vocab_size=1024):
        super().__init__()
        from transformers import HubertModel
        
        self.encoder = HubertModel.from_pretrained("facebook/hubert-base-ls960")
        #self.processor = AutoProcessor.from_pretrained("facebook/hubert-base-ls960")
        self.processor = Wav2Vec2FeatureExtractor.from_pretrained("facebook/hubert-base-ls960")

        # Freeze encoder
        for param in self.encoder.parameters():
            param.requires_grad = False
        
        self.vocab_size = vocab_size
        # K-means centers (learned during preprocessing)
        self.register_buffer('kmeans_centers', torch.randn(vocab_size, 768))
        
    def forward(self, input_values):
        """
        Args:
            input_values: (B, L) audio waveform
        Returns:
            tokens: (B, T) discrete token indices
        """
        with torch.no_grad():
            outputs = self.encoder(input_values)
            features = outputs.last_hidden_state  # (B, T, 768)
        
        # Quantize using k-means
        B, T, D = features.shape
        features_flat = features.reshape(-1, D)
        
        distances = torch.cdist(features_flat, self.kmeans_centers)
        tokens = torch.argmin(distances, dim=-1)
        tokens = tokens.reshape(B, T)
        
        # Downsample by factor of 2 to reduce sequence length
        tokens = tokens[:, ::2]
        
        return tokens
    
    def encode_audio(self, audio_path, sr=16000):
        """
        Encode audio file
        Args:
            audio_path: path to audio file
            sr: sampling rate
        Returns:
            tokens: (1, T)
        """
        audio, _ = librosa.load(audio_path, sr=sr)
        inputs = self.processor(audio, sampling_rate=sr, return_tensors="pt")
        return self.forward(inputs.input_values)