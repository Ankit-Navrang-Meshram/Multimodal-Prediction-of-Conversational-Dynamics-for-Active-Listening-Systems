"""
Unimodal models for text, audio, and video classification.
Uses pretrained transformers (GPT-2, HuBERT, VideoMAE) with classification heads.
"""

import torch
from torch import nn
from transformers import AutoModel


class LanguageModel(nn.Module):
    """GPT-2 based text model for classification."""
    
    def __init__(self, num_classes=3, return_embeddings=False, freeze_encoder=False):
        """
        Args:
            num_classes: Number of output classes (default: 3 for KEEP, TURN, BACKCHANNEL)
            return_embeddings: If True, return embeddings instead of class logits
            freeze_encoder: If True, freeze the pretrained encoder weights
        """
        super().__init__()
        self.encoder = AutoModel.from_pretrained("openai-community/gpt2")
        self.return_embeddings = return_embeddings
        self.hidden_size = 768
        
        # Freeze encoder if specified
        if freeze_encoder:
            for param in self.encoder.parameters():
                param.requires_grad = False
        
        if not return_embeddings:
            self.classifier = nn.Sequential(
                nn.Linear(self.hidden_size, 256),
                nn.ReLU(),
                nn.Dropout(0.1),
                nn.Linear(256, num_classes)
            )
    
    def forward(self, input_ids, attention_mask=None):
        """
        Args:
            input_ids: Token IDs (batch_size, seq_len)
            attention_mask: Attention mask (batch_size, seq_len)
        
        Returns:
            If return_embeddings: (batch_size, hidden_size)
            Else: (batch_size, num_classes)
        """
        outputs = self.encoder(input_ids=input_ids, attention_mask=attention_mask)
        
        # Mean pooling over sequence dimension
        if attention_mask is not None:
            # Masked mean pooling
            mask_expanded = attention_mask.unsqueeze(-1).expand(outputs.last_hidden_state.size()).float()
            sum_embeddings = torch.sum(outputs.last_hidden_state * mask_expanded, 1)
            sum_mask = torch.clamp(mask_expanded.sum(1), min=1e-9)
            pooled = sum_embeddings / sum_mask
        else:
            pooled = outputs.last_hidden_state.mean(dim=1)
        
        if self.return_embeddings:
            return pooled
        return self.classifier(pooled)


class AudioModel(nn.Module):
    """HuBERT based audio model for classification."""
    
    def __init__(self, num_classes=3, return_embeddings=False, freeze_encoder=False):
        """
        Args:
            num_classes: Number of output classes (default: 3 for KEEP, TURN, BACKCHANNEL)
            return_embeddings: If True, return embeddings instead of class logits
            freeze_encoder: If True, freeze the pretrained encoder weights
        """
        super().__init__()
        self.encoder = AutoModel.from_pretrained("facebook/hubert-large-ls960-ft")
        self.return_embeddings = return_embeddings
        self.hidden_size = 1024
        
        # Freeze encoder if specified
        if freeze_encoder:
            for param in self.encoder.parameters():
                param.requires_grad = False
        
        if not return_embeddings:
            self.classifier = nn.Sequential(
                nn.Linear(self.hidden_size, 256),
                nn.ReLU(),
                nn.Dropout(0.1),
                nn.Linear(256, num_classes)
            )
    
    def forward(self, input_values):
        """
        Args:
            input_values: Audio input values (batch_size, sequence_length)
        
        Returns:
            If return_embeddings: (batch_size, hidden_size)
            Else: (batch_size, num_classes)
        """
        outputs = self.encoder(input_values)
        
        # Mean pooling over time dimension
        pooled = outputs.last_hidden_state.mean(dim=1)
        
        if self.return_embeddings:
            return pooled
        return self.classifier(pooled)


class VisionModel(nn.Module):
    """VideoMAE based vision model for classification."""
    
    def __init__(self, num_classes=3, return_embeddings=False, freeze_encoder=False):
        """
        Args:
            num_classes: Number of output classes (default: 3 for KEEP, TURN, BACKCHANNEL)
            return_embeddings: If True, return embeddings instead of class logits
            freeze_encoder: If True, freeze the pretrained encoder weights
        """
        super().__init__()
        self.encoder = AutoModel.from_pretrained("MCG-NJU/videomae-base")
        self.return_embeddings = return_embeddings
        self.hidden_size = 768
        
        # Freeze encoder if specified
        if freeze_encoder:
            for param in self.encoder.parameters():
                param.requires_grad = False
        
        if not return_embeddings:
            self.classifier = nn.Sequential(
                nn.Linear(self.hidden_size, 256),
                nn.ReLU(),
                nn.Dropout(0.1),
                nn.Linear(256, num_classes)
            )
    
    def forward(self, pixel_values):
        """
        Args:
            pixel_values: Video frames (batch_size, num_frames, channels, height, width)
        
        Returns:
            If return_embeddings: (batch_size, hidden_size)
            Else: (batch_size, num_classes)
        """
        outputs = self.encoder(pixel_values)
        
        # Mean pooling over frame dimension
        pooled = outputs.last_hidden_state.mean(dim=1)
        
        if self.return_embeddings:
            return pooled
        return self.classifier(pooled)


def get_model(modal, num_classes=3, return_embeddings=False, freeze_encoder=False):
    """
    Factory function to get the appropriate model based on modality.
    
    Args:
        modal: Modality type ('text', 'audio', or 'video')
        num_classes: Number of output classes
        return_embeddings: If True, return embeddings instead of class logits
        freeze_encoder: If True, freeze the pretrained encoder weights
    
    Returns:
        Model instance
    """
    if modal == "text":
        return LanguageModel(num_classes, return_embeddings, freeze_encoder)
    elif modal == "audio":
        return AudioModel(num_classes, return_embeddings, freeze_encoder)
    elif modal == "video":
        return VisionModel(num_classes, return_embeddings, freeze_encoder)
    else:
        raise ValueError(f"Invalid modal: {modal}. Choose from 'text', 'audio', 'video'")


if __name__ == "__main__":
    # Test models
    print("Testing models...")
    
    # Text model
    text_model = LanguageModel(num_classes=3)
    dummy_input_ids = torch.randint(0, 1000, (2, 512))
    dummy_attention_mask = torch.ones(2, 512)
    text_output = text_model(dummy_input_ids, dummy_attention_mask)
    print(f"Text model output shape: {text_output.shape}")  # Should be (2, 3)
    
    # Audio model
    audio_model = AudioModel(num_classes=3)
    dummy_audio = torch.randn(2, 32000)
    audio_output = audio_model(dummy_audio)
    print(f"Audio model output shape: {audio_output.shape}")  # Should be (2, 3)
    
    # Video model
    video_model = VisionModel(num_classes=3)
    dummy_video = torch.randn(2, 16, 3, 224, 224)
    video_output = video_model(dummy_video)
    print(f"Video model output shape: {video_output.shape}")  # Should be (2, 3)
    
    print("\nAll models working correctly!")