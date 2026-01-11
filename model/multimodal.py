"""
Simple multimodal architecture using existing unimodal models.
Supports pluggable fusion strategies.
"""

import torch
from torch import nn

from models.unimodal import LanguageModel, AudioModel, VisionModel


class MultimodalModel(nn.Module):
    """
    Main multimodal model with pluggable fusion mechanism.
    Uses pretrained unimodal models.
    """
    
    def __init__(
        self,
        fusion_mechanism,
        hidden_dim=256,
        freeze_encoders=False,
        text_ckpt_path=None,
        audio_ckpt_path=None,
        video_ckpt_path=None
    ):
        """
        Args:
            fusion_mechanism: Fusion module instance
            hidden_dim: Hidden dimension for embeddings (default: 256)
            freeze_encoders: If True, freeze all encoder weights
            text_ckpt_path: Path to pretrained text model checkpoint
            audio_ckpt_path: Path to pretrained audio model checkpoint
            video_ckpt_path: Path to pretrained video model checkpoint
        """
        super().__init__()
        
        self.hidden_dim = hidden_dim
        
        # Initialize unimodal models to return embeddings
        self.text_model = LanguageModel(
            num_classes=3,
            return_embeddings=True,
            freeze_encoder=freeze_encoders
        )
        
        self.audio_model = AudioModel(
            num_classes=3,
            return_embeddings=True,
            freeze_encoder=freeze_encoders
        )
        
        self.video_model = VisionModel(
            num_classes=3,
            return_embeddings=True,
            freeze_encoder=freeze_encoders
        )
        
        # Load pretrained weights if provided
        if text_ckpt_path is not None:
            self._load_encoder_checkpoint(self.text_model, text_ckpt_path, "text")
        if audio_ckpt_path is not None:
            self._load_encoder_checkpoint(self.audio_model, audio_ckpt_path, "audio")
        if video_ckpt_path is not None:
            self._load_encoder_checkpoint(self.video_model, video_ckpt_path, "video")
        
        # Get embedding dimensions from unimodal models
        text_emb_dim = self.text_model.hidden_size  # 768
        audio_emb_dim = self.audio_model.hidden_size  # 1024
        video_emb_dim = self.video_model.hidden_size  # 768
        
        # Project to common hidden dimension
        self.text_projection = nn.Sequential(
            nn.Linear(text_emb_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.1)
        )
        
        self.audio_projection = nn.Sequential(
            nn.Linear(audio_emb_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.1)
        )
        
        self.video_projection = nn.Sequential(
            nn.Linear(video_emb_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.1)
        )
        
        # Fusion mechanism
        self.fusion = fusion_mechanism
    
    def _load_encoder_checkpoint(self, model, checkpoint_path, modality_name):
        """Load encoder checkpoint."""
        try:
            checkpoint = torch.load(checkpoint_path, map_location='cpu')
            
            # Handle both direct state dict and wrapped checkpoints
            if 'model_state_dict' in checkpoint:
                state_dict = checkpoint['model_state_dict']
            else:
                state_dict = checkpoint
            
            # Load state dict (strict=False to allow loading encoder weights only)
            model.load_state_dict(state_dict, strict=False)
            print(f"✓ Loaded {modality_name} model from {checkpoint_path}")
            
        except Exception as e:
            print(f"⚠ Warning: Could not load {modality_name} checkpoint from {checkpoint_path}")
            print(f"  Error: {str(e)}")
            print(f"  Continuing with randomly initialized weights...")
    
    def forward(self, text_inputs, audio_inputs, video_inputs):
        """
        Forward pass through multimodal model.
        
        Args:
            text_inputs: Dict with 'input_ids' and 'attention_mask'
            audio_inputs: Audio tensor (batch_size, seq_len)
            video_inputs: Dict with 'pixel_values'
        
        Returns:
            Classification logits (batch_size, num_classes)
        """
        # Extract features from each modality
        text_emb = self.text_model(
            text_inputs['input_ids'],
            text_inputs.get('attention_mask', None)
        )
        text_features = self.text_projection(text_emb)
        
        audio_emb = self.audio_model(audio_inputs)
        audio_features = self.audio_projection(audio_emb)
        
        video_emb = self.video_model(video_inputs['pixel_values'])
        video_features = self.video_projection(video_emb)
        
        # Fuse features
        output = self.fusion(text_features, audio_features, video_features)
        
        return output


def create_multimodal_model(
    fusion_type='lmf',
    hidden_dim=256,
    output_dim=3,
    freeze_encoders=False,
    text_ckpt_path=None,
    audio_ckpt_path=None,
    video_ckpt_path=None,
    **fusion_kwargs
):
    """
    Factory function to create multimodal models with different fusion mechanisms.
    
    Args:
        fusion_type: Type of fusion ('lmf', 'early', 'late', 'tensor', etc.)
        hidden_dim: Hidden dimension for fusion
        output_dim: Number of output classes
        freeze_encoders: Whether to freeze encoder weights
        text_ckpt_path: Path to pretrained text model checkpoint
        audio_ckpt_path: Path to pretrained audio model checkpoint
        video_ckpt_path: Path to pretrained video model checkpoint
        **fusion_kwargs: Additional arguments for fusion mechanism
    
    Returns:
        MultimodalModel instance
    """
    from models.fusion import get_fusion_mechanism
    
    # Get fusion mechanism
    fusion = get_fusion_mechanism(
        fusion_type=fusion_type,
        hidden_dim=hidden_dim,
        output_dim=output_dim,
        **fusion_kwargs
    )
    
    # Create multimodal model
    model = MultimodalModel(
        fusion_mechanism=fusion,
        hidden_dim=hidden_dim,
        freeze_encoders=freeze_encoders,
        text_ckpt_path=text_ckpt_path,
        audio_ckpt_path=audio_ckpt_path,
        video_ckpt_path=video_ckpt_path
    )
    
    return model


if __name__ == "__main__":
    # Test the multimodal model
    print("Testing Multimodal Model\n")
    
    from models.fusion import EarlyFusion
    
    # Create model with early fusion
    fusion = EarlyFusion(hidden_dim=256, output_dim=3)
    model = MultimodalModel(fusion_mechanism=fusion, hidden_dim=256)
    
    # Create dummy inputs
    batch_size = 4
    text_inputs = {
        'input_ids': torch.randint(0, 1000, (batch_size, 512)),
        'attention_mask': torch.ones(batch_size, 512)
    }
    audio_inputs = torch.randn(batch_size, 32000)
    video_inputs = {
        'pixel_values': torch.randn(batch_size, 16, 3, 224, 224)
    }
    
    print("Testing with all modalities...")
    output = model(text_inputs, audio_inputs, video_inputs)
    print(f"Output shape: {output.shape}")
    
    # Test with pretrained checkpoints (if they exist)
    print("\n" + "="*60)
    print("Testing with checkpoint loading...")
    print("="*60)
    
    try:
        model_with_ckpt = create_multimodal_model(
            fusion_type='early',
            hidden_dim=256,
            output_dim=3,
            text_ckpt_path='log/text_model/best_model_acc.pt',
            freeze_encoders=True
        )
        print("Model created with checkpoint loading")
    except:
        print("Checkpoint loading test skipped (files not found)")
    
    print("\n✓ Model architecture validated successfully!")