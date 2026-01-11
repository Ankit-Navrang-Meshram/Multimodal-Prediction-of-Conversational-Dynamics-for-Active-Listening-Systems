"""
Simple fusion mechanisms for multimodal learning.
All mechanisms expect (batch_size, hidden_dim) tensors from all three modalities.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.parameter import Parameter
from torch.nn.init import xavier_normal_


class EarlyFusion(nn.Module):
    """Simple concatenation followed by MLP."""
    
    def __init__(self, hidden_dim=256, output_dim=3, dropout=0.1):
        super().__init__()
        
        self.fc = nn.Sequential(
            nn.Linear(hidden_dim * 3, hidden_dim * 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, output_dim)
        )
    
    def forward(self, text_x, audio_x, video_x):
        fused = torch.cat([text_x, audio_x, video_x], dim=1)
        return self.fc(fused)


class LateFusion(nn.Module):
    """Independent processing followed by weighted combination."""
    
    def __init__(self, hidden_dim=256, output_dim=3, dropout=0.1):
        super().__init__()
        
        # Individual classifiers
        self.text_classifier = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, output_dim)
        )
        
        self.audio_classifier = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, output_dim)
        )
        
        self.video_classifier = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, output_dim)
        )
        
        # Learnable weights
        self.weights = nn.Parameter(torch.ones(3))
    
    def forward(self, text_x, audio_x, video_x):
        text_out = self.text_classifier(text_x)
        audio_out = self.audio_classifier(audio_x)
        video_out = self.video_classifier(video_x)
        
        # Weighted average
        weights = F.softmax(self.weights, dim=0)
        output = weights[0] * text_out + weights[1] * audio_out + weights[2] * video_out
        
        return output


class LMF(nn.Module):
    """Low-rank Multimodal Fusion."""
    
    def __init__(self, hidden_dim=256, output_dim=3, rank=16, dropout=0.1):
        super().__init__()
        self.rank = rank
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        
        self.text_factor = Parameter(torch.Tensor(rank, hidden_dim + 1, output_dim))
        self.audio_factor = Parameter(torch.Tensor(rank, hidden_dim + 1, output_dim))
        self.video_factor = Parameter(torch.Tensor(rank, hidden_dim + 1, output_dim))
        self.fusion_weights = Parameter(torch.Tensor(1, rank))
        self.fusion_bias = Parameter(torch.Tensor(1, output_dim))
        
        xavier_normal_(self.text_factor)
        xavier_normal_(self.audio_factor)
        xavier_normal_(self.video_factor)
        xavier_normal_(self.fusion_weights)
        self.fusion_bias.data.fill_(0)
        
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, text_x, audio_x, video_x):
        batch_size = text_x.shape[0]
        device = text_x.device
        
        # Add bias dimension
        ones = torch.ones(batch_size, 1).to(device)
        text_x = torch.cat([ones, text_x], dim=1)
        audio_x = torch.cat([ones, audio_x], dim=1)
        video_x = torch.cat([ones, video_x], dim=1)
        
        # Low-rank fusion
        fusion_text = torch.matmul(text_x, self.text_factor)
        fusion_audio = torch.matmul(audio_x, self.audio_factor)
        fusion_video = torch.matmul(video_x, self.video_factor)
        
        fusion_zy = fusion_text * fusion_audio * fusion_video
        
        output = torch.matmul(self.fusion_weights, fusion_zy.permute(1, 0, 2)).squeeze()
        output = output + self.fusion_bias
        output = output.view(-1, self.output_dim)
        
        return self.dropout(output)


class TensorFusion(nn.Module):
    """Tensor Fusion Network: Outer product of all modalities."""
    
    def __init__(self, hidden_dim=256, output_dim=3, dropout=0.1):
        super().__init__()
        self.hidden_dim = hidden_dim
        
        post_fusion_dim = (hidden_dim + 1) ** 3
        
        self.post_fusion = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(post_fusion_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, output_dim)
        )
    
    def forward(self, text_x, audio_x, video_x):
        batch_size = text_x.shape[0]
        device = text_x.device
        
        # Add bias dimension
        ones = torch.ones(batch_size, 1).to(device)
        text_x = torch.cat([ones, text_x], dim=1)
        audio_x = torch.cat([ones, audio_x], dim=1)
        video_x = torch.cat([ones, video_x], dim=1)
        
        # Compute outer product
        fusion_tensor = torch.bmm(text_x.unsqueeze(2), audio_x.unsqueeze(1))
        fusion_tensor = fusion_tensor.view(batch_size, -1, 1)
        fusion_tensor = torch.bmm(fusion_tensor, video_x.unsqueeze(1))
        fusion_tensor = fusion_tensor.view(batch_size, -1)
        
        return self.post_fusion(fusion_tensor)


class CrossModalAttention(nn.Module):
    """Cross-Modal Attention Fusion."""
    
    def __init__(self, hidden_dim=256, output_dim=3, num_heads=4, dropout=0.1):
        super().__init__()
        self.num_heads = num_heads
        
        # Multi-head attention
        self.text_audio_attn = nn.MultiheadAttention(hidden_dim, num_heads, dropout=dropout, batch_first=True)
        self.text_video_attn = nn.MultiheadAttention(hidden_dim, num_heads, dropout=dropout, batch_first=True)
        self.audio_video_attn = nn.MultiheadAttention(hidden_dim, num_heads, dropout=dropout, batch_first=True)
        
        self.norm1 = nn.LayerNorm(hidden_dim)
        self.norm2 = nn.LayerNorm(hidden_dim)
        self.norm3 = nn.LayerNorm(hidden_dim)
        
        self.fusion_fc = nn.Sequential(
            nn.Linear(hidden_dim * 3, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, output_dim)
        )
    
    def forward(self, text_x, audio_x, video_x):
        # Add sequence dimension
        text_x = text_x.unsqueeze(1)
        audio_x = audio_x.unsqueeze(1)
        video_x = video_x.unsqueeze(1)
        
        # Cross-modal attention
        ta_out, _ = self.text_audio_attn(text_x, audio_x, audio_x)
        ta_out = self.norm1(ta_out + text_x)
        
        tv_out, _ = self.text_video_attn(text_x, video_x, video_x)
        tv_out = self.norm2(tv_out + text_x)
        
        av_out, _ = self.audio_video_attn(audio_x, video_x, video_x)
        av_out = self.norm3(av_out + audio_x)
        
        fused = torch.cat([ta_out, tv_out, av_out], dim=-1).squeeze(1)
        return self.fusion_fc(fused)


class GatedFusion(nn.Module):
    """Gated Multimodal Unit (GMU)."""
    
    def __init__(self, hidden_dim=256, output_dim=3, dropout=0.1):
        super().__init__()
        
        # Gating mechanisms
        self.text_gate = nn.Sequential(
            nn.Linear(hidden_dim * 3, hidden_dim),
            nn.Sigmoid()
        )
        
        self.audio_gate = nn.Sequential(
            nn.Linear(hidden_dim * 3, hidden_dim),
            nn.Sigmoid()
        )
        
        self.video_gate = nn.Sequential(
            nn.Linear(hidden_dim * 3, hidden_dim),
            nn.Sigmoid()
        )
        
        # Transform layers
        self.text_transform = nn.Linear(hidden_dim, hidden_dim)
        self.audio_transform = nn.Linear(hidden_dim, hidden_dim)
        self.video_transform = nn.Linear(hidden_dim, hidden_dim)
        
        # Classifier
        self.fc = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, output_dim)
        )
    
    def forward(self, text_x, audio_x, video_x):
        # Concatenate for gating
        concat_features = torch.cat([text_x, audio_x, video_x], dim=1)
        
        # Compute gates
        text_gate = self.text_gate(concat_features)
        audio_gate = self.audio_gate(concat_features)
        video_gate = self.video_gate(concat_features)
        
        # Apply gates
        text_h = text_gate * torch.tanh(self.text_transform(text_x))
        audio_h = audio_gate * torch.tanh(self.audio_transform(audio_x))
        video_h = video_gate * torch.tanh(self.video_transform(video_x))
        
        # Fuse
        fused = text_h + audio_h + video_h
        return self.fc(fused)


class MultimodalTransformer(nn.Module):
    """Multimodal Transformer with self-attention across modalities."""
    
    def __init__(self, hidden_dim=256, output_dim=3, num_heads=4, num_layers=2, dropout=0.1):
        super().__init__()
        
        # Modality embeddings
        self.text_embed = nn.Linear(hidden_dim, hidden_dim)
        self.audio_embed = nn.Linear(hidden_dim, hidden_dim)
        self.video_embed = nn.Linear(hidden_dim, hidden_dim)
        
        # Modality type embeddings
        self.modality_embedding = nn.Embedding(3, hidden_dim)
        
        # Transformer
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim,
            nhead=num_heads,
            dim_feedforward=hidden_dim * 4,
            dropout=dropout,
            batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        
        self.fc = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, output_dim)
        )
    
    def forward(self, text_x, audio_x, video_x):
        batch_size = text_x.shape[0]
        device = text_x.device
        
        # Embed each modality
        text_emb = self.text_embed(text_x).unsqueeze(1)
        audio_emb = self.audio_embed(audio_x).unsqueeze(1)
        video_emb = self.video_embed(video_x).unsqueeze(1)
        
        # Concatenate modalities
        x = torch.cat([text_emb, audio_emb, video_emb], dim=1)
        
        # Add modality type embeddings
        modality_ids = torch.tensor([0, 1, 2]).to(device)
        modality_embs = self.modality_embedding(modality_ids).unsqueeze(0).expand(batch_size, -1, -1)
        x = x + modality_embs
        
        # Transformer
        x = self.transformer(x)
        x = x.mean(dim=1)
        
        return self.fc(x)


def get_fusion_mechanism(fusion_type, hidden_dim=256, output_dim=3, **kwargs):
    """
    Factory function to get fusion mechanism by name.
    
    Args:
        fusion_type: Name of fusion mechanism
        hidden_dim: Hidden dimension
        output_dim: Output dimension
        **kwargs: Additional arguments for specific fusion mechanisms
    
    Returns:
        Fusion mechanism instance
    """
    fusion_map = {
        'early': EarlyFusion,
        'late': LateFusion,
        'lmf': LMF,
        'tensor': TensorFusion,
        'tfn': TensorFusion,
        'attention': CrossModalAttention,
        'gated': GatedFusion,
        'gmu': GatedFusion,
        'transformer': MultimodalTransformer
    }
    
    if fusion_type.lower() not in fusion_map:
        raise ValueError(f"Unknown fusion type: {fusion_type}. Choose from {list(fusion_map.keys())}")
    
    fusion_class = fusion_map[fusion_type.lower()]
    return fusion_class(hidden_dim=hidden_dim, output_dim=output_dim, **kwargs)


if __name__ == "__main__":
    # Test all fusion mechanisms
    batch_size = 8
    hidden_dim = 256
    output_dim = 3
    
    text = torch.randn(batch_size, hidden_dim)
    audio = torch.randn(batch_size, hidden_dim)
    video = torch.randn(batch_size, hidden_dim)
    
    print("Testing all fusion mechanisms:\n")
    
    fusion_types = ['early', 'late', 'lmf', 'tensor', 'attention', 'gated', 'transformer']
    
    for fusion_type in fusion_types:
        print(f"{fusion_type.upper()}:")
        fusion = get_fusion_mechanism(fusion_type, hidden_dim, output_dim)
        
        # Test
        out = fusion(text, audio, video)
        print(f"  Output shape: {out.shape}")
        
        # Count parameters
        params = sum(p.numel() for p in fusion.parameters())
        print(f"  Parameters: {params:,}\n")
    
    print("✓ All fusion mechanisms working!")