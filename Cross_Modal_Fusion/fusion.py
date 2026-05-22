import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.modules.normalization import LayerNorm
from torch.nn.modules.activation import GELU


class CrossModalAttention(nn.Module):
    """
    Cross-Modal Attention Fusion
    """
    def __init__(self, hidden_dim=256, output_dim=3, num_heads=4, dropout=0.1):
        super(CrossModalAttention, self).__init__()
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        self.num_heads = num_heads
        
        # Multi-head attention for cross-modal interactions
        self.text_audio_attn = nn.MultiheadAttention(hidden_dim, num_heads, dropout=dropout, batch_first=True)
        self.text_video_attn = nn.MultiheadAttention(hidden_dim, num_heads, dropout=dropout, batch_first=True)
        self.audio_video_attn = nn.MultiheadAttention(hidden_dim, num_heads, dropout=dropout, batch_first=True)
        
        # Layer normalization
        self.norm1 = nn.LayerNorm(hidden_dim)
        self.norm2 = nn.LayerNorm(hidden_dim)
        self.norm3 = nn.LayerNorm(hidden_dim)
        
        # Final fusion layers
        self.fusion_fc = nn.Sequential(
            nn.Linear(hidden_dim * 3, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, output_dim)
        )
        
    def forward(self, text_x, audio_x, video_x):
        batch_size = (text_x if text_x is not None else 
                     audio_x if audio_x is not None else video_x).shape[0]
        device = (text_x if text_x is not None else 
                 audio_x if audio_x is not None else video_x).device
        
        if text_x is None:
            text_x = torch.zeros(batch_size, self.hidden_dim).to(device)
        if audio_x is None:
            audio_x = torch.zeros(batch_size, self.hidden_dim).to(device)
        if video_x is None:
            video_x = torch.zeros(batch_size, self.hidden_dim).to(device)
        
        # Reshape for attention (add sequence dimension)
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
        
        # Concatenate and squeeze
        fused = torch.cat([ta_out, tv_out, av_out], dim=-1).squeeze(1)
        
        output = self.fusion_fc(fused)
        
        return output


if __name__ == "__main__":
    QF = CrossModalAttention()

    batch_size = 16
    Xt = torch.randn(batch_size, 256)
    Xa = torch.randn(batch_size, 256)
    Xv = torch.randn(batch_size, 256)

    print(QF(Xa, Xv, Xt).shape)
    print(QF(None, Xv, Xt).shape)
    print(QF(Xa, None, Xt).shape)
    print(QF(Xa, Xv, None).shape)

    params = sum(p.numel() for p in QF.parameters())
    print(f"  Parameters: {params:,}")
    print()