import torch
import torch.nn as nn


class TinyAntiCorrelator(nn.Module):
    """
    Ultra-lightweight fusion (< 5k parameters).
    Captures positive correlation (Hadamard) and negative correlation (Abs Difference).
    """
    def __init__(self, hidden_dim=256, output_dim=3):
        super(TinyAntiCorrelator, self).__init__()
        # Instead of large matrices, we use a single vector to weight the importance 
        # of each dimension during fusion.
        self.feature_importance = nn.Parameter(torch.ones(1, hidden_dim))
        
        # A tiny MLP to decide the balance between positive and negative correlation
        # This is the ONLY dense part, and it's very small.
        self.correlation_gate = nn.Sequential(
            nn.Linear(hidden_dim, 16),
            nn.ReLU(),
            nn.Linear(16, 2), # [Weight for Positive, Weight for Negative]
            nn.Softmax(dim=-1)
        )
        
        self.classifier = nn.Linear(hidden_dim, output_dim)

    def forward(self, text_x, audio_x, video_x):
        # 1. Fill missing modalities with zeros
        ref = text_x if text_x is not None else audio_x if audio_x is not None else video_x
        device = ref.device
        z_t = text_x if text_x is not None else torch.zeros_like(ref)
        z_a = audio_x if audio_x is not None else torch.zeros_like(ref)
        z_v = video_x if video_x is not None else torch.zeros_like(ref)

        # 2. Positive Correlation (Agreement)
        # We use a simple product. High value = modalities agree in sign/magnitude.
        pos_corr = z_t * z_a * z_v
        
        # 3. Negative Correlation (Conflict)
        # Sum of absolute differences captures how much they disagree.
        neg_corr = torch.abs(z_t - z_a) + torch.abs(z_t - z_v)
        
        # 4. Global Context for Gating
        # Mean pooling across modalities to get a "global state"
        global_state = (z_t + z_a + z_v) / 3
        weights = self.correlation_gate(global_state) # Shape: (batch, 2)
        
        # 5. Dynamic Fusion
        # Mix the agreement and conflict signals based on the gate
        fused = (weights[:, 0:1] * pos_corr) + (weights[:, 1:2] * neg_corr)
        
        # Apply learnable feature scaling (cheap way to mimic weight matrices)
        fused = fused * self.feature_importance
        
        return self.classifier(fused)
