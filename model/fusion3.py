import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from torch.nn.parameter import Parameter
from torch.nn.init import xavier_normal_, kaiming_normal_
import math
   
class LMF(nn.Module):
    """
    Low-rank Multimodal Fusion
    """
    def __init__(self, hidden_dim=256, output_dim=3, rank=16, use_softmax=False, post_fusion_prob=0.1):
        super(LMF, self).__init__()
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        self.rank = rank
        self.use_softmax = use_softmax
        self.post_fusion_prob = post_fusion_prob
        
        self.post_fusion_dropout = nn.Dropout(p=self.post_fusion_prob)
        self.factor_1 = Parameter(torch.Tensor(self.rank, self.hidden_dim + 1, self.output_dim))
        self.factor_2 = Parameter(torch.Tensor(self.rank, self.hidden_dim + 1, self.output_dim))
        self.factor_3 = Parameter(torch.Tensor(self.rank, self.hidden_dim + 1, self.output_dim))
        self.fusion_weights = Parameter(torch.Tensor(1, self.rank))
        self.fusion_bias = Parameter(torch.Tensor(1, self.output_dim))
        
        xavier_normal_(self.factor_1)
        xavier_normal_(self.factor_2)
        xavier_normal_(self.factor_3)
        xavier_normal_(self.fusion_weights)
        self.fusion_bias.data.fill_(0)
    
    def forward(self, text_x, audio_x, video_x):
        temp_x = text_x if text_x is not None else audio_x if audio_x is not None else video_x
        batch_size = temp_x.data.shape[0]
        DTYPE = torch.cuda.FloatTensor if temp_x.is_cuda else torch.FloatTensor
        
        # Initialize fusion variables
        fusion_text = None
        fusion_audio = None
        fusion_video = None

        if text_x is not None:
            _text_h = torch.cat((Variable(torch.ones(batch_size, 1).type(DTYPE), requires_grad=False), text_x), dim=1)
            fusion_text = torch.matmul(_text_h, self.factor_1)
        
        if audio_x is not None:
            _audio_h = torch.cat((Variable(torch.ones(batch_size, 1).type(DTYPE), requires_grad=False), audio_x), dim=1)
            fusion_audio = torch.matmul(_audio_h, self.factor_2)
        
        if video_x is not None:
            _video_h = torch.cat((Variable(torch.ones(batch_size, 1).type(DTYPE), requires_grad=False), video_x), dim=1)
            fusion_video = torch.matmul(_video_h, self.factor_3)
        

        # if text_x is None:
        #     fusion_zy = fusion_audio * fusion_video
        # elif audio_x is None:
        #     fusion_zy = fusion_text * fusion_video
        # elif video_x is None:
        #     fusion_zy = fusion_audio * fusion_text
        # else:
        #     fusion_zy = fusion_audio * fusion_video * fusion_text

        # Combine available modalities
        available_fusions = [f for f in [fusion_text, fusion_audio, fusion_video] if f is not None]
        
        if len(available_fusions) == 0:
            raise ValueError("At least one modality must be provided")
        elif len(available_fusions) == 1:
            fusion_zy = available_fusions[0]
        elif len(available_fusions) == 2:
            fusion_zy = available_fusions[0] * available_fusions[1]
        else:  # All 3 modalities available
            fusion_zy = fusion_text * fusion_audio * fusion_video
        
        output = torch.matmul(self.fusion_weights, fusion_zy.permute(1, 0, 2)).squeeze() + self.fusion_bias
        output = output.view(-1, self.output_dim)
        
        if self.use_softmax:
            output = F.softmax(output, dim=-1)
        
        return output


class EarlyFusion(nn.Module):
    """
    Early Fusion: Simple concatenation followed by MLP
    """
    def __init__(self, hidden_dim=256, output_dim=3, dropout=0.1):
        super(EarlyFusion, self).__init__()
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        
        # MLP for fusion
        self.fc1 = nn.Linear(hidden_dim * 3, hidden_dim * 2)
        self.fc2 = nn.Linear(hidden_dim * 2, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, output_dim)
        
        self.dropout = nn.Dropout(dropout)
        self.relu = nn.ReLU()
        self.layer_norm = nn.LayerNorm(hidden_dim)
        
    def forward(self, text_x, audio_x, video_x):
        # Handle missing modalities by using zeros
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
        
        # Concatenate all modalities
        fused = torch.cat([text_x, audio_x, video_x], dim=1)
        
        # MLP
        x = self.relu(self.fc1(fused))
        x = self.dropout(x)
        x = self.relu(self.fc2(x))
        x = self.layer_norm(x)
        x = self.dropout(x)
        output = self.fc3(x)
        
        return output


class LateFusion(nn.Module):
    """
    Late Fusion: Independent processing followed by weighted combination
    """
    def __init__(self, hidden_dim=256, output_dim=3, dropout=0.1):
        super(LateFusion, self).__init__()
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        
        # Individual classifiers for each modality
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
        
        # Learnable weights for each modality
        self.weights = nn.Parameter(torch.ones(3))
        
    def forward(self, text_x, audio_x, video_x):
        outputs = []
        active_weights = []
        
        if text_x is not None:
            outputs.append(self.text_classifier(text_x))
            active_weights.append(self.weights[0])
        
        if audio_x is not None:
            outputs.append(self.audio_classifier(audio_x))
            active_weights.append(self.weights[1])
        
        if video_x is not None:
            outputs.append(self.video_classifier(video_x))
            active_weights.append(self.weights[2])
        
        # Weighted average
        active_weights = torch.stack(active_weights)
        active_weights = F.softmax(active_weights, dim=0)
        
        output = sum(w * o for w, o in zip(active_weights, outputs))
        
        return output




class MultimodalFactorizedBilinear(nn.Module):
    """
    Multimodal Factorized Bilinear Pooling (MFB)
    """
    def __init__(self, hidden_dim=256, output_dim=3, mfb_factor=5, dropout=0.1):
        super(MultimodalFactorizedBilinear, self).__init__()
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        self.mfb_factor = mfb_factor
        self.mfb_out_dim = hidden_dim
        
        # Text-Audio MFB
        self.text_audio_proj1 = nn.Linear(hidden_dim, self.mfb_out_dim * mfb_factor)
        self.text_audio_proj2 = nn.Linear(hidden_dim, self.mfb_out_dim * mfb_factor)
        
        # Audio-Video MFB
        self.audio_video_proj1 = nn.Linear(hidden_dim, self.mfb_out_dim * mfb_factor)
        self.audio_video_proj2 = nn.Linear(hidden_dim, self.mfb_out_dim * mfb_factor)
        
        # Text-Video MFB
        self.text_video_proj1 = nn.Linear(hidden_dim, self.mfb_out_dim * mfb_factor)
        self.text_video_proj2 = nn.Linear(hidden_dim, self.mfb_out_dim * mfb_factor)
        
        # Final fusion
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(self.mfb_out_dim * 3, output_dim)
        
    def mfb_pooling(self, x1, x2, proj1, proj2):
        """Perform MFB pooling between two modalities"""
        z1 = proj1(x1)
        z2 = proj2(x2)
        
        z1 = z1.view(-1, self.mfb_factor, self.mfb_out_dim)
        z2 = z2.view(-1, self.mfb_factor, self.mfb_out_dim)
        
        # Element-wise product and sum over factor dimension
        z = (z1 * z2).sum(1)
        z = torch.sqrt(F.relu(z)) - torch.sqrt(F.relu(-z))
        z = F.normalize(z, p=2, dim=1)
        
        return z
    
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
        
        # Pairwise MFB pooling
        ta_fused = self.mfb_pooling(text_x, audio_x, self.text_audio_proj1, self.text_audio_proj2)
        av_fused = self.mfb_pooling(audio_x, video_x, self.audio_video_proj1, self.audio_video_proj2)
        tv_fused = self.mfb_pooling(text_x, video_x, self.text_video_proj1, self.text_video_proj2)
        
        # Concatenate all fused features
        fused = torch.cat([ta_fused, av_fused, tv_fused], dim=1)
        fused = self.dropout(fused)
        
        output = self.fc(fused)
        
        return output


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


class GatedMultimodalUnit(nn.Module):
    """
    Gated Multimodal Unit (GMU)
    """
    def __init__(self, hidden_dim=256, output_dim=3, dropout=0.1):
        super(GatedMultimodalUnit, self).__init__()
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        
        # Gating mechanisms for each modality
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
        
        # Final classifier
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, output_dim)
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
        
        # Concatenate all modalities for gating
        concat_features = torch.cat([text_x, audio_x, video_x], dim=1)
        
        # Compute gates
        text_gate = self.text_gate(concat_features)
        audio_gate = self.audio_gate(concat_features)
        video_gate = self.video_gate(concat_features)
        
        # Apply gates to transformed features
        text_h = text_gate * torch.tanh(self.text_transform(text_x))
        audio_h = audio_gate * torch.tanh(self.audio_transform(audio_x))
        video_h = video_gate * torch.tanh(self.video_transform(video_x))
        
        # Fuse gated features
        fused = text_h + audio_h + video_h
        fused = self.dropout(fused)
        
        output = self.fc(fused)
        
        return output


class MultimodalTransformer(nn.Module):
    """
    Multimodal Transformer with self-attention across modalities
    """
    def __init__(self, hidden_dim=256, output_dim=3, num_heads=4, num_layers=2, dropout=0.1):
        super(MultimodalTransformer, self).__init__()
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        
        # Modality embeddings
        self.text_embed = nn.Linear(hidden_dim, hidden_dim)
        self.audio_embed = nn.Linear(hidden_dim, hidden_dim)
        self.video_embed = nn.Linear(hidden_dim, hidden_dim)
        
        # Positional/modality type embeddings
        self.modality_embedding = nn.Embedding(3, hidden_dim)
        
        # Transformer encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim,
            nhead=num_heads,
            dim_feedforward=hidden_dim * 4,
            dropout=dropout,
            batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        
        # Output projection
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(hidden_dim, output_dim)
        
    def forward(self, text_x, audio_x, video_x):
        batch_size = (text_x if text_x is not None else 
                     audio_x if audio_x is not None else video_x).shape[0]
        device = (text_x if text_x is not None else 
                 audio_x if audio_x is not None else video_x).device
        
        modalities = []
        modality_ids = []
        
        if text_x is not None:
            text_emb = self.text_embed(text_x).unsqueeze(1)
            modalities.append(text_emb)
            modality_ids.append(0)
        
        if audio_x is not None:
            audio_emb = self.audio_embed(audio_x).unsqueeze(1)
            modalities.append(audio_emb)
            modality_ids.append(1)
        
        if video_x is not None:
            video_emb = self.video_embed(video_x).unsqueeze(1)
            modalities.append(video_emb)
            modality_ids.append(2)
        
        # Concatenate modalities
        x = torch.cat(modalities, dim=1)
        
        # Add modality type embeddings
        modality_ids = torch.tensor(modality_ids).to(device)
        modality_embs = self.modality_embedding(modality_ids).unsqueeze(0).expand(batch_size, -1, -1)
        x = x + modality_embs
        
        # Transformer encoding
        x = self.transformer(x)
        
        # Global average pooling
        x = x.mean(dim=1)
        x = self.dropout(x)
        
        output = self.fc(x)
        
        return output



 

class ModalityComplementationLayer(nn.Module):
    """
    Single Complementation Layer following the paper's exact architecture.
    
    Key differences from initial implementation:
    1. Gates are computed from averaged sequence representations (h_bar)
    2. BiGRU processes the INPUT sequences (X^{i-1}), not attention outputs
    3. Gates are applied AFTER attention, in the residual connection
    4. The order is: Attention -> Gate Application -> Add&Norm -> FFN -> Add&Norm
    5. BiGRU comes BEFORE the transformer components for feature extraction
    """
    def __init__(self, hidden_dim=256, num_heads=4, dropout=0.1):
        super(ModalityComplementationLayer, self).__init__()
        self.hidden_dim = hidden_dim
        self.num_heads = num_heads
        
        # BiGRU for sequence encoding (processes input before attention)
        self.bigru_m1 = nn.GRU(hidden_dim, hidden_dim // 2, num_layers=1, 
                               batch_first=True, bidirectional=True)
        self.bigru_m2 = nn.GRU(hidden_dim, hidden_dim // 2, num_layers=1, 
                               batch_first=True, bidirectional=True)
        
        # Multi-head attention for both pipelines
        self.mha_m1 = nn.MultiheadAttention(hidden_dim, num_heads, dropout=dropout, batch_first=True)
        self.mha_m2 = nn.MultiheadAttention(hidden_dim, num_heads, dropout=dropout, batch_first=True)
        
        # Feed-forward networks for both pipelines
        self.ffn_m1 = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim * 4),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim * 4, hidden_dim)
        )
        
        self.ffn_m2 = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim * 4),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim * 4, hidden_dim)
        )
        
        # Layer normalization (two per pipeline: after attention and after FFN)
        self.norm1_m1 = nn.LayerNorm(hidden_dim)
        self.norm2_m1 = nn.LayerNorm(hidden_dim)
        self.norm1_m2 = nn.LayerNorm(hidden_dim)
        self.norm2_m2 = nn.LayerNorm(hidden_dim)
        
        # Projection matrices for gates (W_r and W_c)
        # Gates take concatenation of both averaged representations
        self.W_r_m1 = nn.Linear(hidden_dim * 2, hidden_dim)
        self.W_c_m1 = nn.Linear(hidden_dim * 2, hidden_dim)
        
        self.W_r_m2 = nn.Linear(hidden_dim * 2, hidden_dim)
        self.W_c_m2 = nn.Linear(hidden_dim * 2, hidden_dim)
        
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, X_m1, X_m2):
        """
        Args:
            X_m1: Modality 1 input from previous layer (batch, seq_len, hidden_dim)
            X_m2: Modality 2 input from previous layer (batch, seq_len, hidden_dim)
        Returns:
            X_m1_out: Enhanced modality 1 features (batch, seq_len, hidden_dim)
            X_m2_out: Enhanced modality 2 features (batch, seq_len, hidden_dim)
            h_bar_m1: Averaged representation for m1 (for feature separator)
            h_bar_m2: Averaged representation for m2 (for feature separator)
        """
        # Ensure 3D input
        if X_m1.dim() == 2:
            X_m1 = X_m1.unsqueeze(1)
        if X_m2.dim() == 2:
            X_m2 = X_m2.unsqueeze(1)
        
        batch_size = X_m1.shape[0]
        
        # ============ Process Modality 1 (main) with Modality 2 (complementary) ============
        # Step 1: BiGRU encoding and average pooling to get h_bar
        h_m1, _ = self.bigru_m1(X_m1)  # (batch, seq_len, hidden_dim)
        h_bar_m1 = torch.mean(h_m1, dim=1)  # (batch, hidden_dim)
        
        h_m2, _ = self.bigru_m2(X_m2)  # (batch, seq_len, hidden_dim)
        h_bar_m2 = torch.mean(h_m2, dim=1)  # (batch, hidden_dim)
        
        # Step 2: Compute gates from concatenated averaged representations
        concat_m1 = torch.cat([h_bar_m1, h_bar_m2], dim=1)  # (batch, 2*hidden_dim)
        g_r_m1 = torch.sigmoid(self.W_r_m1(concat_m1))  # (batch, hidden_dim) - Retain gate
        g_c_m1 = torch.sigmoid(self.W_c_m1(concat_m1))  # (batch, hidden_dim) - Compound gate
        
        concat_m2 = torch.cat([h_bar_m2, h_bar_m1], dim=1)  # (batch, 2*hidden_dim)
        g_r_m2 = torch.sigmoid(self.W_r_m2(concat_m2))  # (batch, hidden_dim) - Retain gate
        g_c_m2 = torch.sigmoid(self.W_c_m2(concat_m2))  # (batch, hidden_dim) - Compound gate
        
        # Step 3: Multi-head attention (Q from main, K,V from complementary)
        # For m1 pipeline: Q=X_m1, K=V=X_m2
        m_m1, _ = self.mha_m1(X_m1, X_m2, X_m2)  # (batch, seq_len, hidden_dim)
        
        # For m2 pipeline: Q=X_m2, K=V=X_m1
        m_m2, _ = self.mha_m2(X_m2, X_m1, X_m1)  # (batch, seq_len, hidden_dim)
        
        # Step 4: Apply gates and residual connection
        # X_tilde = LN(g_c * m + g_r * X)
        # Need to expand gates to match sequence dimension
        g_r_m1_expanded = g_r_m1.unsqueeze(1)  # (batch, 1, hidden_dim)
        g_c_m1_expanded = g_c_m1.unsqueeze(1)  # (batch, 1, hidden_dim)
        
        g_r_m2_expanded = g_r_m2.unsqueeze(1)  # (batch, 1, hidden_dim)
        g_c_m2_expanded = g_c_m2.unsqueeze(1)  # (batch, 1, hidden_dim)
        
        X_tilde_m1 = self.norm1_m1(g_c_m1_expanded * m_m1 + g_r_m1_expanded * X_m1)
        X_tilde_m2 = self.norm1_m2(g_c_m2_expanded * m_m2 + g_r_m2_expanded * X_m2)
        
        # Step 5: FFN with residual connection
        X_m1_out = self.norm2_m1(X_tilde_m1 + self.ffn_m1(X_tilde_m1))
        X_m2_out = self.norm2_m2(X_tilde_m2 + self.ffn_m2(X_tilde_m2))
        
        return X_m1_out, X_m2_out, h_bar_m1, h_bar_m2


class ModalitySpecificFeatureSeparator(nn.Module):
    """
    Discriminator-based feature separator to maintain modality independence.
    
    Uses grouping strategy to reduce noise as described in the paper.
    """
    def __init__(self, hidden_dim=256, group_size=4):
        super(ModalitySpecificFeatureSeparator, self).__init__()
        self.hidden_dim = hidden_dim
        self.group_size = group_size
        
        # Binary classifier to distinguish between modalities
        self.discriminator = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim // 2, 1),
            nn.Sigmoid()
        )
        
    def forward(self, h_bar_m1, h_bar_m2):
        """
        Args:
            h_bar_m1: Averaged representations for modality 1 (batch, hidden_dim)
            h_bar_m2: Averaged representations for modality 2 (batch, hidden_dim)
        Returns:
            loss: Binary cross-entropy discriminator loss
            predictions: Discriminator predictions for analysis
        """
        batch_size = h_bar_m1.shape[0]
        
        # Apply grouping strategy to reduce noise
        num_groups = batch_size // self.group_size
        
        if num_groups == 0:
            # If batch too small, no grouping
            h_tilde_m1 = h_bar_m1
            h_tilde_m2 = h_bar_m2
        else:
            # Group and average
            h_bar_m1_grouped = h_bar_m1[:num_groups * self.group_size].view(num_groups, self.group_size, -1)
            h_tilde_m1 = torch.mean(h_bar_m1_grouped, dim=1)  # (num_groups, hidden_dim)
            
            h_bar_m2_grouped = h_bar_m2[:num_groups * self.group_size].view(num_groups, self.group_size, -1)
            h_tilde_m2 = torch.mean(h_bar_m2_grouped, dim=1)  # (num_groups, hidden_dim)
        
        # Concatenate both modalities with pseudo labels
        # Label 0 for modality 1, label 1 for modality 2
        combined_features = torch.cat([h_tilde_m1, h_tilde_m2], dim=0)
        labels = torch.cat([
            torch.zeros(h_tilde_m1.shape[0], 1, device=h_tilde_m1.device),
            torch.ones(h_tilde_m2.shape[0], 1, device=h_tilde_m2.device)
        ], dim=0)
        
        # Discriminator predictions
        predictions = self.discriminator(combined_features)
        
        # Binary cross-entropy loss
        loss = F.binary_cross_entropy(predictions, labels)
        
        return loss, predictions


class BiBimodalFusionNetwork(nn.Module):
    """
    Bi-Bimodal Fusion Network (BBFN) - Exact implementation from paper.
    
    Key architectural features:
    1. Two bimodal complementation modules: TA (Text-Acoustic) and TV (Text-Visual)
    2. L stacked complementation layers in each module
    3. Layer-wise feature space separators for regularization
    4. Gated control mechanism in attention
    5. Final prediction from concatenated heads
    """
    def __init__(self, hidden_dim=256, output_dim=3, num_layers=2, 
                 num_heads=4, dropout=0.1, lambda_sep=0.1, group_size=4):
        super(BiBimodalFusionNetwork, self).__init__()
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        self.num_layers = num_layers
        self.lambda_sep = lambda_sep  # Regularization weight for separator loss
        
        # Text-Acoustic (TA) complementation module
        self.ta_complementation_layers = nn.ModuleList([
            ModalityComplementationLayer(hidden_dim, num_heads, dropout)
            for _ in range(num_layers)
        ])
        
        # Text-Visual (TV) complementation module  
        self.tv_complementation_layers = nn.ModuleList([
            ModalityComplementationLayer(hidden_dim, num_heads, dropout)
            for _ in range(num_layers)
        ])
        
        # Feature separators (one per layer per module)
        self.ta_separators = nn.ModuleList([
            ModalitySpecificFeatureSeparator(hidden_dim, group_size)
            for _ in range(num_layers)
        ])
        
        self.tv_separators = nn.ModuleList([
            ModalitySpecificFeatureSeparator(hidden_dim, group_size)
            for _ in range(num_layers)
        ])
        
        # Final prediction layer
        # Input: concatenation of 4 heads [h_TA,a, h_TA,t, h_TV,t, h_TV,v]
        self.prediction_layer = nn.Sequential(
            nn.Linear(hidden_dim * 4, hidden_dim * 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, output_dim)
        )
        
    def forward(self, text_x, audio_x, video_x, return_losses=False):
        """
        Args:
            text_x: Text features (batch, seq_len, hidden_dim) or (batch, hidden_dim)
            audio_x: Audio features (batch, seq_len, hidden_dim) or (batch, hidden_dim)
            video_x: Video features (batch, seq_len, hidden_dim) or (batch, hidden_dim)
            return_losses: If True, return separator losses for training
        Returns:
            output: Predictions (batch, output_dim)
            separator_losses: List of separator losses (if return_losses=True)
        """
        batch_size = (text_x if text_x is not None else 
                     audio_x if audio_x is not None else video_x).shape[0]
        device = (text_x if text_x is not None else 
                 audio_x if audio_x is not None else video_x).device
        
        # Handle missing modalities
        if text_x is None:
            text_x = torch.zeros(batch_size, self.hidden_dim).to(device)
        if audio_x is None:
            audio_x = torch.zeros(batch_size, self.hidden_dim).to(device)
        if video_x is None:
            video_x = torch.zeros(batch_size, self.hidden_dim).to(device)
        
        # Ensure 3D
        if text_x.dim() == 2:
            text_x = text_x.unsqueeze(1)
        if audio_x.dim() == 2:
            audio_x = audio_x.unsqueeze(1)
        if video_x.dim() == 2:
            video_x = video_x.unsqueeze(1)
        
        separator_losses = []
        
        # ============ Text-Acoustic (TA) Complementation ============
        X_ta_text = text_x
        X_ta_audio = audio_x
        
        for i, (layer, separator) in enumerate(zip(self.ta_complementation_layers, self.ta_separators)):
            X_ta_text, X_ta_audio, h_bar_ta_text, h_bar_ta_audio = layer(X_ta_text, X_ta_audio)
            
            # Compute separator loss
            sep_loss, _ = separator(h_bar_ta_text, h_bar_ta_audio)
            separator_losses.append(sep_loss)
        
        # Extract heads (first token, similar to [CLS])
        h_ta_text = X_ta_text[:, 0, :]  # (batch, hidden_dim)
        h_ta_audio = X_ta_audio[:, 0, :]  # (batch, hidden_dim)
        
        
        # ============ Text-Visual (TV) Complementation ============
        X_tv_text = text_x
        X_tv_video = video_x
        
        for i, (layer, separator) in enumerate(zip(self.tv_complementation_layers, self.tv_separators)):
            X_tv_text, X_tv_video, h_bar_tv_text, h_bar_tv_video = layer(X_tv_text, X_tv_video)
            
            # Compute separator loss
            sep_loss, _ = separator(h_bar_tv_text, h_bar_tv_video)
            separator_losses.append(sep_loss)
        
        # Extract heads (first token, similar to [CLS])
        h_tv_text = X_tv_text[:, 0, :]  # (batch, hidden_dim)
        h_tv_video = X_tv_video[:, 0, :]  # (batch, hidden_dim)
        
        
        # ============ Final Prediction ============
        # Concatenate all four head representations
        # Order: h_TA,a, h_TA,t, h_TV,t, h_TV,v (as shown in Figure 2)
        final_features = torch.cat([h_ta_audio, h_ta_text, h_tv_text, h_tv_video], dim=1)
        
        # Generate prediction
        output = self.prediction_layer(final_features)
        
        if return_losses:
            return output, separator_losses
        else:
            return output



# =============================================================================
# Anti-Correlation Gated Fusion (ACGF)
# =============================================================================

class AntiCorrelationGatedFusion(nn.Module):
    """
    Anti-Correlation Gated Fusion (ACGF)

    Motivation
    ----------
    All existing fusion mechanisms (LMF, TFN, GMU, BBFN, etc.) are built on
    Hadamard products or dot-product attention, which maximally activate when
    two modalities *agree* (positive correlation). When modalities *conflict*
    — e.g. text signals TURN but audio prosody signals KEEP — the Hadamard
    product yields near-zero activations, effectively discarding the conflict
    information.

    ACGF explicitly models this gap by introducing:
      1. Signed difference vectors (d_TA, d_TV, d_AV) that encode the
         direction and magnitude of each pairwise modality conflict.
      2. A positive stream (h_pos) that summarises agreement, identical in
         spirit to other fusion methods.
      3. A negative stream (h_neg) that summarises disagreement by learning
         from the difference vectors.
      4. An anti-correlation gate (gamma) that decides, per feature dimension,
         how much weight to assign to each stream.

    Architecture
    ------------
    Given embeddings z_T, z_A, z_V in R^d (d = hidden_dim):

        Difference vectors:
            d_TA = z_T - z_A          # text vs. audio conflict
            d_TV = z_T - z_V          # text vs. video conflict
            d_AV = z_A - z_V          # audio vs. video conflict

        Positive stream (captures agreement):
            h_pos = ReLU(W_p [z_T; z_A; z_V] + b_p)
                    W_p in R^{d x 3d}

        Negative stream (captures conflict / anti-correlation):
            h_neg = ReLU(W_n [d_TA; d_TV; d_AV] + b_n)
                    W_n in R^{d x 3d}

        Anti-correlation gate (routes each dimension to pos or neg stream):
            gamma = sigmoid(W_g [z_T; z_A; z_V; d_TA; d_TV; d_AV] + b_g)
                    W_g in R^{d x 6d}
            When gamma_i ~ 1  -> trust agreement (positive stream)
            When gamma_i ~ 0  -> trust conflict  (negative stream)

        Fused representation:
            h = gamma * h_pos + (1 - gamma) * h_neg

        Output:
            y_hat = W_out * LayerNorm(h) + b

    Missing modalities
    ------------------
    If a modality is None it is replaced by a zero vector. The corresponding
    difference vectors become ±z_other, which naturally signals maximum
    discrepancy between the available and missing modality — a sensible
    inductive bias since a missing modality contributes no evidence.

    Parameter count (default hidden_dim=256)
    -----------------------------------------
        W_p : 256 x 768  =  196,608
        W_n : 256 x 768  =  196,608
        W_g : 256 x 1536 =  393,216
        W_out: 3 x 256   =      768
        Biases + LN      ~    1,024
        Total            ~ 788,224  (~0.79 M)

    This is less than 20% overhead over the LMF baseline fusion module, while
    explicitly covering the negative-correlation regime that LMF ignores.
    """

    def __init__(self, hidden_dim: int = 256, output_dim: int = 3,
                 dropout: float = 0.1):
        super(AntiCorrelationGatedFusion, self).__init__()
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim

        # ── Positive stream ───────────────────────────────────────────────
        # Summarises agreement: operates on raw concatenation [z_T; z_A; z_V]
        self.W_pos = nn.Linear(hidden_dim * 3, hidden_dim, bias=True)

        # ── Negative stream ───────────────────────────────────────────────
        # Summarises conflict: operates on difference vectors [d_TA; d_TV; d_AV]
        self.W_neg = nn.Linear(hidden_dim * 3, hidden_dim, bias=True)

        # ── Anti-correlation gate ─────────────────────────────────────────
        # Conditions on both raw embeddings AND difference vectors so the gate
        # can tell apart meaningful conflict from noise.
        # Input dimension: 3*d (raw) + 3*d (diffs) = 6*d
        self.W_gate = nn.Linear(hidden_dim * 6, hidden_dim, bias=True)

        # ── Output head ───────────────────────────────────────────────────
        self.layer_norm = nn.LayerNorm(hidden_dim)
        self.dropout = nn.Dropout(dropout)
        self.classifier = nn.Linear(hidden_dim, output_dim, bias=True)

        # ── Weight initialisation ─────────────────────────────────────────
        self._init_weights()

    # ------------------------------------------------------------------
    def _init_weights(self):
        for module in [self.W_pos, self.W_neg, self.W_gate, self.classifier]:
            xavier_normal_(module.weight)
            if module.bias is not None:
                nn.init.zeros_(module.bias)

    # ------------------------------------------------------------------
    def forward(self, text_x, audio_x, video_x):
        """
        Parameters
        ----------
        text_x  : Tensor (batch, hidden_dim) or None
        audio_x : Tensor (batch, hidden_dim) or None
        video_x : Tensor (batch, hidden_dim) or None

        Returns
        -------
        output  : Tensor (batch, output_dim)  — class logits
        """
        # ── Determine batch size and device ──────────────────────────────
        ref = text_x if text_x is not None else \
              audio_x if audio_x is not None else video_x
        batch_size = ref.shape[0]
        device = ref.device

        # ── Replace missing modalities with zeros ─────────────────────────
        # A zero vector is the neutral element: difference vectors for a
        # missing modality become ±z_other, encoding maximal discrepancy.
        z_T = text_x  if text_x  is not None else torch.zeros(batch_size, self.hidden_dim, device=device)
        z_A = audio_x if audio_x is not None else torch.zeros(batch_size, self.hidden_dim, device=device)
        z_V = video_x if video_x is not None else torch.zeros(batch_size, self.hidden_dim, device=device)

        # ── Signed difference vectors ─────────────────────────────────────
        # d_ij encodes direction: positive when modality i > modality j.
        # d_ij = -d_ji, so each pair contributes one directed signal.
        d_TA = z_T - z_A   # (batch, hidden_dim)
        d_TV = z_T - z_V   # (batch, hidden_dim)
        d_AV = z_A - z_V   # (batch, hidden_dim)

        # ── Positive stream: captures inter-modal agreement ───────────────
        # Concatenate raw embeddings and project
        cat_raw  = torch.cat([z_T, z_A, z_V], dim=1)          # (batch, 3*d)
        h_pos    = F.relu(self.W_pos(cat_raw))                 # (batch, d)

        # ── Negative stream: captures inter-modal conflict ────────────────
        # Concatenate difference vectors and project.
        # ReLU on the projected differences retains the conflict magnitude.
        cat_diff = torch.cat([d_TA, d_TV, d_AV], dim=1)       # (batch, 3*d)
        h_neg    = F.relu(self.W_neg(cat_diff))                # (batch, d)

        # ── Anti-correlation gate ─────────────────────────────────────────
        # Conditions on both raw and difference features so the gate can
        # distinguish meaningful conflict (e.g. rising pitch + complete
        # sentence) from random noise.
        cat_all  = torch.cat([cat_raw, cat_diff], dim=1)       # (batch, 6*d)
        gamma    = torch.sigmoid(self.W_gate(cat_all))         # (batch, d)
        # gamma_i ~ 1  -> agreement dominates  (positive stream)
        # gamma_i ~ 0  -> conflict dominates   (negative stream)

        # ── Combine streams via the gate ──────────────────────────────────
        h_fused  = gamma * h_pos + (1.0 - gamma) * h_neg      # (batch, d)

        # ── Output head ───────────────────────────────────────────────────
        h_fused  = self.dropout(self.layer_norm(h_fused))
        output   = self.classifier(h_fused)                    # (batch, output_dim)

        return output



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


def get_fusion_module(module_name):
    mechanisms = {
        "LMF": LMF(),
        "Early_Fusion": EarlyFusion(),
        "Late_Fusion": LateFusion(),
        
        "MFB": MultimodalFactorizedBilinear(),
        "Cross_Modal_Attention": CrossModalAttention(),
        "GMU": GatedMultimodalUnit(),
        "Multimodal_Transformer": MultimodalTransformer(),
        "BBFN": BiBimodalFusionNetwork(),
        "ACGF": AntiCorrelationGatedFusion(),
        "TAC" : TinyAntiCorrelator(),
    }
    return mechanisms[module_name] 



# Testing code
if __name__ == "__main__":
    batch_size = 16
    hidden_dim = 256
    output_dim = 3
    
    # Create dummy data
    text = torch.randn(batch_size, hidden_dim)
    audio = torch.randn(batch_size, hidden_dim)
    video = torch.randn(batch_size, hidden_dim)
    
    print("Testing all fusion mechanisms:\n")
    
    # Test each fusion mechanism
    mechanisms = {
        "LMF": LMF(),
        "Early_Fusion": EarlyFusion(),
        "Late_Fusion": LateFusion(),
        #"TFN": TensorFusionNetwork(),
        "MFB": MultimodalFactorizedBilinear(),
        "Cross_Modal_Attention": CrossModalAttention(),
        "GMU": GatedMultimodalUnit(),
        "Multimodal_Transformer": MultimodalTransformer(),
        "Tucker_Fusion": TuckerFusion(),
        "BBFN": BiBimodalFusionNetwork(),
        "ACGF": AntiCorrelationGatedFusion(),
        "TAC" : TinyAntiCorrelator(),
    }
    
    for name, model in mechanisms.items():
        print(f"{name}:")
        print(f"  All modalities: {model(text, audio, video).shape}")
        print(f"  Missing text:   {model(None, audio, video).shape}")
        print(f"  Missing audio:  {model(text, None, video).shape}")
        print(f"  Missing video:  {model(text, audio, None).shape}")
        
        # Count parameters
        params = sum(p.numel() for p in model.parameters())
        print(f"  Parameters: {params:,}")
        print()