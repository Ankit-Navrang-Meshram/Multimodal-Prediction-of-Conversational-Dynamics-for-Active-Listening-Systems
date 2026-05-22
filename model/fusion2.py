import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from torch.nn.parameter import Parameter
from torch.nn.init import xavier_normal_, kaiming_normal_
import math
#from hmef_fusion import HierarchicalMixtureExpertFusion

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


class TensorFusionNetwork(nn.Module):
    """
    Tensor Fusion Network: Outer product of all modalities
    """
    def __init__(self, hidden_dim=256, output_dim=3, dropout=0.1):
        super(TensorFusionNetwork, self).__init__()
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        
        # Post-fusion dimensions
        post_fusion_dim = (hidden_dim + 1) ** 3
        
        self.post_fusion_dropout = nn.Dropout(p=dropout)
        self.post_fusion_layer_1 = nn.Linear(post_fusion_dim, hidden_dim)
        self.post_fusion_layer_2 = nn.Linear(hidden_dim, hidden_dim // 2)
        self.post_fusion_layer_3 = nn.Linear(hidden_dim // 2, output_dim)
        
    def forward(self, text_x, audio_x, video_x):
        batch_size = (text_x if text_x is not None else 
                     audio_x if audio_x is not None else video_x).shape[0]
        device = (text_x if text_x is not None else 
                 audio_x if audio_x is not None else video_x).device
        
        # Add constant 1 dimension for bias
        if text_x is None:
            text_x = torch.zeros(batch_size, self.hidden_dim).to(device)
        if audio_x is None:
            audio_x = torch.zeros(batch_size, self.hidden_dim).to(device)
        if video_x is None:
            video_x = torch.zeros(batch_size, self.hidden_dim).to(device)
        
        # Add the constant 1
        text_x = torch.cat([torch.ones(batch_size, 1).to(device), text_x], dim=1)
        audio_x = torch.cat([torch.ones(batch_size, 1).to(device), audio_x], dim=1)
        video_x = torch.cat([torch.ones(batch_size, 1).to(device), video_x], dim=1)
        
        # Compute outer product
        fusion_tensor = torch.bmm(text_x.unsqueeze(2), audio_x.unsqueeze(1))
        fusion_tensor = fusion_tensor.view(batch_size, -1, 1)
        fusion_tensor = torch.bmm(fusion_tensor, video_x.unsqueeze(1))
        fusion_tensor = fusion_tensor.view(batch_size, -1)
        
        # Post-fusion layers
        x = self.post_fusion_dropout(fusion_tensor)
        x = F.relu(self.post_fusion_layer_1(x))
        x = self.post_fusion_dropout(x)
        x = F.relu(self.post_fusion_layer_2(x))
        output = self.post_fusion_layer_3(x)
        
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


class TuckerFusion(nn.Module):
    """
    Tucker Decomposition Fusion (Generalization of LMF)
    """
    def __init__(self, hidden_dim=256, output_dim=3, rank=(16, 16, 16), dropout=0.1):
        super(TuckerFusion, self).__init__()
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        self.rank = rank
        
        # Tucker decomposition factors
        self.text_factor = Parameter(torch.Tensor(rank[0], hidden_dim + 1))
        self.audio_factor = Parameter(torch.Tensor(rank[1], hidden_dim + 1))
        self.video_factor = Parameter(torch.Tensor(rank[2], hidden_dim + 1))
        
        # Core tensor
        self.core_tensor = Parameter(torch.Tensor(rank[0], rank[1], rank[2]))
        
        # Post-fusion layers
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Sequential(
            nn.Linear(rank[0] * rank[1] * rank[2], hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, output_dim)
        )
        
        # Initialize
        xavier_normal_(self.text_factor)
        xavier_normal_(self.audio_factor)
        xavier_normal_(self.video_factor)
        xavier_normal_(self.core_tensor)
        
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
        
        # Add bias term
        text_x = torch.cat([torch.ones(batch_size, 1).to(device), text_x], dim=1)
        audio_x = torch.cat([torch.ones(batch_size, 1).to(device), audio_x], dim=1)
        video_x = torch.cat([torch.ones(batch_size, 1).to(device), video_x], dim=1)
        
        # Project to low-rank space
        text_proj = torch.matmul(text_x, self.text_factor.t())  # (batch, rank[0])
        audio_proj = torch.matmul(audio_x, self.audio_factor.t())  # (batch, rank[1])
        video_proj = torch.matmul(video_x, self.video_factor.t())  # (batch, rank[2])
        
        # Tucker contraction with core tensor
        fusion = torch.einsum('bi,ijk,bj,bk->b', 
                             text_proj, self.core_tensor, audio_proj, video_proj)
        
        # For better stability, we can also use sequential contraction
        # Reshape for batch processing
        fusion = torch.einsum('bi,bj,bk->bijk', text_proj, audio_proj, video_proj)
        fusion = fusion.view(batch_size, -1)
        
        fusion = self.dropout(fusion)
        output = self.fc(fusion)
        
        return output



import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.parameter import Parameter
from torch.nn.init import xavier_normal_

class GatedBilinearCrossAttention(nn.Module):
    """
    Gated Bilinear Cross-Attention (GBCA)
    Combines Low-rank Bilinear Pooling with Dynamic Gating and Cross-Attention.
    """
    def __init__(self, hidden_dim=256, output_dim=3, rank=32, dropout=0.1):
        super(GatedBilinearCrossAttention, self).__init__()
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        self.rank = rank
        
        # 1. Modality Confidence Gates (Global Context)
        self.gate_fc = nn.Sequential(
            nn.Linear(hidden_dim * 3, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 3),
            nn.Sigmoid()
        )
        
        # 2. Low-Rank Factorization Parameters (similar to LMF but for projection)
        # We project modalities into a common 'rank' space for bilinear interaction
        self.text_factor = nn.Linear(hidden_dim, rank)
        self.audio_factor = nn.Linear(hidden_dim, rank)
        self.video_factor = nn.Linear(hidden_dim, rank)
        
        # 3. Cross-Modal Attention to refine features
        self.cross_attn = nn.MultiheadAttention(rank, num_heads=4, dropout=dropout, batch_first=True)
        
        # 4. Final Fusion weights
        self.fusion_proj = nn.Sequential(
            nn.Linear(rank, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, output_dim)
        )

    def forward(self, text_x, audio_x, video_x):
        batch_size = (text_x if text_x is not None else 
                     audio_x if audio_x is not None else video_x).shape[0]
        device = (text_x if text_x is not None else 
                 audio_x if audio_x is not None else video_x).device
        
        # Handle missing modalities with zero-padding
        if text_x is None: text_x = torch.zeros(batch_size, self.hidden_dim).to(device)
        if audio_x is None: audio_x = torch.zeros(batch_size, self.hidden_dim).to(device)
        if video_x is None: video_x = torch.zeros(batch_size, self.hidden_dim).to(device)
        
        # Step 1: Global Gating
        # Learns which modality is "trustworthy" for this specific sample
        combined_raw = torch.cat([text_x, audio_x, video_x], dim=1)
        gates = self.gate_fc(combined_raw) # (batch, 3)
        
        text_g = text_x * gates[:, 0:1]
        audio_g = audio_x * gates[:, 1:2]
        video_g = video_x * gates[:, 2:3]
        
        # Step 2: Low-Rank Projection
        t_p = self.text_factor(text_g)   # (batch, rank)
        a_p = self.audio_factor(audio_g) # (batch, rank)
        v_p = self.video_factor(video_g) # (batch, rank)
        
        # Step 3: Bilinear interaction via Element-wise Product
        # This captures the "AND" relationship between modalities
        bilinear_fused = t_p * a_p * v_p # (batch, rank)
        
        # Step 4: Refinement with Attention
        # We treat the bilinear fused vector as a query to "attend" back 
        # to the individual modality projections
        modality_stack = torch.stack([t_p, a_p, v_p], dim=1) # (batch, 3, rank)
        query = bilinear_fused.unsqueeze(1) # (batch, 1, rank)
        
        # Cross-modal attention focuses on the most relevant modality for the fused state
        attn_out, _ = self.cross_attn(query, modality_stack, modality_stack)
        
        # Step 5: Final Output
        output = self.fusion_proj(attn_out.squeeze(1))
        
        return output




 
def get_fusion_module(module_name):
    mechanisms = {
        "LMF":                    LMF(),
        "Early_Fusion":           EarlyFusion(),
        "Late_Fusion":            LateFusion(),
        "TFN":                    TensorFusionNetwork(),
        "MFB":                    MultimodalFactorizedBilinear(),
        "Cross_Modal_Attention":  CrossModalAttention(),
        "GMU":                    GatedMultimodalUnit(),
        "Multimodal_Transformer": MultimodalTransformer(),
        "Tucker_Fusion":          TuckerFusion(),
        "GBCA":                   GatedBilinearCrossAttention(),
        #"HMEF":                   HierarchicalMixtureExpertFusion(),  # ← NEW
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
        "TFN": TensorFusionNetwork(),
        "MFB": MultimodalFactorizedBilinear(),
        "Cross_Modal_Attention": CrossModalAttention(),
        "GMU": GatedMultimodalUnit(),
        "Multimodal_Transformer": MultimodalTransformer(),
        "Tucker_Fusion": TuckerFusion(),
        "GBCA" : GatedBilinearCrossAttention()
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