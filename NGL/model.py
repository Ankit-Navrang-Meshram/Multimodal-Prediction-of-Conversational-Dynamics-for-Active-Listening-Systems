import torch
from torch import nn
import torch.nn.functional as F
from transformers import GPT2Model, HubertModel, VideoMAEModel

HUBERT_MIN_INPUT_LENGTH = 400  # HuBERT CNN downsamples by 320x total


class CrossModalAttention(nn.Module):
    def __init__(self, embed_dim=768, num_heads=8, proj_dim=256):
        super(CrossModalAttention, self).__init__()
        self.mha  = nn.MultiheadAttention(embed_dim, num_heads, batch_first=True)
        self.proj = nn.Linear(embed_dim, proj_dim)

    def forward(self, query, key_value):
        attn_out, _ = self.mha(query, key_value, key_value)
        pooled = attn_out.mean(dim=1)   # (B, 768)
        return self.proj(pooled)        # (B, 256)


class MLPHead(nn.Module):
    def __init__(self, in_dim=256, num_classes=3):
        super(MLPHead, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, 256), nn.ReLU(),
            nn.Linear(256, 64),    nn.ReLU(),
            nn.Linear(64, num_classes),
        )

    def forward(self, x):
        return self.net(x)


class LanguageModel(nn.Module):
    def __init__(self, pretrained_model_name_or_path="openai-community/gpt2"):
        super(LanguageModel, self).__init__()
        self.transformer = GPT2Model.from_pretrained(pretrained_model_name_or_path)
        self.proj = nn.Linear(self.transformer.config.n_embd, 256)

    def forward(self, inputs):
        hidden_state      = self.transformer(inputs).last_hidden_state  # (B, T, 768)
        last_hidden_state = hidden_state[:, -1, :]
        return hidden_state, self.proj(last_hidden_state)               # (B,T,768), (B,256)


class AudioModel(nn.Module):
    def __init__(self, pretrained_model_name_or_path="facebook/hubert-base-ls960"):
        super(AudioModel, self).__init__()
        self.hubert   = HubertModel.from_pretrained(pretrained_model_name_or_path)
        self.avg_pool = nn.AdaptiveAvgPool1d(1)
        self.proj     = nn.Linear(self.hubert.config.hidden_size, 256)

    def forward(self, inputs):
        # FIX Bug 4: always float32 — HuBERT Conv1d weights are float32.
        # Under autocast inputs arrive as float16; this prevents both the
        # "no engine" crash and float16 NaN overflow inside the CNN stack.
        inputs = inputs.float()

        # FIX (previous session): pad sequences shorter than CNN minimum
        if inputs.shape[-1] < HUBERT_MIN_INPUT_LENGTH:
            inputs = F.pad(inputs, (0, HUBERT_MIN_INPUT_LENGTH - inputs.shape[-1]))

        hidden_state = self.hubert(inputs).last_hidden_state            # (B, T_audio, 768)
        avg_pooled   = self.avg_pool(hidden_state.transpose(1, 2)).squeeze(-1)
        return hidden_state, self.proj(avg_pooled)                      # (B,T,768), (B,256)


class VisionModel(nn.Module):
    def __init__(self, pretrained_model_name_or_path="MCG-NJU/videomae-base"):
        super(VisionModel, self).__init__()
        self.model    = VideoMAEModel.from_pretrained(pretrained_model_name_or_path)
        self.avg_pool = nn.AdaptiveAvgPool1d(1)
        self.proj     = nn.Linear(self.model.config.hidden_size, 256)

    def forward(self, inputs):
        hidden_state = self.model(inputs).last_hidden_state             # (B, T_vision, 768)
        avg_pooled   = self.avg_pool(hidden_state.transpose(1, 2)).squeeze(-1)
        return hidden_state, self.proj(avg_pooled)                      # (B,T,768), (B,256)


class DNCCEnsembleModel(nn.Module):
    def __init__(self, text_ckpt=None, audio_ckpt=None, vision_ckpt=None):
        super(DNCCEnsembleModel, self).__init__()

        # Layer 1: Uni-modal Encoders
        self.text_encoder   = LanguageModel()
        self.audio_encoder  = AudioModel()
        self.vision_encoder = VisionModel()

        if text_ckpt:
            self.text_encoder.load_state_dict(torch.load(text_ckpt), strict=False)
        if audio_ckpt:
            self.audio_encoder.load_state_dict(torch.load(audio_ckpt), strict=False)
        if vision_ckpt:
            self.vision_encoder.load_state_dict(torch.load(vision_ckpt), strict=False)

        # Layer 2: Bi-modal Cross-Attention Blocks
        self.attn_ta = CrossModalAttention()  # G4: Text   → Audio
        self.attn_at = CrossModalAttention()  # G5: Audio  → Text
        self.attn_tv = CrossModalAttention()  # G6: Text   → Vision
        self.attn_vt = CrossModalAttention()  # G7: Vision → Text
        self.attn_av = CrossModalAttention()  # G8: Audio  → Vision
        self.attn_va = CrossModalAttention()  # G9: Vision → Audio

        # Layer 3: Independent Prediction Heads (one per ensemble member)
        self.heads = nn.ModuleList([MLPHead(num_classes=3) for _ in range(9)])

    def forward(self, text_inputs, audio_inputs, vision_inputs):

        # ── Step 1: Frozen encoder forward ───────────────────────────────
        with torch.no_grad():
            # FIX Bug 5: DO NOT call .eval() inside forward().
            # torch.no_grad() alone stops gradient computation for frozen layers.
            # Calling .eval() here permanently sets sub-encoder training flags
            # to False — model.train() on the outer model cannot undo it,
            # silently breaking BatchNorm running-stat updates if encoders
            # are ever unfrozen.
            h_T, z_T = self.text_encoder(text_inputs)
            h_A, z_A = self.audio_encoder(audio_inputs)
            h_V, z_V = self.vision_encoder(vision_inputs)

        # FIX Bug 2: cast encoder outputs to float32 before cross-attention.
        # Under torch.cuda.amp.autocast() the encoders run in float16, so
        # h_T / h_A / h_V are float16.  MultiheadAttention with d_model=768
        # computes QK^T values that can overflow float16 (max ≈ 65504),
        # producing Inf → softmax(Inf) = NaN which propagates through the
        # entire forward pass.  Casting here is cheap (no copy on CPU).
        h_T, z_T = h_T.float(), z_T.float()
        h_A, z_A = h_A.float(), z_A.float()
        h_V, z_V = h_V.float(), z_V.float()

        # ── Step 2: Cross-modal projections (G4–G9) ──────────────────────
        z_TA_t = self.attn_ta(query=h_T, key_value=h_A)
        z_TA_a = self.attn_at(query=h_A, key_value=h_T)
        z_TV_t = self.attn_tv(query=h_T, key_value=h_V)
        z_TV_v = self.attn_vt(query=h_V, key_value=h_T)
        z_AV_a = self.attn_av(query=h_A, key_value=h_V)
        z_AV_v = self.attn_va(query=h_V, key_value=h_A)

        # ── Step 3: Aggregate 9 representations ──────────────────────────
        z_ensemble = [z_T, z_A, z_V, z_TA_t, z_TA_a, z_TV_t, z_TV_v, z_AV_a, z_AV_v]

        # ── Step 4: Predict from each head → (9, B, 3) ───────────────────
        ensemble_logits = torch.stack(
            [head(z) for head, z in zip(self.heads, z_ensemble)]
        )

        if self.training:
            return ensemble_logits          # DNCCLoss needs all 9 individual logits
        else:
            probs = torch.softmax(ensemble_logits, dim=-1)
            return probs.mean(dim=0)        # (B, 3) averaged ensemble probability