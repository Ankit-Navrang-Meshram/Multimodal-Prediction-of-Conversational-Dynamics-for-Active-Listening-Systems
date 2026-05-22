"""
Uni-modal encoder backbones.

Each encoder produces a fixed 256-dimensional representation that is shared
across all fusion experiments. Encoders are pretrained once (Stage 1) and
held frozen during fusion training (Stage 2), so that benchmark differences
reflect the quality of the fusion strategy, not the encoder.

Backbones (768-dim hidden states, projected to 256):

    Text  : GPT-2  (openai-community/gpt2)
            - last token of the autoregressive hidden state
    Audio : HuBERT (facebook/hubert-base-ls960)
            - frame-level features, mean-pooled over time
    Video : VideoMAE (MCG-NJU/videomae-base)
            - patch tokens from 16 sampled frames, mean-pooled

All three encoders expose a `return_embeddings` flag. When True they output
the 256-d projection (used by the fusion module). When False they output
3-class logits via an attached classification head (used during Stage 1
uni-modal pre-training).
"""

import torch
import torch.nn as nn
from transformers import (
    GPT2Model,
    HubertModel,
    VideoMAEModel,
    AutoTokenizer,
    AutoProcessor,
    AutoImageProcessor,
)


PROJECTION_DIM = 256
NUM_CLASSES = 3


class LanguageModel(nn.Module):
    """GPT-2 text encoder + 256-d projection + optional 3-class head."""

    def __init__(
        self,
        pretrained_model_name_or_path: str = "openai-community/gpt2",
        return_embeddings: bool = False,
    ):
        super().__init__()
        self.transformer = GPT2Model.from_pretrained(pretrained_model_name_or_path)
        self.return_embeddings = return_embeddings

        hidden_size = self.transformer.config.n_embd
        self.proj = nn.Linear(hidden_size, PROJECTION_DIM)
        self.out_layer = nn.Linear(PROJECTION_DIM, NUM_CLASSES)

    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        # Take the last hidden state at the final token position — captures
        # the full autoregressive context of the dialogue history.
        hidden = self.transformer(input_ids).last_hidden_state  # (B, T, 768)
        last_hidden = hidden[:, -1, :]                          # (B, 768)
        z = self.proj(last_hidden)                              # (B, 256)
        if self.return_embeddings:
            return z
        return self.out_layer(z)


class AudioModel(nn.Module):
    """HuBERT audio encoder + 256-d projection + optional 3-class head."""

    def __init__(
        self,
        pretrained_model_name_or_path: str = "facebook/hubert-base-ls960",
        return_embeddings: bool = False,
    ):
        super().__init__()
        self.hubert = HubertModel.from_pretrained(pretrained_model_name_or_path)
        self.return_embeddings = return_embeddings

        hidden_size = self.hubert.config.hidden_size
        self.avg_pool = nn.AdaptiveAvgPool1d(1)
        self.proj = nn.Linear(hidden_size, PROJECTION_DIM)
        self.out_layer = nn.Linear(PROJECTION_DIM, NUM_CLASSES)

    def forward(self, waveform: torch.Tensor) -> torch.Tensor:
        # Frame-level hidden states then mean-pool over time.
        hidden = self.hubert(waveform).last_hidden_state            # (B, T_a, 768)
        pooled = self.avg_pool(hidden.transpose(1, 2)).squeeze(-1)  # (B, 768)
        z = self.proj(pooled)
        if self.return_embeddings:
            return z
        return self.out_layer(z)


class VisionModel(nn.Module):
    """VideoMAE video encoder + 256-d projection + optional 3-class head."""

    def __init__(
        self,
        pretrained_model_name_or_path: str = "MCG-NJU/videomae-base",
        return_embeddings: bool = False,
    ):
        super().__init__()
        self.model = VideoMAEModel.from_pretrained(pretrained_model_name_or_path)
        self.return_embeddings = return_embeddings

        hidden_size = self.model.config.hidden_size
        self.avg_pool = nn.AdaptiveAvgPool1d(1)
        self.proj = nn.Linear(hidden_size, PROJECTION_DIM)
        self.out_layer = nn.Linear(PROJECTION_DIM, NUM_CLASSES)

    def forward(self, pixel_values: torch.Tensor) -> torch.Tensor:
        # Spatio-temporal patch tokens, then mean-pool.
        hidden = self.model(pixel_values).last_hidden_state         # (B, T_v, 768)
        pooled = self.avg_pool(hidden.transpose(1, 2)).squeeze(-1)
        z = self.proj(pooled)
        if self.return_embeddings:
            return z
        return self.out_layer(z)


# ---------------------------------------------------------------------------
# Pre-/post-processing helpers
# ---------------------------------------------------------------------------

def normalise_text(text: str) -> str:
    """Lowercase and strip basic punctuation before tokenisation."""
    text = text.strip().lower()
    for sym in [".", ",", "!", "?", ":", ";", "(", ")", "[", "]", "{", "}",
                "<", ">", '"', "'"]:
        text = text.replace(sym, "")
    return text


def load_processors():
    """Load the HuggingFace tokenizer / processors used by all three encoders.

    Returns
    -------
    (tokenizer, text_normaliser, audio_processor, video_processor)
    """
    tokenizer = AutoTokenizer.from_pretrained("openai-community/gpt2")
    tokenizer.pad_token = tokenizer.eos_token
    audio_processor = AutoProcessor.from_pretrained("facebook/hubert-large-ls960-ft")
    video_processor = AutoImageProcessor.from_pretrained("MCG-NJU/videomae-base")
    return tokenizer, normalise_text, audio_processor, video_processor
