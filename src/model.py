"""
End-to-end multimodal model: frozen encoders + selectable fusion module.

The :class:`LanguageAudioVisionModel` plugs together the three uni-modal
encoders (text, audio, video) and a fusion mechanism chosen by name from
:func:`src.fusion.get_fusion_module`. During Stage 2 fusion training the
encoders are typically frozen (see ``scripts/train_fusion.py``); this class
does not perform the freezing itself so that the training script remains
responsible for that policy.
"""

from __future__ import annotations
from typing import Optional

import torch
import torch.nn as nn

from .encoders import LanguageModel, AudioModel, VisionModel
from .fusion import get_fusion_module, FUSION_REGISTRY


class LanguageAudioVisionModel(nn.Module):
    """Compose the three encoders with a fusion module.

    Parameters
    ----------
    text_ckpt_path, audio_ckpt_path, vision_ckpt_path : str, optional
        Paths to Stage 1 uni-modal checkpoints. If provided, each encoder
        is initialised from the corresponding ``.pt`` file (non-strict
        loading so the temporary Stage 1 classification head can be
        discarded silently).
    fusion_module : str, optional
        Key from :data:`src.fusion.FUSION_REGISTRY`. Defaults to ``"LMF"``.
    fusion_kwargs : dict, optional
        Forwarded to the fusion module constructor (e.g. for rank or
        dropout overrides).
    """

    def __init__(
        self,
        text_ckpt_path: Optional[str] = None,
        audio_ckpt_path: Optional[str] = None,
        vision_ckpt_path: Optional[str] = None,
        fusion_module: Optional[str] = None,
        fusion_kwargs: Optional[dict] = None,
    ):
        super().__init__()

        # Uni-modal encoders configured to return 256-d embeddings.
        self.text_model = LanguageModel(return_embeddings=True)
        if text_ckpt_path is not None:
            print(f"[model] Loading text encoder from {text_ckpt_path}")
            self.text_model.load_state_dict(
                torch.load(text_ckpt_path, map_location="cpu"), strict=False
            )

        self.audio_model = AudioModel(return_embeddings=True)
        if audio_ckpt_path is not None:
            print(f"[model] Loading audio encoder from {audio_ckpt_path}")
            self.audio_model.load_state_dict(
                torch.load(audio_ckpt_path, map_location="cpu"), strict=False
            )

        self.vision_model = VisionModel(return_embeddings=True)
        if vision_ckpt_path is not None:
            print(f"[model] Loading vision encoder from {vision_ckpt_path}")
            self.vision_model.load_state_dict(
                torch.load(vision_ckpt_path, map_location="cpu"), strict=False
            )

        # Fusion module — defaults to LMF if not specified.
        name = fusion_module if fusion_module is not None else "LMF"
        if name not in FUSION_REGISTRY:
            raise ValueError(
                f"Unknown fusion module '{name}'. "
                f"Available: {sorted(FUSION_REGISTRY.keys())}"
            )
        print(f"[model] Fusion module: {name}")
        self.fusion = get_fusion_module(name, **(fusion_kwargs or {}))

    # ------------------------------------------------------------------
    def forward(self, text_inputs, audio_inputs, vision_inputs):
        """
        Parameters
        ----------
        text_inputs   : (B, T) int tensor of token ids, or None
        audio_inputs  : (B, L) float tensor of waveform, or None
        vision_inputs : (B, F, 3, H, W) float tensor of frame pixels, or None

        Returns
        -------
        logits : (B, 3) class logits over {KEEP, TURN, BACKCHANNEL}
        """
        z_t = self.text_model(text_inputs)    if text_inputs   is not None else None
        z_a = self.audio_model(audio_inputs)  if audio_inputs  is not None else None
        z_v = self.vision_model(vision_inputs) if vision_inputs is not None else None
        return self.fusion(z_t, z_a, z_v)

    # ------------------------------------------------------------------
    def freeze_encoders(self):
        """Disable gradient updates for all three encoders."""
        for p in self.text_model.parameters():   p.requires_grad = False
        for p in self.audio_model.parameters():  p.requires_grad = False
        for p in self.vision_model.parameters(): p.requires_grad = False
