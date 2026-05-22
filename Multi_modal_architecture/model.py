# ============================================================================
# File: model.py
# Multimodal Token LLM Model
# ============================================================================

import torch
import torch.nn as nn
from transformers import GPT2LMHeadModel, GPT2Config, AutoTokenizer


class MultimodalTokenLLM(nn.Module):
    """
    Multimodal Token LLM for Turn-Taking Prediction
    
    Architecture:
    1. Tokenize video, audio, text into discrete tokens
    2. Embed tokens into LLM space
    3. Concatenate: [VID][AUD][TXT][PREDICT]
    4. Pass through LLM
    5. Classify final token: keep/turn/backchannel
    """
    def __init__(
        self, 
        llm_model_name="gpt2",
        video_vocab_size=8192,
        audio_vocab_size=1024,
        num_classes=3,
        freeze_llm=False,
        use_pretrained_tokenizers=True
    ):
        super().__init__()
        
        # Load or initialize tokenizers
        if use_pretrained_tokenizers:
            self.video_tokenizer = VideoTokenizer(vocab_size=video_vocab_size)
            self.audio_tokenizer = SpeechTokenizer(vocab_size=audio_vocab_size)
        else:
            self.video_tokenizer = None
            self.audio_tokenizer = None
        
        self.text_tokenizer = AutoTokenizer.from_pretrained(llm_model_name)
        self.text_tokenizer.pad_token = self.text_tokenizer.eos_token
        
        # Load LLM
        self.llm = GPT2LMHeadModel.from_pretrained(llm_model_name)
        self.hidden_dim = self.llm.config.n_embd  # 768 for GPT2
        
        if freeze_llm:
            for param in self.llm.parameters():
                param.requires_grad = False
        
        # Modality embeddings
        self.video_embedding = nn.Embedding(video_vocab_size, self.hidden_dim)
        self.audio_embedding = nn.Embedding(audio_vocab_size, self.hidden_dim)
        
        # Special tokens
        self.predict_token = nn.Parameter(torch.randn(1, 1, self.hidden_dim))
        
        # Modality type embeddings (helps LLM distinguish modalities)
        self.modality_type_embedding = nn.Embedding(4, self.hidden_dim)  # video, audio, text, predict
        
        # Classifier head
        self.classifier = nn.Sequential(
            nn.Linear(self.hidden_dim, self.hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(self.hidden_dim // 2, num_classes)
        )
        
        # Initialize embeddings
        nn.init.normal_(self.video_embedding.weight, std=0.02)
        nn.init.normal_(self.audio_embedding.weight, std=0.02)
        nn.init.normal_(self.modality_type_embedding.weight, std=0.02)
        
    def forward(self, video_tokens=None, audio_tokens=None, text_ids=None, 
                attention_mask=None, return_embeddings=False):
        """
        Args:
            video_tokens: (B, T_v) discrete video tokens
            audio_tokens: (B, T_a) discrete audio tokens  
            text_ids: (B, T_t) text token ids
            attention_mask: (B, total_length) attention mask
        Returns:
            logits: (B, 3) classification logits
        """
        B = video_tokens.shape[0] if video_tokens is not None else \
            audio_tokens.shape[0] if audio_tokens is not None else text_ids.shape[0]
        device = video_tokens.device if video_tokens is not None else \
                 audio_tokens.device if audio_tokens is not None else text_ids.device
        
        embeddings_list = []
        modality_ids_list = []
        
        # Video embeddings
        if video_tokens is not None:
            video_embeds = self.video_embedding(video_tokens)  # (B, T_v, H)
            embeddings_list.append(video_embeds)
            modality_ids_list.append(torch.zeros(video_tokens.shape, dtype=torch.long, device=device))
        
        # Audio embeddings
        if audio_tokens is not None:
            audio_embeds = self.audio_embedding(audio_tokens)  # (B, T_a, H)
            embeddings_list.append(audio_embeds)
            modality_ids_list.append(torch.ones(audio_tokens.shape, dtype=torch.long, device=device))
        
        # Text embeddings
        if text_ids is not None:
            text_embeds = self.llm.transformer.wte(text_ids)  # (B, T_t, H)
            embeddings_list.append(text_embeds)
            modality_ids_list.append(torch.full(text_ids.shape, 2, dtype=torch.long, device=device))
        
        # Predict token
        predict_embed = self.predict_token.expand(B, -1, -1)  # (B, 1, H)
        embeddings_list.append(predict_embed)
        modality_ids_list.append(torch.full((B, 1), 3, dtype=torch.long, device=device))
        
        # Concatenate all embeddings
        sequence_embeds = torch.cat(embeddings_list, dim=1)  # (B, total_length, H)
        modality_ids = torch.cat(modality_ids_list, dim=1)  # (B, total_length)
        
        # Add modality type embeddings
        modality_type_embeds = self.modality_type_embedding(modality_ids)
        sequence_embeds = sequence_embeds + modality_type_embeds
        
        # Create attention mask if not provided
        if attention_mask is None:
            attention_mask = torch.ones(sequence_embeds.shape[:2], device=device)
        
        if return_embeddings:
            return sequence_embeds
        
        # Pass through LLM
        outputs = self.llm(
            inputs_embeds=sequence_embeds,
            attention_mask=attention_mask,
            output_hidden_states=True
        )
        
        # Get final token representation (the PREDICT token)
        final_hidden = outputs.hidden_states[-1][:, -1, :]  # (B, H)
        
        # Classify
        logits = self.classifier(final_hidden)  # (B, 3)
        
        return logits