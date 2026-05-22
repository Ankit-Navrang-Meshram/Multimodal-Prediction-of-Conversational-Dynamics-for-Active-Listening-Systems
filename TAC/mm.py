# import torch
# from torch import nn

# from transformers import AutoTokenizer, GPT2Model
# from transformers import AutoProcessor, HubertModel
# from transformers import AutoImageProcessor, VideoMAEModel

# from fusion import TinyAntiCorrelator


# class LanguageModel(nn.Module):
#     def __init__(self, pretrained_model_name_or_path="openai-community/gpt2", return_embeddings=False):
#         super(LanguageModel, self).__init__()
#         self.transformer = GPT2Model.from_pretrained(pretrained_model_name_or_path)
#         self.return_embeddings = return_embeddings
#         hidden_size = self.transformer.config.n_embd
#         self.proj = nn.Linear(hidden_size, 256)
#         self.out_layer = nn.Linear(256, 3)


#     def forward(self, inputs):
#         hidden_state = self.transformer(inputs).last_hidden_state #(bs, len, 768)
#         last_hidden_state = hidden_state[:, -1, :]
#         proj = self.proj(last_hidden_state)
#         if self.return_embeddings: return proj
#         out = self.out_layer(proj)
#         return out
    

# class AudioModel(nn.Module):
#     def __init__(self, pretrained_model_name_or_path="facebook/hubert-base-ls960", return_embeddings=False):
#         super(AudioModel, self).__init__()
#         self.hubert = HubertModel.from_pretrained(pretrained_model_name_or_path)
#         self.return_embeddings = return_embeddings
#         hidden_size = self.hubert.config.hidden_size
#         self.avg_pool = nn.AdaptiveAvgPool1d(1)
#         self.proj = nn.Linear(hidden_size, 256)
#         self.out_layer = nn.Linear(256, 3)


#     def forward(self, inputs):
#         hidden_state = self.hubert(inputs).last_hidden_state
#         avg_pooled = self.avg_pool(hidden_state.transpose(1, 2)).squeeze(-1)
#         proj = self.proj(avg_pooled)
#         if self.return_embeddings: return proj
#         out = self.out_layer(proj)
#         return out
    

# class VisionModel(nn.Module):
#     def __init__(self, pretrained_model_name_or_path="MCG-NJU/videomae-base", return_embeddings=False):
#         super(VisionModel, self).__init__()
#         self.model = VideoMAEModel.from_pretrained(pretrained_model_name_or_path)
#         self.return_embeddings = return_embeddings
#         hidden_size = self.model.config.hidden_size
#         self.avg_pool = nn.AdaptiveAvgPool1d(1)
#         self.proj = nn.Linear(hidden_size, 256)
#         self.out_layer = nn.Linear(256, 3)


#     def forward(self, inputs):
#         hidden_state = self.model(inputs).last_hidden_state
#         avg_pooled = self.avg_pool(hidden_state.transpose(1, 2)).squeeze(-1)
#         proj = self.proj(avg_pooled)
#         if self.return_embeddings: return proj
#         out = self.out_layer(proj)
#         return out
    

# class LanguageAudioVisionModel(nn.Module):
#     def __init__(self, text_ckpt_path=None, audio_ckpt_path=None, vision_ckpt_path=None , fusion_module = None):
#         super(LanguageAudioVisionModel, self).__init__()

#         self.text_model = LanguageModel(return_embeddings=True)
#         if text_ckpt_path is not None:
#             print("text model loaded succesfully ...")
#             self.text_model.load_state_dict(torch.load(text_ckpt_path), strict=False)
#         self.audio_model = AudioModel(return_embeddings=True)
#         if audio_ckpt_path is not None:
#             print("audio model loaded succesfully ...")
#             self.audio_model.load_state_dict(torch.load(audio_ckpt_path), strict=False)
#         self.vision_model = VisionModel(return_embeddings=True)
#         if vision_ckpt_path is not None:
#             print("vision model loaded succesfully ...")
#             self.vision_model.load_state_dict(torch.load(vision_ckpt_path), strict=False)

#         self.fusion = TinyAntiCorrelator()

#     def forward(self, text_inputs, audio_inputs, vision_inputs):
#         text_proj = self.text_model(text_inputs) if text_inputs is not None else None
#         audio_proj = self.audio_model(audio_inputs) if audio_inputs is not None else None
#         vision_proj = self.vision_model(vision_inputs) if vision_inputs is not None else None

#         out = self.fusion(text_proj, audio_proj, vision_proj)
#         return out
            
    

# def load_mm_model(args):
#     model = LanguageAudioVisionModel(args.fusion_module).to(args.device)
#     model.load_state_dict(torch.load(args.ckpt_path), strict=False)
#     model.eval()
#     return model


# def text_processor(text):
#     text = text.strip().lower()
#     symbols = [".", ",", "!", "?", ":", ";", "(", ")", "[", "]", "{", "}", "<", ">", "\"", "'"]
#     for symbol in symbols:
#         text = text.replace(symbol, "")
#     return text


# def load_processors():
#     tokenizer = AutoTokenizer.from_pretrained("openai-community/gpt2")
#     tokenizer.pad_token = tokenizer.eos_token
#     audio_processor = AutoProcessor.from_pretrained("facebook/hubert-large-ls960-ft")
#     video_processor = AutoImageProcessor.from_pretrained("MCG-NJU/videomae-base")
#     return tokenizer, text_processor, audio_processor, video_processor


import torch
from torch import nn

from transformers import AutoTokenizer, GPT2Model
from transformers import AutoProcessor, HubertModel
from transformers import AutoImageProcessor, VideoMAEModel

from peft import get_peft_model, LoraConfig
from fusion import TinyAntiCorrelator

class LanguageModel(nn.Module):
    def __init__(self, pretrained_model_name_or_path="openai-community/gpt2", return_embeddings=False):
        super(LanguageModel, self).__init__()
        base_model = GPT2Model.from_pretrained(pretrained_model_name_or_path)
        
        # Inject LoRA into attention blocks
        lora_config = LoraConfig(
            r=8, lora_alpha=16, target_modules=["c_attn"], lora_dropout=0.1
        )
        self.transformer = get_peft_model(base_model, lora_config)
        
        self.return_embeddings = return_embeddings
        hidden_size = self.transformer.config.n_embd
        self.proj = nn.Linear(hidden_size, 256) # Aligned to dim=128

    def forward(self, inputs):
        hidden_state = self.transformer(inputs).last_hidden_state
        last_hidden_state = hidden_state[:, -1, :]
        return self.proj(last_hidden_state)
    

class AudioModel(nn.Module):
    def __init__(self, pretrained_model_name_or_path="facebook/hubert-base-ls960", return_embeddings=False):
        super(AudioModel, self).__init__()
        base_model = HubertModel.from_pretrained(pretrained_model_name_or_path)
        
        lora_config = LoraConfig(
            r=16, lora_alpha=16, target_modules=["q_proj", "v_proj"], lora_dropout=0.1
        )
        self.hubert = get_peft_model(base_model, lora_config)
        
        self.return_embeddings = return_embeddings
        hidden_size = self.hubert.config.hidden_size
        self.avg_pool = nn.AdaptiveAvgPool1d(1)
        self.proj = nn.Linear(hidden_size, 256) # Aligned to dim=128

    def forward(self, inputs):
        hidden_state = self.hubert(inputs).last_hidden_state
        avg_pooled = self.avg_pool(hidden_state.transpose(1, 2)).squeeze(-1)
        return self.proj(avg_pooled)
    

class VisionModel(nn.Module):
    def __init__(self, pretrained_model_name_or_path="MCG-NJU/videomae-base", return_embeddings=False):
        super(VisionModel, self).__init__()
        base_model = VideoMAEModel.from_pretrained(pretrained_model_name_or_path)
        
        lora_config = LoraConfig(
            r=16, lora_alpha=16, target_modules=["query", "value"], lora_dropout=0.1
        )
        self.model = get_peft_model(base_model, lora_config)
        
        self.return_embeddings = return_embeddings
        hidden_size = self.model.config.hidden_size
        self.avg_pool = nn.AdaptiveAvgPool1d(1)
        self.proj = nn.Linear(hidden_size, 256) # Aligned to dim=128

    def forward(self, inputs):
        hidden_state = self.model(inputs).last_hidden_state
        avg_pooled = self.avg_pool(hidden_state.transpose(1, 2)).squeeze(-1)
        return self.proj(avg_pooled)
    

class LanguageAudioVisionModel(nn.Module):
    def __init__(self, text_ckpt_path=None, audio_ckpt_path=None, vision_ckpt_path=None, fusion_module=None):
        super(LanguageAudioVisionModel, self).__init__()

        self.text_model = LanguageModel(return_embeddings=True)
        self.audio_model = AudioModel(return_embeddings=True)
        self.vision_model = VisionModel(return_embeddings=True)

        print("Fusion module : TinyAntiCorrelation")
        self.fusion = TinyAntiCorrelator(hidden_dim=256, output_dim=3)

    def forward(self, text_inputs, audio_inputs, vision_inputs):
        text_proj = self.text_model(text_inputs) if text_inputs is not None else None
        audio_proj = self.audio_model(audio_inputs) if audio_inputs is not None else None
        vision_proj = self.vision_model(vision_inputs) if vision_inputs is not None else None

        out = self.fusion(text_proj, audio_proj, vision_proj)
        return out



def text_processor(text):
    text = text.strip().lower()
    symbols = [".", ",", "!", "?", ":", ";", "(", ")", "[", "]", "{", "}", "<", ">", "\"", "'"]
    for symbol in symbols:
        text = text.replace(symbol, "")
    return text

def load_processors():
    tokenizer = AutoTokenizer.from_pretrained("openai-community/gpt2")
    tokenizer.pad_token = tokenizer.eos_token
    audio_processor = AutoProcessor.from_pretrained("facebook/hubert-large-ls960-ft")
    video_processor = AutoImageProcessor.from_pretrained("MCG-NJU/videomae-base")
    return tokenizer, text_processor, audio_processor, video_processor