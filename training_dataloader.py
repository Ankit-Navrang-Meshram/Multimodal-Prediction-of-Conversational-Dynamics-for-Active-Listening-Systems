import os
import pandas as pd
import torch
from torch.utils.data import DataLoader, Dataset
from transformers import AutoTokenizer, AutoProcessor, AutoImageProcessor

# Initialize processors (for compatibility)
tokenizer = AutoTokenizer.from_pretrained("openai-community/gpt2")
tokenizer.pad_token = tokenizer.eos_token
audio_processor = AutoProcessor.from_pretrained("facebook/hubert-large-ls960-ft")
video_processor = AutoImageProcessor.from_pretrained("MCG-NJU/videomae-base")

class MultiModalDataset(Dataset):
    def __init__(self, data_root, split, modal="all", tokenizer=None, audio_processor=None, video_processor=None):
        assert split in ["train", "val", "test"]
        assert modal in ["audio", "video", "text", "all"]
        self.modal = modal
        
        # Load metadata
        features_dir = os.path.join(data_root, "features", split)
        metadata_path = os.path.join(features_dir, "metadata.csv")
        if not os.path.exists(metadata_path):
            raise FileNotFoundError(
                f"Metadata file not found at {metadata_path}. "
                f"Please run feature extraction first."
            )
        
        df = pd.read_csv(metadata_path)
        self.data_list = []
        for _, row in df.iterrows():
            feature_path = os.path.join(features_dir, row["feature_file"])
            self.data_list.append({
                "feature_path": feature_path,
                "label": row["label"]
            })
    
    def __len__(self):
        return len(self.data_list)
    
    def __getitem__(self, idx):
        data = self.data_list[idx]
        
        # Load precomputed features
        features = torch.load(data["feature_path"], weights_only=False)
        
        out_dict = {"label": features["label"]}
        
        if self.modal in ["text", "all"]:
            out_dict["text"] = features["text"]
        
        if self.modal in ["audio", "all"]:
            out_dict["audio"] = features["audio"]
        
        if self.modal in ["video", "all"]:
            out_dict["video"] = features["video"]
        
        return out_dict


def collate_fn(batch):
    out_tuple = ()
    
    if "text" in batch[0]:
        # Already tensors, just extract and pad
        input_ids = [x["text"]["input_ids"] if isinstance(x["text"]["input_ids"], torch.Tensor) 
                     else torch.tensor(x["text"]["input_ids"], dtype=torch.long) for x in batch]
        attention_mask = [x["text"]["attention_mask"] if isinstance(x["text"]["attention_mask"], torch.Tensor)
                          else torch.tensor(x["text"]["attention_mask"], dtype=torch.long) for x in batch]
        
        text = {
            "input_ids": torch.nn.utils.rnn.pad_sequence(input_ids, batch_first=True, padding_value=0),
            "attention_mask": torch.nn.utils.rnn.pad_sequence(attention_mask, batch_first=True, padding_value=0),
        }
        out_tuple += (text,)
    
    if "audio" in batch[0]:
        # Fix: Use clone() instead of torch.tensor() to avoid the warning
        audio_seqs = []
        for x in batch:
            audio_data = x["audio"]
            if isinstance(audio_data, torch.Tensor):
                audio_seqs.append(audio_data.clone().detach().float())
            else:
                audio_seqs.append(torch.tensor(audio_data, dtype=torch.float))
        
        audio = torch.nn.utils.rnn.pad_sequence(audio_seqs, batch_first=True, padding_value=0.0)
        out_tuple += (audio,)
    
    if "video" in batch[0]:
        # Video should already be tensors from preprocessing
        video_data = []
        for x in batch:
            vid = x["video"]
            if isinstance(vid, torch.Tensor):
                video_data.append(vid)
            else:
                video_data.append(torch.tensor(vid, dtype=torch.float))
        
        video = {
            "pixel_values": torch.stack(video_data),
        }
        out_tuple += (video,)
    
    label = torch.tensor([x["label"] for x in batch], dtype=torch.long)
    out_tuple += (label,)
    
    return out_tuple


if __name__ == "__main__":
    data_root = "./dataset"
    test_set = MultiModalDataset(data_root, "test", modal="all", 
                                  tokenizer=tokenizer, 
                                  audio_processor=audio_processor, 
                                  video_processor=video_processor)
    test_loader = DataLoader(test_set, batch_size=2, shuffle=False, collate_fn=collate_fn)
    
    text, audio, video, label = next(iter(test_loader))
    print(text["input_ids"].shape)
    print(audio.shape)
    print(video["pixel_values"].shape)
    print(label)