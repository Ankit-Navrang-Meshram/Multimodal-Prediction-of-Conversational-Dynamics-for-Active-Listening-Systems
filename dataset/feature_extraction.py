from dataset import VideoConversationDataset , collate_fn
from transformers import AutoTokenizer, AutoProcessor, AutoImageProcessor, AutoModel
from torch.utils.data import DataLoader

import argparse
    
parser = argparse.ArgumentParser(description='Multi-Modal Video Conversation Dataset Feature Extraction Testing')
# Data paths
parser.add_argument('--csv_file', type=str, default='../data/splits/train.csv',
                    help='Path to CSV file')
parser.add_argument('--transcript_folder', type=str, default='../data/transcripts/',
                    help='Folder containing transcript CSV files')
parser.add_argument('--audio_folder', type=str, default='audio/',
                    help='Folder containing audio files')
parser.add_argument('--video_folder', type=str, default='videos/',
                    help='Folder containing video files')

# Dataset parameters
parser.add_argument('--t', type=float, default=2.0,
                    help='Time window in seconds')
parser.add_argument('--modal', type=str, default='all',
                    choices=['audio', 'video', 'text', 'all'],
                    help='Modality to use')
parser.add_argument('--max_frames', type=int, default=16,
                    help='Number of frames to extract')

# DataLoader parameters
parser.add_argument('--batch_size', type=int, default=4,
                    help='Batch size')
parser.add_argument('--num_workers', type=int, default=2,
                    help='Number of workers')


args = parser.parse_args()


# Initialize processors
tokenizer = AutoTokenizer.from_pretrained("openai-community/gpt2")
tokenizer.pad_token = tokenizer.eos_token
audio_processor = AutoProcessor.from_pretrained("facebook/hubert-large-ls960-ft")
video_processor = AutoImageProcessor.from_pretrained("MCG-NJU/videomae-base")

# Create dataset
dataset = VideoConversationDataset(
    csv_file=args.csv_file,
    transcript_folder=args.transcript_folder,
    audio_folder=args.audio_folder,
    video_folder=args.video_folder,
    t=2.0,
    modal="all",
    tokenizer=tokenizer,
    audio_processor=audio_processor,
    video_processor=video_processor
)

# Load models for feature extraction
text_model = AutoModel.from_pretrained("openai-community/gpt2")
audio_model = AutoModel.from_pretrained("facebook/hubert-large-ls960-ft")
video_model = AutoModel.from_pretrained("MCG-NJU/videomae-base")
dataloader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        collate_fn=collate_fn
    )
# Extract features in training loop
for batch in dataloader:
    # Text features
    text_outputs = text_model(**batch['text'])
    text_features = text_outputs.last_hidden_state
    
    # Audio features
    audio_outputs = audio_model(batch['audio'])
    audio_features = audio_outputs.last_hidden_state
    
    # Video features
    video_outputs = video_model(**batch['video'])
    video_features = video_outputs.last_hidden_state

    print(f"text features shape : {text_features.shape}")
    print(f"audio features shape : {audio_features.shape}")
    print(f"video features shape : {video_features.shape}")
    break