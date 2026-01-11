import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
import cv2
import librosa
import os
from tqdm import tqdm
from transformers import AutoTokenizer, AutoProcessor, AutoImageProcessor


class VideoConversationDataset(Dataset):
    """
    Multi-modal dataset for video conversation instances with temporal context.
    Returns preprocessed data ready for feature extraction.
    
    For each instance at time t, provides data from window [time-t, time]:
    - Preprocessed text (tokenized)
    - Preprocessed audio (processed input values)
    - Preprocessed video (processed pixel values)
    """
    
    def __init__(
        self,
        csv_file: str,
        transcript_folder: str,
        audio_folder: str,
        video_folder: str,
        t: float = 2.0,
        modal: str = "all",
        max_frames: int = 16,
        tokenizer=None,
        audio_processor=None,
        video_processor=None
    ):
        """
        Args:
            csv_file: Path to CSV file (train.csv, val.csv, or test.csv)
            transcript_folder: Folder containing transcript CSV files for each video
            audio_folder: Folder containing audio files (.wav, .mp3, etc.)
            video_folder: Folder containing video files (.mp4, .avi, etc.)
            t: Time window in seconds before the instance timestamp
            modal: Modality to use - "audio", "video", "text", or "all"
            max_frames: Number of frames to extract for video
            tokenizer: Tokenizer for text (e.g., GPT2)
            audio_processor: Processor for audio (e.g., HuBERT)
            video_processor: Processor for video (e.g., VideoMAE)
        """
        assert modal in ["audio", "video", "text", "all"]
        
        self.transcript_folder = transcript_folder
        self.audio_folder = audio_folder
        self.video_folder = video_folder
        self.t = t
        self.modal = modal
        self.max_frames = max_frames
        
        # Load CSV data
        self.data = pd.read_csv(csv_file)
        
        # Initialize processors based on modality
        if self.modal in ["text", "all"]:
            assert tokenizer is not None, "Tokenizer required for text modality"
            self.tokenizer = tokenizer
        
        if self.modal in ["audio", "all"]:
            assert audio_processor is not None, "Audio processor required for audio modality"
            self.audio_processor = audio_processor
            self.sampling_rate = self.audio_processor.feature_extractor.sampling_rate
        
        if self.modal in ["video", "all"]:
            assert video_processor is not None, "Video processor required for video modality"
            self.video_processor = video_processor
        
        # Cache for transcripts
        self.transcript_cache = {}
        
        # Build data list with preprocessing info and validate files
        print("Building dataset index and validating files...")
        self.data_list = []
        skipped_count = 0
        
        for idx in tqdm(range(len(self.data)), desc="Indexing dataset"):
            row = self.data.iloc[idx]
            
            video_id = row['video_id']
            instance_id = row['instance_id']
            label = row['label']
            time = row['time']
            speaker = row['speaker']
            listener = row['listener']
            
            # Validate that required files exist
            skip_instance = False
            
            if self.modal in ["audio", "all"]:
                audio_exists = False
                for ext in ['.wav', '.mp3', '.flac', '.m4a']:
                    if os.path.exists(os.path.join(self.audio_folder, f"{video_id}{ext}")):
                        audio_exists = True
                        break
                if not audio_exists:
                    skip_instance = True
            
            if self.modal in ["video", "all"]:
                video_exists = False
                for ext in ['.mp4', '.avi', '.mov', '.mkv']:
                    if os.path.exists(os.path.join(self.video_folder, f"{video_id}{ext}")):
                        video_exists = True
                        break
                if not video_exists:
                    skip_instance = True
            
            if skip_instance:
                skipped_count += 1
                continue
            
            # Map label to integer
            label_map = {'KEEP': 0, 'TURN': 1, 'BACKCHANNEL': 2}
            label_int = label_map.get(label, -1)
            
            data_item = {
                'video_id': video_id,
                'instance_id': instance_id,
                'label': label_int,
                'time': time,
                'speaker': speaker,
                'listener': listener
            }
            
            self.data_list.append(data_item)
        
        if skipped_count > 0:
            print(f"Skipped {skipped_count} instances due to missing files")
        
        print(f"Loaded dataset with {len(self.data_list)} instances")
        print(f"Time window: {t} seconds")
        print(f"Modality: {modal}")
        print(f"Max frames: {max_frames}")
    
    def __len__(self):
        return len(self.data_list)
    
    def _load_transcript(self, video_id: str) -> pd.DataFrame:
        """Load transcript CSV for a video (with caching)."""
        if video_id not in self.transcript_cache:
            transcript_path = os.path.join(self.transcript_folder, f"{video_id}.csv")
            if os.path.exists(transcript_path):
                self.transcript_cache[video_id] = pd.read_csv(transcript_path)
            else:
                # Try without .csv extension in case it's already included
                raise FileNotFoundError(f"Transcript not found: {transcript_path}")
        return self.transcript_cache[video_id]
    
    def _get_text_in_window(self, video_id: str, end_time: float) -> str:
        """Extract text transcription in the time window [end_time - t, end_time]."""
        transcript = self._load_transcript(video_id)
        start_time = max(0, end_time - self.t)
        
        # Filter words in the time window
        in_window = transcript[
            (transcript['end'] >= start_time) & 
            (transcript['end'] <= end_time)
        ]
        
        # Concatenate text in chronological order
        text = ' '.join(in_window['text'].astype(str).values)
        return text if text.strip() else "silence"
    
    def _load_audio(self, video_id: str, end_time: float):
        """Load and process audio segment."""
        # Try different audio file extensions
        audio_path = None
        for ext in ['.wav', '.mp3', '.flac', '.m4a']:
            potential_path = os.path.join(self.audio_folder, f"{video_id}{ext}")
            if os.path.exists(potential_path):
                audio_path = potential_path
                break
        
        if audio_path is None:
            raise FileNotFoundError(f"Audio file not found for video_id: {video_id}")
        
        # Load audio
        audio, _ = librosa.load(audio_path, sr=self.sampling_rate)
        
        # Calculate start and end samples
        start_time = max(0, end_time - self.t)
        start_sample = int(start_time * self.sampling_rate)
        end_sample = int(end_time * self.sampling_rate)
        
        # Extract segment
        audio_segment = audio[start_sample:end_sample]
        
        # Pad if segment is shorter than expected
        expected_length = int(self.t * self.sampling_rate)
        if len(audio_segment) < expected_length:
            audio_segment = np.pad(
                audio_segment,
                (expected_length - len(audio_segment), 0),
                mode='constant'
            )
        
        # Process with audio processor
        processed_audio = self.audio_processor(
            audio_segment,
            sampling_rate=self.sampling_rate,
            return_tensors="pt"
        ).input_values
        
        return processed_audio
    
    def _load_video_frames(self, video_id: str, end_time: float):
        """Load and process video frames."""
        # Try different video file extensions
        video_path = None
        for ext in ['.mp4', '.avi', '.mov', '.mkv']:
            potential_path = os.path.join(self.video_folder, f"{video_id}{ext}")
            if os.path.exists(potential_path):
                video_path = potential_path
                break
        
        if video_path is None:
            raise FileNotFoundError(f"Video file not found for video_id: {video_id}")
        
        # Open video
        cap = cv2.VideoCapture(video_path)
        original_fps = cap.get(cv2.CAP_PROP_FPS)
        
        # Calculate start and end frame numbers
        start_time = max(0, end_time - self.t)
        start_frame = int(start_time * original_fps)
        end_frame = int(end_time * original_fps)
        
        # Calculate total frames in window
        total_frames_in_window = end_frame - start_frame
        
        if total_frames_in_window <= 0:
            cap.release()
            raise ValueError(f"No frames in window for video_id: {video_id}")
        
        # Sample frames evenly to get exactly max_frames
        if total_frames_in_window >= self.max_frames:
            frame_indices = np.linspace(start_frame, end_frame - 1, self.max_frames, dtype=int)
        else:
            # If fewer frames available, repeat some frames
            frame_indices = np.linspace(start_frame, end_frame - 1, total_frames_in_window, dtype=int)
        
        frames = []
        for frame_idx in frame_indices:
            cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
            ret, frame = cap.read()
            if ret:
                # Convert BGR to RGB
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                frames.append(frame)
        
        cap.release()
        
        # Pad frames if needed
        while len(frames) < self.max_frames:
            frames.append(frames[-1] if frames else np.zeros((224, 224, 3), dtype=np.uint8))
        
        if len(frames) == 0:
            raise ValueError(f"No frames extracted for video_id: {video_id}")
        
        # Process with video processor
        processed_video = self.video_processor(frames, return_tensors="pt")
        
        return processed_video
    
    def __getitem__(self, idx: int) -> dict:
        """Get a single preprocessed instance."""
        data = self.data_list[idx]
        
        video_id = data['video_id']
        time = data['time']
        
        out_dict = {
            'video_id': video_id,
            'instance_id': data['instance_id'],
            'label': data['label'],
            'time': time,
            'speaker': data['speaker'],
            'listener': data['listener']
        }
        
        # Load and preprocess each modality with error handling
        try:
            if self.modal in ["text", "all"]:
                text = self._get_text_in_window(video_id, time)
                text_inputs = self.tokenizer(
                    text,
                    return_tensors="pt",
                    padding='max_length',
                    truncation=True,
                    max_length=512
                )
                out_dict['text'] = text_inputs
            
            if self.modal in ["audio", "all"]:
                audio_inputs = self._load_audio(video_id, time)
                out_dict['audio'] = audio_inputs
            
            if self.modal in ["video", "all"]:
                video_inputs = self._load_video_frames(video_id, time)
                out_dict['video'] = video_inputs
                
        except Exception as e:
            print(f"\nError processing video_id {video_id} at time {time}: {str(e)}")
            raise
        
        return out_dict


def collate_fn(batch):
    """Custom collate function for DataLoader."""
    out_dict = {
        'video_id': [x['video_id'] for x in batch],
        'instance_id': [x['instance_id'] for x in batch],
        'label': torch.tensor([x['label'] for x in batch]),
        'time': torch.tensor([x['time'] for x in batch]),
        'speaker': [x['speaker'] for x in batch],
        'listener': [x['listener'] for x in batch]
    }
    
    # Collate text if present
    if 'text' in batch[0]:
        out_dict['text'] = {
            'input_ids': torch.cat([x['text']['input_ids'] for x in batch], dim=0),
            'attention_mask': torch.cat([x['text']['attention_mask'] for x in batch], dim=0)
        }
    
    # Collate audio if present
    if 'audio' in batch[0]:
        out_dict['audio'] = torch.cat([x['audio'] for x in batch], dim=0)
    
    # Collate video if present
    if 'video' in batch[0]:
        out_dict['video'] = {
            'pixel_values': torch.cat([x['video']['pixel_values'] for x in batch], dim=0)
        }
    
    return out_dict


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Multi-Modal Video Conversation Dataset')
    
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
    
    # Testing
    parser.add_argument('--test', action='store_true',
                        help='Run test')
    
    args = parser.parse_args()
    
    print("="*60)
    print("Initializing Processors")
    print("="*60)
    
    # Initialize processors
    tokenizer = None
    audio_processor = None
    video_processor = None
    
    if args.modal in ["text", "all"]:
        print("Loading GPT-2 tokenizer...")
        tokenizer = AutoTokenizer.from_pretrained("openai-community/gpt2")
        tokenizer.pad_token = tokenizer.eos_token
    
    if args.modal in ["audio", "all"]:
        print("Loading HuBERT processor...")
        audio_processor = AutoProcessor.from_pretrained("facebook/hubert-large-ls960-ft")
    
    if args.modal in ["video", "all"]:
        print("Loading VideoMAE processor...")
        video_processor = AutoImageProcessor.from_pretrained("MCG-NJU/videomae-base")
    
    print("\n" + "="*60)
    print("Creating Dataset")
    print("="*60)
    
    dataset = VideoConversationDataset(
        csv_file=args.csv_file,
        transcript_folder=args.transcript_folder,
        audio_folder=args.audio_folder,
        video_folder=args.video_folder,
        t=args.t,
        modal=args.modal,
        max_frames=args.max_frames,
        tokenizer=tokenizer,
        audio_processor=audio_processor,
        video_processor=video_processor
    )
    
    dataloader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        collate_fn=collate_fn
    )
    
    if args.test:
        print("\n" + "="*60)
        print("Testing DataLoader")
        print("="*60)
        
        batch = next(iter(dataloader))
        
        print(f"\nBatch keys: {list(batch.keys())}")
        print(f"Batch size: {len(batch['video_id'])}")
        print(f"Video IDs: {batch['video_id']}")
        print(f"Labels: {batch['label']}")
        
        if 'text' in batch:
            print(f"\nText input_ids shape: {batch['text']['input_ids'].shape}")
            print(f"Text attention_mask shape: {batch['text']['attention_mask'].shape}")
        
        if 'audio' in batch:
            print(f"\nAudio shape: {batch['audio'].shape}")
        
        if 'video' in batch:
            print(f"\nVideo pixel_values shape: {batch['video']['pixel_values'].shape}")
        
        print("\n" + "="*60)
        print("Test completed successfully!")
        print("="*60)
    else:
        print("\nDataset created successfully!")
        print("Use --test flag to test loading batches")
        print(f"\nExample command:")
        print(f"python dataset.py --csv_file train.csv --modal all --test")

"""

# Test with all modalities
python dataset.py \
  --csv_file train.csv \
  --transcript_folder transcripts/ \
  --audio_folder audio/ \
  --video_folder videos/ \
  --modal all \
  --t 2.0 \
  --batch_size 4 \
  --test

# Use only audio
python dataset.py --modal audio --test

# Custom time window
python dataset.py --t 3.0 --max_frames 32 --test


"""