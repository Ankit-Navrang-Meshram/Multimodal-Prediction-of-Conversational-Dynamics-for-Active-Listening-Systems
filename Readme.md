# Multimodal Turn-Taking Detection in Conversations

A PyTorch-based framework for detecting turn-taking behaviors in face-to-face conversations using multimodal fusion of text, audio, and video data.

## 📋 Table of Contents
- [Overview](#overview)
- [Features](#features)
- [Installation](#installation)
- [Dataset Structure](#dataset-structure)
- [Quick Start](#quick-start)
- [Training Pipeline](#training-pipeline)
- [Model Architecture](#model-architecture)
- [Fusion Mechanisms](#fusion-mechanisms)
- [Results](#results)
- [Citation](#citation)

## 🎯 Overview

This repository implements a comprehensive multimodal learning framework for turn-taking detection in conversations. The system classifies conversation instances into three categories:
- **KEEP**: Speaker continues speaking
- **TURN**: Turn-taking occurs (speaker changes)
- **BACKCHANNEL**: Listener provides feedback without taking turn

The framework supports:
- ✅ Unimodal baselines (Text, Audio, Video)
- ✅ Multiple fusion strategies (Early, Late, LMF, Tensor Fusion, etc.)
- ✅ Temporal context windows for dynamic behavior modeling
- ✅ Transfer learning from pretrained encoders

## ✨ Features

- **Modular Architecture**: Easy to extend with new fusion mechanisms
- **Pretrained Encoders**: 
  - GPT-2 for text
  - HuBERT for audio
  - VideoMAE for video
- **7 Fusion Mechanisms**:
  - Early Fusion
  - Late Fusion
  - Low-rank Multimodal Fusion (LMF)
  - Tensor Fusion Network (TFN)
  - Cross-Modal Attention
  - Gated Multimodal Unit (GMU)
  - Multimodal Transformer
- **Temporal Windowing**: Configurable time windows for context
- **Comprehensive Metrics**: Accuracy, F1, Precision, Recall per class

## 🛠️ Installation

### Requirements
- Python 3.8+
- CUDA 11.0+ (for GPU training)
- PyTorch 2.0+

### Setup

```bash
# Clone repository
git clone https://github.com/yourusername/multimodal-turn-taking.git
cd multimodal-turn-taking

# Create virtual environment
conda create -n mmtt python=3.10
conda activate mmtt

# Install dependencies
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
pip install transformers librosa opencv-python pandas scikit-learn tensorboard tqdm
```

## 📁 Dataset Structure

```
project/
├── data/
│   ├── transcripts/           # Transcript CSV files (one per video)
│   │   ├── video1.csv
│   │   └── video2.csv
│   └── splits/               # Train/val/test splits
│       ├── train.csv
│       ├── val.csv
│       └── test.csv
├── audio/                    # Audio files
│   ├── video1.wav
│   └── video2.wav
├── videos/                   # Video files
│   ├── video1.mp4
│   └── video2.mp4
└── ...
```

### Data Preparation

1. **Prepare Transcripts**: Each video should have a corresponding transcript CSV with columns:
   ```
   video_id, sentence_id, start, end, text, speaker_label, label
   ```

2. **Curate Dataset**:
   ```bash
   python data/curate_dataset.py \
     --input_folder raw_transcripts/ \
     --output_file data/curated_dataset.csv
   ```

3. **Create Splits**:
   ```bash
   python data/split_dataset.py \
     --input_file data/curated_dataset.csv \
     --output_dir data/splits \
     --train_ratio 0.7 \
     --val_ratio 0.15 \
     --test_ratio 0.15
   ```

## 🚀 Quick Start

### 1. Train Unimodal Models

```bash
# Text model
python train_unimodal.py \
  --modal text \
  --train_csv data/splits/train.csv \
  --val_csv data/splits/val.csv \
  --transcript_folder data/transcripts/ \
  --audio_folder /path/to/audio/ \
  --video_folder /path/to/videos/ \
  --batch_size 16 \
  --n_epoch 50

# Audio model
python train_unimodal.py \
  --modal audio \
  --batch_size 8 \
  --n_epoch 50 \
  --audio_folder /path/to/audio/ \
  --video_folder /path/to/videos/

# Video model
python train_unimodal.py \
  --modal video \
  --batch_size 4 \
  --n_epoch 50 \
  --audio_folder /path/to/audio/ \
  --video_folder /path/to/videos/
```

### 2. Train Multimodal Model

```bash
# With pretrained encoders
python train_multimodal.py \
  --fusion_type lmf \
  --text_ckpt log/text_model/best_model_acc.pt \
  --audio_ckpt log/audio_model/best_model_acc.pt \
  --video_ckpt log/video_model/best_model_acc.pt \
  --freeze_encoders \
  --batch_size 8 \
  --n_epoch 50 \
  --audio_folder /path/to/audio/ \
  --video_folder /path/to/videos/

# From scratch
python train_multimodal.py \
  --fusion_type early \
  --batch_size 8 \
  --n_epoch 50 \
  --audio_folder /path/to/audio/ \
  --video_folder /path/to/videos/
```

## 🏗️ Model Architecture

### Unimodal Models
Each modality uses a pretrained encoder with a classification head:

```python
# Text: GPT-2 (768-dim) → Projection (256-dim) → Classifier
# Audio: HuBERT (1024-dim) → Projection (256-dim) → Classifier  
# Video: VideoMAE (768-dim) → Projection (256-dim) → Classifier
```

### Multimodal Architecture

```
┌─────────┐  ┌─────────┐  ┌─────────┐
│  Text   │  │  Audio  │  │  Video  │
│ Encoder │  │ Encoder │  │ Encoder │
└────┬────┘  └────┬────┘  └────┬────┘
     │            │            │
     ▼            ▼            ▼
┌─────────┐  ┌─────────┐  ┌─────────┐
│Project  │  │Project  │  │Project  │
│to 256-d │  │to 256-d │  │to 256-d │
└────┬────┘  └────┬────┘  └────┬────┘
     │            │            │
     └────────┬───┴────────────┘
              ▼
       ┌─────────────┐
       │   Fusion    │
       │ Mechanism   │
       └──────┬──────┘
              ▼
       ┌─────────────┐
       │ Classifier  │
       │   (3 cls)   │
       └─────────────┘
```

## 🔀 Fusion Mechanisms

| Fusion Type | Description | Parameters |
|------------|-------------|------------|
| **Early** | Concatenate features → MLP | `--fusion_type early` |
| **Late** | Independent classifiers → Weighted sum | `--fusion_type late` |
| **LMF** | Low-rank multimodal fusion | `--fusion_type lmf --rank 16` |
| **Tensor** | Outer product fusion | `--fusion_type tensor` |
| **Attention** | Cross-modal attention | `--fusion_type attention --num_heads 4` |
| **Gated** | Gated multimodal unit | `--fusion_type gated` |
| **Transformer** | Self-attention across modalities | `--fusion_type transformer --num_heads 4 --num_layers 2` |

### Example: Testing Different Fusion Mechanisms

```bash
for fusion in early late lmf tensor attention gated transformer; do
  python train_multimodal.py \
    --fusion_type $fusion \
    --batch_size 8 \
    --n_epoch 50 \
    --audio_folder /path/to/audio/ \
    --video_folder /path/to/videos/
done
```

## 📊 Training Pipeline

### 1. Data Curation
Extract instances from raw transcripts with temporal context:
```bash
python data/curate_dataset.py
```

### 2. Dataset Splitting
Split by videos (ensures no data leakage):
```bash
python data/split_dataset.py
```

### 3. Unimodal Training
Train individual modality baselines:
```bash
python train_unimodal.py --modal [text|audio|video]
```

### 4. Multimodal Training
Fuse modalities with different strategies:
```bash
python train_multimodal.py --fusion_type [early|late|lmf|...]
```

## 📈 Results

Example results on conversation dataset:

| Model | Accuracy | Macro-F1 | Keep-F1 | Turn-F1 | BC-F1 |
|-------|----------|----------|---------|---------|-------|
| Text | 0.XX | 0.XX | 0.XX | 0.XX | 0.XX |
| Audio | 0.XX | 0.XX | 0.XX | 0.XX | 0.XX |
| Video | 0.XX | 0.XX | 0.XX | 0.XX | 0.XX |
| Early Fusion | 0.XX | 0.XX | 0.XX | 0.XX | 0.XX |
| LMF | 0.XX | 0.XX | 0.XX | 0.XX | 0.XX |
| Transformer | 0.XX | 0.XX | 0.XX | 0.XX | 0.XX |

*Note: Fill in with your actual results*

## 🎛️ Key Parameters

### Dataset Parameters
- `--t`: Temporal window size (default: 2.0 seconds)
- `--max_frames`: Number of video frames (default: 16)

### Training Parameters
- `--batch_size`: Batch size (default: 8)
- `--n_epoch`: Number of epochs (default: 100)
- `--lr`: Learning rate (default: 1e-5)
- `--freeze_encoders`: Freeze pretrained encoders

### Fusion Parameters
- `--rank`: Rank for LMF (default: 16)
- `--num_heads`: Attention heads (default: 4)
- `--num_layers`: Transformer layers (default: 2)
- `--dropout`: Dropout rate (default: 0.1)

## 📝 File Structure

```
.
├── data/
│   ├── curate_dataset.py      # Dataset curation
│   ├── split_dataset.py       # Train/val/test splitting
│   └── dataset.py             # PyTorch dataset class
├── model/
│   ├── unimodal.py            # Unimodal models
│   ├── multimodal.py          # Multimodal architecture
│   └── fusion.py              # Fusion mechanisms
├── train_unimodal.py          # Unimodal training script
├── train_multimodal.py        # Multimodal training script
└── README.md
```

## 🔧 Advanced Usage

### Custom Fusion Mechanism

Add your own fusion in `model/fusion.py`:

```python
class CustomFusion(nn.Module):
    def __init__(self, hidden_dim=256, output_dim=3):
        super().__init__()
        # Your architecture here
        
    def forward(self, text_x, audio_x, video_x):
        # Your fusion logic here
        return output
```

Then register it in `get_fusion_mechanism()`:
```python
fusion_map = {
    'custom': CustomFusion,
    # ... other fusions
}
```

### Fine-tuning Pretrained Models

```bash
# Fine-tune all layers
python train_multimodal.py \
  --fusion_type lmf \
  --text_ckpt path/to/text.pt \
  --audio_ckpt path/to/audio.pt \
  --video_ckpt path/to/video.pt
  # Don't use --freeze_encoders

# Freeze encoders, train only fusion
python train_multimodal.py \
  --fusion_type lmf \
  --text_ckpt path/to/text.pt \
  --freeze_encoders
```

## 📚 Citation

If you use this code in your research, please cite:

```bibtex
@software{multimodal_turn_taking_2025,
  author = {Your Name},
  title = {Multimodal Turn-Taking Detection Framework},
  year = {2025},
  url = {https://github.com/yourusername/multimodal-turn-taking}
}
```

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## 📄 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 🙏 Acknowledgments

- GPT-2: OpenAI
- HuBERT: Facebook AI
- VideoMAE: MCG-NJU
- Fusion mechanisms inspired by various multimodal learning papers

## 📧 Contact

For questions or issues, please open an issue on GitHub or contact [your.email@example.com]

---

**Happy Training! 🚀**