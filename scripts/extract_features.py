"""
Pre-compute and cache tokenised text, audio waveforms, and video tensors
to ``.pt`` files for faster Stage 2 training.

Each call processes one CSV split (``train`` / ``val`` / ``test``) and
writes one ``.pt`` file per valid sample under
``<data_root>/features/<split>/<sentence_id>.pt``, plus a ``metadata.csv``
listing the surviving samples.

Samples missing any of the 16 expected video frames are silently skipped
so that downstream batches stay shape-consistent.

Example
-------
::

    python -m scripts.extract_features --data_root dataset/
"""

from __future__ import annotations

import argparse
import os
import sys

import cv2
import librosa
import pandas as pd
import torch
from tqdm import tqdm
from transformers import AutoTokenizer, AutoProcessor, AutoImageProcessor

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

N_VIDEO_FRAMES = 16


def build_processors():
    tokenizer = AutoTokenizer.from_pretrained("openai-community/gpt2")
    tokenizer.pad_token = tokenizer.eos_token
    audio_processor = AutoProcessor.from_pretrained("facebook/hubert-large-ls960-ft")
    video_processor = AutoImageProcessor.from_pretrained("MCG-NJU/videomae-base")
    return tokenizer, audio_processor, video_processor


def extract_split(data_root: str, split: str):
    """Extract and cache features for one CSV split."""
    print(f"[features] Processing {split} split")
    tokenizer, audio_proc, video_proc = build_processors()

    df = pd.read_csv(os.path.join(data_root, f"{split}.csv"), sep="\t")
    sids   = df["sentence_id"].values
    texts  = df["text"].values
    labels = df["label"].values

    features_dir = os.path.join(data_root, "features", split)
    os.makedirs(features_dir, exist_ok=True)

    audio_root = os.path.join(data_root, "audio")
    video_root = os.path.join(data_root, "video")
    sr = audio_proc.feature_extractor.sampling_rate

    valid = []
    for sid, text, label in tqdm(list(zip(sids, texts, labels)), total=len(sids)):
        video_id = "_".join(sid.split("_")[:-2])

        # Require all N video frames.
        img_paths = [os.path.join(video_root, video_id, sid, f"{i}.jpg")
                     for i in range(N_VIDEO_FRAMES)]
        if not all(os.path.exists(p) for p in img_paths):
            continue

        # Tokenise text.
        text_ids = tokenizer(text)
        text_features = {
            "input_ids":      text_ids["input_ids"],
            "attention_mask": text_ids["attention_mask"],
        }

        # Audio waveform -> processor tensor.
        audio_path = os.path.join(audio_root, video_id, f"{sid}.mp3")
        waveform, _ = librosa.load(audio_path, sr=sr)
        audio_features = audio_proc(
            waveform, sampling_rate=sr, return_tensors="pt"
        ).input_values.squeeze(0)

        # 16 video frames -> processor tensor (BGR -> RGB).
        imgs = [cv2.imread(p)[:, :, ::-1] for p in img_paths]
        video_features = video_proc(imgs, return_tensors="pt")["pixel_values"].squeeze(0)

        # Persist.
        out_path = os.path.join(features_dir, f"{sid}.pt")
        torch.save({
            "text":  text_features,
            "audio": audio_features,
            "video": video_features,
            "label": label,
        }, out_path)

        valid.append({"sentence_id": sid, "label": label, "feature_file": f"{sid}.pt"})

    pd.DataFrame(valid).to_csv(os.path.join(features_dir, "metadata.csv"), index=False)
    print(f"[features] Saved {len(valid)} samples to {features_dir}")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--data_root", default="dataset/")
    p.add_argument("--splits", nargs="+", default=["train", "val", "test"])
    args = p.parse_args()

    for split in args.splits:
        extract_split(args.data_root, split)


if __name__ == "__main__":
    main()
