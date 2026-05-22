"""
MM-F2F PyTorch dataset and collate function.

Expects the on-disk layout documented in :file:`data/README.md`::

    <data_root>/
        train.csv  val.csv  test.csv     (tab-separated, see columns below)
        audio/<video_id>/<sentence_id>.mp3
        video/<video_id>/<sentence_id>/{0..15}.jpg

CSV columns
-----------
``video_id``, ``sentence_id``, ``text``, ``label``, ``start``, ``end``, ``speaker``

Each label is one of::

    0 -> KEEP        (speaker continues; listener stays silent)
    1 -> TURN        (turn-taking; listener takes the floor)
    2 -> BACKCHANNEL (listener emits brief feedback, e.g. "mm-hmm")

Constructor flag ``modal``
--------------------------
``"text"``, ``"audio"``, ``"video"`` for Stage 1 uni-modal training, or
``"all"`` for Stage 2 fusion training. When ``"all"``, samples that lack
the full 16 video frames are silently skipped — keeping the resulting
dataset aligned across modalities.
"""

from __future__ import annotations

import os
from typing import Optional

import pandas as pd
import librosa
import cv2
import torch
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm


N_VIDEO_FRAMES = 16


class MultiModalDataset(Dataset):
    """MM-F2F dataset returning text-token / audio-waveform / video-frame dicts."""

    def __init__(
        self,
        data_root: str,
        split: str,
        modal: str = "all",
        tokenizer=None,
        audio_processor=None,
        video_processor=None,
    ):
        assert split in {"train", "val", "test"}
        assert modal in {"text", "audio", "video", "all"}
        self.modal = modal

        df = pd.read_csv(os.path.join(data_root, f"{split}.csv"), sep="\t")
        sid_arr   = df["sentence_id"].values
        text_arr  = df["text"].values
        label_arr = df["label"].values

        # Stash the processors we actually need.
        if self.modal in {"text", "all"}:
            assert tokenizer is not None, "tokenizer required for text/all modality"
            self.tokenizer = tokenizer

        if self.modal in {"audio", "all"}:
            assert audio_processor is not None, "audio_processor required for audio/all modality"
            audio_root = os.path.join(data_root, "audio")
            self.audio_processor = audio_processor
            self.sampling_rate = getattr(audio_processor, "sampling_rate", 16000)

        if self.modal in {"video", "all"}:
            assert video_processor is not None, "video_processor required for video/all modality"
            video_root = os.path.join(data_root, "video")
            self.video_processor = video_processor

        # Pre-tokenise text and pre-resolve audio / video file paths so that
        # __getitem__ becomes IO-bound but otherwise cheap.
        self.data_list = []
        iterator = zip(sid_arr, text_arr, label_arr)
        for sid, text, label in tqdm(iterator, total=len(sid_arr),
                                     desc=f"Index {split}"):
            sample = {"label": label}
            video_id = "_".join(sid.split("_")[:-2])

            if self.modal in {"text", "all"}:
                sample["text"] = self.tokenizer(text)

            if self.modal in {"audio", "all"}:
                sample["audio"] = os.path.join(audio_root, video_id, f"{sid}.mp3")

            if self.modal in {"video", "all"}:
                img_paths = []
                for i in range(N_VIDEO_FRAMES):
                    p = os.path.join(video_root, video_id, sid, f"{i}.jpg")
                    if not os.path.exists(p):
                        break
                    img_paths.append(p)
                # Drop samples that don't have all 16 frames so that
                # batches remain consistently shaped.
                if len(img_paths) < N_VIDEO_FRAMES:
                    continue
                sample["video"] = img_paths

            self.data_list.append(sample)

    # ------------------------------------------------------------------
    def _load_audio(self, path: str):
        waveform, _ = librosa.load(path, sr=self.sampling_rate)
        return self.audio_processor(
            waveform, sampling_rate=self.sampling_rate, return_tensors="pt"
        ).input_values

    def _load_images(self, paths):
        imgs = [cv2.imread(p)[:, :, ::-1] for p in paths]  # BGR -> RGB
        return self.video_processor(imgs, return_tensors="pt")

    # ------------------------------------------------------------------
    def __len__(self):
        return len(self.data_list)

    def __getitem__(self, idx):
        data = self.data_list[idx]
        out = {"label": data["label"]}
        if "text" in data:
            out["text"] = data["text"]
        if "audio" in data:
            out["audio"] = self._load_audio(data["audio"])
        if "video" in data:
            out["video"] = self._load_images(data["video"])
        return out


# ---------------------------------------------------------------------------
def collate_fn(batch):
    """Pad text/audio sequences and stack video frames into a single batch.

    Returns a tuple in the order ``(text?, audio?, video?, label)`` —
    modalities that are absent from the dataset's ``modal`` setting are
    omitted. ``train_fusion.py`` expects the four-tuple form (all three
    modalities present), while ``train_unimodal.py`` requests only one.
    """
    out = ()

    if "text" in batch[0]:
        input_ids = [torch.tensor(x["text"]["input_ids"], dtype=torch.long) for x in batch]
        attn_mask = [torch.tensor(x["text"]["attention_mask"], dtype=torch.long) for x in batch]
        text = {
            "input_ids": torch.nn.utils.rnn.pad_sequence(input_ids, batch_first=True, padding_value=0),
            "attention_mask": torch.nn.utils.rnn.pad_sequence(attn_mask, batch_first=True, padding_value=0),
        }
        out += (text,)

    if "audio" in batch[0]:
        audio_seqs = [torch.tensor(x["audio"][0], dtype=torch.float) for x in batch]
        audio = torch.nn.utils.rnn.pad_sequence(audio_seqs, batch_first=True, padding_value=0.0)
        out += (audio,)

    if "video" in batch[0]:
        video = {
            "pixel_values": torch.stack([x["video"]["pixel_values"][0] for x in batch]),
        }
        out += (video,)

    labels = torch.tensor([x["label"] for x in batch])
    out += (labels,)
    return out
