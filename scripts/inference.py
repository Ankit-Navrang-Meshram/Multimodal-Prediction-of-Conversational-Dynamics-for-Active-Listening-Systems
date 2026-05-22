"""
End-to-end inference on a raw video file.

Pipeline
--------
1. Decode the video with OpenCV (frames + fps) and the audio with librosa.
2. Run WhisperX to produce word-level transcripts with timestamps.
3. For each word boundary, build the trailing audio + face-frame windows
   and call the trained model to predict the listener's next action
   (KEEP / TURN-TAKING / BACKCHANNEL).

The script prints one prediction per word as it streams through.

Requires
--------
* ``whisperx``   for word-aligned ASR
* ``batch-face`` for face detection (used by :mod:`src.face_utils`)

Example
-------
::

    python -m scripts.inference \\
        --input_path  example/demo.mp4 \\
        --ckpt_path   log/<run>/epoch_9.pt \\
        --fusion_module ACGF
"""

from __future__ import annotations

import argparse
import os
import sys

import cv2
import librosa
import numpy as np
import torch
from torch import nn

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.encoders import load_processors
from src.model import LanguageAudioVisionModel
from src.fusion import FUSION_REGISTRY


# -----------------------------------------------------------------------
# Face detection — lazy import so the dependency is optional for the rest
# of the repo.
# -----------------------------------------------------------------------
def detect_faces(frames):
    from batch_face import RetinaFace
    detector = RetinaFace(gpu_id=0)
    if len(np.array(frames).shape) == 3:
        frames = [frames]
    all_faces = detector(frames)
    out = []
    for faces, frame in zip(all_faces, frames):
        if len(faces) == 0:
            return None
        bbox, _landmarks, _score = faces[0]
        x1, y1, x2, y2 = list(map(int, bbox))
        out.append(frame[y1:y2, x1:x2, ::-1])
    return out


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--input_path", required=True, help="Path to .mp4 with audio")
    p.add_argument("--ckpt_path",  required=True, help="Trained model checkpoint")
    p.add_argument("--fusion_module", required=True, choices=sorted(FUSION_REGISTRY.keys()))
    p.add_argument("--device", default="cuda")
    # Video / audio params.
    p.add_argument("--n_imgs", type=int, default=16)
    p.add_argument("--samplerate", type=int, default=16000)
    p.add_argument("--max_audio_length", type=int, default=400000)
    return p.parse_args()


# -----------------------------------------------------------------------
class WhisperXManager:
    """Thin wrapper around WhisperX for word-level transcription."""

    def __init__(self, device: str, language: str = "en"):
        import whisperx
        self.device = device
        self.model = whisperx.load_model(
            "large-v3", device, compute_type="float16", language=language,
            asr_options={"suppress_numerals": True},
        )
        self.model_a, self.metadata = whisperx.load_align_model(
            language_code=language, device=device
        )

    def transcribe(self, audio, return_words: bool = False):
        import whisperx
        segments = self.model.transcribe(audio, batch_size=16, chunk_size=10)["segments"]
        if return_words:
            aligned = whisperx.align(
                segments, self.model_a, self.metadata, audio, self.device,
                return_char_alignments=False,
            )
            return aligned["word_segments"]
        return segments


# -----------------------------------------------------------------------
class TurnTakingManager:
    """Loads the trained model + processors and runs prediction per word."""

    def __init__(self, args):
        self.args = args
        self.tokenizer, self.text_processor, self.audio_processor, self.video_processor = load_processors()
        self.sampling_rate = self.audio_processor.feature_extractor.sampling_rate

        self.model = LanguageAudioVisionModel(
            fusion_module=args.fusion_module
        ).to(args.device)
        self.model.load_state_dict(torch.load(args.ckpt_path, map_location=args.device),
                                   strict=False)
        self.model.eval()

        self.whisperx = WhisperXManager(args.device)

    def load_text(self, text):
        if isinstance(text, str):
            text = self.text_processor(text)
            if text == "":
                return None
            text = self.tokenizer(text, return_tensors="pt").input_ids
        return text

    def transcribe(self, audio, return_words: bool = False):
        return self.whisperx.transcribe(audio, return_words=return_words)

    def predict(self, text_ids, audio_window, face_frames):
        audio_input = self.audio_processor(
            audio_window, sampling_rate=self.sampling_rate, return_tensors="pt"
        ).input_values
        video_input = self.video_processor(face_frames).pixel_values

        text_input  = text_ids.to(self.args.device)
        audio_input = audio_input.to(self.args.device)
        video_input = torch.tensor(video_input).to(self.args.device)

        with torch.no_grad():
            logits = self.model(text_input, audio_input, video_input)
            probs = nn.Sigmoid()(logits).cpu().numpy()

        p = probs[0]
        return p / p.sum()


# -----------------------------------------------------------------------
def read_video(path):
    cap = cv2.VideoCapture(path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    frames = []
    while cap.isOpened():
        ok, frame = cap.read()
        if not ok:
            break
        frames.append(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    cap.release()
    return frames, fps


def read_audio(path: str, sr: int):
    wav, _ = librosa.load(path, sr=sr)
    return wav


def idx2action(idx: int) -> str:
    return {0: "keep", 1: "turn-taking", 2: "backchannel"}[idx]


# -----------------------------------------------------------------------
def stream_predictions(manager: TurnTakingManager, audio, frames, fps, args):
    """Word-level streaming prediction loop."""
    word_segments = manager.transcribe(audio, return_words=True)
    if len(word_segments) == 0:
        return

    text_seq = ""
    for seg in word_segments:
        try:
            end = seg["end"]
            text_seq += seg["word"] + " "
            text_ids = manager.load_text(text_seq)
            if text_ids is None:
                continue

            audio_window = audio[:int(end * args.samplerate)][-args.max_audio_length:]
            frame_end_idx = int(end * fps)
            if frame_end_idx < args.n_imgs:
                continue
            video_window = frames[frame_end_idx - args.n_imgs:frame_end_idx]
            if len(video_window) < args.n_imgs:
                continue

            face_frames = detect_faces(video_window)
            if face_frames is None:
                continue

            probs = manager.predict(text_ids, audio_window, face_frames)
            action = idx2action(int(probs.argmax()))
            print("=" * 40)
            print(f"INPUT : {text_seq.strip()}")
            print(f"ACTION: {action}   probs={probs}")
        except Exception as e:
            # Streaming inference is best-effort — skip individual word failures.
            print(f"[warn] skipping word: {e}")


def main():
    args = parse_args()
    frames, fps = read_video(args.input_path)
    audio = read_audio(args.input_path, args.samplerate)
    manager = TurnTakingManager(args)
    stream_predictions(manager, audio, frames, fps, args)


if __name__ == "__main__":
    main()
