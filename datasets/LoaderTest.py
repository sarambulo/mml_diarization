
import re
import torch
import os
import torch
from torch.utils.data import Dataset
from pathlib import Path
import numpy as np
import random
import pandas as pd
from typing import List, Dict, Tuple
from math import floor
import re
# from pairs.utils import list_s3_files, s3_load_numpy
# from pairs.config import S3_BUCKET_NAME
from tqdm import tqdm
# import s3fs
# import ast

IMG_WIDTH = 112
IMG_HEIGHT = 112





def parse_rttm(rttm_path: str) -> Dict[str, List[Tuple[float, float, str]]]:
    """
    Parse an RTTM file and return a dict:
        { video_id: [(start, end, speaker_id), ...], ... }
    """
    intervals = {}
    with open(rttm_path, "r") as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) < 8 or parts[0] != "SPEAKER":
                continue
            file_id    = parts[1]     # e.g. "00001"
            start_time = float(parts[3])
            duration   = float(parts[4])
            speaker_id = parts[7]
            end_time   = start_time + duration
            if file_id not in intervals:
                intervals[file_id] = []
            intervals[file_id].append((start_time, end_time, speaker_id))

    return intervals



def extract_audio_segment(mel_tensor: torch.Tensor, frame_idx: int, total_frames: int, desired_length: int = 22) -> torch.Tensor:
    """
    Extracts a segment from the full mel spectrogram corresponding to the given frame.
    
    There are two cases:
      1. If mel_tensor has shape [n_mels, desired_length, total_frames] (e.g. [30, 22, 20]),
         then the last dimension indexes the video frames. In that case, simply select
         the slice corresponding to frame_idx.
      2. Otherwise, if mel_tensor has shape [n_mels, A], we do a proportional slice.
         
    Finally, the function adds an extra channel dimension so the output shape is [1, n_mels, desired_length].
    """
    if mel_tensor.ndim == 3:
        # Assume shape is [n_mels, desired_length, total_frames]
        if frame_idx >= mel_tensor.shape[2]:
            raise ValueError("frame_idx exceeds available frames in mel_tensor")
        segment = mel_tensor[:, :, frame_idx]  # shape: [n_mels, desired_length]
    else:
        # Fallback: assume shape is [n_mels, A]
        mel_tensor = mel_tensor.squeeze()  # now expected shape [n_mels, A]
        if mel_tensor.ndim != 2:
            raise ValueError(f"Expected mel_tensor to have 2 dimensions after squeeze, got {mel_tensor.shape}")
        n_mels, A = mel_tensor.shape
        start_idx = int(frame_idx * A / total_frames)
        end_idx = int((frame_idx + 1) * A / total_frames)
        if end_idx <= start_idx:
            end_idx = start_idx + 1
        segment = mel_tensor[:, start_idx:end_idx]
        # Pad/truncate if needed to desired_length
        current_length = segment.shape[1]
        if current_length < desired_length:
            pad_size = desired_length - current_length
            segment = torch.nn.functional.pad(segment, (0, pad_size))
        elif current_length > desired_length:
            segment = segment[:, :desired_length]
    # Add channel dimension → [1, n_mels, desired_length]
    # segment = segment.squeeze()
    return segment


class TestDataset(Dataset):
    """
    Test dataset that loads individual face frames and their corresponding audio segments.
    
    For each chunk folder in a video directory, this dataset:
      - Loads a face track (e.g., face_{speaker_id}.npy) of shape [T, C, H, W]
      - Loads the full mel spectrogram (e.g., melspectrogram.npy) of shape (30, 22, 20)
      - Reads the ground truth labels from is_speaking.csv (columns: face_id, frame_id, is_speaking)
      
    It then creates one sample per frame in the face track. Each sample returns:
       face_tensor: a single frame (shape [C, H, W])
       audio_segment: the corresponding audio segment for that frame (shape [1, 30, 22])
       label: 0 or 1 for that frame
       metadata: dict with video_id, chunk_id, speaker_id, frame_idx, total_frames (T)
    """
    def __init__(self, root_dir: str, transform=None):
        super().__init__()
        self.root_dir = Path(root_dir)
        self.transform = transform
        self.samples = []
       
        # Iterate over each video folder
        for video_dir in sorted(self.root_dir.iterdir()):
            if not video_dir.is_dir():
                continue
            video_id = video_dir.name
            # For each chunk folder
            for chunk_dir in sorted(video_dir.iterdir()):
                if not chunk_dir.is_dir() or not chunk_dir.name.startswith("Chunk_"):
                    continue
                chunk_id = chunk_dir.name.split("_")[-1]
                
                # Path to the mel spectrogram
                mel_path = chunk_dir / "melspectrogram.npy"
                if not mel_path.exists():
                    print(f"Missing melspectrogram.npy in {chunk_dir}. Skipping chunk.")
                    continue
                
                # Parse is_speaking.csv for labels (if exists)
                csv_path = chunk_dir / "is_speaking.csv"
                speak_map = {}
                if csv_path.exists():
                    df = pd.read_csv(csv_path)
                    for _, row in df.iterrows():
                        face_id = int(row["face_id"])
                        frame_id = int(row["frame_id"])
                        is_spk = int(row["is_speaking"])
                        speak_map[(face_id, frame_id)] = is_spk
                else:
                    print(f"Warning: {csv_path} not found. Defaulting labels to 0.")
                
                # For each face file (assume one face per speaker for testing)
                for face_file in chunk_dir.glob("face_*.npy"):
                    if "_bboxes" in face_file.name:
                        continue
                    parts = face_file.stem.split("_")
                    if len(parts) < 2:
                        continue
                    speaker_id = int(parts[1])
                    
                    # Load the face track once to get T (total frames)
                    face_arr = np.load(str(face_file))  # shape: [T, C, H, W]
                    T = face_arr.shape[0]
                    
                    # For each frame in the face track, create a sample
                    for frame_idx in range(T):
                        label = speak_map.get((speaker_id, frame_idx), 0)
                        metadata = {
                            "video_id": video_id,
                            "chunk_id": chunk_id,
                            "speaker_id": speaker_id,
                            "frame_idx": frame_idx,
                            "total_frames": T
                        }
                        self.samples.append({
                            "face_path": str(face_file),
                            "mel_path": str(mel_path),
                            "metadata": metadata,
                            "label": label
                        })

    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        sample = self.samples[idx]
        metadata = sample["metadata"]
        frame_idx = metadata["frame_idx"]
        T = metadata["total_frames"]
        
        # Load the full face track and extract the specific frame
        face_arr = np.load(sample["face_path"])   # shape: [T, C, H, W]
        face_frame = face_arr[frame_idx]            # shape: [C, H, W]
        face_tensor = torch.from_numpy(face_frame).float()
        if self.transform:
            face_tensor = self.transform(face_tensor)
        
        # Load the full mel spectrogram for the chunk
        full_mel_arr = np.load(sample["mel_path"])  # expected shape: (30, 22, 20)
        full_mel_tensor = torch.from_numpy(full_mel_arr).float()
        # Extract the corresponding audio segment using our helper
        audio_segment = extract_audio_segment(full_mel_tensor, frame_idx, T, desired_length=22)
        # print(audio_segment.shape)
        label = float(sample["label"])
        return face_tensor, audio_segment, label, metadata
