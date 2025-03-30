import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import pandas as pd
from torch.utils.data import Dataset, DataLoader
from pathlib import Path
from collections import defaultdict
from convert_csv_to_rttms import convert_csvs_to_rttm
from evaluate_rttms import evaluate_rttms

def parse_rttm(rttm_path: str):
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
    segment = segment.squeeze()
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
        
        label = float(sample["label"])
        return face_tensor, audio_segment, label, metadata


import torch
from torch.utils.data import DataLoader
from collections import defaultdict

FRAME_SIZE = 0.25  # Seconds per frame

def generate_rttm_from_model(model, dataset, output_path="predictions.rttm", threshold=0.5, device="cpu"):
    """
    Runs the model on the dataset and generates an RTTM file with predicted speech segments.
    """
    model.eval()
    model.to(device)

    loader = DataLoader(dataset, batch_size=1, shuffle=False)
    predictions = defaultdict(list)

    with torch.no_grad():
        for face_tensor, audio_tensor, _, metadata in loader:
            if audio_tensor.ndim == 3:
                audio_tensor = audio_tensor.unsqueeze(1)  # [1, 1, 30, 22]
            audio_tensor = audio_tensor.to(device)

            _, probs = model(audio_tensor)
            pred = (probs > threshold).int().item()

            video_id = metadata["video_id"][0]
            speaker_id = metadata["speaker_id"][0]
            frame_idx = metadata["frame_idx"][0]

            predictions[(video_id, speaker_id)].append((frame_idx, pred))

    # Convert predictions to RTTM segments
    with open(output_path, "w") as f:
        for (video_id, speaker_id), frame_preds in predictions.items():
            frame_preds = sorted(frame_preds, key=lambda x: x[0])
            in_speech = False
            start_idx = 0

            for i, (frame_idx, pred) in enumerate(frame_preds):
                if pred == 1 and not in_speech:
                    in_speech = True
                    start_idx = frame_idx
                elif pred == 0 and in_speech:
                    end_idx = frame_idx
                    start_time = start_idx * FRAME_SIZE
                    duration = (end_idx - start_idx) * FRAME_SIZE
                    f.write(f"SPEAKER {video_id} 1 {start_time:.3f} {duration:.3f} <NA> <NA> {speaker_id} <NA> <NA>\n")
                    in_speech = False

            # If the last frame was still in speech
            if in_speech:
                end_idx = frame_preds[-1][0] + 1
                start_time = start_idx * FRAME_SIZE
                duration = (end_idx - start_idx) * FRAME_SIZE
                f.write(f"SPEAKER {video_id} 1 {start_time:.3f} {duration:.3f} <NA> <NA> {speaker_id} <NA> <NA>\n")

    print(f"✅ RTTM written to {output_path}")

class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1, dropout_rate=0.1):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.dropout1 = nn.Dropout2d(dropout_rate)  # Added dropout
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.dropout2 = nn.Dropout2d(dropout_rate)  # Added dropout
        
        # Shortcut connection to match dimensions
        self.shortcut = nn.Sequential()
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(out_channels)
            )
    
    def forward(self, x):
        residual = x
        
        out = self.conv1(x)
        out = self.bn1(out)
        out = F.relu(out)
        out = self.dropout1(out)  # Apply dropout
        
        out = self.conv2(out)
        out = self.bn2(out)
        out = self.dropout2(out)  # Apply dropout
        
        out += self.shortcut(residual)
        out = F.relu(out)
        
        return out


class SqueezeExcitation(nn.Module):
    def __init__(self, channel, reduction=16):
        super().__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(channel, channel // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channel // reduction, channel, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        return x * y.expand_as(x)


class CompactAudioEmbedding(nn.Module):
    def __init__(self, input_dim=40, embedding_dim=256, dropout_rate=0.3):
        super().__init__()
        
        # Initial convolutional layers - reduced filters
        self.conv_init = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=5, stride=2, padding=2, bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.Dropout2d(dropout_rate/2),  # Add dropout
            nn.MaxPool2d(kernel_size=2, stride=2)
        )
        
        # Residual blocks with moderate channel counts and dropout
        self.layer1 = nn.Sequential(
            ResidualBlock(32, 64, dropout_rate=dropout_rate),
            ResidualBlock(64, 64, dropout_rate=dropout_rate),
            SqueezeExcitation(64)
        )
        
        self.layer2 = nn.Sequential(
            ResidualBlock(64, 128, stride=2, dropout_rate=dropout_rate),
            ResidualBlock(128, 128, dropout_rate=dropout_rate),
            SqueezeExcitation(128)
        )
        
        self.layer3 = nn.Sequential(
            ResidualBlock(128, 256, stride=2, dropout_rate=dropout_rate),
            ResidualBlock(256, 256, dropout_rate=dropout_rate),
            SqueezeExcitation(256)
        )
        
        self.layer4 = nn.Sequential(
            ResidualBlock(256, 512, stride=1, dropout_rate=dropout_rate),
            SqueezeExcitation(512)
        )
        
        # Adaptive pooling to get fixed size
        self.adaptive_pool = nn.AdaptiveAvgPool2d((2, 2))
        
        # Flatten and project
        self.flatten_dim = 512 * 2 * 2
        
        # Final embedding layers with increased dropout
        self.fc1 = nn.Linear(self.flatten_dim, 512)
        self.dropout = nn.Dropout(dropout_rate)
        self.fc2 = nn.Linear(512, embedding_dim)
        
    def forward(self, x):
        # Handle 3D input (batch, freq, time)
        if x.dim() == 3:
            x = x.unsqueeze(1)
        
        # Convolutional blocks
        x = self.conv_init(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        
        # Pooling and flattening
        x = self.adaptive_pool(x)
        x = x.view(x.size(0), -1)
        
        # Final embedding
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)
        
        # L2 normalization for embedding
        x = F.normalize(x, p=2, dim=1)
        
        return x


class AudioActiveSpeakerModel(nn.Module):
    def __init__(self, base_model, embedding_dim=256, num_classes=1):  # Using num_classes=1 for binary classification
        super().__init__()
        self.encoder = base_model
        self.classifier = nn.Linear(embedding_dim, num_classes)  # Single output for binary classification

    def forward(self, x):
        embedding = self.encoder(x)
        logits = self.classifier(embedding)
        probs = torch.sigmoid(logits.squeeze(-1))
        return embedding, probs  # Squeeze to get scalar output for binary classification

# Load trained model
base_model = CompactAudioEmbedding(input_dim=40, embedding_dim=256, dropout_rate=0.3)
model = AudioActiveSpeakerModel(base_model, embedding_dim=256)
model.load_state_dict(torch.load("../models/best_audio_model.pth"))

# Load test dataset
test_dataset = TestDataset(root_dir="../test_preprocessed")

# Generate RTTM file
generate_rttm_from_model(model, test_dataset, output_path="audio_predictions.rttm")

convert_csvs_to_rttm("../test_preprocessed", output_rttm="ground_truth.rttm")
evaluate_rttms("ground_truth.rttm", "audio_predictions.rttm")
