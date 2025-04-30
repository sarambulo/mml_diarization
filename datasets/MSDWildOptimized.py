import boto3
import io
import numpy as np
import json
import numpy as np
import torch
from torch.utils.data import Dataset
import os
from tqdm import tqdm


def load_npz_from_s3(bucket: str, key: str, visual_type):
    """
    Loads a .npz file from S3, unpacks arrays and metadata.

    Args:
        bucket (str): S3 bucket name (e.g. 'my-bucket')
        key (str): S3 key to the .npz file (e.g. 'data/triplet_batch_00001.npz')

    Returns:
        dict: {
            'visual_data': np.ndarray,
            'audio_data': np.ndarray,
            'is_speaking': np.ndarray,
            'metadata': list[dict]  # if present
        }
    """
    s3 = boto3.client("s3")
    response = s3.get_object(Bucket=bucket, Key=key)
    npz_bytes = io.BytesIO(response["Body"].read())
    data = np.load(npz_bytes, allow_pickle=True)

    result = {
        "visual_data": data[f"{visual_type}_data"],
        "audio_data": data["audio_data"],
        "metadata": data["metadata"],
    }

    return result


class LazyNPZDataset(Dataset):
    """
    A PyTorch Dataset that lazily loads samples from multiple .npz files,
    loading one file at a time into memory and optionally shuffling within-file.
    """

    def __init__(
        self,
        npz_dir: str,
        batch_size: int,
        bucket: str,
        shuffle_within_file: bool = False,
        visual_type: str = "face",
        start=0,
        end=-1,
    ):
        s3 = boto3.client("s3")
        paginator = s3.get_paginator("list_objects_v2")
        self.npz_file_paths = [
            obj["Key"]
            for page in paginator.paginate(
                Bucket="mmml-proj", Prefix=npz_dir + "/triplet_batch"
            )
            if "Contents" in page
            for obj in page["Contents"]
            if obj["Key"].endswith(".npz")
        ]
        if end == -1:
            end = len(self.npz_file_paths) + 1
        self.npz_file_paths = self.npz_file_paths[start:end]
        self.shuffle_within_file = shuffle_within_file
        self.samples_per_file = batch_size
        self.total_samples = self.samples_per_file * len(self.npz_file_paths)
        self.bucket = bucket
        self.visual_type = visual_type

        # State for current file cache
        self._current_file_index = None
        self._current_data = None
        self._permutation = None

    def __len__(self):
        return self.total_samples

    def __getitem__(self, idx: int):
        # Determine which .npz file and local index
        file_idx = idx // self.samples_per_file
        local_idx = idx % self.samples_per_file

        # Load file if not already cached
        if file_idx != self._current_file_index:
            self._load_file(file_idx)

        # Apply in-file shuffle if requested
        if self.shuffle_within_file:
            local_idx = int(self._permutation[local_idx])

        visual = torch.tensor(self._current_data["visual_data"][local_idx])
        audio = torch.tensor(self._current_data["audio_data"][local_idx])
        label = int(self._current_data["metadata"][local_idx]["is_speaking"])
        return visual, audio, label

    def _load_file(self, file_idx: int):
        # Load and cache the .npz file
        self._current_data = load_npz_from_s3(
            bucket=self.bucket,
            key=self.npz_file_paths[file_idx],
            visual_type=self.visual_type,
        )
        self._current_file_index = file_idx
        if self.shuffle_within_file:
            self._permutation = np.random.permutation(self.samples_per_file)

    def collate_fn(self, batch):
        # Extract each feature: do the zip thing
        video_data, audio_data, is_speaking = list(zip(*batch))
        video_data = torch.stack(video_data)
        audio_data = torch.stack(audio_data)
        is_speaking = torch.tensor(is_speaking)
        batch_data = {
            "video_data": video_data,
            "audio_data": audio_data,
            "labels": is_speaking,
        }
        return batch_data


class UpfrontNPZDataset(Dataset):
    """
    A PyTorch Dataset that loads all samples from multiple .npz files into memory
    at initialization. Best for smaller datasets or when you have enough RAM.
    """

    def __init__(
        self,
        npz_dir: str,
        bucket: str,
        visual_type: str = "face",
        start=0,
        end=-1,
    ):
        self.samples = []
        self.visual_type = visual_type
        # Load all files up front
        s3 = boto3.client("s3")
        paginator = s3.get_paginator("list_objects_v2")
        self.npz_file_paths = [
            obj["Key"]
            for page in paginator.paginate(
                Bucket="mmml-proj", Prefix=npz_dir + "/triplet_batch"
            )
            if "Contents" in page
            for obj in page["Contents"]
            if obj["Key"].endswith(".npz")
        ]
        if end == -1:
            end = len(self.npz_file_paths) + 1
        self.npz_file_paths = self.npz_file_paths[start:end]
        
        for i, path in enumerate(self.npz_file_paths):
            print(bucket, path)
            data = load_npz_from_s3(
                bucket=bucket, key=path, visual_type=self.visual_type
            )
            visuals = data["visual_data"]
            audios = data["audio_data"]
            labels = data["metadata"]
            for i in tqdm(range(len(labels)), desc=f"Unpacking batch {i}"):
                video = torch.tensor(visuals[i])
                audio = torch.tensor(audios[i])
                label = int(labels[i]["is_speaking"])
                self.samples.append((video, audio, label))

        self.length = len(self.samples)

    def __len__(self):
        return self.length

    def __getitem__(self, idx: int):
        return self.samples[idx]

    def collate_fn(self, batch):
        # Extract each feature: do the zip thing
        video_data, audio_data, is_speaking = list(zip(*batch))
        video_data = torch.stack(video_data)
        audio_data = torch.stack(audio_data)
        is_speaking = torch.tensor(is_speaking)
        batch_data = {
            "video_data": video_data,
            "audio_data": audio_data,
            "labels": is_speaking,
        }
        return batch_data
