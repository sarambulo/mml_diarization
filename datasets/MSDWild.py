"""
This script provides 2 sampling strategies, 2 video inputs and 2 label alternatives:

Sampling strategies:
1) Video and audio segments (possibly the entire video)
2) Individual video frame and audio segment

Video inputs:
1) Complete images
2) Cropped face

Label alternatives:
1) Is active speaker (boolean)
2) Is the anchor the active speaker (boolean) + Triplets (anchor, positive pair, negative pair)

The video and audio segments are used for recurrent models, the invididual
video frames and audio segments for time insensitive models.

The active speaker label is used for for active speaker detection (binary classification),
and the triplets for diarization (active speaker detection + clustering).

MSDWildBase is in charge of parsing the corresponding rttm file for a given dataset partition
and identifyin each video and audio file in the data directory. Given a video ID, it returns
the video and audio as streams, and the speaker IDs at the video frame rates.

MSDWildFrames is in charge of identifying each video frame. For a given frame id, it returns the
video frame, audio segment and speaker IDs (in plural because of possible overlap). Positive and
negative pairs are extracted from the same video and correspond to the next frame where the anchor
is not speaking and there is another active speaker.

MSDWildVideos is in charge for extracting the corresponding sequence of frames for a time interval
(context + target frame) or for the whole video. Positive and negative pairs are extracted from
the same video and correspond to the next segment where the anchor is not speaking and there is
another active speaker.
"""

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
from pairs.utils import list_s3_files, s3_load_numpy
from pairs.config import S3_BUCKET_NAME
from tqdm import tqdm
import s3fs
import ast

IMG_WIDTH = 112
IMG_HEIGHT = 112

s3 = s3fs.S3FileSystem()


def load_numpy(path):
    with s3.open(path, "rb") as f:
        data = np.load(f)
    return data


class MSDWildChunks(Dataset):
    def __init__(
        self,
        data_path: str,
        partition_path: str,
        subset: float = 1,
        data_bucket=None,
        refresh_fileset=False,
    ):
        if refresh_fileset:
            all_files = list_s3_files(S3_BUCKET_NAME, data_path)
            self.all_pairs = set([p for p in all_files if p.endswith("pair.npy")])
            with open("pairs_files.txt", "w") as f:
                f.write(str(self.all_pairs))

        with open("pairs_files.txt", "r") as f:
            content = f.read()
            self.all_pairs = (
                set() if content == str(set()) else ast.literal_eval(content)
            )

        print("Directory has", len(self.all_pairs), "pairs")
        self.data_path = data_path
        self.bucket = data_bucket
        self.subset = subset
        self.video_names = self.get_partition_video_ids(partition_path)
        self.pairs_info = self.load_pairs_info(video_names=self.video_names)
        pairs = min(len(self.all_pairs), len(self.pairs_info))
        N = floor(pairs * subset)
        print("Reducing number of pairs from", pairs, "to", N)
        self.paths_list = self.load_triplets_optimized(N=N)
        self.length = len(self.paths_list)
        print("Initialized Dataset with", self.length, "samples")

    def get_partition_video_ids(self, partition_path: str) -> List[str]:
        """
        Returns a list of video ID. For example: ['00001', '000002']
        """
        video_ids = set()
        with open(partition_path, "r") as f:
            for line in f:
                parts = line.strip().split()
                video_ids.add(parts[1])
        return sorted(list(video_ids))

    def load_pairs_info(self, video_names: List[str]) -> Dict:
        """
        Returns a dictionary with pair metadata.
        """
        all_pairs = {}

        for video_id in tqdm(video_names, desc="Loading Pair Metadata for Videos"):
            pairs_csv_path = os.path.join("preprocessed", video_id, "pairs.csv")
            if self.bucket:
                pairs_csv_path = os.path.join(
                    "s3://" + self.bucket, self.data_path, video_id, "pairs.csv"
                )

            try:
                df = pd.read_csv(pairs_csv_path)
                if df.empty:
                    raise Exception()
            except Exception:
                continue

            for _, row in df.iterrows():
                key = (
                    video_id,
                    int(row["chunk_id"]),
                    int(row["frame_id"]),
                    int(row["speaker_id"]),
                )
                all_pairs[key] = int(row["is_speaking"])

        print("Loaded metadata for", len(all_pairs), "pairs")
        return all_pairs

    def load_triplets_optimized(self, N: int):
        pairs_lst = list(self.pairs_info.items())
        random.shuffle(pairs_lst)
        
        paths = []
        index = 0
        pbar = tqdm(total=N, desc="Loading Triplet Files")
        while len(paths) < N and index < len(pairs_lst):
            (video_id, chunk_id, frame_id, speaker_id), is_speaking = pairs_lst[index]
            index += 1
            if frame_id % 4 != 0:
                continue

            visual_path = os.path.join(
                self.data_path,
                video_id,
                "visual_pairs",
                f"chunk{chunk_id}_speaker{speaker_id}_frame{frame_id}_pair.npy",
            )
            audio_path = os.path.join(
                self.data_path,
                video_id,
                "melspectrogram_audio_pairs",
                f"chunk{chunk_id}_frame{frame_id}_pair.npy",
            )

            if audio_path not in self.all_pairs or visual_path not in self.all_pairs:
                continue

            try:
                visual_data = s3_load_numpy(self.bucket, visual_path)
                audio_data = s3_load_numpy(self.bucket, audio_path)
            except Exception as e:
                print("Could not find", audio_path, str(e))
                continue


            paths.append((visual_data, audio_data, is_speaking))
            pbar.update(1)

        pbar.close()
        return paths

    def __len__(self):
        return self.length

    def __getitem__(self, index):
        """
        Returns a triplet of tensors.
        """
        visual_data, audio_data, is_speaking = self.paths_list[index]
        visual_data, audio_data = map(torch.tensor, (visual_data, audio_data))
        return visual_data, audio_data, is_speaking

    def build_batch(self, batch_examples: List[Tuple[torch.Tensor, torch.Tensor, int]]):
        """
        Returns a batch of tensors.
        """
        video_data, audio_data, is_speaking = list(zip(*batch_examples))
        video_data = torch.stack(video_data)
        audio_data = torch.stack(audio_data)
        is_speaking = torch.tensor(is_speaking)
        return video_data, audio_data, is_speaking
