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

# from .utils import get_streams, parse_rttm, read_audio, read_video
import numpy as np
import random
import torch
import pandas as pd

# import torchvision.transforms.v2 as ImageTransforms
# import torchaudio.transforms as AudioTransforms
from typing import List, Dict, Tuple
from math import floor
import re

IMG_WIDTH = 112
IMG_HEIGHT = 112


class MSDWildChunks(Dataset):
   def __init__(self, data_path: str, partition_path: str, subset: float = 1):
      self.data_path = data_path
      self.subset = subset
      self.video_names = self.get_partition_video_ids(partition_path)
      self.pairs_info = self.load_pairs_info(video_names=self.video_names)
      N = floor(len(self.pairs_info) * subset)
      self.triplets = self.load_triplets(data_path=data_path, pairs_info=self.pairs_info, N=N)
      self.length = len(self.triplets)

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


   def load_pairs_info(self, video_names: List[str]) -> List[Dict]:
      """
      video names is the video ID, not the path
      Returns: [
         {'video_id': 1, 'chunk_id': 1, 'speaker_id': 0, 'is_speaking': 1 ,'frame_id': 2,},
         {'video_id': 1, 'chunk_id': 1, 'speaker_id': 0, 'is_speaking': 1, 'frame_id': 2 }
      ]
      """
      # For each video
         # Load pairs.csv 
      # Concat all pairs.csv
      all_pairs = {}

      for video_id in video_names:
        pairs_csv_path = os.path.join("../preprocessed", video_id, "pairs.csv")
        if not os.path.isfile(pairs_csv_path):
            print(f"Warning: pairs.csv not found for video {video_id}")
            continue

        df = pd.read_csv(pairs_csv_path)

        for _, row in df.iterrows():
            key = (
                video_id,
                int(row["chunk_id"]),
                int(row["frame_id"]),
                int(row["speaker_id"])  # convert to string as requested
            )
            all_pairs[key] = int(row["is_speaking"])

      return all_pairs
   


   def load_triplets(self, data_path: str, pairs_info: List[Dict], N: int) -> List[Tuple[torch.Tensor, torch.Tensor, int]]:
      """
      Loads all triplets stored within each video and chunk directory inside
      `data_path`. Looks for that video, chunk, frame and speaker in `pairs_info`
      to determine the value of `is_speaking` for the anchor

      Return: 
         List where each element is a Tuple = (visual_triplet_data, audio_triplet_data, is_speaking)
      """
      visual_path_pattern = re.compile(r'chunk(\d+)_speaker(\d+)_frame(\d+)_pair.npy')
      triplets = []
      counter = 0
      # Videos
      for video_path in Path(data_path).iterdir():
         video_id = video_path.name
         visual_triplets_paths = Path(video_path, 'visual_pairs')
         # Load visual data
         for path in visual_triplets_paths.iterdir():
            visual_data = np.load(str(path))
            filename = path.name
            match_result = visual_path_pattern.match(filename)
            if not match_result:
               raise ValueError(f'Visual pair {filename} does not match pattern {visual_path_pattern.pattern}')
            chunk_id, speaker_id, frame_id = map(int, match_result.groups())
            # Look for corresponding melspectrogram
            audio_path = Path(video_path, 'melspectrogram_audio_pairs', f"chunk{chunk_id}_frame{frame_id}_pair.npy")
            audio_data = np.load(str(audio_path))
            visual_data, audio_data = map(torch.tensor, (visual_data, audio_data))
            if (video_id, chunk_id, frame_id, speaker_id) not in pairs_info:
               print(f'Missing info for {(video_id, chunk_id, frame_id, speaker_id)} in pairs_info')
               print('Skipping that triplet')
               continue
            is_speaking = pairs_info[(video_id, chunk_id, frame_id, speaker_id)]
            triplets.append((visual_data, audio_data, is_speaking))
            counter += 1
            if counter >= N:
               break
      return triplets
   def __len__(self):
      return self.length
   def __getitem__(self, index):
      """
      video_data: torch.Tensor of dim (3, C, H, W)
      audio_data: torch.Tensor of dim (3, B, T)
      is_speaking: int NOTE: This is only for the anchor
      """
      # Index anchor, positive and negative
      triplet = self.triplets[index]
      return triplet
   def build_batch(self, batch_examples: List[Tuple[torch.Tensor, torch.Tensor, int]]):
      """
      Returns a tuple
      video_data (N, 3, C, H, W), audio_data (N, 3, B, T), is_speaking (N,)
      """
      # Extract each feature: do the zip thing
      video_data, audio_data, is_speaking = list(zip(*batch_examples))
      # Padding: NOTE: Not necessary
      # Stack: 
      video_data = torch.stack(video_data)
      audio_data = torch.stack(audio_data)
      is_speaking = torch.tensor(is_speaking)
      # Return tuple((N, video_data, melspectrogram), (N, video_data, melspectrogram), (N, video_data, melspectrogram))
      # (N, C, H, W), (N, Bands, T) x3 (ask Prachi)
      return video_data, audio_data, is_speaking




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



class TestDataLoader(Dataset):
    """
    A test-only dataset:
    - Reads chunk-based preprocessed data from data_path/<video_id>/...
    - Each chunk file is named e.g. "chunk(\d+)_speaker(\d+)_frame(\d+)_pair.npy"
    - Optionally uses RTTM for ground-truth references. 
    - Returns (visual_data, audio_data, metadata), where metadata has all info to rebuild an RTTM line.
    """
    def __init__(self, data_path: str, rttm_path: str = None):
        super().__init__()
        self.data_path = data_path
        
        # 1) If you want ground-truth intervals from an RTTM file, parse them
        self.intervals = {}
        if rttm_path is not None and os.path.isfile(rttm_path):
            self.intervals = parse_rttm(rttm_path)  # {video_id: [(start,end,spkid), ...]}
        
        # 2) Regex to match chunk/speaker/frame triple
        self.file_pattern = re.compile(r'chunk(\d+)_speaker(\d+)_frame(\d+)_pair\.npy')

        # We'll store a list of samples; each sample is a dict describing the chunk
        self.samples = []
        
        # 3) Scan data_path
        for video_dir in Path(data_path).iterdir():
            if not video_dir.is_dir():
                continue
            video_id = video_dir.name  # e.g. "00001"
            
            # We'll look in "visual_pairs" folder or adapt as needed
            visual_pairs_dir = video_dir / "visual_pairs"
            if not visual_pairs_dir.exists():
                print(f"No visual_pairs in {video_dir}")
                continue

            # Iterate the files
            for npy_file in visual_pairs_dir.glob("*.npy"):
                filename = npy_file.name
                match = self.file_pattern.match(filename)
                if not match:
                    continue
                chunk_id_str, speaker_id_str, frame_id_str = match.groups()
                chunk_id  = int(chunk_id_str)
                speaker_id= int(speaker_id_str)
                frame_id  = int(frame_id_str)

                # Also find the audio file e.g. "melspectrogram_audio_pairs" if needed
                audio_dir = video_dir / "melspectrogram_audio_pairs"
                audio_file = audio_dir / f"chunk{chunk_id}_frame{frame_id}_pair.npy"
                if not audio_file.exists():
                    print(f"Missing audio file {audio_file}")
                    continue

                # For RTTM-based ground truth, you might want start_time or is_speaking. 
                # If your chunk has a known start_time, you can store it in metadata. 
                # For now, let's store the raw key in a dictionary
                metadata = {
                    "video_id": video_id,
                    "chunk_id": chunk_id,
                    "frame_id": frame_id,
                    "speaker_id": speaker_id,
                }

                # If we want to store a ground truth "is_speaking" (0/1) from RTTM:
                # we could do a mapping from intervals or pairs.csv. 
                # For pure test, we might skip. Or if you want to do an offline eval, you can do:
                # is_speaking = ???

                sample = {
                    "visual_npy": str(npy_file),
                    "audio_npy":  str(audio_file),
                    "metadata":   metadata
                }
                self.samples.append(sample)

        # Sort or shuffle as needed
        # e.g. self.samples.sort(key=lambda x: (x["metadata"]["video_id"], x["metadata"]["chunk_id"], ...))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        """
        Returns a tuple: (visual_data, audio_data, metadata_dict)
        where
          visual_data: e.g. shape (3, C, H, W)
          audio_data: e.g. shape (Time, ...) 
          metadata_dict: { video_id, chunk_id, frame_id, speaker_id, etc.}
        """
        sample = self.samples[idx]
        visual_data = np.load(sample["visual_npy"])     # shape e.g. (3, C, H, W)
        audio_data  = np.load(sample["audio_npy"])      # shape e.g. (Freq, Time) if mel-spectrogram
        # Convert to torch
        visual_data = torch.from_numpy(visual_data)
        audio_data  = torch.from_numpy(audio_data)

        metadata = sample["metadata"]
        return (visual_data, audio_data, metadata)




# class MSDWildBase(Dataset):
#    def __init__(self, data_path: str, partition: str, subset: float = 1):
#       """
#       :param data_path str: path to the directory where the data is stored
#       :param partition str: few_train, few_val or many_val
#       :param subset float: portion of the data to use, from 0 to 1
#       """
#       super().__init__()
#       self.data_path = data_path

#       # Parse the rttm file to extract the file ids and labels
#       rttm_filename = f"{partition}.rttm"
#       rttm_path = Path(data_path, rttm_filename)
#       rttm_data = parse_rttm(rttm_path)
#       self.video_names = list(rttm_data.keys())

#       # Subset the data
#       N = len(self.video_names)
#       self.video_names = self.video_names[:int(N * subset)]

#       self.video_durations, self.video_fps = self.get_video_metadata(self.video_names)
#       self.video_num_frames = np.floor(self.video_durations * self.video_fps).astype(int)
#       self.rttm_data = rttm_data # keys: video_names, items: labels
#    def __len__(self):
#       return len(self.video_names)
#    def get_video_metadata(self, video_names):
#       root = Path(self.data_path, 'msdwild_boundingbox_labels')
#       all_metadata = []
#       for video_name in video_names:
#          video_path = root / f'{video_name}.mp4'
#          metadata = get_streams(video_path)[2]
#          all_metadata.append((metadata['video']['duration'], metadata['video']['fps']))
#       duration, fps = list(zip(*all_metadata))
#       return np.array(duration), np.array(fps)
#    def parse_bounding_boxes(self, index):
#       video_name = self.video_names[index]  # Convert file_id to a zero-padded string
#       data_path = Path(self.data_path)  # Convert self.data_path to a Path object if it's a string
#       csv_path = data_path / 'msdwild_boundingbox_labels' / f'{video_name}.csv'  # Ensure correct path

#       print(f"DEBUG: Looking for bounding box file at: {csv_path}")  # Debugging

#       if csv_path.exists():
#          # Read CSV with no headers (ensure all rows are treated as data)
#          df = pd.read_csv(csv_path, header=None)

#          # Manually assign column names
#          df.columns = ["frame_id", "face", "face_id", "x1", "y1", "x2", "y2", "fixed"]

#       #   print(f"Bounding Boxes Parsed Successfully for {file_id}:\n", df.head())  # Debugging
#          return df

#       print(f"ERROR: Bounding boxes file {csv_path} not found!")
#       return None

#    def __getitem__(self, index):
#       video_name = self.video_names[int(index)]
#       root = Path(self.data_path, 'msdwild_boundingbox_labels')
#       video_path = root / f'{video_name}.mp4'
#       csv_path = root / f'{video_name}.csv'
#       if not video_path.exists():
#          raise FileNotFoundError(f"Video file not found: {video_path}")
#       video_stream, audio_stream, metadata = get_streams(video_path)
#       labels = self.rttm_data.get(video_name, [])
#       bounding_boxes = None
#       if csv_path.exists():
#          bounding_boxes = pd.read_csv(csv_path,header=None, skiprows=0)
#          bounding_boxes.columns = ["frame_id", "face", "face_id", "x1", "y1", "x2", "y2", "fixed"]

#       return video_stream, audio_stream, labels, bounding_boxes

# class MSDWildFrames(MSDWildBase):
#    def __init__(self, data_path: str, partition: str, transforms = None, subset: float = 1):
#       """
#       :param data_path str: path to the directory where the data is stored
#       :param partition str: few_train, few_val or many_val
#       """
#       super().__init__(data_path, partition, subset)
#       # Adding IDs to frames
#       # Frame IDs are the position of each frame in this list
#       # Each element of the list contains the file ID and the frame timestamp (for seek)
#       self.video_last_frame_id = np.cumsum(self.video_num_frames)
#       # If transforms is provided, check that is a dictionary with
#       # keys: 'video_frame', 'face', and 'audio_segment'
#       self.transforms = None
#       if transforms:
#          assert isinstance(transforms, dict)
#          assert 'video_frame' in transforms
#          assert 'face' in transforms
#          assert 'audio_segment' in transforms
#          self.transforms = transforms
#       else:
#          transforms = {}
#          image_transform = ImageTransforms.Compose([
#             ImageTransforms.Resize((IMG_HEIGHT, IMG_WIDTH)),
#             ImageTransforms.ToDtype(torch.float32, scale=True),
#             # ImageTransforms.RandomHorizontalFlip(p = 0.5),
#             # ImageTransforms.RandomAffine(degrees=20, translate=(0.1,0.1), scale=(0.9,1.1)),
#             # ImageTransforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
#             ImageTransforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
#          ])
#          transforms['video_frame'] = image_transform
#          transforms['face'] = image_transform
#          transforms['audio_segment'] = lambda x: torch.mean(x, dim=-1)
#          self.transforms = transforms

#    def __len__(self):
#       return self.video_last_frame_id[-1]

#    def get_video_index(self, frame_id, start = 0, end = None):
#       """
#       Binary search over the last frame id for each video
#       Example
#       self.video_last_frame_id = [5, 10, 15]
#       video_indexes = [0, 1, 2]
#       frame_id = 12
#       video_index = 2
#       """
#       if end is None:
#          end = len(self.video_last_frame_id) - 1
#       if start > end:
#          raise ValueError('Frame id not found')
#       mid_index = (start + end) // 2
#       last_frame_id = self.video_last_frame_id[mid_index]
#       # Edge case: Mid video could be the first video
#       first_frame_id = 0
#       if mid_index > 0:
#          first_frame_id = self.video_last_frame_id[mid_index - 1] + 1
#       # Found: Frame is between the first and last frame for the mid video
#       if first_frame_id <= frame_id <= last_frame_id:
#          return mid_index
#       # Search right
#       elif frame_id > last_frame_id:
#          return self.get_video_index(frame_id, mid_index + 1, end)
#       # Search left
#       else:
#          return self.get_video_index(frame_id, start, mid_index - 1)

#    def get_frame_loc(self, frame_id):
#       video_index = self.get_video_index(frame_id)
#       # Edge case: First video
#       first_frame_id = 0
#       if video_index > 0:
#          first_frame_id = self.video_last_frame_id[video_index - 1] + 1
#       frame_offset = frame_id - first_frame_id
#       frame_timestamp = frame_offset / self.video_fps[video_index].item()
#       return video_index, frame_offset, frame_timestamp

#    def get_speakers_at_ts(self, data, timestamp) -> np.ndarray:
#       time_intervals, speaker_ids = data
#       start_times = time_intervals[:,0]
#       durations=time_intervals[:, 1]
#       end_times = start_times+ durations
#       active_speaker_ids = [speaker_ids[i] for i in range(len(start_times)) if start_times[i] <= timestamp < end_times[i]]
#       if not active_speaker_ids:
#         max_speaker_id = max(speaker_ids, default=0)  # Avoid error if speaker_ids is empty
#         return torch.zeros(max_speaker_id + 1, dtype=int)
#       max_speaker_id = max(speaker_ids)  # Get max speaker ID for array size
#       speaker_vector = torch.zeros(max_speaker_id + 1, dtype=int)
#       for speaker_id in active_speaker_ids:
#          speaker_vector[speaker_id] = 1
#       return speaker_vector

#    def extract_faces_from_frame(self, frame, bounding_boxes, frame_offset):
#       if bounding_boxes is None:
#          return {}

#       frame_boxes = bounding_boxes[bounding_boxes["frame_id"] == frame_offset]
#       cropped_faces = {}

#       for _, row in frame_boxes.iterrows():
#          face_id = int(row["face_id"])
#          x1, y1, x2, y2 = int(row["x1"]), int(row["y1"]), int(row["x2"]), int(row["y2"])
#          x1, y1, x2, y2 = max(x1, 0), max(y1, 0), max(x2, 0), max(y2, 0)
#          if x2 > x1 and y2 > y1:
#             cropped_faces[face_id] = frame[:, y1:y2, x1:x2]
#       return cropped_faces


#    def get_audio_segment(self, audio_stream, frame_id):
#       frame_id = int(frame_id)
#       current_file_id, current_offset, current_timestamp = self.get_frame_loc(frame_id)
#       # Edge case: first frame
#       if frame_id == 0:
#          start = 0
#       else:
#          prev_file_id, prev_offset, prev_timestamp = self.get_frame_loc(frame_id - 1)
#          # Case: first frame in video
#          if prev_file_id != current_file_id:
#             prev_timestamp = 0
#          start = (prev_timestamp + current_timestamp) / 2
#       # Edge case: last frame
#       if frame_id >= len(self):
#          end = float('inf')
#       else:
#          next_file_id, next_offset, next_timestamp = self.get_frame_loc(frame_id + 1)
#          # Case: last frame in video
#          if current_file_id != next_file_id:
#             next_timestamp = float('inf')
#          end = (current_timestamp + next_timestamp) / 2
#       audio_frames = read_audio(audio_stream, start, end)
#       return audio_frames

#    def get_features(self, frame_id):
#       file_id, frame_offset, frame_timestamp = self.get_frame_loc(frame_id)
#       video_stream, audio_stream, labels, bounding_boxes = super().__getitem__(file_id)
#       # Get frame from video stream
#       result = next(iter(video_stream.seek(frame_timestamp)))
#       video_frame = result['data']
#       cropped_faces = self.extract_faces_from_frame(video_frame, bounding_boxes, frame_offset)
#       audio_segment = self.get_audio_segment(audio_stream, frame_id)
#       # Transform features
#       if self.transforms:
#          video_frame = self.transforms['video_frame'](video_frame)
#          if cropped_faces:
#             cropped_faces = {face_id: self.transforms['face'](cropped_faces[face_id]) for face_id in cropped_faces}
#          audio_segment = self.transforms['audio_segment'](audio_segment)
#       labels = self.get_speakers_at_ts(labels, frame_timestamp) if labels else None
#       # print(labels)
#       features = (video_frame, audio_segment, labels, cropped_faces)
#       return features

#    def get_positive_sample(self, frame_id, anchor_speaker_id):
#       # Case: Last frame
#       next_frame_offset = self.get_frame_loc(frame_id + 1)[1]
#       cropped_faces = {}
#       candidate_frame = frame_id
#       if next_frame_offset == 0:
#          # Use the previous frame
#          while anchor_speaker_id not in cropped_faces:
#             candidate_frame -= 1
#             video_frame, audio_segment, labels, cropped_faces = self.get_features(candidate_frame)
#       else:
#          # Use the next frame
#          while anchor_speaker_id not in cropped_faces:
#             candidate_frame += 1
#             video_frame, audio_segment, labels, cropped_faces = self.get_features(candidate_frame)
#       positive_sample = video_frame, audio_segment, cropped_faces[anchor_speaker_id]
#       return positive_sample

#    def get_negative_sample(self, file_id, face_id):
#       if file_id is None:
#          raise ValueError("file_id cannot be None")
#       random_file_id = None
#       while random_file_id == file_id:
#          random_frame = random.randint(0, len(self) - 1)
#       anchor, _ = self.get_features(random_frame)
#       return anchor

#    def __getitem__(self, index):
#       video_frame, audio_segment, labels, cropped_faces = self.get_features(index)
#       if len(cropped_faces) < 2:
#          if index < len(self) - 1:
#             return self.__getitem__(index + 1)
#          return None
#       face_ids = list(cropped_faces.keys())
#       random.shuffle(face_ids)
#       anchor_speaker_id, negative_sample_speaker_id = face_ids[:2]
#       anchor = video_frame, audio_segment, cropped_faces[anchor_speaker_id]
#       negative_pair = video_frame, audio_segment, cropped_faces[negative_sample_speaker_id]
#       positive_pair = self.get_positive_sample(index, anchor_speaker_id)
#       # print(labels)
#       # print(anchor_speaker_id)
#       if anchor_speaker_id >= len(labels):
#          anchor_speaker_id = 0
#       label = labels[anchor_speaker_id]
#       return anchor, positive_pair, negative_pair, label

#    def build_batch(self, batch_examples: list):
#       # batch_examples: [(anchor1, pos1, neg1, label1), (anchor2, pos2, neg2, label2)]
#       batch_examples = [ex for ex in batch_examples if ex is not None]
#       batch_size = len(batch_examples)
#       features = []
#       # features: (feature, element, example)
#       for feature_index in range(3):
#          feature = []
#          for element in range(3):
#             for example in range(batch_size):
#                feature.append(batch_examples[example][element][feature_index])
#          features.append(feature)
#       labels = torch.tensor([batch_examples[i][3] for i in range(batch_size)])

#       if len(batch_examples) == 0:
#          return None

#       padded_features = []
#       for feature in features:
#          feature = [torch.tensor(sample) if sample is not None else torch.zeros(1) for sample in feature]
#          feature = torch.nn.utils.rnn.pad_sequence(feature, batch_first=True)
#          padded_features.append(feature)

#       padded_elements = []
#       for i in range(3):
#          element = []
#          for feature in padded_features:
#             element.append(feature[i * batch_size: (i + 1) * batch_size])
#          padded_elements.append(element)

#       return tuple(padded_elements + [labels])

# class MSDWildVideos(MSDWildFrames):
#    def __init__(self, data_path: str, partition: str, transforms, subset: float = 1, max_frames = 30):
#       """
#       :param data_path str: path to the directory where the data is stored
#       :param partition str: few_train, few_val or many_val
#       """
#       super().__init__(data_path, partition, transforms, subset)
#       self.max_frames = max_frames
#    def __len__(self):
#       return len(self.video_names)
#    def __getitem__(self, index):
#       video_stream, audio_stream, labels, bounding_boxes = super(MSDWildFrames, self).__getitem__(index)
#       # Get frames from video stream
#       all_video_frames = []
#       all_audio_segments = []
#       all_labels = []
#       all_faces = []
#       all_timestamps = []
#       if index == 0:
#          frame_id = 0
#       else:
#          frame_id = self.video_last_frame_id[index - 1] + 1
#       for frame_offset, data in enumerate(video_stream):
#          if frame_offset >= self.max_frames:
#             break
#          video_frame, frame_timestamp = data['data'], data['pts']
#          faces = self.extract_faces_from_frame(video_frame, bounding_boxes, frame_offset)
#          audio_segment = self.get_audio_segment(audio_stream, frame_id + frame_offset)
#          # Transform features
#          if self.transforms and faces:
#             video_frame = self.transforms['video_frame'](video_frame)
#             faces = [self.transforms['face'](faces[face]) for face in faces]
#             audio_segment = self.transforms['audio_segment'](audio_segment)
#          label = self.get_speakers_at_ts(labels, frame_timestamp) if labels else None
#          # Accumulate features for all frames
#          all_video_frames.append(video_frame)
#          all_audio_segments.append(audio_segment)
#          all_labels.append(label)
#          all_faces.append(faces)
#          all_timestamps.append(frame_timestamp)
#       return all_video_frames, all_audio_segments, all_labels, all_faces, all_timestamps, self.video_names[index]
