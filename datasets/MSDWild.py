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

import torch
from torch.utils.data import Dataset
from pathlib import Path
from .utils import get_streams, parse_rttm, read_audio, read_video
import numpy as np
import random
import torch
import pandas as pd
import torchvision.transforms.v2 as ImageTransforms
import torchaudio.transforms as AudioTransforms

IMG_WIDTH = 112
IMG_HEIGHT = 112

class MSDWildBase(Dataset):
   def __init__(self, data_path: str, partition: str, subset: float = 1):
      """
      :param data_path str: path to the directory where the data is stored 
      :param partition str: few_train, few_val or many_val
      :param subset float: portion of the data to use, from 0 to 1
      """
      super().__init__()
      self.data_path = data_path
      # Parse the rttm file to extract the file ids and labels
      rttm_filename = f"{partition}.rttm"
      rttm_path = Path(data_path, rttm_filename)
      rttm_data = parse_rttm(rttm_path)
      self.video_names = list(rttm_data.keys())
      random.shuffle(self.video_names)
      N = len(self.video_names)
      self.video_names = self.video_names[:int(N * subset)]
      self.video_durations, self.video_fps = self.get_video_metadata(self.video_names)
      self.video_num_frames = np.floor(self.video_durations * self.video_fps).astype(int)
      self.rttm_data = rttm_data # keys: video_names, items: labels
   def __len__(self):
      return len(self.video_names)
   def get_video_metadata(self, video_names):
      root = Path(self.data_path, 'msdwild_boundingbox_labels')
      all_metadata = []
      for video_name in video_names:
         video_path = root / f'{video_name}.mp4'
         metadata = get_streams(video_path)[2]
         all_metadata.append((metadata['video']['duration'], metadata['video']['fps']))
      duration, fps = list(zip(*all_metadata))
      return np.array(duration), np.array(fps)
   def parse_bounding_boxes(self, index):
      video_name = self.video_names[index]  # Convert file_id to a zero-padded string
      data_path = Path(self.data_path)  # Convert self.data_path to a Path object if it's a string
      csv_path = data_path / 'msdwild_boundingbox_labels' / f'{video_name}.csv'  # Ensure correct path

      print(f"DEBUG: Looking for bounding box file at: {csv_path}")  # Debugging

      if csv_path.exists():
         # Read CSV with no headers (ensure all rows are treated as data)
         df = pd.read_csv(csv_path, header=None)

         # Manually assign column names
         df.columns = ["frame_id", "face", "face_id", "x1", "y1", "x2", "y2", "fixed"]

      #   print(f"Bounding Boxes Parsed Successfully for {file_id}:\n", df.head())  # Debugging
         return df

      print(f"ERROR: Bounding boxes file {csv_path} not found!")
      return None
   
   def __getitem__(self, index):
      video_name = self.video_names[int(index)]
      root = Path(self.data_path, 'msdwild_boundingbox_labels')
      video_path = root / f'{video_name}.mp4'
      csv_path = root / f'{video_name}.csv'
      if not video_path.exists():
         raise FileNotFoundError(f"Video file not found: {video_path}")
      video_stream, audio_stream, metadata = get_streams(video_path)
      labels = self.rttm_data.get(video_name, [])
      bounding_boxes = None
      if csv_path.exists():
         bounding_boxes = pd.read_csv(csv_path,header=None, skiprows=0)
         bounding_boxes.columns = ["frame_id", "face", "face_id", "x1", "y1", "x2", "y2", "fixed"]

      return video_stream, audio_stream, labels, bounding_boxes

class MSDWildFrames(MSDWildBase):
   def __init__(self, data_path: str, partition: str, transforms = None, subset: float = 1):
      """
      :param data_path str: path to the directory where the data is stored 
      :param partition str: few_train, few_val or many_val
      """
      super().__init__(data_path, partition, subset)
      # Adding IDs to frames
      # Frame IDs are the position of each frame in this list
      # Each element of the list contains the file ID and the frame timestamp (for seek)
      self.video_last_frame_id = np.cumsum(self.video_num_frames)
      # If transforms is provided, check that is a dictionary with
      # keys: 'video_frame', 'face', and 'audio_segment'
      self.transforms = None
      if transforms:
         assert isinstance(transforms, dict)
         assert 'video_frame' in transforms
         assert 'face' in transforms
         assert 'audio_segment' in transforms
         self.transforms = transforms
      else:
         transforms = {}
         image_transform = ImageTransforms.Compose([
            ImageTransforms.Resize((IMG_HEIGHT, IMG_WIDTH)),
            ImageTransforms.ToDtype(torch.float32, scale=True),
            # ImageTransforms.RandomHorizontalFlip(p = 0.5),
            # ImageTransforms.RandomAffine(degrees=20, translate=(0.1,0.1), scale=(0.9,1.1)),
            # ImageTransforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
            ImageTransforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
         ])
         transforms['video_frame'] = image_transform
         transforms['face'] = image_transform
         transforms['audio_segment'] = lambda x: torch.mean(x, dim=-1)
         self.transforms = transforms
        
   def __len__(self):
      return self.video_last_frame_id[-1]
   
   def get_video_index(self, frame_id, start = 0, end = None):
      """
      Binary search over the last frame id for each video
      Example
      self.video_last_frame_id = [5, 10, 15]
      video_indexes = [0, 1, 2]
      frame_id = 12
      video_index = 2
      """
      if end is None:
         end = len(self.video_last_frame_id)
      if start > end:
         raise ValueError('Frame id not found')
      mid_index = (start + end) // 2
      last_frame_id = self.video_last_frame_id[mid_index]
      # Edge case: Mid video could be the first video
      first_frame_id = 0
      if mid_index > 0:
         first_frame_id = self.video_last_frame_id[mid_index - 1] + 1
      # Found: Frame is between the first and last frame for the mid video
      if first_frame_id <= frame_id <= last_frame_id:
         return mid_index
      # Search right
      elif frame_id > last_frame_id:
         return self.get_video_index(frame_id, mid_index + 1, end)
      # Search left
      else:
         return self.get_video_index(frame_id, start, mid_index - 1)
   
   def get_frame_loc(self, frame_id):
      video_index = self.get_video_index(frame_id)
      # Edge case: First video
      first_frame_id = 0
      if video_index > 0:
         first_frame_id = self.video_last_frame_id[video_index - 1] + 1
      frame_offset = frame_id - first_frame_id
      frame_timestamp = frame_offset / self.video_fps[video_index].item()
      return video_index, frame_offset, frame_timestamp
   
   def get_speakers_at_ts(self, data, timestamp) -> np.ndarray:
      time_intervals, speaker_ids = data  
      start_times = time_intervals[:,0]
      durations=time_intervals[:, 1]
      end_times = start_times+ durations
      active_speaker_ids = [speaker_ids[i] for i in range(len(start_times)) if start_times[i] <= timestamp < end_times[i]]
      if not active_speaker_ids:
        max_speaker_id = max(speaker_ids, default=0)  # Avoid error if speaker_ids is empty
        return np.zeros(max_speaker_id + 1, dtype=int)  
      max_speaker_id = max(speaker_ids)  # Get max speaker ID for array size
      speaker_vector = np.zeros(max_speaker_id + 1, dtype=int)  
      for speaker_id in active_speaker_ids:
         speaker_vector[speaker_id] = 1 
      return speaker_vector

   def extract_faces_from_frame(self, frame, bounding_boxes, frame_offset):
      if bounding_boxes is None:
         return {}

      frame_boxes = bounding_boxes[bounding_boxes["frame_id"] == frame_offset]
      cropped_faces = {}

      for _, row in frame_boxes.iterrows():
         face_id = int(row["face_id"])
         x1, y1, x2, y2 = int(row["x1"]), int(row["y1"]), int(row["x2"]), int(row["y2"])
         x1, y1, x2, y2 = max(x1, 0), max(y1, 0), max(x2, 0), max(y2, 0)
         if x2 > x1 and y2 > y1:
            cropped_faces[face_id] = frame[:, y1:y2, x1:x2]  
      return cropped_faces


   def get_audio_segment(self, audio_stream, frame_id):
      frame_id = int(frame_id)
      current_file_id, current_offset, current_timestamp = self.get_frame_loc(frame_id)
      # Edge case: first frame
      if frame_id == 0:
         start = 0
      else: 
         prev_file_id, prev_offset, prev_timestamp = self.get_frame_loc(frame_id - 1)
         # Case: first frame in video
         if prev_file_id != current_file_id:
            prev_timestamp = 0
         start = (prev_timestamp + current_timestamp) / 2
      # Edge case: last frame
      if frame_id >= len(self):
         end = float('inf')
      else:
         next_file_id, next_offset, next_timestamp = self.get_frame_loc(frame_id + 1)
         # Case: last frame in video
         if current_file_id != next_file_id:
            next_timestamp = float('inf')
         end = (current_timestamp + next_timestamp) / 2
      audio_frames = read_audio(audio_stream, start, end)
      return audio_frames
   
   def get_features(self, frame_id):
      file_id, frame_offset, frame_timestamp = self.get_frame_loc(frame_id)
      video_stream, audio_stream, labels, bounding_boxes = super().__getitem__(file_id)
      # Get frame from video stream
      result = next(iter(video_stream.seek(frame_timestamp)))
      video_frame = result['data']
      cropped_faces = self.extract_faces_from_frame(video_frame, bounding_boxes, frame_offset)
      audio_segment = self.get_audio_segment(audio_stream, frame_id)
      # Transform features
      if self.transforms:
         video_frame = self.transforms['video_frame'](video_frame)
         if cropped_faces:
            cropped_faces = {face_id: self.transforms['face'](cropped_faces[face_id]) for face_id in cropped_faces}
         audio_segment = self.transforms['audio_segment'](audio_segment)
      labels = self.get_speakers_at_ts(labels, frame_timestamp) if labels else None
      # print(labels)
      features = (video_frame, audio_segment, labels, cropped_faces)
      return features

   def get_positive_sample(self, frame_id, anchor_speaker_id):
      # Case: Last frame
      next_frame_offset = self.get_frame_loc(frame_id + 1)[1]
      cropped_faces = {}
      candidate_frame = frame_id
      if next_frame_offset == 0:
         # Use the previous frame
         while anchor_speaker_id not in cropped_faces:
            candidate_frame -= 1
            video_frame, audio_segment, labels, cropped_faces = self.get_features(candidate_frame)
      else:
         # Use the next frame
         while anchor_speaker_id not in cropped_faces:
            candidate_frame += 1
            video_frame, audio_segment, labels, cropped_faces = self.get_features(candidate_frame)
      positive_sample = video_frame, audio_segment, cropped_faces[anchor_speaker_id]
      return positive_sample
   
   def get_negative_sample(self, file_id, face_id):
      if file_id is None:
         raise ValueError("file_id cannot be None")
      random_file_id = None 
      while random_file_id == file_id:
         random_frame = random.randint(0, len(self) - 1)
      anchor, _ = self.get_features(random_frame)
      return anchor

   def __getitem__(self, index):
      video_frame, audio_segment, labels, cropped_faces = self.get_features(index)
      if len(cropped_faces) < 2:
         return self.__getitem__(index + 1)
      face_ids = list(cropped_faces.keys())
      random.shuffle(face_ids)
      anchor_speaker_id, negative_sample_speaker_id = face_ids[:2]
      anchor = video_frame, audio_segment, cropped_faces[anchor_speaker_id]
      negative_pair = video_frame, audio_segment, cropped_faces[negative_sample_speaker_id]
      positive_pair = self.get_positive_sample(index, anchor_speaker_id) 
      # print(labels)
      # print(anchor_speaker_id)
      if anchor_speaker_id >= len(labels):  
         anchor_speaker_id = 0
      label = labels[anchor_speaker_id]
      return anchor, positive_pair, negative_pair, label
   
   def build_batch(self, batch_examples: list):
      # batch_examples: [(anchor1, pos1, neg1, label1), (anchor2, pos2, neg2, label2)]
      batch_examples = [ex for ex in batch_examples if ex is not None]
      batch_size = len(batch_examples)
      features = []
      # features: (feature, element, example)
      for feature_index in range(3):
         feature = []
         for element in range(3):
            for example in range(batch_size):
               feature.append(batch_examples[example][element][feature_index])
         features.append(feature)
      labels = torch.tensor([batch_examples[i][3] for i in range(batch_size)])

      if len(batch_examples) == 0:
         return None

      padded_features = []
      for feature in features:
         feature = [torch.tensor(sample) if sample is not None else torch.zeros(1) for sample in feature]
         feature = torch.nn.utils.rnn.pad_sequence(feature, batch_first=True)
         padded_features.append(feature)
      
      padded_elements = []
      for i in range(3):
         element = []
         for feature in padded_features:
            element.append(feature[i * batch_size: (i + 1) * batch_size])
         padded_elements.append(element)

      return tuple(padded_elements + [labels])

class MSDWildVideos(MSDWildFrames):
   def __init__(self, data_path: str, partition: str, transforms):
      """
      :param data_path str: path to the directory where the data is stored 
      :param partition str: few_train, few_val or many_val
      """
      super().__init__(data_path, partition, transforms)
      self.starting_frame_ids = self.get_starting_frames()
   def get_starting_frames(self):
      starting_frame_ids_path = Path(self.data_path, 'starting_frame_ids.csv')
      if starting_frame_ids_path.exists:
         return pd.read_csv(starting_frame_ids_path)
      else:
         # Iterate over file ids
         counter = 0
         starting_frame_ids = []
         for file_id in self.video_names:
            starting_frame_ids.append(counter)
            video_stream = super(MSDWildBase).__getitem__(file_id)[0]
            for frame in video_stream:
               counter += 1
         starting_frame_ids = np.array(starting_frame_ids)
         np.savetxt(starting_frame_ids_path, starting_frame_ids, delimiter=',')
         return starting_frame_ids
   def __len__(self):
      return len(self.video_names)
   def __getitem__(self, index):
      file_id = self.video_names[index]
      video_stream, audio_stream, labels, bounding_boxes = super().__getitem__(file_id)
      # Get frames from video stream
      frame_id = self.starting_frame_ids[index]
      all_video_frames = []
      all_audio_segments = []
      all_labels = []
      all_faces = []
      for data in video_stream:
         video_frame, frame_timestamp = data['data'], data['pts']
         faces = self.extract_faces_from_frame(video_frame, bounding_boxes, frame_id)
         audio_segment = self.get_audio_segment(audio_stream, frame_timestamp)
         # Transform features
         if self.transforms and faces:
            video_frame = self.transforms['video_frame'](video_frame)
            faces = [self.transforms['face'](face) for face in faces]
            audio_segment = self.transforms['audio_segment'](audio_segment)
         label = self.get_speakers_at_ts(labels, frame_timestamp) if labels else None
         # Accumulate features for all frames
         all_video_frames.append(video_frame)
         all_audio_segments.append(audio_segment)
         all_labels.append(label)
         all_faces.append(faces)
      return all_video_frames, all_audio_segments, all_labels, all_faces




