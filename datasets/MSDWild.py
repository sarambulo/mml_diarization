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
import torchaudio
import pandas as pd
import torchvision.transforms.v2 as Transforms

IMG_WIDTH = 112
IMG_HEIGHT = 112

class MSDWildBase(Dataset):
   def __init__(self, data_path: str, partition = str):
      """
      :param data_path str: path to the directory where the data is stored 
      :param partition str: few_train, few_val or many_val
      """
      super().__init__()
      self.data_path = data_path
      # Parse the rttm file to extract the file ids and labels
      rttm_filename = f"{partition}.rttm"
      rttm_path = Path(data_path, rttm_filename)
      rttm_data = parse_rttm(rttm_path)
      self.file_ids = list(rttm_data.keys())
      self.rttm_data = rttm_data # items: labels
   def __len__(self):
      return len(self.file_ids)
   
   def __getitem__(self, index):
      file_id = self.file_ids[index]
      root = Path(self.data_path, 'msdwild_boundingbox_labels')
      video_path = root / f'{file_id}.mp4'
      csv_path = root / f'{file_id}.csv'
      # Get video and audio streams
      video_stream, audio_stream, metadata = get_streams(video_path)
      # TODO: extract labels from rttm file
      labels = self.rttm_data[file_id]
      # Load bounding boxes from csv
      bounding_boxes =self.parse_bounding_boxes(csv_path)
      return video_stream, audio_stream, labels, bounding_boxes



class MSDWildFrames(MSDWildBase):
   def __init__(self, data_path: str, partition: str, transforms):
      """
      :param data_path str: path to the directory where the data is stored 
      :param partition str: few_train, few_val or many_val
      """
      super().__init__(data_path, partition)
      # Adding IDs to frames
      # Frame IDs are the position of each frame in this list
      # Each element of the list contains the file ID and the frame timestamp (for seek)
      self.frame_ids = self.get_frame_ids()
      # If transforms is provided, check that is a dictionary with
      # keys: 'video_frame', 'face', and 'audio_segment'
      self.transform = None
      if transforms:
         assert isinstance(transforms, dict)
         assert 'video_frame' in transforms
         assert 'face' in transforms
         assert 'audio_segment' in transforms
         self.transform = transforms
      else:
         transforms = {}
         image_transform = Transforms.Compose([
            Transforms.Resize((IMG_HEIGHT, IMG_WIDTH)),
            Transforms.ToDtype(torch.float32, scale=True),
            Transforms.RandomHorizontalFlip(p = 0.5),
            Transforms.RandomAffine(degrees=20, translate=(0.1,0.1), scale=(0.9,1.1)),
            Transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
            Transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
         ])
         transforms['video_frame'] = image_transform
         transforms['face'] = image_transform
         transforms['audio_segment'] = Transforms.Identity()

   def get_frame_ids(self):
      frame_ids_path = Path(self.data_path, 'frame_ids.csv')
      if frame_ids_path.exists:
         return pd.read_csv(frame_ids_path)
      else:
         # Iterate over file ids
         frame_ids = []
         for file_id in self.file_ids:
            video_stream = super(MSDWildBase).__getitem__(file_id)[0]
            # Append file ID and timestamp for each frame
            for frame in video_stream:
               frame_ids.append((file_id, frame['pts']))
         frame_ids = np.array(frame_ids)
         return frame_ids
        
   def __len__(self):
      return len(self.frame_ids)
   
   def get_speakers_at_ts(self, data, timestamp):
      time_intervals, speaker_ids = data  
      start_times = time_intervals[:,0]
      durations=time_intervals[:, 1]
      end_times = start_times+ durations
      active_speaker_ids = [speaker_ids[i] for i in range(len(start_times)) if start_times[i] <= timestamp < end_times[i]]
      if not active_speaker_ids:
        return np.zeros(1, dtype=int)  # Return a single zero if no speakers are active
      max_speaker_id = max(speaker_ids)  # Get max speaker ID for array size
      speaker_vector = np.zeros(max_speaker_id + 1, dtype=int)  
      for speaker_id in active_speaker_ids:
         speaker_vector[speaker_id] = 1 
      return speaker_vector
   
   def extract_faces_from_frame(self, frame, bounding_boxes, frame_id):
      if bounding_boxes is None:
         return {}
      frame_boxes = bounding_boxes[bounding_boxes["frame_id"] == frame_id]
      cropped_faces = {}
      for _, row in frame_boxes.iterrows():
         x1, y1, x2, y2 = int(row["x1"]), int(row["y1"]), int(row["x2"]), int(row["y2"])
         cropped_faces[row["face_id"]] = frame[y1:y2, x1:x2]  
      return cropped_faces
       
   def get_audio_segment(self, audio_stream, frame_id):
      prev_file_id, prev_timestamp = self.frame_ids[frame_id - 1]
      current_file_id, current_timestamp = self.frame_ids[frame_id]
      next_file_id, next_timestamp = self.frame_ids[frame_id + 1]
      # Case: first frame in video
      if prev_file_id != current_file_id:
         prev_timestamp = 0
      # Case: last frame in video
      elif current_file_id != next_file_id:
         next_timestamp = float('inf')
      start = (prev_timestamp + current_timestamp) / 2
      end = (current_timestamp + next_timestamp) / 2
      audio_frames = read_audio(audio_stream, start, end)
      return audio_frames

   def get_positive_sample(self, file_id, frame_id, face_id): #TODO
      bounding_boxes = self.parse_bounding_boxes(file_id)
      if bounding_boxes is None:
         return None
      next_frame = bounding_boxes[(bounding_boxes["frame_id"] > frame_id) & (bounding_boxes["face_id"] == face_id)]
      if not next_frame.empty:
         next_frame_id = next_frame.iloc[0]["frame_id"]
         anchor,_=self.get_features(self.frame_ids.index((file_id, next_frame_id)))
         return anchor
      return None
   
   def get_negative_sample(self, file_id, face_id): #TODO
      random_frame_id = random.choice(self.frame_ids)
      while self.frame_ids[random_frame_id][0]==file_id:
         random_frame_id = random.choice(self.frame_ids)
      anchor, _ =self.get_features(random_frame_id)
      return anchor
   
   def get_features(self, index):
         file_id, frame_timestamp = self.frame_ids[index]
         video_stream, audio_stream, labels, bounding_boxes = super(MSDWildFrames).__getitem__(file_id)
         # Get frame from video stream
         video_frame = next(iter(video_stream.seek(frame_timestamp)))
         cropped_faces = self.extract_faces_from_frame(video_frame, bounding_boxes, index)
         face, face_id = (None, None)
         if cropped_faces:
            face_id = random.randint(0, len(cropped_faces) - 1)
            face = cropped_faces[face_id]
         audio_segment = self.get_audio_segment(audio_stream, frame_timestamp)
         # Transform features
         if self.transform:
            video_frame = self.transforms['video_frame'](video_frame)
            face = self.transforms['face'](face)
            audio_segment = self.transforms['audio_segment'](audio_segment)
         labels = self.get_speakers_at_ts(labels, frame_timestamp) if labels else None
         anchor = (video_frame, audio_segment, labels, face)
         return anchor, face_id

   def __getitem__(self, index):
      file_id, frame_timestamp = self.frame_ids[index]
      anchor, face_id = self.get_anchor(index)
      positive_pair = self.get_positive_sample(file_id, frame_timestamp, face_id) 
      negative_pair = self.get_negative_sample(file_id, face_id)
      return anchor, positive_pair, negative_pair
   
   def build_batch(self, batch_examples: list):
      batch_examples = [ex for ex in batch_examples if ex is not None]

      if len(batch_examples) == 0:
         return None
      features = list(zip(*batch_examples))
      padded_features = []

      for feature in features:
         feature = [torch.tensor(f) if f is not None else torch.zeros(1) for f in feature]
         feature = torch.nn.utils.rnn.pad_sequence(feature, batch_first=True)
         padded_features.append(feature)

      return tuple(padded_features)

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
         for file_id in self.file_ids:
            starting_frame_ids.append(counter)
            video_stream = super(MSDWildBase).__getitem__(file_id)[0]
            for frame in video_stream:
               counter += 1
         starting_frame_ids = np.array(starting_frame_ids)
         np.savetxt(starting_frame_ids_path, starting_frame_ids, delimiter=',')
         return starting_frame_ids
   def __len__(self):
      return len(self.file_ids)
   def __getitem__(self, index):
      file_id = self.file_ids[index]
      video_stream, audio_stream, labels, bounding_boxes = super(MSDWildBase).__getitem__(file_id)
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
         if self.transform:
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




