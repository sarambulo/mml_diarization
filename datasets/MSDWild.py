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
from .utils import get_streams, parse_rttm, read_video
import numpy as np
from ultralytics import YOLO
import cv2
import random
import torch
import torchaudio
import pandas as pd

class MSDWildBase(Dataset):
   def __init__(self, data_path: str, partition = str):
      """
      :param data_path str: path to the directory where the data is stored 
      :param partition str: few_train, few_val or many_val
      """
      super(MSDWildBase).__init__()
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
      # TODO: parse csv
      csv = None
      # Get video and audio streams
      video_stream, audio_stream, metadata = get_streams(video_path)
      # TODO: extract labels from rttm file
      labels = self.rttm_data[file_id]
      # TODO: load csv
      bounding_boxes = None
      return video_stream, audio_stream, labels, bounding_boxes



class MSDWildFrames(MSDWildBase):
   def __init__(self, data_path: str, partition: str, transforms):
      """
      :param data_path str: path to the directory where the data is stored 
      :param partition str: few_train, few_val or many_val
      """
      super(MSDWildFrames).__init__(data_path, partition)
      # TODO: Adding IDs to frames
      # Frame IDs are the position of each frame in this list
      self.frame_ids = []
      self._populate_frame_ids()
      # Each position holds the file ID and the frame timestamp (for seek)
      # Iterate over file ids
      # Append file ID and timestamp for each frame
      # TODO: If transforms is provided, check that is a dictionary with
      # keys: 'video_frame', 'face', and 'audio_segment'
      self.transform = transforms

   def _populate_frame_ids(self):
        # TODO
        return NotImplemented
        
   def __len__(self):
      return len(self.frame_ids)
   
   def parse_bounding_boxes(self, file_id):
        csv_path = self.data_path / 'msdwild_boundingbox_labels' / f'{file_id}.csv'
        if csv_path.exists():
            df = pd.read_csv(csv_path, header=None) 
            df.columns = ["frame_id", "face", "face_id", "x1", "y1", "x2", "y2", "fixed"]
            return df
        return None
   
   def extract_faces_from_frame(self, frame, bounding_boxes, frame_id):
      if bounding_boxes is None:
         return {}
      frame_boxes =bounding_boxes[bounding_boxes["frame_id"] == frame_id]
      cropped_faces = {}
      for _, row in frame_boxes.iterrows():
         x1, y1, x2, y2 = int(row["x1"]), int(row["y1"]), int(row["x2"]), int(row["y2"])
         cropped_faces[row["face_id"]] = frame[y1:y2, x1:x2]  
      return cropped_faces
       
   def get_audio_segment(self, audio_stream, timestamp, segment_duration=1.0):
      #   if not audio_stream:
      #       return None

      #   waveform, sample_rate = torchaudio.load(audio_stream)  # Load as tensor
      #   start_sample = int(timestamp * sample_rate)
      #   end_sample = start_sample + int(segment_duration * sample_rate)

      #   return waveform[:, start_sample:end_sample]  # Extract segment

   def get_positive_sample(self, file_id, frame_id, face_id): #TODO
      bounding_boxes = self.parse_bounding_boxes(file_id)
      if bounding_boxes is None:
         return None
      next_frame = bounding_boxes[(bounding_boxes["frame_id"] > frame_id) & (bounding_boxes["face_id"] == face_id)]
      if not next_frame.empty:
         next_frame_id = next_frame.iloc[0]["frame_id"]
         return self.get_anchor(self.frame_ids.index((file_id, next_frame_id)))
      return None
   
   def get_negative_sample(self, file_id, face_id): #TODO
      negative_candidates = []
      for fid in self.file_ids:
         bounding_boxes = self.parse_bounding_boxes(fid)
         if bounding_boxes is None:
            continue
         different_faces = bounding_boxes[bounding_boxes["face_id"] != face_id]
         if not different_faces.empty:
            negative_candidates.append(different_faces)

      if negative_candidates:
         neg_sample = random.choice(negative_candidates).iloc[0]
         return self.__getitem__(self.frame_ids.index((neg_sample["frame_id"], neg_sample["face_id"])))
      return None
   
   def get_anchor(self, index):
         file_id, frame_timestamp = self.frame_ids[index]
         video_stream, audio_stream, labels, bounding_boxes = super(MSDWildFrames).__getitem__(file_id)
         # TODO: get frame from video stream
         video_frame =next(iter(video_stream.seek(frame_timestamp)))
         cropped_faces = self.extract_faces_from_frame(video_frame, bounding_boxes, index)
         face, face_id = (None, None)
         if cropped_faces:
               face_id = random.randint(0, len(cropped_faces) - 1)
               face = cropped_faces[face_id]
         audio_segment =self.get_audio_segment(audio_stream, frame_timestamp)
         # TODO: transform features
         if self.transform:
            video_frame = self.transforms['video_frame'](video_frame)
            face = self.transforms['face'](face)
            audio_segment = self.transforms['audio_segment'](audio_segment)
         label = labels[0] if labels else None
         anchor = (video_frame, audio_segment, label, face)
         return anchor, face_id

   def __getitem__(self, index):
      file_id, frame_timestamp = self.frame_ids[index]
      anchor, face_id= self.get_anchor(index)
      positive_pair =self.get_positive_sample(file_id, frame_timestamp, face_id) 
      negative_pair =self.get_negative_sample(file_id, face_id)
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

# class MSDWildVideos(MSDWildBase):
#    def __init__(self, data_path: str, partition: str, transforms, max_length = None, max_video_frames = None):
#       """
#       :param data_path str: path to the directory where the data is stored 
#       :param partition str: few_train, few_val or many_val
#       :param max_length float: Each video and audio will be clipped to this duration in seconds
#       :param max_video_frames int: If `max_video_frames / FPS` is lower than `max_length`, clip videos
#          and audios to `max_video_frames / FPS` seconds
#       """
#       super(MSDWildFrames).__init__(data_path, partition)
#       # Store configuration for generating segments
#       self.transform = transform
#       self.max_video_frames = max_video_frames
#       self.max_length = max_length # Also frames / fps
#    def __len__(self):
#       return len(self.file_ids)
#    def __getitem__(self, index):
#       return NotImplemented
#    def build_batch(self, batch_examples: list):
#       # TODO: Add padding
#       return NotImplemented
#       features = list(zip(*batch_examples))
#       for feature in features:
#          feature = [torch.tensor(example) for example in feature]
#          feature = torch.stack(feature, axis=0)
#       return tuple([feature for feature in features])



