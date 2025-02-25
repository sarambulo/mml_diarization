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
from .utils import parse_rttm, read_video
import numpy as np

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
      # TODO: correct path and retrieve csv
      video_path = Path(self.data_path, 'msdwild_boundingbox_labels', file_id, f'{file_id}.mp4')
      # TODO: get video and audio streams
      video_stream, audio_stream, metadata = (None, None, None)
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
      # TODO: Assing IDs to frames
      # Frame IDs are the position of each frame in this list
      self.frame_ids = []
      # Each position holds the file ID and the frame timestamp (for seek)
      # Iterate over file ids
      # Append file ID and timestamp for each frame
      # TODO: If transforms is provided, check that is a dictionary with
      # keys: 'video_frame', 'face', and 'audio_segment'
      self.transform = transforms
   def __len__(self):
      return len(self.frame_ids)
   def __getitem__(self, index):
      file_id, frame_timestamp = self.frame_ids[index]
      video_stream, audio_stream, labels, bounding_boxes = super(MSDWildFrames).__getitem__(file_id)
      # TODO: get frame from video stream
      video_frame = None
      # TODO: extract faces from frame using bounding box
      cropped_faces = None
      # TODO: select face randomly
      face, face_id = (None, None)
      # TODO: get audio segment from audio stream
      audio_segment = None
      # TODO: transform features
      if self.transform:
         video_frame = self.transforms['video_frame'](video_frame)
         face = self.transforms['face'](face)
         audio_segment = self.transforms['audio_segment'](audio_segment)
      # TODO: extract label
      label = None
      # TODO: Select positive and negative pairs
      anchor = (video_frame, audio_segment, label, face)
      positive_pair = None
      negative_pair = None
      return anchor, positive_pair, negative_pair
   def build_batch(self, batch_examples: list):
      # TODO: Add padding
      return NotImplemented
      features = list(zip(*batch_examples))
      for feature in features:
         feature = [torch.tensor(example) for example in feature]
         feature = torch.stack(feature, axis=0)
      return tuple([feature for feature in features])

class MSDWildVideos(MSDWildBase):
   def __init__(self, data_path: str, partition: str, transforms, max_length = None, max_video_frames = None):
      """
      :param data_path str: path to the directory where the data is stored 
      :param partition str: few_train, few_val or many_val
      :param max_length float: Each video and audio will be clipped to this duration in seconds
      :param max_video_frames int: If `max_video_frames / FPS` is lower than `max_length`, clip videos
         and audios to `max_video_frames / FPS` seconds
      """
      super(MSDWildFrames).__init__(data_path, partition)
      # Store configuration for generating segments
      self.transform = transform
      self.max_video_frames = max_video_frames
      self.max_length = max_length # Also frames / fps
   def __len__(self):
      return len(self.file_ids)
   def __getitem__(self, index):
      return NotImplemented
   def build_batch(self, batch_examples: list):
      # TODO: Add padding
      return NotImplemented
      features = list(zip(*batch_examples))
      for feature in features:
         feature = [torch.tensor(example) for example in feature]
         feature = torch.stack(feature, axis=0)
      return tuple([feature for feature in features])