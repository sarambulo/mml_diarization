from datasets.MSDWild import MSDWildBase, MSDWildFrames
from pathlib import Path
import torch
from torch.utils.data import DataLoader
import numpy as np
import pandas as pd

DATA_PATH = 'data_sample'

class TestMSDWildBase():
   def test_init(self):
      msdwild = MSDWildBase(Path(DATA_PATH), 'few_train')
      assert len(msdwild) == 5
      msdwild = MSDWildBase(Path(DATA_PATH), 'few_val')
      assert len(msdwild) == 5
      msdwild = MSDWildBase(Path(DATA_PATH), 'many_val')
      assert len(msdwild) == 5

   def test_item(self):
      msdwild = MSDWildBase(Path(DATA_PATH), 'many_val')
      for data in msdwild:
         video_stream, audio_stream, rttm, faces_bounding_boxes = data
         # First element is the video stream
         result = next(iter(video_stream))
         video_frame, timestamp = result['data'], result['pts']
         assert isinstance(video_frame, torch.Tensor)
         assert isinstance(timestamp, float)
         # Second element is the audio stream
         result = next(iter(audio_stream))
         audio_frame, timestamp = result['data'], result['pts']
         assert isinstance(audio_frame, torch.Tensor)
         assert isinstance(timestamp, float)
         # Third element is the parsed rttm
         timestamps, speaker_ids = rttm
         assert isinstance(timestamps, np.ndarray)
         assert isinstance(speaker_ids, np.ndarray)
         # Fourth element is the bounding boxes for the faces
         assert isinstance(faces_bounding_boxes, pd.DataFrame)

class TestMSDWildFrames():
   def test_init(self):
      msdwild = MSDWildFrames(Path(DATA_PATH), 'few_train')
      assert len(msdwild) >= 5 * 30 * 5 # num_videos * fps * seconds
      msdwild = MSDWildFrames(Path(DATA_PATH), 'few_val')
      assert len(msdwild) >= 5 * 30 * 5 # num_videos * fps * seconds
      msdwild = MSDWildFrames(Path(DATA_PATH), 'many_val')
      assert len(msdwild) >= 5 * 30 * 5 # num_videos * fps * seconds

   def test_item(self):
      msdwild = MSDWildFrames(Path(DATA_PATH), 'many_val')
      for data in msdwild:
         anchor, positive_pair, negative_pair, label = data
         assert self.triplet_element(anchor)
         assert self.triplet_element(positive_pair)
         assert self.triplet_element(negative_pair)
         # assert label == 1

   def test_batch(self):
      msdwild = MSDWildFrames(Path(DATA_PATH), 'many_val')
      dataloader = DataLoader(msdwild, batch_size=2, shuffle=True, collate_fn=msdwild.build_batch)
      batch = next(iter(dataloader))
      assert batch is not None

   def triplet_element(self, element):
      video_frame, audio_segment, face = element
      # First element is a video frame
      assert isinstance(video_frame, torch.Tensor)
      assert video_frame.dim() == 3
      assert video_frame.shape == (3, 112, 112)
      # Second element is the audio stream
      assert isinstance(audio_segment, torch.Tensor)
      # Third element is the face
      assert isinstance(face, torch.Tensor)
      assert video_frame.dim() == 3
      assert video_frame.shape == (3, 112, 112)
      return True

      