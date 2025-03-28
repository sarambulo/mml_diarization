from datasets.MSDWild import MSDWildChunks
from pathlib import Path
import torch
from torch.utils.data import DataLoader
import numpy as np
import pandas as pd

DATA_PATH = 'preprocessed'

class TestMSDWildBase():
   def test_init(self):
      msdwild = MSDWildChunks(Path(DATA_PATH), Path('data_sample', 'few_train.rttm'))
      assert len(msdwild) == 5
      msdwild = MSDWildChunks(Path(DATA_PATH), Path('data_sample', 'few_val.rttm'))
      assert len(msdwild) == 5
      msdwild = MSDWildChunks(Path(DATA_PATH), Path('data_sample', 'many_val.rttm'))
      assert len(msdwild) == 5

   def test_item(self):
      msdwild = MSDWildChunks(Path(DATA_PATH), 'few_train.rttm')
      for data in msdwild:
         video_data, audio_data, is_speaking = data
         # First element is the video data
         assert isinstance(video_data, torch.Tensor)
         assert video_data.shape == (3, 3, 112, 112)
         # Second element is the audio stream
         assert isinstance(audio_data, torch.Tensor)
         assert audio_data.shape == (3, 30, 22)
         # Third element is the parsed rttm
         assert isinstance(is_speaking, int)

# class TestMSDWildFrames():
#    def test_init(self):
#       msdwild = MSDWildFrames(Path(DATA_PATH), 'few_train')
#       assert len(msdwild) >= 5 * 30 * 5 # num_videos * fps * seconds
#       msdwild = MSDWildFrames(Path(DATA_PATH), 'few_val')
#       assert len(msdwild) >= 5 * 30 * 5 # num_videos * fps * seconds
#       msdwild = MSDWildFrames(Path(DATA_PATH), 'many_val')
#       assert len(msdwild) >= 5 * 30 * 5 # num_videos * fps * seconds

#    def test_item(self):
#       msdwild = MSDWildFrames(Path(DATA_PATH), 'many_val')
#       for data in msdwild:
#          anchor, positive_pair, negative_pair, label = data
#          assert self.triplet_element(anchor)
#          assert self.triplet_element(positive_pair)
#          assert self.triplet_element(negative_pair)
#          # assert label == 1

#    def test_batch(self):
#       msdwild = MSDWildFrames(Path(DATA_PATH), 'many_val')
#       dataloader = DataLoader(msdwild, batch_size=2, shuffle=True, collate_fn=msdwild.build_batch)
#       batch = next(iter(dataloader))
#       assert batch is not None

#    def triplet_element(self, element):
#       video_frame, audio_segment, face = element
#       # First element is a video frame
#       assert isinstance(video_frame, torch.Tensor)
#       assert video_frame.dim() == 3
#       assert video_frame.shape == (3, 112, 112)
#       # Second element is the audio stream
#       assert isinstance(audio_segment, torch.Tensor)
#       # Third element is the face
#       assert isinstance(face, torch.Tensor)
#       assert video_frame.dim() == 3
#       assert video_frame.shape == (3, 112, 112)
#       return True
   
#    def test_face_crop(self):
#       msdwild = MSDWildFrames(Path('data'), 'few_train')
#       msdwild.get_features(23796)

      