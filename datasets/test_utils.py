from .utils import parse_rttm, get_streams
from pathlib import Path
import torch
import numpy as np

class TestParseRTTM():
   def test_ids(self):
      data_path = Path('data')
      rttm_files = ['few_train', 'few_val', 'many_val']
      for file in rttm_files:
         file_path = data_path / f'{file}.rttm'
         rttm = parse_rttm(file_path)
         # rttm is a dictionary with file ids as keys
         assert isinstance(rttm, dict)
         # each entry has a tuple with the timestamps and the speaker ids
         for file_id in rttm:
            data = rttm[file_id]
            assert len(data) == 2
            assert isinstance(data[0], np.ndarray)
            assert isinstance(data[1], np.ndarray)

class TestGetStream():
   def test_data(self):
      video_path = Path('data_sample', 'msdwild_boundingbox_labels', '00001.mp4')
      video_stream, audio_stream, metadata = get_streams(video_path)
      # Test video stream
      result = next(iter(video_stream))
      video_frame, video_frame_timestamp = result['data'], result['pts']
      assert isinstance(video_frame, torch.Tensor)
      assert video_frame.dim() == 3
      assert isinstance(video_frame_timestamp, float)
      # Test audio stream
      result =next(iter(audio_stream)) 
      audio_frame, audio_frame_timestamp =result['data'], result['pts']
      assert isinstance(audio_frame, torch.Tensor)
      assert audio_frame.dim() == 2
      assert isinstance(audio_frame_timestamp, float)
      # Test metadata
      assert isinstance(metadata, dict)
      assert 'video' in metadata
      assert 'audio' in metadata
