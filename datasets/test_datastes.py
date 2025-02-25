from datasets.MSDWild import MSDWild
from pathlib import Path
import torch
from torch.utils.data import DataLoader

class TestMSDWild():
   def test_init(self):
      msdwild = MSDWild(Path('data'), 'few_train')
      assert len(msdwild) == 2476
      msdwild = MSDWild(Path('data'), 'few_val')
      assert len(msdwild) == 490
      msdwild = MSDWild(Path('data'), 'many_val')
      assert len(msdwild) == 177

   def test_item(self):
      msdwild = MSDWild(Path('data'), 'many_val')
      data = next(iter(msdwild))

      # First element is the video data as a tensor
      video_data = data[0]
      assert isinstance(video_data, torch.Tensor)
      # Shape of the video data is (time, channels, height, width)
      assert video_data.dim() == 4
      # Three color channels
      assert video_data.shape[1] == 3

      # Third element is the audio data as a tensor
      audio_data = data[2]
      assert isinstance(audio_data, torch.Tensor)
      # Shape of the audio data is (time, channels)
      assert audio_data.dim() == 2
      # Two audio channels (stereo)
      assert audio_data.shape[1] == 2

   def test_batch(self):
      msdwild = MSDWild(Path('data'), 'many_val', max_video_frames=60)
      dataloader = DataLoader(msdwild, batch_size=1, shuffle=True, collate_fn=msdwild.build_batch)
      batch = next(iter(dataloader))
      assert batch is not None


      