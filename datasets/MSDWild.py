import torch
from torch.utils.data import DataLoader, Dataset
from pathlib import Path
from .utils import parse_rttm, read_video
import numpy as np

class MSDWild(Dataset):
   def __init__(self, data_path: str, partition = str, clip_length = None, frames = None, fps = 25):
      """
      :param partition str: few_train, few_val or many_val
      """
      self.data_path = data_path
      # Parse the rttm file to extract the file ids and labels
      rttm_filename = f"{partition}.rttm"
      rttm_path = Path(data_path, rttm_filename)
      rttm_data = parse_rttm(rttm_path)
      if rttm_data is None:
         print(f"rttm file {rttm_filename} not found in {data_path}")
         raise FileNotFoundError
      self.file_ids = list(rttm_data.keys())
      self.rttm_data = rttm_data # items: labels

      # Store configuration from generating time steps
      self.frames = frames
      self.fps = fps
      self.clip_length = clip_length # Also frames / fps
   
   def __len__(self):
      return len(self.file_ids)
   
   def __getitem__(self, index):
      file_id = self.file_ids[index]
      # audio_path = Path(self.data_path, 'wav', f'{file_id}.wav')
      video_path = Path(self.data_path, 'mp4', f'{file_id}.mp4')
      video_frames, audio_frames, timestamps, metadata = read_video(video_path, start_sec=0, end_sec=self.clip_length)
      labels = self.rttm_data[file_id]
      return video_frames, audio_frames, timestamps, video_path, labels