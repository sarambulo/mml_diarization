from typing import Tuple
import torch

def read_video(video_path: str, seconds: float = 3) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
   """
   Reads a video file and returns video data, audio data, and timestamps.

   :param video_path: Path to the video file.
   :param seconds: Duration of the video to read in seconds.

   :return: A generator that yields a tuple containing:

      - video_data (torch.Tensor): Shape (Frames, C, H, W)
      - audio_data (torch.Tensor): Shape (Frames, SamplingRate, C)
      - timestamps (torch.Tensor): Shape (Frames,)
      - frame_ids  (torch.Tensor): Shape (Frames,)
   """
   return

def downsample_video(video_frames: torch.Tensor, timestamps: torch.Tensor, frame_ids: torch.Tensor, factor: int = 5) -> Tuple[torch.Tensor, torch.Tensor]:
   """
   Downsamples a video and returns the remaining frames, timestamps and frame IDs.
   The number of remaining frames is `ceil(Frames / factor)`
   
   :param video_frames: Shape (Frames, FPS, C, H, W)
   :param timestamps: Shape (Frames,

  :return: A tuple containing:
      - video_data (torch.Tensor): Shape (ceil(Frames / factor), C, H, W)
      - timestamps (torch.Tensor): Shape (ceil(Frames / factor),)
      - frame_ids  (torch.Tensor): Shape (ceil(Frames / factor),)
   """
   return

def extract_faces(
      video_frames: torch.Tensor, frame_ids: torch.Tensor, bounding_boxes: torch.Tensor
   ) ->  Tuple[torch.Tensor, torch.Tensor]:
   """
   Extract the faces identified by the provided bounding boxes

   :param video_frames: Shape (Frames, C, H, W)
   :param frame_ids: Shape (Frames,)
   :param bounding_boxes: List of length Frames with dictionaries. Each dictionary has
   the face ID as a key and the bounding box as the value

   :return: A dictionary with face_ids as keys and extracted bounding boxes as values
   """
   return

def transform_video(
   video_frames: torch.Tensor, height: int = 112, width: int = 112, scale: int = True
) -> torch.Tensor:
   """
   Set all frames to the same size and data type, and scales the values to
   [0, 1]

   :param video_frames: Shape (Frames, C, H, W)
   :param height: Number of rows in the resulting frames
   :param width: Number of columns in the resulting frames
   :param scale: Whether to scale values from [0, 255] to [0, 1]

   :return: Transformed video frames
   """
   return