import torch
from utils.video import read_video, downsample_video, parse_bounding_boxes, extract_faces, transform_video
from math import ceil

def test_read_video():
   video_path = './data_sample/msdwild_boundingbox_labels/00001.mp4'
   video_reader = read_video(video_path=video_path, seconds=1)
   for video_chunk in video_reader:
      assert len(video_chunk) == 4
      assert all([isinstance(item, torch.Tensor) for item in video_chunk])
      video_data, audio_data, timestamps, frame_ids = video_chunk
      assert video_data.dim() == 4 # (Frames, C, H, W)
      assert audio_data.dim() == 3 # (Frames, S, C)
      assert timestamps.dim() == 1 # (Frames, )
      assert frame_ids.dim() == 1 # (Frames, )
   return

def test_downsample_video():
   F, C, H, W = (10, 3, 5, 5)  # (Frames, Channels, Height, Width)
   video_frames = torch.rand((F, C, H, W))
   DURATION = 2 # seconds
   timestamps = torch.arange(0, DURATION, DURATION / F)
   frame_ids = torch.arange(0, F)
   FACTOR = 2
   result = downsample_video(
      video_frames=video_frames,
      timestamps=timestamps,
      frame_ids=frame_ids,
      factor = FACTOR
   )
   assert len(result) == 3
   assert all([isinstance(item, torch.Tensor) for item in result])
   video_frames, timestamps, frame_ids = result
   assert len(video_frames) == ceil(F / FACTOR)
   assert len(timestamps) == ceil(F / FACTOR)
   assert len(frame_ids) == ceil(F / FACTOR)
   return

def test_parse_bounding_boxes():
   bounding_boxes_path = './data_sample/msdwild_boundingbox_labels/00001.csv'
   box_1_reference = {'frame_id': 0, 'face_id': 2, 'coords': [366, 442, 142, 253]}
   box_2_reference = {'frame_id': 1, 'face_id': 0, 'coords': [725, 795, 181, 275]}
   bounding_boxes = parse_bounding_boxes(bounding_boxes_path=bounding_boxes_path)
   for reference in [box_1_reference, box_2_reference]:
      # Frame is present
      frame_id = reference['frame_id']
      assert frame_id in bounding_boxes
      # Face is present in that frame
      bounding_boxes_in_frame = bounding_boxes[frame_id]
      face_id = reference['face_id']
      assert face_id in bounding_boxes_in_frame
      # Coordinates match
      box_coords = bounding_boxes_in_frame[face_id]
      assert (box_coords == reference['coords']).all()
   return

def test_extract_faces():
   bounding_boxes = {
      0: {
         2: [350, 400, 120, 200]
      },
      4: {
         0: [600, 700, 450, 550],
         1: [700, 800, 180, 280],
      },
   }
   video_frames = torch.randint(low=0, high=255, size=(2, 3, 1000, 1000))
   frame_ids = torch.tensor([0, 4])
   faces = extract_faces(video_frames=video_frames, frame_ids=frame_ids, bounding_boxes=bounding_boxes)
   face_ids = [0, 1, 2]
   num_frames = [1, 1, 1]
   assert len(faces) == len(face_ids)
   for face_id, N in zip(face_ids, num_frames):
      assert face_id in faces
      assert isinstance(faces[face_id], torch.Tensor)
      assert len(faces[face_id]) == N

def test_transform_video():
   B = 2 # Batch size
   video_frames = torch.randint(low=0, high=255, size=(B, 3, 1000, 1000))
   H, W, scale = 100, 100, True
   video_frames = transform_video(video_frames=video_frames, height=H, width=W, scale=scale)
   assert isinstance(video_frames, torch.Tensor)
   assert video_frames.shape == (B, 3, H, W)
   assert torch.all((-2 <= video_frames) & (video_frames <= 2))
