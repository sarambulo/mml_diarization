from .video import read_video, downsample_video, parse_bounding_boxes, extract_faces, transform_video
from .audio import flatten_audio, transform_audio
from .rttm import get_rttm_labels
import os
import numpy as np
MAX_CHUNKS = 100

def build_chunks(
      video_path: str, bounding_boxes_path: str, rttm_path: str, seconds: int = 3,
      downsampling_factor: int = 5, img_height: int = 112, img_width: int = 112, scale: bool = True, chunk_idx_start: int = 0
   ) -> None:

   video_name = os.path.splitext(os.path.basename(video_path))[0]
   base_dir = os.path.join("preprocessed", video_name)
   os.makedirs(base_dir, exist_ok=True)
   # Get video reader (generator)
   video_reader = read_video(video_path=video_path, seconds=seconds)
   bounding_boxes = parse_bounding_boxes(bounding_boxes_path)
   #parse_rttm outside loop
#    chunks = []
   chunk_idx = chunk_idx_start
   for chunk in video_reader:
      chunk_idx+=1
      if chunk_idx - chunk_idx_start > MAX_CHUNKS:
        break

      chunk_dir = os.path.join(base_dir, f"Chunk_{chunk_idx}")
      os.makedirs(chunk_dir, exist_ok=True)

      video_data, audio_data, timestamps, frame_ids=chunk
      audio_data = flatten_audio(audio_data)
      video_data, timestamps, frame_ids = downsample_video(
         video_frames=video_data, timestamps=timestamps,
         frame_ids=frame_ids, factor=downsampling_factor
      )
      try:
         faces = extract_faces(
            video_frames=video_data, frame_ids=frame_ids,
            bounding_boxes=bounding_boxes
         )
      except:
        #  raise ValueError(f"Error extracting faces for video {video_path}")
        continue
      for speaker_id in faces:
         faces[speaker_id] = transform_video(
            video_frames=faces[speaker_id], height=img_height, width=img_width, scale=scale
         )
      melspectrogram = transform_audio(audio_data) # (Frequencies, Time)
      speaker_ids = list(faces.keys())
      csv_path = os.path.join(chunk_dir, "is_speaking.csv")
      is_speaking = get_rttm_labels(rttm_path, timestamps, speaker_ids=speaker_ids, csv_path=csv_path)
      mel_file = os.path.join(chunk_dir, "melspectrogram.npy")
      np.save(mel_file, melspectrogram)
      for speaker_id, face_dict in faces.items():
            # Bounding boxes
            bbox_array = face_dict 
            bbox_file = os.path.join(chunk_dir, f"face_{speaker_id}.npy")
            np.save(bbox_file, bbox_array)
      # Store chunk
    #   chunks.append((faces, melspectrogram, is_speaking))
   return chunk_idx
