from .video import read_video, downsample_video, parse_bounding_boxes, extract_faces, transform_video
from .audio import flatten_audio, transform_audio
from .rttm import get_rttm_labels

def build_chunks(
      video_path: str, bounding_boxes_path: str, rttm_path: str, seconds: int = 3,
      downsampling_factor: int = 5, img_height: int = 112, img_width: int = 112, scale: bool = True
   ) -> None:
   # Get video reader (generator)
   video_reader = read_video(video_path=video_path, seconds=seconds)
   bounding_boxes = parse_bounding_boxes(bounding_boxes_path)
   #parse_rttm outside loop
   chunks = []
   for chunk in video_reader:
      # Video data (F, C, H, W), Audio data (F, SamplesPerFrame, C)
      video_data, audio_data, timestamps, frame_ids = chunk
      audio_data = flatten_audio(audio_data)
      video_data, timestamps, frame_ids = downsample_video(
         video_frames=video_data, timestamps=timestamps,
         frame_ids=frame_ids, factor=downsampling_factor
      )
      faces = extract_faces(
         video_frames=video_data, frame_ids=frame_ids,
         bounding_boxes=bounding_boxes
      )
      for speaker_id in faces:
         faces[speaker_id] = transform_video(
            video_frames=faces[speaker_id], height=img_height, width=img_width, scale=scale
         )
      melspectogram = transform_audio(audio_data) # (Frequencies, Time)
      speaker_ids = list(faces.keys())
      is_speaking = get_rttm_labels(rttm_path, timestamps, speaker_ids=speaker_ids)
      # Store chunk
      chunks.append((faces, melspectogram, is_speaking))
   return chunks

def main():
   # Go to the data directory
   # Scan the files to identify video ids
   # For each video id
      # Call build chunks
      # Store the results on disk
   return

if __name__ == '__main__':
   # Parse args like data directory
   main()
