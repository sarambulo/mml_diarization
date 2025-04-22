from typing import List, Tuple
from .video import (
    read_video,
    downsample_video,
    parse_bounding_boxes,
    extract_faces,
    transform_video,
)
from .audio import flatten_audio, transform_audio
from .rttm import get_rttm_labels
import math


def build_chunks(
    video_path: str,
    bounding_boxes_path: str,
    rttm_path: str,
    video_id,
    seconds: int = 3,
    downsampling_factor: int = 5,
    img_height: int = 112,
    img_width: int = 112,
    scale: bool = True,
    max_chunks: int = 60,
) -> List[Tuple]:
    """
    Returns a list of chunks. Each chunk is a tuple of three elements:
    - Faces: dictionary with face IDs as keys and sequences of cropped faces as values
    - Melspectrogram
    - Is Speaking: pd.DataFrame with cols "face_id", "frame_id", "is_speaking"
    """

    # Get video reader (generator)
    video_reader, metadata = read_video(video_path=video_path, seconds=seconds)
    bounding_boxes = parse_bounding_boxes(bounding_boxes_path)
    chunks = []
    for i, chunk in enumerate(video_reader):
        # Stop after max_chunks
        if i >= max_chunks:
            break

        video_data, audio_data, timestamps, frame_ids = chunk

        # Audio
        audio_data = flatten_audio(audio_data)  # (N, C)
        melspectrogram, mfcc = transform_audio(
            audio_data, sr=metadata["sampling_rate"], n_bands=30, target_tf=440
        )  # (Frequencies, Time)
        #   print(melspectrogram.shape)
        # Video
        video_data, timestamps, frame_ids = downsample_video(
            video_frames=video_data,
            timestamps=timestamps,
            frame_ids=frame_ids,
            factor=math.ceil(metadata["fps"] / 4),
        )
        try:
            faces = extract_faces(
                video_frames=video_data,
                frame_ids=frame_ids,
                bounding_boxes=bounding_boxes,
            )
        except:
            raise ValueError(f"Failed extracting faces for video {video_path}")
        for speaker_id in faces:
            faces[speaker_id] = transform_video(
                video_frames=faces[speaker_id],
                height=img_height,
                width=img_width,
                scale=scale,
            )

        # Labels
        speaker_ids = list(faces.keys())
        is_speaking = get_rttm_labels(
            rttm_path, timestamps, speaker_ids=speaker_ids, video_id=video_id
        )
        chunks.append((faces, melspectrogram, mfcc, is_speaking))
    return chunks
