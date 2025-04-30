from typing import Dict, Tuple, List
import torch
from pathlib import Path
from torchvision.io import VideoReader
import itertools
import pandas as pd
import numpy as np
import torchvision.transforms.v2 as ImageTransforms
import boto3
import os

s3 = boto3.client("s3")
bucket_name = "mmml-proj"


def read_video(
    video_path: str, seconds: float = 3
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
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
    # Check inputs
    if seconds <= 0:
        raise ValueError("seconds should be >0")

    # if not Path(video_path).exists():
    #     raise FileExistsError(f"file {video_path} not found")
    video_path = str(video_path)
    # print(video_path)
    s3.download_file(bucket_name, video_path, video_path)

    print(video_path)
    # Create a streams to read the video and audio
    video_stream = VideoReader(video_path, stream="video")
    audio_stream = VideoReader(video_path, stream="audio")
    metadata = video_stream.get_metadata()
    metadata = {
        "fps": metadata["video"]["fps"][0],
        "duration": metadata["video"]["duration"][0],
        "sampling_rate": metadata["audio"]["framerate"][0],
    }
    print(metadata)
    chunk_generator = generate_chunks(
        video_stream=video_stream, audio_stream=audio_stream, seconds=seconds
    )
    os.remove(video_path)
    return chunk_generator, metadata


def generate_chunks(
    video_stream: VideoReader, audio_stream: VideoReader, seconds: float
):
    """
    Generator that yields consecutive chunks of 'seconds' duration
    from the given video_stream. Each yield is:
       (video_frames, audio_frames, timestamps, frame_ids)
    where:
      - video_frames: shape (T, C, H, W) as a torch.Tensor (if frames exist)
      - audio_frames: shape (N, ...) as a torch.Tensor (depends on your audio shape)
      - timestamps: 1D torch.Tensor of length T (frame timestamps in seconds)
      - frame_ids: 1D torch.Tensor of length T (global frame indices)
    """

    start = 0.0  # where our current chunk begins
    frame_counter = 0
    duration = video_stream.get_metadata()["video"]["duration"][0]
    # print("Start Chunk Generation")
    while True:
        if start > duration:
            break
        end = start + seconds  # the end time for this chunk
        # Gather video frames
        video_frames = []
        video_timestamps = []
        video_frame_ids = []
        for video_frame in itertools.takewhile(lambda x: x["pts"] < end, video_stream):
            video_frames.append(video_frame["data"])
            video_timestamps.append(video_frame["pts"])
            video_frame_ids.append(frame_counter)
            frame_counter += 1

        # Gather audio frames
        # NOTE: In practice, you'd handle audio carefully to match the chunk boundaries.
        audio_frames = []
        for audio_frame in itertools.takewhile(lambda x: x["pts"] < end, audio_stream):
            audio_frames.append(audio_frame["data"])

        # If we didn't get any video frames, we've likely reached the end
        if not video_frames:
            break

        # Convert lists to tensors
        video_frames = torch.stack(video_frames) if video_frames else torch.empty(0)
        audio_frames = torch.stack(audio_frames) if audio_frames else torch.empty(0)
        video_timestamps = torch.tensor(video_timestamps, dtype=torch.float32)
        video_frame_ids = torch.tensor(video_frame_ids, dtype=torch.int64)

        # Yield the current chunk
        yield video_frames, audio_frames, video_timestamps, video_frame_ids

        # Advance start time for the next chunk
        start = end
    # print("Chunk Generation Complete")


def downsample_video(
    video_frames: torch.Tensor,
    timestamps: torch.Tensor,
    frame_ids: torch.Tensor,
    factor: int,
) -> Tuple[torch.Tensor, torch.Tensor]:
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
    video_data = video_frames[::factor]
    timestamps = timestamps[::factor]
    frame_ids = frame_ids[::factor]
    return video_data, timestamps, frame_ids


def parse_bounding_boxes(bounding_boxes_path: str) -> Dict[int, Dict[int, Dict]]:
    """
    Read the file with the bounding boxes and return a nested dictionary with frame_id as first level keys,
    face_ids as second level keys and bounding boxes coordinates as values
    """
    # if not Path(bounding_boxes_path).exists():
    #     raise FileExistsError(f"Bounding boxes file {bounding_boxes_path} not found")
    # Read CSV with no headers (ensure all rows are treated as data)
    df = pd.read_csv(bounding_boxes_path, header=None)

    # Manually assign column names
    df.columns = [
        "frame_id",
        "face",
        "face_id",
        "x_start",
        "y_start",
        "x_end",
        "y_end",
        "fixed",
    ]

    # Clean data
    cols_to_keep = ["frame_id", "face_id", "x_start", "x_end", "y_start", "y_end"]
    df = df.loc[:, cols_to_keep]
    df = df.astype(int)
    coord_cols = cols_to_keep[-4:]
    df.loc[:, coord_cols] = np.where(
        df.loc[:, coord_cols] >= 0, df.loc[:, coord_cols], 0
    )
    valid_rows = (df.loc[:, "x_start"] <= df.loc[:, "x_end"]) & (
        df.loc[:, "y_start"] <= df.loc[:, "y_end"]
    )
    df = df.loc[valid_rows, :]

    # Indices
    frame_ids = df.loc[:, "frame_id"].unique()
    face_ids = df.loc[:, "face_id"].unique()

    # Return type
    bounding_boxes = {
        frame_id: {
            face_id: df.loc[
                (df["frame_id"] == frame_id) & (df["face_id"] == face_id), coord_cols
            ]
            .values.reshape(-1)
            .tolist()
            for face_id in face_ids
        }
        for frame_id in frame_ids
    }

    return bounding_boxes


def extract_faces(
    video_frames: torch.Tensor, frame_ids: torch.Tensor, bounding_boxes: List[Dict]
) -> Dict[int, List[torch.Tensor]]:
    """
    Extract the faces identified by the provided bounding boxes

    :param video_frames: Shape (Frames, C, H, W)
    :param frame_ids: Shape (Frames,)
    :param bounding_boxes: List of length Frames with dictionaries. Each dictionary has
    the face ID as a key and the bounding box as the value

    :return: A dictionary with face_ids as keys and a list of extracted bounding boxes as values
    """
    if bounding_boxes is None:
        raise ValueError("bounding_boxes cannot be empty")

    face_frames = {}
    for i, frame_id in enumerate(frame_ids):
        # video_frames and frame_ids are already downsampled so we cannot use
        # the frame_ids to index into video_frames by position
        frame = video_frames[i]
        if frame_id.item() not in bounding_boxes:
            raise ValueError(f"Frame ID {frame_id} not present in bounding_boxes")
        bounding_boxes_in_frame = bounding_boxes[frame_id.item()]
        for face_id in bounding_boxes_in_frame:
            # Crop face
            bounding_box = bounding_boxes_in_frame[face_id]
            if bounding_box:
                x_start, x_end, y_start, y_end = bounding_box
                cropped_face = frame[:, y_start:y_end, x_start:x_end]
            else:
                # Some people are not in every frame
                cropped_face = torch.zeros((3, 100, 100))
            # Store cropped face
            if face_id in face_frames:
                face_frames[face_id].append(cropped_face)
            else:
                face_frames[face_id] = [cropped_face]
    return face_frames


def transform_video(
    video_frames: List[torch.Tensor],
    height: int = 112,
    width: int = 112,
    scale: int = True,
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
    transformations = ImageTransforms.Compose(
        [
            ImageTransforms.Resize((height, width)),
            ImageTransforms.ToDtype(torch.float32, scale=scale),
            ImageTransforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]),
        ]
    )
    video_frames = torch.stack([transformations(frame) for frame in video_frames])
    return video_frames
