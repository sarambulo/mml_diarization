import numpy as np
from pathlib import Path
from torchvision.io import VideoReader
import torch
import itertools
import re

def parse_rttm(path) -> np.array:
    """
    :return records np.array: file_id, onset, duration, speaker_id
    """
    path = Path(path)
    if not path.exists():
        raise FileExistsError(f'file {path} not found')
    with open(path, "r") as file:
        records = file.readlines()
    records = [line.split(" ") for line in records]
    records = [(line[1], line[3], line[4], line[7]) for line in records if line[7].isnumeric()]
    data = {}
    for record in records:
        file_id = record[0]
        if file_id not in data:
            data[file_id] = [record[1:]]
        else:
            data[file_id].append(record[1:])
    data = {file_id: (np.array(data[file_id])[:, :2].astype(float), np.array(data[file_id])[:, 2].astype(int)) for file_id in data} #tuple ([start time, time spoken], speaker_id)
    return data


def read_video(path: str, start_sec=0, end_sec=None, max_frames=None, return_video=True, return_audio=True):
    """
    :return data: (
        video_frames,
        audio_frames,
        (video_timestamps, audio_timestamps),
        metadata
    )
    """
    # Check inputs
    if end_sec is None:
        end_sec = float("inf")
    if end_sec < start_sec:
        raise ValueError(
            "end time should be larger than start time, got "
            "start time={} and end time={}".format(start_sec, end_sec)
        )
    if not Path(path).exists:
        raise FileExistsError(
            f"file {path} not found"
        )
    path = str(path)

    # Create a stream to read the video
    stream_type = "video"
    stream = VideoReader(path, stream_type)

    # Store the video data
    video_frames = torch.empty(0)
    video_timestamps = []
    if return_video:
        stream.set_current_stream("video")
        if max_frames is not None:
            fps = stream.get_metadata()['video']['fps'][0]
            end_sec = min(end_sec, max_frames / fps)
        frames = []
        # Read from start to end
        for frame in itertools.takewhile(
            lambda x: x["pts"] <= end_sec, stream.seek(start_sec)
        ):
            frames.append(frame["data"])
            video_timestamps.append(frame["pts"])
        if len(frames) > 0:
            video_frames = torch.stack(frames, 0)
    video_timestamps = torch.tensor(video_timestamps)

    audio_frames = torch.empty(0)
    audio_timestamps = []
    if return_audio:
        stream.set_current_stream("audio")
        frames = []
        for frame in itertools.takewhile(
            lambda x: x["pts"] <= end_sec, stream.seek(start_sec)
        ):
            frames.append(frame["data"])
            audio_timestamps.append(frame["pts"])
        if len(frames) > 0:
            audio_frames = torch.cat(frames, 0)
    audio_timestamps = torch.tensor(audio_timestamps)
    return (
        video_frames,
        video_timestamps,
        audio_frames,
        audio_timestamps,
        stream.get_metadata(),
    )

def get_streams(path):
    path = str(path)
    video_stream = VideoReader(path, 'video')
    audio_stream = VideoReader(path, 'audio') 
    metadata = video_stream.get_metadata()
    return video_stream, audio_stream, metadata
