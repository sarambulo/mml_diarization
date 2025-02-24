import numpy as np
from pathlib import Path
from torchvision.io import VideoReader
import torch
import itertools

def parse_rttm(path) -> np.array:
    """
    :return records np.array: file_id, onset, duration, speaker_id
    """
    path = Path(path)
    if not path.exists():
        return None
    with open(path, "r") as file:
        records = file.readlines()
    records = [line.split(" ") for line in records]
    records = [(line[1], line[3], line[4], line[7]) for line in records]
    data = {}
    for record in records:
        file_id = record[0]
        if file_id not in data:
            data[file_id] = [record[1:]]
        else:
            data[file_id].append(record[1:])
    data = {file_id: np.array(data[file_id]) for file_id in data}
    return data


def read_video(path: str, start_sec=0, end_sec=None, return_video=True, return_audio=True):
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
        frames = []
        # Read from start to end
        for frame in itertools.takewhile(
            lambda x: x["pts"] <= end_sec, stream.seek(start_sec)
        ):
            frames.append(frame["data"])
            video_timestamps.append(frame["pts"])
        if len(frames) > 0:
            video_frames = torch.stack(frames, 0)

    audio_frames = torch.empty(0)
    audio_timestamps = []
    if return_audio:
        stream.set_current_stream("audio")
        frames = []
        for frame in itertools.takewhile(
            lambda x: x["pts"] <= end_sec, stream.seek(start_sec)
        ):
            frames.append(frame["data"])
            video_timestamps.append(frame["pts"])
        if len(frames) > 0:
            audio_frames = torch.cat(frames, 0)
    return (
        video_frames,
        audio_frames,
        (video_timestamps, audio_timestamps),
        stream.get_metadata(),
    )
