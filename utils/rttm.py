import os
import pandas as pd
from .data import rttm_to_annotations
from pyannote.metrics.diarization import GreedyDiarizationErrorRate, JaccardErrorRate
from typing import Dict
from pyannote.core import Annotation, Segment
from pathlib import Path

FRAME_DURATION = 0.25  # seconds per frame


def get_rttm_labels(
    rttm_path: str,
    timestamps: list,
    speaker_ids: list,
    video_id
):
    """
    Parse an RTTM file in the format:
        SPEAKER file_id chan start dur <NA> <NA> speaker_id <NA> <NA>
    and determine whether each face_id (speaker_id) is speaking at each
    frame timestamp. Returns a DataFrame of:

        face_id, frame_id, is_speaking (boolean)

    The resulting DataFrame is automatically saved to CSV at `csv_path`.

    Parameters
    ----------
    rttm_path : str
        Path to the RTTM file.
    timestamps : list of float
        A list of timestamps (in seconds) for each video frame in this chunk.
    speaker_ids : list of str
        A list of speaker/face IDs that appear in this chunk (keys in `faces`).
        These should match the RTTM's speaker_id in column 7 (e.g., "2", "0", "1", etc.).
    csv_path : str
        The path where the resulting CSV should be written. 
        Example: "path/to/chunk_x/is_speaking.csv"

    Returns
    -------
    pd.DataFrame
        DataFrame with columns ["face_id", "frame_id", "is_speaking"].
    """

    speakers= [str(x) for x in speaker_ids]
    
    intervals = {}
    with open(rttm_path, "r") as f:
        for line in f:
            parts = line.strip().split()
            # Skip lines that don't start with 'SPEAKER' or aren't long enough
            if parts[0] != "SPEAKER":
                continue
            if parts[1]!=video_id:
                continue
            # parts layout:
            #  0: SPEAKER
            #  1: file_id
            #  2: chan
            #  3: start
            #  4: dur
            #  5: <NA>
            #  6: <NA>
            #  7: speaker_id
            #  8: <NA>
            #  9: <NA>
            rttm_speaker_id = parts[7]
            start_time = float(parts[3])
            # print(start_time)
            duration = float(parts[4])
            end_time = start_time + duration
            # print(end_time)

            if rttm_speaker_id not in intervals:
                intervals[rttm_speaker_id] = []
            intervals[rttm_speaker_id].append((start_time, end_time))

    # 2) For each frame time, determine if each speaker_id is speaking
    rows = []
    for frame_id, t in enumerate(timestamps):
        for face_id in speakers:
            # print(speakers)
            speaking_flag = False
            if face_id in intervals:
                # print("REACH")
                for (start, end) in intervals[face_id]:
                    if start <= t < end:
                        speaking_flag = True
                        break
            rows.append((face_id, frame_id, speaking_flag))

    # 3) Convert to DataFrame
    df = pd.DataFrame(rows, columns=["face_id", "frame_id", "is_speaking"])
    # print(df)

    return df



def greedy_speaker_matching(reference_rttm_path, predicted_rttm_path) -> Dict[str, str]:
    """
    Returns: Dictionary with predicted ID as keys and reference ID as
    values
    """
    greedyDER = GreedyDiarizationErrorRate()
    reference_annotation = rttm_to_annotations(reference_rttm_path)
    reference_annotation = list(reference_annotation.values())[0]  # Extract only value
    predicted_annotation = rttm_to_annotations(predicted_rttm_path)
    predicted_annotation = list(predicted_annotation.values())[0]  # Extract only value
    mapping = greedyDER.greedy_mapping(
        reference=reference_annotation, hypothesis=predicted_annotation
    )
    return mapping


def rttm_to_annotations(path) -> Dict[str, Annotation]:
    """
    Returns a dictionary with video ID as keys and Annotation as values
    """
    d = load_rttm_by_video(path)
    annotations = {}
    for videoId in d:
        ann = Annotation()
        for seg in d[videoId]:
            ann[Segment(start=seg["startTime"], end=seg["endTime"])] = seg["speakerId"]
        annotations[videoId] = ann
    return annotations


def load_rttm_by_video(path):
    data = {}
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            fields = line.strip().split()
            if len(fields) == 10 and fields[0] == "SPEAKER":
                file_id, start, duration, speaker = (
                    fields[1],
                    float(fields[3]),
                    float(fields[4]),
                    fields[7],
                )
                if file_id not in data:
                    data[file_id] = []
                data[file_id].append(
                    {
                        "speakerId": speaker,
                        "startTime": start,
                        "endTime": start + duration,
                        "duration": duration,
                    }
                )
    return data


def csv_to_rttm(csv_path: str, output_rttm_path: str) -> None:
    if not Path(csv_path).exists():
        raise FileExistsError(f"File {csv_path} does not exist")
    data = pd.read_csv(csv_path)
    data = data.rename(columns={
        'is_speaking_pred': 'Speaking',
        'speaker_id': 'Speaker ID',
        'video_id': 'Video ID',
        'frame_idx': 'Frame Offset',
        'chunk_id': 'Chunk ID',
    })
    data['Timestamp'] = (data['Frame Offset'] + data['Chunk ID'] * 5) * FRAME_DURATION
    video_id = data['Video ID'][0]
    data = data.sort_values(['Speaker ID', 'Timestamp'])
    data['Interval Flag'] = (data['Speaking'] == 1) & (data.groupby('Speaker ID')['Speaking'].shift(1, fill_value=0) == 0)
    data['Interval ID'] = data.groupby('Speaker ID')['Interval Flag'].cumsum()
    data = data[data['Speaking'] == 1]
    data = data.groupby(['Speaker ID', 'Interval ID']).agg(**{
        'Start': ('Timestamp', lambda x: x.min()),
        'End': ('Timestamp', lambda x: x.max() + FRAME_DURATION),
    })
    data = data.reset_index()
    data['Duration'] = data['End'] - data['Start']
    speaker_ids = data['Speaker ID'].astype(int)
    start = data['Start'].astype(float)
    duration = data['Duration'].astype(float)

    # Save to RTTM file
    nrows = data.shape[0]
    with open(output_rttm_path, "w") as f:
        for i in range(nrows):
            f.write(f"SPEAKER {int(video_id):05d} 0 {start[i]:.2f} {duration[i]:.2f} <NA> <NA> {speaker_ids[i]} <NA> <NA>\n")

def csvs_to_rttms(input_dir: str, output_dir: str):
    """
    Traverse the preprocessed directory structure and convert all is_speaking.csv files to RTTM lines.
    """
    input_dir = Path(input_dir)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    for csv_path in sorted(input_dir.iterdir()):
        if csv_path.suffix == '.csv':
            video_id = csv_path.stem
            output_rttm_path = output_dir / f'{video_id}.rttm'
            csv_to_rttm(
                csv_path=str(csv_path),
                output_rttm_path=output_rttm_path
            )
