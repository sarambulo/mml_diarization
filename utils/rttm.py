import os
import pandas as pd
from .data import rttm_to_annotations
from pyannote.metrics.diarization import GreedyDiarizationErrorRate
from typing import Dict
from collections import defaultdict

def get_rttm_labels(
    rttm_path: str,
    timestamps: list,
    speaker_ids: list
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
    reference_annotation = list(reference_annotation.values())[0] # Extract only value
    predicted_annotation = rttm_to_annotations(predicted_rttm_path)
    predicted_annotation = list(predicted_annotation.values())[0] # Extract only value
    mapping = greedyDER.greedy_mapping(
        reference=reference_annotation, hypothesis=predicted_annotation
    )
    return mapping