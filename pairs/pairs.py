from utils import *
from config import *
from collections import defaultdict
import numpy as np
import pandas as pd
import random
import os


def create_lookups(df):
    # speaker -> list of (chunk_id, frame_id, is_speaking) tuples
    speaker_frames = defaultdict(list)
    # chunk id -> list of (speaker_id, frame_id, is_speaking) tuples
    chunk_speakers = defaultdict(list)

    for _, row in df.iterrows():
        spk = row["speaker_id"]
        ch = row["chunk_id"]
        fr = row["frame_id"]
        is_speaking = row["is_speaking"]
        speaker_frames[spk].append((ch, fr, is_speaking))
        chunk_speakers[ch].append((spk, fr, is_speaking))

    return speaker_frames, chunk_speakers


def get_positive_pair(current_chunk, current_frame, anchor_frames):
    diff_chunk_frames = [(ch, fr) for ch, fr, _ in anchor_frames if ch != current_chunk]
    if diff_chunk_frames:
        pos_chunk_id, pos_frame_id = random.choice(diff_chunk_frames)
    else:
        same_chunk_frames = [
            (ch, fr)
            for ch, fr, _ in anchor_frames
            if ch == current_chunk and fr != current_frame
        ]
        if same_chunk_frames:
            pos_chunk_id, pos_frame_id = random.choice(same_chunk_frames)
        else:
            pos_chunk_id, pos_frame_id = None, None

    return pos_chunk_id, pos_frame_id


def get_speaking_negative_pair(
    anchor_speaker_id, current_chunk, chunk_speakers, sample_diff_speaker_prob
):
    is_speaking_val = 1 if np.random.uniform(0, 1) <= sample_diff_speaker_prob else 0
    speakers_by_chunk = chunk_speakers[current_chunk]

    other_speaker_frames = [
        (spk, fr)
        for spk, fr, is_speaking in speakers_by_chunk
        if spk != anchor_speaker_id and is_speaking == is_speaking_val
    ]
    if other_speaker_frames:  # check if found in same chunk
        neg_face_id, neg_frame_id = random.choice(other_speaker_frames)
        return current_chunk, neg_face_id, neg_frame_id
    else:
        remaining_chunk_ids = [
            ch for ch in chunk_speakers.keys() if ch != current_chunk
        ]
        for chunk_id in remaining_chunk_ids:
            speakers_by_chunk = chunk_speakers[chunk_id]
            other_speaker_frames = [
                (spk, fr, is_speaking)
                for spk, fr, is_speaking in speakers_by_chunk
                if spk != anchor_speaker_id and is_speaking == is_speaking_val
            ]
            if other_speaker_frames:
                neg_face_id, neg_frame_id, _ = random.choice(other_speaker_frames)
                return chunk_id, neg_face_id, neg_frame_id
    return None, None, None


def get_anchor_speaking_frames(speakers_by_chunk, anchor_id):
    same_speaker_frames = [
        fr
        for spk, fr, is_speaking in speakers_by_chunk
        if spk == anchor_id and is_speaking == 1
    ]
    return same_speaker_frames


def get_non_speaking_negative_pair(anchor_speaker_id, current_chunk, chunk_speakers):
    speakers_by_chunk = chunk_speakers[current_chunk]
    anchor_speaking_frames_ids = get_anchor_speaking_frames(
        speakers_by_chunk, anchor_speaker_id
    )

    other_speaker_frames = [
        (spk, fr)
        for spk, fr, _ in speakers_by_chunk
        if spk != anchor_speaker_id
        and fr
        in anchor_speaking_frames_ids  # check that speaker is not anchor (for correct face) and filter for anchor speaking frames (for correct audio)
    ]
    if other_speaker_frames:  # check if found in same chunk
        neg_face_id, neg_frame_id = random.choice(other_speaker_frames)
        return current_chunk, neg_face_id, neg_frame_id
    else:
        remaining_chunk_ids = [
            ch for ch in chunk_speakers.keys() if ch != current_chunk
        ]
        for chunk_id in remaining_chunk_ids:
            speakers_by_chunk = chunk_speakers[chunk_id]
            anchor_speaking_frames_ids = get_anchor_speaking_frames(
                speakers_by_chunk, anchor_speaker_id
            )
            other_speaker_frames = [
                (spk, fr)
                for spk, fr, _ in speakers_by_chunk
                if spk != anchor_speaker_id and fr in anchor_speaking_frames_ids
            ]
            if other_speaker_frames:
                neg_face_id, neg_frame_id = random.choice(other_speaker_frames)
                return chunk_id, neg_face_id, neg_frame_id
    return None


# INPUT: is_speaking.csv
# output: pairs.csv with 7 new columns:  PosChunkID, PosFrameID, NegChunkID, NegFrameID, NegSpeakerID, Video_Flag, Audio_Flag
def choose_and_save_pairs_for_video(input_file_path, output_file_path):
    df = pd.read_csv(
        input_file_path,
        header=0,
        # face_id,frame_id,is_speaking,video_id,chunk_id
        names=["speaker_id", "frame_id", "is_speaking", "video_id", "chunk_id"],
    )
    df = df.loc[:, df.columns != "timestamp"].astype(int)

    speaker_frames, chunk_speakers = create_lookups(df)
    # print(speaker_frames)
    pair_info = {
        "chunk_id": [],
        "speaker_id": [],
        "is_speaking": [],
        "frame_id": [],
        "pos_chunk_id": [],
        "pos_frame_id": [],
        "neg_chunk_id": [],
        "neg_speaker_id": [],
        "neg_frame_id": [],
    }

    for anchor_speaker in speaker_frames:  # iterate through speakers
        anchor_frames = speaker_frames[anchor_speaker]  # get frames for current speaker
        # split anchor_frames into speaking/not speaking
        anchor_speaking_frames = [
            (ch, fr, is_speaking)
            for (ch, fr, is_speaking) in anchor_frames
            if is_speaking == 1
        ]
        anchor_non_speaking_frames = [
            (ch, fr, is_speaking)
            for (ch, fr, is_speaking) in anchor_frames
            if is_speaking == 0
        ]
        # exit()
        print(f"producing pairs for speaker {anchor_speaker}...")
        for current_chunk, current_frame, _ in anchor_speaking_frames:
            # print(
            #     f"Pairing chunk {current_chunk} frame {current_frame} speaker {anchor_speaker}"
            # )
            pos_chunk, pos_frame = get_positive_pair(
                current_chunk, current_frame, anchor_speaking_frames
            )  # get positive pair from speaking_frames only
            neg_chunk, neg_speaker, neg_frame = get_speaking_negative_pair(
                anchor_speaker,
                current_chunk,
                chunk_speakers,
                AUDIO_SAMPLE_DIFFERENT_SPEAKER_PROB,
            )  # get negative pair for different face, speaking vs not speaking depends on probability
            if pos_chunk and neg_chunk:
                pair_info["chunk_id"].append(current_chunk)
                pair_info["speaker_id"].append(anchor_speaker)
                pair_info["is_speaking"].append(1)  # always speaking
                pair_info["frame_id"].append(current_frame)
                pair_info["pos_chunk_id"].append(pos_chunk)
                pair_info["pos_frame_id"].append(pos_frame)
                pair_info["neg_chunk_id"].append(neg_chunk)
                pair_info["neg_speaker_id"].append(neg_speaker)
                pair_info["neg_frame_id"].append(neg_frame)
            else:
                print(
                    f"skipping anchor {anchor_speaker} chunk {current_chunk} frame {current_frame} from speaking frames"
                )

        # non-speaking
        for current_chunk, current_frame, is_speaking in anchor_non_speaking_frames:
            # print(f"Non-speaking pairing chunk {current_chunk} frame {current_frame}")
            # get positive pair from any anchor frame
            pos_chunk, pos_frame = get_positive_pair(
                current_chunk, current_frame, anchor_non_speaking_frames
            )
            neg_chunk, neg_speaker, neg_frame = get_non_speaking_negative_pair(
                anchor_speaker, current_chunk, chunk_speakers
            )  # get negative pair without speaking restriction
            if pos_chunk and neg_chunk:
                pair_info["chunk_id"].append(current_chunk)
                pair_info["speaker_id"].append(anchor_speaker)
                pair_info["is_speaking"].append(0)
                pair_info["frame_id"].append(current_frame)
                pair_info["pos_chunk_id"].append(pos_chunk)
                pair_info["pos_frame_id"].append(pos_frame)
                pair_info["neg_chunk_id"].append(neg_chunk)
                pair_info["neg_speaker_id"].append(neg_speaker)
                pair_info["neg_frame_id"].append(neg_frame)
            # else:
            #     print(
            #         f"No pair found: Pos Found = {pos_chunk!=None} Neg Found = {neg_chunk!=None}"
            #     )
    save_pair_info(pair_info, output_file_path)
    return pair_info


def save_pair_info(pair_info_dict, output_file_path):
    pairs_df = pd.DataFrame(pair_info_dict).astype(int)
    # if os.path.exists(output_file_path):
    #     current_pairs = pd.read_csv(output_file_path)
    #     pairs_df = current_pairs + pairs_df
    # # print(pairs_df)
    pairs_df.to_csv(output_file_path, index=False)


# testing
# choose_and_save_pairs_for_video("../preprocessed/00001/is_speaking.csv", "pairs_test.csv")
