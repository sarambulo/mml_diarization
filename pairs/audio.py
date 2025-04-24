import os
import numpy as np
import pandas as pd
from utils import visualize_mel_spectrogram, s3_load_numpy, s3_save_numpy, paginator
from config import VIDEO_FPS
import re


def load_audio_frames(bucket_name, vid, audio_type):
    prefix = os.path.join("preprocessed_2", vid)
    audio_pattern = re.compile(rf"^{prefix}/Chunk_(\d+)/{audio_type}.npy$")
    temp = {}
    for page in paginator.paginate(Bucket=bucket_name, Prefix=prefix):
        for obj in page.get("Contents", []):
            key = obj["Key"]
            m = audio_pattern.match(key)
            if not m:
                continue
            chunk_id = int(m.group(1))
            arr = s3_load_numpy(bucket_name, key)
            temp[chunk_id] = arr
    return temp


def build_audio_pair(
    audio_frames,
    chunk_id,
    frame_id,
    pos_chunk_id,
    pos_frame_id,
    neg_chunk_id,
    neg_frame_id,
):
    pair = np.array(
        [
            audio_frames[chunk_id][:, :, frame_id],
            audio_frames[pos_chunk_id][:, :, pos_frame_id],
            audio_frames[neg_chunk_id][:, :, neg_frame_id],
        ]
    )
    return pair


def build_audio_pairs(bucket, vid, pairs_path, audio_type, visualize=False):
    pairs_dir = os.path.join("preprocessed_2", vid, f"{audio_type}_audio_pairs")

    pairs_df = pd.read_csv(pairs_path)

    curr_anchor_chunk = None
    for i, row in pairs_df.iterrows():
        (
            chunk_id,
            speaker_id,
            is_speaking,
            frame_id,
            pos_chunk_id,
            pos_frame_id,
            neg_chunk_id,
            neg_speaker_id,
            neg_frame_id,
            video_flag,
            audio_flag,
        ) = row.values
        if audio_flag == 1 and frame_id % VIDEO_FPS == 0:
            pos_path = os.path.join(
                "preprocessed_2", vid, f"Chunk_{pos_chunk_id}", f"{audio_type}.npy"
            )
            neg_path = os.path.join(
                "preprocessed_2", vid, f"Chunk_{neg_chunk_id}", f"{audio_type}.npy"
            )

            if chunk_id != curr_anchor_chunk:
                curr_anchor_chunk = chunk_id
                anchor_path = os.path.join(
                    "preprocessed_2",
                    vid,
                    f"Chunk_{curr_anchor_chunk}",
                    f"{audio_type}.npy",
                )
                anchor = s3_load_numpy(bucket, anchor_path)
                if visualize:
                    visualize_mel_spectrogram(
                        anchor,
                        os.path.join(dir, f"Chunk_{curr_anchor_chunk}"),
                        saveas=f"{audio_type}.png",
                    )

            pos = s3_load_numpy(bucket, pos_path)
            neg = s3_load_numpy(bucket, neg_path)

            pair = np.array(
                [
                    anchor[:, :, frame_id],
                    pos[:, :, pos_frame_id],
                    neg[:, :, neg_frame_id],
                ]
            )

            outfile = os.path.join(
                pairs_dir, f"chunk{curr_anchor_chunk}_frame{frame_id}_pair.npy"
            )

            # np.save(outpath, pair)
            s3_save_numpy(pair, bucket, outfile)
