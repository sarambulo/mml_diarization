import os
import pandas as pd
import numpy as np


def build_visual_pairs(dir, pairs_path):
    pairs_dir = os.path.join(dir, "visual_pairs")
    os.makedirs(pairs_dir, exist_ok=True)

    pairs_df = pd.read_csv(pairs_path)

    curr_anchor_chunk = None
    curr_anchor_speaker = None
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
        if video_flag == 1:
            pos_path = os.path.join(
                dir, f"Chunk_{pos_chunk_id}", f"face_{speaker_id}.npy"
            )
            neg_path = os.path.join(
                dir, f"Chunk_{neg_chunk_id}", f"face_{neg_speaker_id}.npy"
            )

            if chunk_id != curr_anchor_chunk or speaker_id != curr_anchor_speaker:
                curr_anchor_chunk = chunk_id
                curr_anchor_speaker = speaker_id
                anchor_path = os.path.join(
                    dir, f"Chunk_{curr_anchor_chunk}", f"face_{curr_anchor_speaker}.npy"
                )
                anchor = np.load(anchor_path)

            pos = np.load(pos_path)
            neg = np.load(neg_path)

            pair = np.array(
                [
                    anchor[frame_id],
                    pos[pos_frame_id],
                    neg[neg_frame_id],
                ]
            )

            outpath = os.path.join(
                pairs_dir,
                f"chunk{curr_anchor_chunk}_speaker{curr_anchor_speaker}_frame{frame_id}_pair.npy",
            )
            np.save(outpath, pair)
