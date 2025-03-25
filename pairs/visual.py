import os
import pandas as pd
import numpy as np


def build_visual_pairs(dir, pairs_path):
    pairs_dir = os.path.join(dir, "pairs")
    os.makedirs(pairs_dir, exist_ok=True)

    pairs_df = pd.read_csv(pairs_path)

    curr_anchor_chunk = None
    curr_anchor_speaker = None
    for row in pairs_df.iterrows():
        pos_path = os.path.join(
            dir, f"Chunk_{row['pos_chunk_id']}", f"face_{row['speaker_id']}.npy"
        )
        neg_path = os.path.join(
            dir, f"Chunk_{row['neg_chunk_id']}", f"face_{row['neg_speaker_id']}.npy"
        )

        if (
            row["chunk_id"] != curr_anchor_chunk
            or row["speaker_id"] != curr_anchor_speaker
        ):
            curr_anchor_chunk = row["chunk_id"]
            curr_anchor_speaker = row["speaker_id"]
            anchor_path = os.path.join(
                dir, f"Chunk_{curr_anchor_chunk}", f"face_{curr_anchor_speaker}.npy"
            )
            anchor = np.load(anchor_path)

        pos = np.load(pos_path)
        neg = np.load(neg_path)

        pair = np.array(
            [
                anchor[row["frame_id"]],
                pos[row["pos_frame_id"]],
                neg[row["neg_frame_id"]],
            ]
        )

        outpath = os.path.join(
            pairs_dir, f"chunk{curr_anchor_chunk}_speaker{curr_anchor_speaker}_pair.npy"
        )
        np.save(outpath, pair)
