import os
import numpy as np
import pandas as pd
from utils import visualize_mel_spectrogram
from config import VIDEO_FPS


def build_audio_pairs(dir, pairs_path, audio_type, visualize=False):
    pairs_dir = os.path.join(dir, f"{audio_type}_audio_pairs")
    os.makedirs(pairs_dir, exist_ok=True)

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
            pos_path = os.path.join(dir, f"Chunk_{pos_chunk_id}", f"{audio_type}.npy")
            neg_path = os.path.join(dir, f"Chunk_{neg_chunk_id}", f"{audio_type}.npy")

            if chunk_id != curr_anchor_chunk:
                curr_anchor_chunk = chunk_id
                anchor_path = os.path.join(
                    dir, f"Chunk_{curr_anchor_chunk}", f"{audio_type}.npy"
                )
                anchor = np.load(anchor_path)
                if visualize:
                    visualize_mel_spectrogram(
                        anchor,
                        os.path.join(dir, f"Chunk_{curr_anchor_chunk}"),
                        saveas=f"{audio_type}.png",
                    )

            pos = np.load(pos_path)
            neg = np.load(neg_path)

            pair = np.array(
                [
                    anchor[:, :, frame_id],
                    pos[:, :, pos_frame_id],
                    neg[:, :, neg_frame_id],
                ]
            )

            outpath = os.path.join(
                pairs_dir,
                f"chunk{curr_anchor_chunk}_frame{frame_id}_pair.npy",
            )
            np.save(outpath, pair)
