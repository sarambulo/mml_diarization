from visual import load_video_frames, build_visual_pair
from audio import load_audio_frames, build_audio_pair
from config import *
import os
import pandas as pd
from utils import upload_npz


def build_combined_pairs(bucket, vid, pairs_path, video_buf, audio_buf, meta_buf, batch_size, outpath, audio_type):
    pairs_dir = os.path.join("preprocessed", vid, "visual_pairs")
    pairs_df = pd.read_csv(pairs_path)

    video_frames = load_video_frames(bucket, vid)
    audio_frames = load_audio_frames(bucket, vid, audio_type)

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

        
        if video_flag==1 and audio_flag==1 and frame_id % VIDEO_FPS == 0:
            visual_data = build_visual_pair(video_frames, chunk_id, frame_id, speaker_id, pos_chunk_id, pos_frame_id, neg_chunk_id, neg_frame_id, neg_speaker_id)
            audio_data = build_audio_pair(audio_frames, chunk_id, frame_id, pos_chunk_id, pos_frame_id, neg_chunk_id, neg_frame_id)
            metadata = {
                "video_id": vid,
                "chunk_id": chunk_id,
                "frame_id": frame_id,
                "speaker_id": speaker_id,
                "is_speaking": is_speaking,
                "pos_chunk_id": pos_chunk_id,
                "pos_frame_id": pos_frame_id,
                "neg_chunk_id": neg_chunk_id,
                "neg_speaker_id": neg_speaker_id,
                "neg_frame_id": neg_frame_id,
            }

            video_buf.append(visual_data)
            audio_buf.append(audio_data)
            meta_buf.append(metadata)

            if len(v_buf) >= BATCH_SIZE:
                out_key = os.path.join(OUTPUT_PREFIX, f"triplet_batch_{batch_idx:05d}.npz")
                upload_npz(
                    bucket, out_key, np.stack(video_buf), np.stack(audio_buf), np.array(meta_buf)
                )
                print(f"✅ Uploaded {out_key}")
                v_buf.clear()
                a_buf.clear()
                l_buf.clear()
                batch_idx += 1


