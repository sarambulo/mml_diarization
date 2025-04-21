from visual import load_video_frames, build_visual_pair
from audio import load_audio_frames, build_audio_pair
from config import *
import os
import pandas as pd
from utils import (
    upload_npz,
    visualize_visual_triplet,
    visualize_audio_triplet,
    plot_images_from_array,
)
from tqdm import tqdm
import numpy as np


def build_combined_pairs(
    bucket,
    vid,
    pairs_path,
    face_buf,
    lip_buf,
    audio_buf,
    meta_buf,
    batch_size,
    outpath,
    audio_type,
    batch_idx,
):
    pairs_dir = os.path.join("preprocessed", vid, "visual_pairs")
    pairs_df = pd.read_csv(pairs_path)

    face_frames = load_video_frames(bucket, vid, visual_type="face")
    lip_frames = load_video_frames(bucket, vid, visual_type="lip")
    audio_frames = load_audio_frames(bucket, vid, audio_type)

    # for chunk_id in face_frames:
    #     for speaker_id in face_frames[chunk_id]:
    #         plot_images_from_array(face_frames[chunk_id][speaker_id], f"eda/chunks_new/{vid}_Chunk{chunk_id}_Speaker{speaker_id}_faces.png")
    #         plot_images_from_array(lip_frames[chunk_id][speaker_id], f"eda/chunks_new/{vid}_Chunk{chunk_id}_Speaker{speaker_id}_lips.png")

    for i, row in tqdm(pairs_df.iterrows(), total=len(pairs_df), desc="Building Pairs"):
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

        if video_flag == 1 and audio_flag == 1 and frame_id % 4 == 0:
            # print(
            #     f"i: {i}, "
            #     f"chunk_id: {chunk_id}, "
            #     f"frame_id: {frame_id}, "
            #     f"speaker_id: {speaker_id}, "
            #     f"pos_chunk_id: {pos_chunk_id}, "
            #     f"pos_frame_id: {pos_frame_id}, "
            #     f"neg_chunk_id: {neg_chunk_id}, "
            #     f"neg_frame_id: {neg_frame_id}, "
            #     f"neg_speaker_id: {neg_speaker_id}"
            # )
            face_data = build_visual_pair(
                face_frames,
                chunk_id,
                frame_id,
                speaker_id,
                pos_chunk_id,
                pos_frame_id,
                neg_chunk_id,
                neg_frame_id,
                neg_speaker_id,
            )
            # print("Face Shape:", face_data.shape)
            # visualize_visual_triplet(face_data, "eda", f"face")

            lip_data = build_visual_pair(
                lip_frames,
                chunk_id,
                frame_id,
                speaker_id,
                pos_chunk_id,
                pos_frame_id,
                neg_chunk_id,
                neg_frame_id,
                neg_speaker_id,
            )
            # print("Lip Shape:", lip_data.shape)
            # visualize_visual_triplet(lip_data, "eda", f"lip")

            audio_data = build_audio_pair(
                audio_frames,
                chunk_id,
                frame_id,
                pos_chunk_id,
                pos_frame_id,
                neg_chunk_id,
                neg_frame_id,
            )
            # print("Audio Shape:", audio_data.shape)
            # visualize_audio_triplet(audio_data, "eda")

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

            face_buf.append(face_data)
            lip_buf.append(lip_data)
            audio_buf.append(audio_data)
            meta_buf.append(metadata)

            if len(face_buf) >= batch_size:
                out_key = os.path.join(outpath, f"triplet_batch_{batch_idx:05d}.npz")
                upload_npz(
                    bucket,
                    out_key,
                    np.stack(face_buf),
                    np.stack(lip_buf),
                    np.stack(audio_buf),
                    np.array(meta_buf),
                )
                print(f"✅ Uploaded {out_key}")
                face_buf.clear()
                lip_buf.clear()
                audio_buf.clear()
                meta_buf.clear()
                batch_idx += 1

    return batch_idx
