import os
import pandas as pd
import numpy as np
from config import VIDEO_FPS
from utils import visualize_visual_triplet, s3_load_numpy, s3_save_numpy, paginator
import io
import re



def load_video_frames(bucket_name, vid):
    prefix = os.path.join("preprocessed", vid)
    face_pattern = re.compile(fr'^{prefix}/Chunk_(\d+)/face_(\d+)\.npy$')
    temp = {}
    for page in paginator.paginate(Bucket=bucket_name, Prefix=prefix):
        for obj in page.get('Contents', []):
            key = obj['Key']
            m = face_pattern.match(key)
            if not m:
                continue
            chunk_id = int(m.group(1))
            speaker_id = int(m.group(2))

            arr = s3_load_numpy(bucket_name, key)
            print(arr.shape)
            temp.setdefault(chunk_id, {})[speaker_id] = arr

    num_chunks = max(temp.keys()) + 1
    num_faces = max(max(faces.keys()) for faces in temp.values()) + 1
    print(num_faces)
    
    # Initialize list to collect per-chunk arrays
    all_chunks = []
    
    for chunk in range(num_chunks):
        faces = temp.get(chunk, {})
        # Collect faces in order
        print(faces)
        face_list = [faces[i] for i in faces]
        # Stack faces along new axis 0: shape (num_faces, ...)
        stacked_faces = np.stack(face_list, axis=0)
        print(stacked_faces.shape)
        all_chunks.append(stacked_faces)
    
    # Final stacking across chunks: shape (num_chunks, num_faces, ...)
    result = np.stack(all_chunks, axis=0)
    
    # Now you can access elements like result[chunk_idx, face_idx, ...]
    print(result.shape)
    return result

def build_visual_pair(video_frames, chunk_id, frame_id, speaker_id, pos_chunk_id, pos_frame_id, neg_chunk_id, neg_frame_id, neg_speaker_id):
    pair = np.array(
        [
            video_frames[chunk_id, speaker_id, frame_id],
            video_frames[pos_chunk_id, speaker_id, pos_frame_id],
            video_frames[neg_chunk_id, neg_speaker_id, neg_frame_id],
        ]
    )

    return pair
    

def build_visual_pairs(bucket, vid, pairs_path, visualize=False):
    pairs_dir = os.path.join("preprocessed", vid, "visual_pairs")

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
        if video_flag == 1 and frame_id % VIDEO_FPS == 0:
            pos_path = os.path.join(
                "preprocessed", vid, f"Chunk_{pos_chunk_id}", f"face_{speaker_id}.npy"
            )
            neg_path = os.path.join(
                "preprocessed",
                vid,
                f"Chunk_{neg_chunk_id}",
                f"face_{neg_speaker_id}.npy",
            )

            if chunk_id != curr_anchor_chunk or speaker_id != curr_anchor_speaker:
                curr_anchor_chunk = chunk_id
                curr_anchor_speaker = speaker_id
                anchor_path = os.path.join(
                    "preprocessed",
                    vid,
                    f"Chunk_{curr_anchor_chunk}",
                    f"face_{curr_anchor_speaker}.npy",
                )
                anchor = s3_load_numpy(bucket, anchor_path)

            pos = s3_load_numpy(bucket, pos_path)
            neg = s3_load_numpy(bucket, neg_path)

            pair = np.array(
                [
                    anchor[frame_id],
                    pos[pos_frame_id],
                    neg[neg_frame_id],
                ]
            )

            if visualize:
                visualize_visual_triplet(
                    images=pair,
                    dir=pairs_dir,
                    name=f"chunk{curr_anchor_chunk}_speaker{curr_anchor_speaker}_frame{frame_id}_pair",
                )

            outfile = os.path.join(
                pairs_dir,
                f"chunk{curr_anchor_chunk}_speaker{curr_anchor_speaker}_frame{frame_id}_pair.npy",
            )

            # np.save(outpath, pair)
            s3_save_numpy(pair, bucket, outfile)
