import boto3
import io
import numpy as np
import pandas as pd
from tqdm import tqdm
import os

s3 = boto3.client("s3")

# --- CONFIG ---
BUCKET = "mmml-proj"
PREFIX = "preprocessed"
BATCH_SIZE = 1000
OUTPUT_PREFIX = "test_batched_triplets"  # Where to save output .npz files

def list_videos(bucket, prefix):
    # paginator = s3.get_paginator("list_objects_v2")
    # pages = paginator.paginate(Bucket=bucket, Prefix=prefix, Delimiter='/')
    # return [p['Prefix'] for page in pages for p in page.get('CommonPrefixes', [])]
    return [str(video_id).zfill(5) for video_id in range(0, 2251)]

def read_s3_numpy(bucket, key):
    response = s3.get_object(Bucket=bucket, Key=key)
    return np.load(io.BytesIO(response['Body'].read()))

def read_s3_csv(bucket, key):
    response = s3.get_object(Bucket=bucket, Key=key)
    return pd.read_csv(io.BytesIO(response['Body'].read()))

def upload_npz(bucket, key, visual_data, audio_data, labels):
    buffer = io.BytesIO()
    np.savez_compressed(buffer, visual_data=visual_data, audio_data=audio_data, is_speaking=labels)
    buffer.seek(0)
    s3.upload_fileobj(buffer, Bucket=bucket, Key=key)

# --- MAIN PROCESS ---
video_ids = list_videos(BUCKET, PREFIX)
batch_idx = 0
v_buf, a_buf, m_buf = [], [], []

for video_id in tqdm(video_ids):
    try:
        pairs_df = read_s3_csv(BUCKET, os.path.join(PREFIX, video_id, "pairs.csv"))
    except:
        continue

    pairs_info = {
        (video_id, int(row["chunk_id"]), int(row["frame_id"]), int(row["speaker_id"])): int(row["is_speaking"])
        for _, row in pairs_df.iterrows()
    }

    base_prefix = os.path.join(PREFIX, video_id)
    response = s3.list_objects_v2(Bucket=BUCKET, Prefix=os.path.join(base_prefix, "visual_pairs"))
    for obj in response.get('Contents', []):
        filename = obj["Key"].split("/")[-1]
        m = re.match(r"chunk(\d+)_speaker(\d+)_frame(\d+)_pair.npy", filename)
        if not m:
            continue

        chunk_id, speaker_id, frame_id = map(int, m.groups())
        key = (video_id, chunk_id, frame_id, speaker_id)
        # if key not in pairs_info:
        #     continue

        # Load visual & audio
        visual_data = read_s3_numpy(BUCKET, obj["Key"])
        audio_key = os.path.join(base_prefix, "melspectrogram_audio_pairs", f"chunk{chunk_id}_frame{frame_id}_pair.npy")
        try:
            audio_data = read_s3_numpy(BUCKET, audio_key)
        except:
            continue

        metadata = {
            "video_id": video_id,
            "chunk_id": chunk_id,
            "frame_id": frame_id,
            "speaker_id": speaker_id,
            "is_speaking": pairs_info[key]
        }
        # Buffer
        v_buf.append(visual_data)
        a_buf.append(audio_data)
        l_buf.append(metadata)

        if len(v_buf) >= BATCH_SIZE:
            out_key = os.path.join(OUTPUT_PREFIX, f"triplet_batch_{batch_idx:05d}.npz")
            upload_npz(BUCKET, out_key, np.stack(v_buf), np.stack(a_buf), np.array(l_buf))
            print(f"✅ Uploaded {out_key}")
            v_buf.clear()
            a_buf.clear()
            l_buf.clear()
            batch_idx += 1

# Final flush
if v_buf:
    out_key = f"{OUTPUT_PREFIX}triplet_batch_{batch_idx:05d}.npz"
    upload_npz(BUCKET, out_key, np.stack(v_buf), np.stack(a_buf), np.array(l_buf))
    print(f"✅ Uploaded final {out_key}")
