import os

from pairs import choose_and_save_pairs_for_video
from utils import (
    get_speaking_csv_files_s3,
    create_numbered_file,
    s3_load_numpy,
    upload_npz,
)
from visual import build_visual_pairs, build_visual_pair
from audio import build_audio_pairs, build_audio_pair
from combined_pairs import build_combined_pairs
from config import *
import numpy as np
import traceback
import sys

START = 93
UPLOAD_BATCH_SIZE = 5000
OUTPUT_PREFIX = "test_batched_triplets_with_lips"


def create_pairs() -> None:
    # chunked_dirs = set([dir for dir, _ in get_speaking_csv_files_s3(S3_BUCKET_NAME, S3_VIDEO_DIR, S3_SPEAKING_CSV_NAME)])
    traceback.print_exc(file=sys.stdout)
    video_ids = list(range(START, 100))
    speaking_filename = "is_speaking.csv"
    face_buf, lip_buf, audio_buf, meta_buf = [], [], [], []
    batch_idx = 0
    for video_id in video_ids:
        try:
            video_id = str(video_id).zfill(5)
            video_dir = f"s3://mmml-proj/preprocessed/{video_id}"

            print("Building Pairs for Video", video_id)
            pairs_filename = (
                "pairs.csv"  # create_numbered_file(video_dir, "pairs", "csv")
            )

            pairs_csv_path = os.path.join(video_dir, pairs_filename)
            speaking_csv_path = os.path.join(video_dir, speaking_filename)
            choose_and_save_pairs_for_video(speaking_csv_path, pairs_csv_path)

            # # visual pairs = (3, num_channels=3, height=128, width=128)
            # build_visual_pairs(S3_BUCKET_NAME, video_id, pairs_csv_path)

            # # audio pairs = (3, num_bands=30, time_steps=22)
            # build_audio_pairs(
            #     "mmml-proj", video_id, pairs_csv_path, audio_type="melspectrogram"
            # )

            batch_idx = build_combined_pairs(
                S3_BUCKET_NAME,
                video_id,
                pairs_csv_path,
                face_buf,
                lip_buf,
                audio_buf,
                meta_buf,
                batch_size=UPLOAD_BATCH_SIZE,
                outpath=OUTPUT_PREFIX,
                audio_type="melspectrogram",
                batch_idx=batch_idx,
            )

        except Exception as e:
            traceback.print_exc(file=sys.stdout)
            print(f"Error fetching Video {video_id}:", str(e))

    # if face_buf:
    #     out_key = f"{OUTPUT_PREFIX}/triplet_batch_{batch_idx:05d}.npz"
    #     upload_npz(
    #         S3_BUCKET_NAME,
    #         out_key,
    #         np.stack(face_buf),
    #         np.stack(lip_buf),
    #         np.stack(audio_buf),
    #         np.array(meta_buf),
    #     )
    #     print(f"✅ Uploaded final {out_key}")


if __name__ == "__main__":
    create_pairs()
    # build_combined_pairs(S3_BUCKET_NAME, "00005", "s3://mmml-proj/preprocessed/00001/pairs.csv", [], [], [], [], batch_size=100, outpath="test", audio_type="melspectrogram")
    # files = get_speaking_csv_files_s3(S3_BUCKET_NAME, S3_VIDEO_DIR, S3_SPEAKING_CSV_NAME)
    # print(len(files))
    # print(files[:10])

    # arr = s3_load_numpy("mmml-proj", "preprocessed/00005/Chunk_38/lip_0.npy")
    # print(arr.shape)
