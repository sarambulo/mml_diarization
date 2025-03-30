import os

from pairs import choose_and_save_pairs_for_video
from utils import get_speaking_csv_files_s3, create_numbered_file
from visual import build_visual_pairs
from audio import build_audio_pairs
from config import *


def create_pairs() -> None:
    chunked_dirs = set([dir for dir, _ in get_speaking_csv_files_s3(S3_BUCKET_NAME, S3_VIDEO_DIR, S3_SPEAKING_CSV_NAME)])
    video_ids = list(reversed(range(2250, 2251)))
    speaking_filename = "is_speaking.csv"
    for video_id in video_ids:
        video_id = str(video_id).zfill(5)
        video_dir = f"s3://mmml-proj/preprocessed/{video_id}"
        if video_dir not in chunked_dirs:
            print("Can not find", video_dir)
            continue
        print("Building Pairs for Video", video_id)
        pairs_filename = "pairs.csv"  # create_numbered_file(video_dir, "pairs", "csv")
        pairs_csv_path = os.path.join(video_dir, pairs_filename)
        speaking_csv_path = os.path.join(video_dir, speaking_filename)
        choose_and_save_pairs_for_video(speaking_csv_path, pairs_csv_path)

        video_id = video_dir.split("/")[-1]
        
        # visual pairs = (3, num_channels=3, height=128, width=128)
        build_visual_pairs(S3_BUCKET_NAME, video_id, pairs_csv_path)

        # audio pairs = (3, num_bands=30, time_steps=22)
        build_audio_pairs(S3_BUCKET_NAME, video_id, pairs_csv_path, audio_type="melspectrogram")


if __name__ == "__main__":
    create_pairs()
    # files = get_speaking_csv_files_s3(S3_BUCKET_NAME, S3_VIDEO_DIR, S3_SPEAKING_CSV_NAME)
    # print(len(files))
    # print(files[:10])
