import os

from pairs import choose_and_save_pairs_for_video
from utils import get_speaking_csv_files, create_numbered_file
from visual import build_visual_pairs
from audio import build_audio_pairs
from config import *


def create_pairs(input_dir: str) -> None:
    for video_dir, speaking_filename in get_speaking_csv_files(input_dir):
        pairs_filename = "pairs.csv"  # create_numbered_file(video_dir, "pairs", "csv")
        pairs_csv_path = os.path.join(video_dir, pairs_filename)
        speaking_csv_path = os.path.join(video_dir, speaking_filename)
        choose_and_save_pairs_for_video(speaking_csv_path, pairs_csv_path)
        # visual pairs = (3, num_channels=3, height=128, width=128)
        build_visual_pairs(video_dir, pairs_csv_path)

        # audio pairs = (3, num_bands=30, time_steps=22)
        build_audio_pairs(video_dir, pairs_csv_path, audio_type="melspectrogram")


if __name__ == "__main__":
    create_pairs(INPUT_DIR)
