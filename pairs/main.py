import os

from pairs import build_pairs_for_video
from utils import get_all_csv_files, create_numbered_file
from config import *


def create_pairs(input_dir: str, output_dir: str) -> None:
    os.makedirs(output_dir, exist_ok=True)
    pairs_csv_filename = create_numbered_file(output_dir, "pairs", "csv")
    pairs_csv_path = os.path.join(output_dir, pairs_csv_filename)
    for path in get_all_csv_files(input_dir):
        build_pairs_for_video(path, pairs_csv_path)


if __name__ == "__main__":
    create_pairs(INPUT_DIR, OUTPUT_DIR)
