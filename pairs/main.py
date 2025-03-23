from typing import Literal
from video import create_video_pairs
from audio import create_audio_pairs
from utils import crawl_chunks_dir
from config import *


def create_pairs(input_dir: str, output_dir: str) -> None:
    chunks = crawl_chunks_dir(input_dir)
    create_audio_pairs(chunks, output_dir)
    create_video_pairs(chunks, output_dir)


if __name__ == "__main__":
    create_pairs(INPUT_DIR, OUTPUT_DIR)
