from typing import Literal
from video import create_visual_pairs
from audio import create_audio_pairs

CHUNK_LENGTH_SECS = 3
AUDIO_LENGTH_SECS = 0.2


def create_pairs(dir: str, type: Literal["audio", "visual"]) -> None:
    if type == "audio":
        create_audio_pairs(dir)
    else:
        create_visual_pairs(dir)


def create_audio_pairs(dir):
    pass


def create_video_pairs(dir):
    pass


if __name__ == "__main__":
    print()
