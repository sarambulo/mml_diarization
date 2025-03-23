from utils import read_speaker_file, read_spectrogram
from config import *


def get_speaker_status_by_frame(path):
    """Returns a list of speaker status's by frame"""
    speakers = read_speaker_file(path)


def divide_mel_into_frames(mel):
    pass


def sample_speaker_is_speaking(speaker):
    pass


def sample_speaker_not_speaking(speaker):
    pass


def create_audio_pairs(chunks, output_dir):
    for chunk in chunks:
        speaker_status_list = get_speaker_status_by_frame(chunk["is_speaking"])
        mel = read_spectrogram(chunk["spectrogram"])
        mel_frames = divide_mel_into_frames(mel)
        for frame_id, frame_status in enumerate(speaker_status_list):
            for speaker_id, status in enumerate(frame_status):
                is_speaking = bool(status)
                if is_speaking:
                    anchor = mel_frames[frame_id]
                    pos_frame_id = sample_speaker_is_speaking(speaker_id)
                    neg_frame_id = sample_speaker_not_speaking(speaker_id)
                    if pos_frame_id != None and neg_frame_id != None:
                        pos = mel_frames[pos_frame_id]
                        neg = mel_frames[neg_frame_id]

                        path = save_triplet(output_dir, (anchor, pos, neg))
                        save_pair_to_csv(speaker_id, status, path)
