from utils import read_speaker_file, read_spectrogram
from config import *


def get_speaker_status_by_frame(path):
    """Returns a list of speaker status's by frame"""
    speakers = read_speaker_file(path)


def divide_mfcc_into_frames(mfcc):
    pass


def sample_speaker_is_speaking(speaker):
    pass


def sample_speaker_not_speaking(speaker):
    pass


def sample_positive_pair(speaker, status):
    is_speaking = bool(status)
    if is_speaking:
        return sample_speaker_is_speaking(speaker)
    else:
        return sample_speaker_not_speaking(speaker)


def sample_negative_pair(speaker, status):
    is_speaking = bool(status)
    if is_speaking:
        return sample_speaker_not_speaking(speaker)
    else:
        return sample_speaker_is_speaking(speaker)


def create_audio_pairs(chunks):
    pairs = []
    for chunk in chunks:
        speaker_status_list = get_speaker_status_by_frame(chunk["is_speaking"])
        mfcc = read_spectrogram(chunk["spectrogram"])
        mfcc_frames = divide_mfcc_into_frames(mfcc)
        for frame_id, frame_status in enumerate(speaker_status_list):
            for speaker_id, status in enumerate(frame_status):
                anchor = mfcc_frames[frame_id]
                pos_frame_id = sample_positive_pair(speaker_id, status)
                neg_frame_id = sample_negative_pair(speaker_id, status)
                if pos_frame_id != None and neg_frame_id != None:
                    pos = mfcc_frames[pos_frame_id]
                    neg = mfcc_frames[neg_frame_id]
                    pairs.append((anchor, pos, neg))
