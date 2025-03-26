from typing import Dict, Tuple, List
import torch
import numpy as np
import librosa

# def read_audio(video_path: str, seconds: float = 3) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
#    """
#    Reads a video file and returns video data, audio data, and timestamps.

#    :param video_path: Path to the video file.
#    :param seconds: Duration of the video to read in seconds.

#    :return: A generator that yields a tuple containing:

#       - video_data (torch.Tensor): Shape (Frames, C, H, W)
#       - audio_data (torch.Tensor): Shape (Frames, SamplingRate, C)
#       - timestamps (torch.Tensor): Shape (Frames,)
#       - frame_ids  (torch.Tensor): Shape (Frames,)
#    """
#    return

# def downsample_audio(video_frames: torch.Tensor, timestamps: torch.Tensor, frame_ids: torch.Tensor, factor: int = 5) -> Tuple[torch.Tensor, torch.Tensor]:
#    """
#    Downsamples a video and returns the remaining frames, timestamps and frame IDs.
#    The number of remaining frames is `ceil(Frames / factor)`

#    :param video_frames: Shape (Frames, FPS, C, H, W)
#    :param timestamps: Shape (Frames,

#   :return: A tuple containing:
#       - video_data (torch.Tensor): Shape (ceil(Frames / factor), C, H, W)
#       - timestamps (torch.Tensor): Shape (ceil(Frames / factor),)
#       - frame_ids  (torch.Tensor): Shape (ceil(Frames / factor),)
#    """
#    return


def flatten_audio(audio_data: np.ndarray) -> np.ndarray:
    """
    Flattens a multi-channel audio input into a single channel (mono) signal
    by averaging across channels. If the input is already mono, it is returned
    unchanged.

    Parameters:
        audio_data (np.ndarray): Audio data array.
                                 Shape could be (samples,) or (channels, samples).

    Returns:
        np.ndarray: Flattened mono audio data (shape: (samples,)).
    """
    # If audio has more than one dimension (e.g., [channels, samples]),
    # average across channels to get a single channel.
    #    if isinstance(audio_data, torch.Tensor):
    #         audio_data = audio_data.detach().cpu().numpy()
    audio_shape = audio_data.shape
    #    print(audio_shape)
    audio_data = audio_data.reshape((-1, audio_shape[-1]))
    return audio_data


def transform_audio(
    audio_data: np.ndarray,
    output_type: str,
    sr: int,
    n_bands: int,
    n_fft=1024,
    hop_length=512,
) -> np.ndarray:
    """
    Transforms audio into a log-mel spectrogram using librosa.

    Parameters:
        audio_data (np.ndarray): Mono audio data (1D array).
        output_type (str): mfcc or melspectrogram
        sr (int): Sampling rate of the audio. Approx 44k.
        n_bands (int): Number of output bands to generate ~30
        fmax (int): Maximum frequency when converting to Mel scale. Typically sr/2.

    Returns:
        np.ndarray: The log-mel spectrogram of shape (n_mels, time_frames= approx 10ms).
    """
    audio_data = audio_data.transpose(1, 0)  # Put the channel first
    if output_type == "mfcc":
        audio_data = audio_data.detach().cpu().numpy()
        if audio_data.ndim > 1:
            audio_data = np.mean(audio_data, axis=0)
        mfcc = librosa.feature.mfcc(
            y=audio_data, sr=sr, n_mfcc=n_bands, n_fft=n_fft, hop_length=hop_length
        )
        return mfcc
    elif output_type == "melspectrogram":
        mel_spec = librosa.feature.melspectrogram(
            y=audio_data, sr=sr, n_mels=n_bands, n_fft=n_fft, hop_length=hop_length
        )
        mel_spec_db = librosa.power_to_db(mel_spec, ref=np.max)
        return mel_spec_db
    else:
        raise ValueError(f"Invalid output type")
