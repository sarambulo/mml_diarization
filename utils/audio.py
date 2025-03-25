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
   if isinstance(audio_data, torch.Tensor):
        audio_data = audio_data.detach().cpu().numpy()

   if audio_data.ndim > 1:
        audio_data = np.mean(audio_data, axis=0)
   return audio_data


def transform_audio(
    audio_data: np.ndarray,
    sr: int = 16000,
    n_mels: int = 128,
    fmax: int = 8000
) -> np.ndarray:
    """
    Transforms audio into a log-mel spectrogram using librosa.
    
    Parameters:
        audio_data (np.ndarray): Mono audio data (1D array).
        sr (int): Sampling rate of the audio. Default is 16k.
        n_mels (int): Number of Mel bands to generate.
        fmax (int): Maximum frequency when converting to Mel scale. Typically sr/2.

    Returns:
        np.ndarray: The log-mel spectrogram of shape (n_mels, time_frames).
    """

    mel_spec = librosa.feature.melspectrogram(y=audio_data, sr=sr, n_mels=n_mels, fmax=fmax)
    mel_spec_db = librosa.power_to_db(mel_spec, ref=np.max)

    return mel_spec_db