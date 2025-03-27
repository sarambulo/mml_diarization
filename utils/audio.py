from typing import Dict, Tuple, List
import torch
import numpy as np
import librosa

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
   audio_shape=audio_data.shape
#    print(audio_shape)
   audio_data=audio_data.reshape((-1,audio_shape[-1])) 
   return audio_data


def transform_audio(
    audio_data: np.ndarray,
    sr: int,
    n_bands: int,
    target_tf: int,
    n_fft=1024,
    hop_length=512

) -> np.ndarray:
    """
    Transforms audio into a log-mel spectrogram using librosa.
    
    Parameters:
        audio_data (np.ndarray): Mono audio data (1D array).
        sr (int): Sampling rate of the audio. Approx 44k.
        n_bands (int): Number of output bands to generate ~30
        fmax (int): Maximum frequency when converting to Mel scale. Typically sr/2.

    Returns:
        np.ndarray: The log-mel spectrogram of shape (n_mels, time_frames per segment, segments).
    """

    audio_data = audio_data.transpose(1, 0) # Put the channel first
    audio_data = audio_data.detach().cpu().numpy()
    if audio_data.ndim > 1:
        audio_data = np.mean(audio_data, axis=0)
    mfcc = librosa.feature.mfcc(y=audio_data, sr=sr, n_mfcc=n_bands, n_fft=n_fft, hop_length=hop_length)
        

    mel_spec = librosa.feature.melspectrogram(
            y=audio_data, sr=sr, n_mels=n_bands, n_fft=n_fft, hop_length=hop_length
        )
    mel_spec_db = librosa.power_to_db(mel_spec, ref=np.max)

    if mfcc.shape[1]<target_tf:
        pad_width=target_tf-mfcc.shape[1]
        mfcc=np.pad(mfcc, ((0, 0), (0, pad_width)), mode='constant')
        mel_spec_db=np.pad(mel_spec_db, ((0, 0), (0, pad_width)), mode='constant')
    elif mfcc.shape[1]>target_tf:
        mfcc=mfcc[:,:target_tf]
        mel_spec_db= mel_spec_db[:,:target_tf]
    mel_spec_db=mel_spec_db.reshape(n_bands, int(target_tf/20), 20)
    mfcc=mfcc.reshape(n_bands, int(target_tf/20), 20)
    return mel_spec_db, mfcc
    
       # (30,440) pad/truncate
   # reshape (30, 22, 20)