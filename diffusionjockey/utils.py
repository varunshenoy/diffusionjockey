import torchaudio
import torch
import numpy as np
from torchaudio import transforms as AT
from diffusers.pipelines.audio_diffusion.mel import Mel

TARGET_SAMPLE_RATE = 32000

# SDXL resolution
X_RES = 1024
Y_RES = 1024


def resample_audio(audio, sample_rate=TARGET_SAMPLE_RATE):
    if audio.shape[0] != TARGET_SAMPLE_RATE:
        resampler = AT.Resample(orig_freq=sample_rate, new_freq=TARGET_SAMPLE_RATE)
        audio = resampler(audio)
    return audio


def audio_to_spectrogram(audio_file, resample=False):
    # load audio from mp3 file
    audio, sample_rate = torchaudio.load(audio_file)

    mel = Mel(X_RES, Y_RES, sample_rate)

    audio = audio[0]  # just get one channel

    # resample audio to 32000 Hz
    if resample:
        audio = resample_audio(audio)

    mel.load_audio(raw_audio=np.array(audio))
    num_slices = mel.get_number_of_slices()

    slices = []
    for i in range(num_slices):
        mel_slice = mel.audio_slice_to_image(i)
        slices.append(mel_slice)

    return slices, sample_rate


def spectrogram_to_audio(spectrogram, sample_rate=TARGET_SAMPLE_RATE):
    mel = Mel(X_RES, Y_RES, sample_rate)
    audio = mel.image_to_audio(spectrogram)
    return audio


def save_audio(audio, filename, sample_rate=TARGET_SAMPLE_RATE):
    torchaudio.save(filename, torch.Tensor([audio, audio]), sample_rate, format="mp3")
