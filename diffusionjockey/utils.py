import torchaudio
import torch
import numpy as np
from torchaudio import transforms as AT
# from diffusers.pipelines.audio_diffusion.mel import Mel
from diffusionjockey.riff.spectrogram_image_converter import SpectrogramImageConverter
from diffusionjockey.riff.spectrogram_params import SpectrogramParams
import pydub
from PIL import Image


TARGET_SAMPLE_RATE = 44100

# SDXL resolution
X_RES = 512
Y_RES = 512


def resample_audio(audio, sample_rate=TARGET_SAMPLE_RATE):
    if audio.shape[0] != TARGET_SAMPLE_RATE:
        resampler = AT.Resample(orig_freq=sample_rate, new_freq=TARGET_SAMPLE_RATE)
        audio = resampler(audio)
    return audio


# def audio_to_spectrogram(audio_file, resample=False):
#     # load audio from mp3 file
#     audio, sample_rate = torchaudio.load(audio_file, format="mp3")

#     mel = Mel(X_RES, Y_RES, sample_rate)

#     # Get the channel 
#     audio = audio[0]

#     # resample audio to 32000 Hz
#     if resample:
#         audio = resample_audio(audio)

#     mel.load_audio(raw_audio=np.array(audio))
#     num_slices = mel.get_number_of_slices()

#     slices = []
#     for i in range(num_slices):
#         mel_slice = mel.audio_slice_to_image(i)
#         slices.append(mel_slice)

#     return slices, sample_rate

    
# Audio to Spectrogram using code from the riffusion repo.
def audio_to_spectrogram_riffusion(audio_file, resample=False):
    audio = audio_file

    device = "cuda"
    step_size_ms: int = 10
    num_frequencies: int = 512
    min_frequency: int = 0
    max_frequency: int = 10000
    window_duration_ms: int = 100
    padded_duration_ms: int = 400
    power_for_image: float = 0.25
    stereo: bool = False
    device: str = "cuda"
    sample_rate = 44100

    segment = pydub.AudioSegment.from_file(audio)
    params = SpectrogramParams(
        sample_rate=segment.frame_rate,
        stereo=stereo,
        window_duration_ms=window_duration_ms,
        padded_duration_ms=padded_duration_ms,
        step_size_ms=step_size_ms,
        min_frequency=min_frequency,
        max_frequency=max_frequency,
        num_frequencies=num_frequencies,
        power_for_image=power_for_image,
    )

    converter = SpectrogramImageConverter(params=params, device=device)
    pil_image = converter.spectrogram_image_from_audio(segment)
    pil_image_tensor = torch.tensor(np.array(pil_image))

    slices = []
    for i in range(0, pil_image_tensor.shape[1], 512):
        if i + 512 >= pil_image_tensor.shape[1]:
            continue
        print(pil_image_tensor[:, i:i+512, :].shape)
        pil_image = Image.fromarray(pil_image_tensor[:, i:i+512, :].numpy())
        slices.append(pil_image)

    return slices, sample_rate


def spectrogram_to_audio(spectrogram, sample_rate=TARGET_SAMPLE_RATE):
    # Get parameters from image exif
    pil_image = spectrogram
    device = "cuda"

    img_exif = pil_image.getexif()
    assert img_exif is not None

    try:
        params = SpectrogramParams.from_exif(exif=img_exif)
    except (KeyError, AttributeError):
        print("WARNING: Could not find spectrogram parameters in exif data. Using defaults.")
        params = SpectrogramParams()

    converter = SpectrogramImageConverter(params=params, device=device)
    segment = converter.audio_from_spectrogram_image(pil_image)

    return segment


def save_audio(audio, filename, sample_rate=TARGET_SAMPLE_RATE):
    torchaudio.save(filename, torch.Tensor([audio, audio]), sample_rate, format="mp3")
