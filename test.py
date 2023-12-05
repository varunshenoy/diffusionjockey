from diffusionjockey.utils import *
from matplotlib import pyplot as plt

slices, sample_rate = audio_to_spectrogram_riffusion("audio.mp3")
print(sample_rate)

image = slices[0]
image.save("spectrogram.png")

sample_rate = 44100

# display spectrogram
audio = spectrogram_to_audio(image, sample_rate)
audio.export("audio_reconstructi.mp3", format="mp3")

# save_audio(audio, "audio_reconstructed.mp3", sample_rate=sample_rate)
