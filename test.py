from diffusionjockey.utils import *
from matplotlib import pyplot as plt
from PIL import Image

# slices, sample_rate = audio_to_spectrogram_riffusion("softtune.png")
# print(sample_rate)

# image = slices[0]
# image.save("spectrogram.png")
image  = Image.open("26.png")

sample_rate = 44100

# display spectrogram
audio = spectrogram_to_audio(image, sample_rate)
audio.export("audio_reconstructis.mp3", format="mp3")

# save_audio(audio, "audio_reconstructed.mp3", sample_rate=sample_rate)
