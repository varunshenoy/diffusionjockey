from diffusionjockey.utils import *
from matplotlib import pyplot as plt

slices, sample_rate = audio_to_spectrogram("examples/audio.mp3")

image = slices[0]
# display spectrogram
plt.imshow(image, cmap="gray")
plt.show()


audio = spectrogram_to_audio(image, sample_rate)
print(audio.shape)

save_audio(audio, "examples/audio_reconstructed.mp3", sample_rate=sample_rate)
