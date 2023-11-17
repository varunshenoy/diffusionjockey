# Training Harness for the diffusion model

import argparse

from .. import diffusionjockey

def read_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--image_folder', type=str, help='Description of arg1')
    return parser.parse_args()

args = read_args()

spectogram, sampling_rate = audio_to_spectrogram("./examples/audio.mp3")

import PIL.Image as Image

spectogram_image = Image.fromarray(spectogram)
spectogram_image.show()

