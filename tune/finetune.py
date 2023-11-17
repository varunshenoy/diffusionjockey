# Training Harness for the diffusion model

import argparse

from diffusionjockey.utils import audio_to_spectrogram

def read_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--image_folder', type=str, help='Description of arg1')
    return parser.parse_args()

args = read_args()

import PIL.Image as Image
def finetune():
    print("hallo")
    spectogram, sampling_rate = audio_to_spectrogram("./examples/audio.mp3")
    print(spectogram)
    spectogram[0].show()
