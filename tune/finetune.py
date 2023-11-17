# Training Harness for the diffusion model

import argparse
from diffusionjockey.utils import audio_to_spectrogram
import os
import subprocess
import PIL.Image as Image

def read_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--image_folder', type=str, help='Description of arg1')
    return parser.parse_args()

args = read_args()

def finetune():
    print("hallo")
    spectogram, sampling_rate = audio_to_spectrogram("./examples/audio.mp3")
    print(spectogram)
    spectogram[0].show()

    image_folder = "./spectogram_images"
    os.makedirs(image_folder, exist_ok=True)
    spectogram[0].save(os.path.join(image_folder, "img_0.png"))

    subprocess.call([
        "python",
        "tune/train_text_to_image_lora_sdxl.py"
    ])