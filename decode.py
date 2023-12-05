#
# Decodes a folder of spectograms
#

import argparse
import torch
from PIL import Image
from diffusionjockey.utils import spectrogram_to_audio, save_audio
from torchvision import transforms

parser = argparse.ArgumentParser()
parser.add_argument("-i", "--input_dir", help="input directory")
parser.add_argument("-o", "--output_dir", help="output directory")
args = parser.parse_args()

import os

for file in os.listdir(args.input_dir):
    print(file)
    # Process each file in input_dir
    file_path = os.path.join(args.input_dir, file)
    # Rest of the code to process the file goes here
    image = Image.open(file_path)
    img_exif = pil_image.getexif()
    image_tensor = transforms.ToTensor()(image).mean(axis=0)
    image = transforms.ToPILImage()(image_tensor)

    # Rest of the code to process the file goes here
    audio = spectrogram_to_audio(image, sample_rate=44100)

    output_file_path = os.path.join(args.output_dir, file)
    output_file_path = output_file_path.replace('.png', '.mp3')
    save_audio(audio, output_file_path)
