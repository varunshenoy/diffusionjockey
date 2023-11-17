from datasets import Dataset
from datasets.features.image import Image
import torchvision
import torch

def load_custom_dataset():
    # Directory for training inputs
    WRITE_DIR = 'input/'
    AUDIO_DIR = 'fma/data/fma_small/'

    # Turn it into huggingface dataset format

    descriptions = torch.load(WRITE_DIR + 'descriptions.pt')

    print(descriptions)

    image_list, text_list = [], []

    for track_id in descriptions:
        spectrogram_path = f"{WRITE_DIR}{track_id}.pt"
        spectrogram = torch.load(spectrogram_path)[0]
        spectrogram = spectrogram.unsqueeze(0)
        spectrogram = spectrogram.expand(3, -1, -1)
        text = descriptions[track_id]

        print(spectrogram.shape)
        image = torchvision.transforms.ToPILImage()(spectrogram)

        image_list.append(image)
        text_list.append(text)

    image_data = Dataset.from_dict({"image": image_list, "text": text_list})
    return image_data

dataset = load_custom_dataset()
dataset.push_to_hub("punwaiw/DiffusionJockey")

