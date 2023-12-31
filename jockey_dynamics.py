from datasets import Dataset, load_dataset
from datasets.features.image import Image
import torchvision
import torch
import os

def load_custom_dataset():
    dataset = load_dataset("punwaiw/DiffusionJockey")
    return dataset["train"]

def load_custom_dataset_start():
    # Directory for training inputs
    WRITE_DIR = 'input/'

    # Turn it into huggingface dataset format

    descriptions = torch.load(WRITE_DIR + 'descriptions.pt')

    image_list, text_list = [], []

    print(descriptions)

    for folder in os.listdir(WRITE_DIR):
        print(f"LOADING FOLDER {folder}")
        item_path = os.path.join(WRITE_DIR, folder)

        if os.path.isdir(item_path):
            for filename in os.listdir(item_path):
                spectrogram_path = os.path.join(item_path, filename)
                spectrogram_tensor = torch.load(spectrogram_path)
                batch_size = spectrogram_tensor.shape[0]
                for i in range(min(batch_size, 8)):
                    spectrogram = spectrogram_tensor[i]
                    spectrogram = spectrogram.unsqueeze(0)
                    spectrogram = spectrogram.expand(3, -1, -1)

                    track_id = int(filename[:-3])
                    text = descriptions[track_id][0]
                    image = torchvision.transforms.ToPILImage()(spectrogram)
                    image_list.append(image)
                    text_list.append(text)

    image_data = Dataset.from_dict({"image": image_list, "text": text_list})
    return image_data

# dataset = load_custom_dataset_start()
# dataset.push_to_hub("punwaiw/DiffusionJockey")

