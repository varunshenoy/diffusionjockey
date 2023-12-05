from datasets import Dataset, load_dataset
from datasets.features.image import Image
import torchvision
import torch
import os
import numpy as np

def load_custom_dataset():
    dataset = load_dataset("punwaiw/DiffusionJockey")
    return dataset["train"]

def load_custom_dataset_start():
    # Directory for training inputs
    WRITE_DIR = 'input-512/'

    # Turn it into huggingface dataset format

    descriptions = torch.load(WRITE_DIR + 'descriptions.pt')

    image_list, text_list, dynamics_list = [], [], []

    print(descriptions)

    counter = 0
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
                    spectrogram = spectrogram.permute((2, 0, 1))
                    spectrogram = spectrogram.expand(3, -1, -1)

                    track_id = int(filename[:-3])
                    text = descriptions[track_id][0]
                    image = torchvision.transforms.ToPILImage()(spectrogram)

                    # Apply the transform to the PIL image
                    transform = torchvision.transforms.ToTensor()
                    tensor_image = transform(image)

                    # Compute dynamics
                    mean_spectrogram = tensor_image.mean(axis=1)
                    MASK_LENGTH = 11
                    mask = np.ones(MASK_LENGTH)
                    convolved_output = np.convolve(mean_spectrogram[0], mask) / MASK_LENGTH

                    HALF_MASK = int((MASK_LENGTH - 1) / 2)

                    mean_spectrogram = torch.tensor(convolved_output[HALF_MASK:-HALF_MASK])
                    mean_spectrogram = mean_spectrogram.unsqueeze(0)
                    mean_spectrogram = mean_spectrogram.unsqueeze(0)
                    mean_spectrogram = mean_spectrogram.expand(3, 512, -1)

                    dynamics_image = torchvision.transforms.ToPILImage()(mean_spectrogram)

                    image_list.append(image)
                    text_list.append(text)
                    dynamics_list.append(dynamics_image)

    print(len(image_list))
    image_data = Dataset.from_dict({"image": image_list, "text": text_list, "dynamics": dynamics_list })
    return image_data

dataset = load_custom_dataset_start()
dataset.push_to_hub("punwaiw/diffusionJockeyDynamicsTight")

