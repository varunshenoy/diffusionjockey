#
# Code for turning the generated dataset into the format
# specified by Huggingface.
#

from datasets import Dataset
import torch

# Directory for training inputs
WRITE_DIR = 'input/'
# Turn it into huggingface dataset format

descriptions = torch.load(WRITE_DIR + 'descriptions.pt')


ds_creator = Dataset.from_dict({
    "image": [
        "bulbasaur", "squirtle"
    ], 
    "text": [
        "grass", "water"
    ]}
)
