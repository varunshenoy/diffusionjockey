import torch
import numpy as np
import pandas as pd

# FMA utilities
import fma.utils

# Project utilities
import diffusionjockey.utils as dj

# FMA mp3 directory
AUDIO_DIR = 'fma/data/fma_small/'

# Directory for training inputs
WRITE_DIR = 'input/'

# These ten tracks all have nonempty tags
SAMPLE_TRACK_IDS = [114399, 112583, 74387, 75908, 86793, 91160, 123509, 54753, 147956, 133274]

# Returns view of DataFrame with only nonempty track tags
def select_nonempty(df):
    df[df['track', 'tags'].map(lambda x: len(x) != 0)]

# Given a track_id read the audio file from the FMA dataset and convert it into
# spectrograms using the diffusionjockey utils. Write out the raw spectrogram
# data as pytorch tensors.
def make_spectrograms(track_id):
    audio_path = fma.utils.get_audio_path(AUDIO_DIR, track_id)
    write_path = fma.utils.get_audio_path(WRITE_DIR, track_id).replace('.mp3', '.pt')

    slices, sample_rate = dj.audio_to_spectrogram(audio_path)

    t = list()
    for image in slices:
        # image is 8-bit greyscale so this creates a tensor of torch.uint8
        t.append(torch.tensor(np.asarray(image)))

    # Creates a tensor of shape (slice id, spectro height, spectro width)
    t = torch.stack(t)
    print("Writing tensor of {} with {} to {}".format(t.dtype, t.size(), write_path))
    torch.save(t, write_path)

# Format string containing all genres and any track tags
def format_description(genres, metadat):
    tokens = list()

    for g in metadat['track', 'genres_all']:
        tokens.append(genres.loc[g].title)

    tokens.extend(metadat['track', 'tags'])

    return ', '.join(tokens)

# Load tensor of spectrograms for a track. This should be moved to the diffusionjockey
# utils file, which will then create a dependency on fma.utils
def load_spectrograms(input_dir, track_id):
    tensor_path = fma.utils.get_audio_path(input_dir, track_id).replace('.mp3', '.pt')
    return torch.load(tensor_path)

# To populate the directory hierarchy under "input/" run the following
# commands from the repo root, otherwise torch.save() will fail:
#
# mkdir input
# for TR in $(ls -1 "fma/data/fma_small"); do mkdir "input/$TR"; done
# rmdir input/README.txt/
# rmdir input/checksums/
#
if __name__ == '__main__':
    # FMA metadata
    tracks = fma.utils.load('fma/data/fma_metadata/tracks.csv')
    genres = fma.utils.load('fma/data/fma_metadata/genres.csv')

    train_select = (tracks['set', 'subset'] == 'small') & (tracks['set', 'split'] == 'training')
    data = tracks[train_select]

    # Dictionary of (track_id, string with genres and tags)
    descriptions = dict()

    # To iterate over whole training set, replace with
    #for track_id, metadat in data[data.index >= 99134].iterrows():
    for track_id, metadat in data.iterrows():
        try:
            make_spectrograms(track_id)
            descriptions[track_id] = format_description(genres, metadat)
        except RuntimeError:
            print("Error reading or processing ", track_id)

    torch.save(descriptions, WRITE_DIR + 'descriptions.pt')

    # Load descriptions into dictionary with
    # torch.load(WRITE_DIR + 'descriptions.pt')

    # Load a stack of spectrograms with
    # load_spectrograms(WRITE_DIR, SAMPLE_TRACK_IDS[0])