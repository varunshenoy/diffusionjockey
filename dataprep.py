import torch
import numpy as np
import pandas as pd
import os

# FMA utilities
import fma.utils

# SDD and MTG utilities
import sdd.utils

# Project utilities
import diffusionjockey.utils as dj

# FMA mp3 directory
FMA_AUDIO_DIR = 'fma/data/fma_small/'

# SDD mp3 directory
SDD_AUDIO_DIR = 'sdd/data/audio/'

# Directory for training inputs
WRITE_DIR = 'input-512/'

# These ten tracks all have nonempty tags
SAMPLE_TRACK_IDS = [114399, 112583, 74387, 75908, 86793, 91160, 123509, 54753, 147956, 133274]

# Returns view of DataFrame with only nonempty track tags
def select_nonempty(df):
    df[df['track', 'tags'].map(lambda x: len(x) != 0)]

# Given a track_id read the audio file from the FMA dataset and convert it into
# spectrograms using the diffusionjockey utils. Write out the raw spectrogram
# data as pytorch tensors. sdd_path must be set to the "path" column of the sdd
# csv if and only if the source audio is from SDD, in which case track_id is the
# song's relabeled index.
def make_spectrograms(track_id, sdd_path = None):
    if sdd_path is None:
        audio_path = fma.utils.get_audio_path(FMA_AUDIO_DIR, track_id)
    else:
        audio_path = SDD_AUDIO_DIR + sdd_path.replace('.mp3', '.2min.mp3')

    write_path = fma.utils.get_audio_path(WRITE_DIR, track_id).replace('.mp3', '.pt')

    slices, sample_rate = dj.audio_to_spectrogram_riffusion(audio_path)

    t = list()
    for image in slices:
        # image is 8-bit greyscale so this creates a tensor of torch.uint8
        t.append(torch.tensor(np.asarray(image)))

    # Creates a tensor of shape (slice id, spectro height, spectro width)
    t = torch.stack(t)

    # torch will refuse to save if the directory does not exist
    write_dir = os.path.dirname(write_path)
    if not os.path.isdir(write_dir):
        os.mkdir(write_dir, mode=0o775)

    print("Writing tensor of {} with {} to {}".format(t.dtype, t.size(), write_path))
    torch.save(t, write_path)

# Format string containing all genres and any track tags
def fma_format_description(genres, metadat):
    tokens = list()

    for g in metadat['track', 'genres_all']:
        tokens.append(genres.loc[g].title)

    tokens.extend(metadat['track', 'tags'])

    return ', '.join(tokens)

def sdd_make_description(metadat):
    tokens = (metadat['genres'] +
        metadat['instruments'] +
        metadat['moods_themes'])

    return ', '.join(tokens)

# Load tensor of spectrograms for a track
def load_spectrograms(input_dir, track_id):
    tensor_path = fma.utils.get_audio_path(input_dir, track_id).replace('.mp3', '.pt')
    return torch.load(tensor_path)

if __name__ == '__main__':
    # Dictionary of (track_id, list of string with genres, tags, and/or captions)
    descriptions = dict()

    # Create directory for processed data if it does not exist so torch.save() works
    try:
        os.mkdir(WRITE_DIR, mode=0o775)
    except FileExistsError:
        pass

    # FMA metadata
    tracks = fma.utils.load('fma/data/fma_metadata/tracks.csv')
    genres = fma.utils.load('fma/data/fma_metadata/genres.csv')

    # Relabel tracks used in SDD starting after the *full* FMA dataset (not just the small subset).
    # In particular, increment the leading three digits to start a new paritition of the directory.
    # These ids are the ones used to store spectrograms and to reference descriptions.
    sdd_relabel_index = 1000 * (int('{:06d}'.format(tracks.index.max() + 1)[:3]) + 1)

    train_select = (tracks['set', 'subset'] == 'small') & (tracks['set', 'split'] == 'training')
    data = tracks[train_select]

    # Process the FMA training set first
    for track_id, metadat in data.iterrows():
    #for track_id, metadat in data[data.index >= 99134].iterrows():
        try:
            make_spectrograms(track_id)
            descriptions[track_id] = [ fma_format_description(genres, metadat) ]
        except RuntimeError as err:
            print("Error reading or processing FMA", track_id, ":", err)

    # SDD/MTG metadata
    sddat = pd.read_csv('sdd/data/song_describer.csv')
    sddat.fillna(False, inplace=True)
    mtg = sdd.utils.load_metadata('sdd/data/raw_30s_cleantags_50artists.tsv')

    train_select = sddat['is_valid_subset'] == True
    data = sddat[train_select]

    sdd_relabel = dict()

    for _, metadat in data.iterrows():
        sdd_id = metadat['track_id']
        try:
            if sdd_id not in sdd_relabel:
                make_spectrograms(sdd_relabel_index, metadat['path'])
                sdd_relabel[sdd_id] = sdd_relabel_index
                descriptions[sdd_relabel_index] = [ metadat['caption'] ]
                # Don't increment the relabel index until the spectrograms have successfully
                # been created. Otherwise, control will jump to the except clause below.
                sdd_relabel_index += 1
            else:
                descriptions[sdd_relabel[sdd_id]].append(metadat['caption'])
        except RuntimeError as err:
            print("Error reading or processing SDD", sdd_id, ":", err)

    for sdd_id in data['track_id'].unique():
        # Don't try adding the tags unless the audio could be processed in the first place
        if sdd_id in sdd_relabel:
            descriptions[sdd_relabel[sdd_id]].append(
                sdd_make_description(mtg.loc[sdd_id]))

    torch.save(descriptions, WRITE_DIR + 'descriptions.pt')

    # Load descriptions into dictionary with
    # torch.load(WRITE_DIR + 'descriptions.pt')

    # Load a stack of spectrograms with
    # load_spectrograms(WRITE_DIR, SAMPLE_TRACK_IDS[0])
