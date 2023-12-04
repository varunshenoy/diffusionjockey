import numpy as np
import pandas as pd
import soundfile as sf
import torch
import resampy
import os

from transformers import ClapFeatureExtractor, ClapAudioModelWithProjection

# FMA utilities
import fma.utils

# FMA mp3 directory
FMA_AUDIO_DIR = 'fma/data/fma_small/'

# Directory for validation references
WRITE_DIR = 'eval/'

def get_toplevel_genres(genres, id_list):
    s = set()

    for genre_id in id_list:
        toplevel = genre_id
        while genres.loc[toplevel]['parent'] != 0:
            toplevel = genres.loc[toplevel]['parent']
        s.add(toplevel)

    return s

def compute_embedding(model, extractor, model_sr, track_id):
    audio_path = fma.utils.get_audio_path(FMA_AUDIO_DIR, track_id)
    write_path = fma.utils.get_audio_path(WRITE_DIR, track_id).replace('.mp3', '.pt')

    # The data returned is in the range [-1.0, +1.0]
    wav, sr = sf.read(audio_path)

    if wav.ndim == 2:
        data = resampy.resample(wav[:, 0], sr, model_sr)
    else:
        data = resampy.resample(wav, sr, model_sr)

    features = extractor(data, sampling_rate=model_sr, return_tensors='pt')

    output = model(**features)

    return torch.squeeze(output.audio_embeds.detach())

def record_statistics(genres, embeddings, histo, genre_list, vec):
    # Add the embedding to all its top level genres
    for g in genre_list:
        if g not in embeddings:
            embeddings[g] = [vec]
        else:
            embeddings[g].append(vec)

    # Collect all tokens, including all genre names and update histograms.
    # The genres_all key here is different from genre_list, which is a list
    # of only top-level genres that the track is categorized under. genres_all
    # is part of the collection of keywords for this track.
    tokens = list()
    for g in metadat['track', 'genres_all']:
        tokens.append(genres.loc[g].title)
    tokens.extend(metadat['track', 'tags'])

    # Update histogram counts
    for g in genre_list:
        # Create histogram for genre if not yet seen
        if g not in histo:
            histo[g] = dict()

        for t in tokens:
            if t not in histo[g]:
                histo[g][t] = 1
            else:
                histo[g][t] += 1

if __name__ == '__main__':
    # Create directory for processed data
    try:
        os.mkdir(WRITE_DIR, mode=0o775)
    except FileExistsError:
        pass

    # FMA metadata
    tracks = fma.utils.load('fma/data/fma_metadata/tracks.csv')
    genres = fma.utils.load('fma/data/fma_metadata/genres.csv')

    valid_select = (tracks['set', 'subset'] == 'small') & (tracks['set', 'split'] == 'validation')
    data = tracks[valid_select]

    validation_embeddings = dict()

    validation_histograms = dict()

    # clap_config = ClapAudioConfig.from_pretrained(name_or_path='laion/larger_clap_music')
    clap_model = ClapAudioModelWithProjection.from_pretrained('laion/larger_clap_music')
    clap_extractor = ClapFeatureExtractor.from_pretrained('laion/larger_clap_music')
    clap_sr = clap_extractor.sampling_rate

    print('CLAP sampling_rate:', clap_sr)

    clap_model.eval()

    # Read in the validation set and extract embeddings and compute statistics for FAD
    for track_id, metadat in data.iterrows():
        try:
            print("Embeddings for track", track_id, end=',')

            vec = compute_embedding(
                clap_model,
                clap_extractor,
                clap_sr,
                track_id
            )

            toplevel_genres = get_toplevel_genres(genres, metadat['track', 'genres'])
            print(" genres", toplevel_genres)
            
            record_statistics(
                genres,
                validation_embeddings,
                validation_histograms,
                toplevel_genres,
                vec
            )
        
            #write_dir = os.path.dirname(write_path)
            #if not os.path.isdir(write_dir):
            #    os.mkdir(write_dir, mode=0o775)

        except RuntimeError as err:
            print("Error reading or processing FMA", track_id, ":", err)

    validation_statistics = dict()
    for g, embeddings in validation_embeddings.items():
        if len(embeddings) < 11:
            print("Skipping genre", g, "because too few samples were seen")
        else:
            stack = torch.stack(embeddings, dim=1)
            mu = torch.mean(stack, dim=1)
            sigma = torch.cov(stack)
            validation_statistics[g] = (mu, sigma)

    torch.save(validation_statistics, WRITE_DIR + 'statistics.pt')

    torch.save(validation_histograms, WRITE_DIR + 'histograms.pt')
    