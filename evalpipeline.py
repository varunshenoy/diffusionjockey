import numpy as np
import pandas as pd
import soundfile as sf
import torch
import resampy
import os

from transformers import ClapFeatureExtractor, ClapAudioModelWithProjection

# FMA utilities
import fma.utils

# Numerically stable computation of Frechet audio distance 
# from https://github.com/microsoft/fadtk
from frechet_distance import calc_frechet_distance

# FMA mp3 directory
FMA_AUDIO_DIR = 'fma/data/fma_small/'

# Directory for validation references
WRITE_DIR = 'evalpipeline/'

AUDIO_DIR = 'eval_audio/'

# Keyed by top-level genre id
PROMPTS = {
    38: ['Experimental, Noise, Improv, space',
         'Electroacoustic, Novelty, Sound Poetry',
         'Minimalism, outer space',
         'acid jazz',
         'trip hop'
    ],
    12: ['Rock, Power-Pop, Punk',
         'Hardcore, Death-Metal, Grindcore, Rock',
         'Lo-Fi, Garage, New Wave',
         'Post-Rock, psychedelic'
    ],
    17: ['Folk, Singer-Songwriter, strum, soft, morning, string'],
    10: ['Pop, electronic',
         'experimental pop, Synth pop'
    ],
    2: ['Hip-Hop, Rap, Hip-Hop Beats',
        'Breakbeat, Alternative Hip-Hop'
    ],
    15: ['Electronic, Techno',
         'Electronic, Trip-Hop, Chip Music',
         'Ambient Electronic, Chill-out, House',
          'Bigbeat, Dance, italy, House'
    ],
    21: ['Hip-Hop, Rap, Hip-Hop Beats',
         'Breakbeat, Alternative Hip-Hop'
    ],
    1235: ['Soundtrack, Instrumental, Ambient, atmosphere']
}

# How many samples to generate from each prompt above. 12 total per category.
PROMPT_COUNT = {
    38: [2, 2, 2, 3, 3],
    12: [3, 3, 3, 3],
    17: [12],
    10: [6, 6],
    2: [3, 3, 3, 3],
    15: [3, 3, 3, 3],
    21: [6, 6],
    1235: [12]
}

# Names of top-level genres
GENRE_NAME = {
    38: 'Experimental',
    12: 'Rock',
    17: 'Folk',
    10: 'Pop',
    2: 'International',
    15: 'Electronic',
    21: 'Hip-Hop',
    1235: 'Instrumental'
}

clap_model = None
clap_extractor = None
clap_sr = None

def get_toplevel_genres(genres, id_list):
    s = set()

    for genre_id in id_list:
        toplevel = genre_id
        while genres.loc[toplevel]['parent'] != 0:
            toplevel = genres.loc[toplevel]['parent']
        s.add(toplevel)

    return s

def get_clap_embedding(audio_path):
    # The data returned is in the range [-1.0, +1.0]
    wav, sr = sf.read(audio_path)

    if wav.ndim == 2:
        data = resampy.resample(wav[:, 0], sr, clap_sr)
    else:
        data = resampy.resample(wav, sr, clap_sr)

    features = clap_extractor(data, sampling_rate=clap_sr, return_tensors='pt')

    output = clap_model(**features)

    return torch.squeeze(output.audio_embeds.detach())

def record_statistics(genres, embeddings, histo, genre_list, metadat, track_id, vec):
    # Add the embedding to all its top level genres
    for g in genre_list:
        if g not in embeddings:
            embeddings[g] = [(track_id, vec)]
        else:
            embeddings[g].append((track_id, vec))

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

def compute_validation_stats():
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

    # Read in the validation set and extract embeddings and compute statistics for FAD
    for track_id, metadat in data.iterrows():
        try:
            print("Embeddings for track", track_id, end=',')

            vec = get_clap_embedding(fma.utils.get_audio_path(FMA_AUDIO_DIR, track_id))

            toplevel_genres = get_toplevel_genres(genres, metadat['track', 'genres'])
            print(" genres", toplevel_genres)
            
            record_statistics(
                genres,
                validation_embeddings,
                validation_histograms,
                toplevel_genres,
                metadat,
                track_id,
                vec
            )
        
            #write_dir = os.path.dirname(write_path)
            #if not os.path.isdir(write_dir):
            #    os.mkdir(write_dir, mode=0o775)

        except RuntimeError as err:
            print("Error reading or processing FMA", track_id, ":", err)

    torch.save(validation_embeddings, WRITE_DIR + 'valid_embeddings.pt')
    torch.save(validation_histograms, WRITE_DIR + 'valid_histograms.pt')

    return validation_embeddings

def compute_frechet_audio_dist(validation_embeddings):
    stats = dict()
    for g, embeddings in validation_embeddings.items():
        if len(embeddings) < 11:
            print("Skipping genre", g, "because too few samples were seen")
        else:
            stack = torch.stack([v for (i, v) in embeddings]).numpy()
            stats[g] = (np.mean(stack, axis=0), np.cov(stack, rowvar=False))

    eval_embeddings = dict()

    info = list()

    for category, prompt_list in PROMPTS.items():
        eval_embeddings[category] = list()

        # Get the audio file
        for i, prompt in enumerate(prompt_list):
            for count in range(PROMPT_COUNT[category][i]):
                    name = "{}_{}_{}".format(category, i, count)

                    print("Embeddings for sample", name, "in", GENRE_NAME[category])

                    embedding = get_clap_embedding(AUDIO_DIR + name + '.mp3')
                    eval_embeddings[category].append((name, embedding))

                    info.append({
                        'name': name,
                        'prompt': prompt,
                        'target_genre': GENRE_NAME[category],
                        'sample_no': count
                    })

    torch.save(eval_embeddings, WRITE_DIR + 'eval_embeddings.pt')

    info = pd.DataFrame(info)
    info.to_csv(WRITE_DIR + 'info.csv', index=False)

    results = {
        'Genre': [],
        'Closest': [],
        'Self-Distance': []
    }

    # Populate genre names
    for category in GENRE_NAME:
        results[GENRE_NAME[category]] = []

    for category, sample_list in eval_embeddings.items():
        arr = torch.stack([v for (i, v) in sample_list]).numpy()

        # Generated samples in a category
        mu1 = np.mean(arr, axis=0)
        sigma1 = np.cov(arr, rowvar=False)

        results['Genre'].append(GENRE_NAME[category])

        closest_category = 0 # Which category was the closest?
        closest_distance = float('inf') # How far from the category it was generated from?

        for g, s in stats.items():
            # Validation set statistics
            mu2 = s[0]
            sigma2 = s[1]

            dist = calc_frechet_distance(mu1, sigma1, mu2, sigma2)
            if dist < closest_distance:
                closest_distance = dist
                closest_category = g

            results[GENRE_NAME[g]].append(dist)

            if g == category:
                results['Self-Distance'].append(dist)

        results['Closest'].append(GENRE_NAME[closest_category])

    results = pd.DataFrame(results)
    results.to_csv(WRITE_DIR + 'results.csv', index=False)

if __name__ == '__main__':
    # Create directory for processed data
    try:
        os.mkdir(WRITE_DIR, mode=0o775)
    except FileExistsError:
        pass

    clap_model = ClapAudioModelWithProjection.from_pretrained('laion/larger_clap_music')
    clap_extractor = ClapFeatureExtractor.from_pretrained('laion/larger_clap_music')
    clap_sr = clap_extractor.sampling_rate

    print('CLAP sampling_rate:', clap_sr)

    clap_model.eval()

    valid_emb = compute_validation_stats()
    compute_frechet_audio_dist(valid_emb)