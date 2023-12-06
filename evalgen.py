import numpy as np
import torch
import os
import argparse
import resampy
import pandas as pd
import soundfile as sf

# Numerically stable computation of Frechet audio distance 
# from https://github.com/microsoft/fadtk
from frechet_distance import calc_frechet_distance

WRITE_DIR = 'eval_spectro/'

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

FINETUNE_CHECKPOINT = 'outputs/rank512-output-2/checkpoint-9000'

def compute_frechet_audio_dist():
    from transformers import ClapFeatureExtractor, ClapAudioModelWithProjection

    stats = torch.load('eval/statistics.pt')

    eval_samples = dict()

    info = list()

    clap_model = ClapAudioModelWithProjection.from_pretrained('laion/larger_clap_music')
    clap_extractor = ClapFeatureExtractor.from_pretrained('laion/larger_clap_music')
    clap_sr = clap_extractor.sampling_rate

    clap_model.eval()

    for category, prompt_list in PROMPTS.items():
        eval_samples[category] = list()

        for i, prompt in enumerate(prompt_list):
            for count in range(PROMPT_COUNT[category][i]):
                    filename = "{}/{}_{}_{}.mp3".format(AUDIO_DIR, category, i, count)
                    wav, sr = sf.read(filename)
                    if wav.ndim == 2:
                        data = resampy.resample(wav[:, 0], sr, clap_sr)
                    else:
                        data = resampy.resample(wav, sr, clap_sr)
                    features = clap_extractor(data, sampling_rate=clap_sr, return_tensors='pt')
                    output = clap_model(**features)
                    nparr = torch.squeeze(output.audio_embeds.detach()).numpy()
                    eval_samples[category].append(nparr)

                    info.append({
                        'name': "{}_{}_{}".format(category, i, count),
                        'prompt': prompt,
                        'target_genre': GENRE_NAME[category],
                        'sample_no': count
                    })

    info = pd.DataFrame(info)
    info.to_csv(AUDIO_DIR + 'info.csv', index=False)

    results = {
        'Genre': [],
        'Closest': [],
        'Self-Distance': []
    }

    # Populate genre names
    for category in GENRE_NAME:
        results[GENRE_NAME[category]] = []

    for category, sample_list in eval_samples.items():
        arr = np.stack(sample_list)

        # Generated samples in a category
        mu1 = np.mean(arr, axis=0)
        sigma1 = np.cov(arr, rowvar=False)

        results['Genre'].append(GENRE_NAME[category])

        closest_category = 0 # Which category was the closest?
        closest_distance = float('inf') # How far from the category it was generated from?

        for g, s in stats.items():
            # Validation set statistics
            mu2 = s[0].numpy()
            sigma2 = s[1].numpy()

            dist = calc_frechet_distance(mu1, sigma1, mu2, sigma2)
            if dist < closest_distance:
                closest_distance = dist
                closest_category = g

            results[GENRE_NAME[g]].append(dist)

            if g == category:
                results['Self-Distance'].append(dist)

        results['Closest'].append(GENRE_NAME[closest_category])

    results = pd.DataFrame(results)
    results.to_csv(AUDIO_DIR + 'results.csv', index=False)

def create_samples():
    from diffusers import DiffusionPipeline, DPMSolverMultistepScheduler

    pipe = DiffusionPipeline.from_pretrained("stabilityai/stable-diffusion-xl-base-1.0", torch_dtype=torch.float16, use_safetensors=True, variant="fp16")
    pipe.to("cuda")
    pipe.scheduler = DPMSolverMultistepScheduler.from_config(pipe.scheduler.config)
    pipe.load_lora_weights(FINETUNE_CHECKPOINT)

    # if using torch < 2.0
    # pipe.enable_xformers_memory_efficient_attention()

    for category, prompt_list in PROMPTS.items():
        for i, prompt in enumerate(prompt_list):
            for count in range(PROMPT_COUNT[category][i]):
                image = pipe(prompt=prompt).images[0]
                filename = "{}/{}_{}_{}.png".format(WRITE_DIR, category, i, count)
                image.save(filename)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Generate evaluation set and measure audio distance.')
    parser.add_argument('--gen', action='store_true', help='generate samples')
    parser.add_argument('--eval', action='store_true', help='compute frechet distance')
    args = parser.parse_args()

    if not (args.gen or args.eval):
        print('Must specify one of --gen or --eval')
        exit()
    elif args.gen and args.eval:
        print('Generating and evaluating in the same invocation is not supported')
        exit()

    # Create directory for processed data
    try:
        os.mkdir(WRITE_DIR, mode=0o775)
    except FileExistsError:
        pass

    # This is the histogram of tags for each category
    # histo = torch.load('eval/histograms.pt')

    if args.gen:
        create_samples()
    elif args.eval:
        compute_frechet_audio_dist()