#!/bin/bash
wget -P data https://zenodo.org/record/10072001/files/song_describer.csv \
    https://zenodo.org/record/10072001/files/audio.zip \
    https://github.com/MTG/mtg-jamendo-dataset/raw/master/data/raw_30s_cleantags_50artists.tsv \
    https://github.com/MTG/mtg-jamendo-dataset/raw/master/data/autotagging_genre.tsv \
    https://github.com/MTG/mtg-jamendo-dataset/raw/master/data/autotagging_instrument.tsv \
    https://github.com/MTG/mtg-jamendo-dataset/raw/master/data/autotagging_moodtheme.tsv
unzip data/audio.zip -d data/audio