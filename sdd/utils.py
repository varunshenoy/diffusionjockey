#
# The MTG TSV files are poorly formatted so this script is to do the following:
#   1. The column values are prefixed, i.e. "track_0000214" instead of just "0000214"
#      in the TRACK_ID column. Same thing for genres, so we do additional processing
#      and pack everything in a dictionary that is handed off to pandas.
#   2. Rows have all their tags dumped at the end and separated by the column
#      delimiter (tab character), so pandas cannot parse the rows because they appear
#      to have differing numbers of entries and do not align with the headers. We
#      collect all the tags by category (genre, instrument, mood/theme) into a list
#      for use with pandas.
#
import csv
import pandas as pd

def format_tags(processed, row):
    genres = []
    instruments = []
    moods_themes = []

    # Tags are found in the sixth column onwards.
    # Take each "category---value" string and split it at the
    # "---" and bucket values based on category.
    for i in range(5, len(row)):
        tag = row[i].split('---')
        if tag[0] == 'genre':
            genres.append(tag[1])
        elif tag[0] == 'instrument':
            instruments.append(tag[1])
        elif tag[0] == 'mood/theme':
            moods_themes.append(tag[1])

    processed['genres'].append(genres)
    processed['instruments'].append(instruments)
    processed['moods_themes'].append(moods_themes)

def load_metadata(tsv_path):
    processed = {
        'track_id': [],
        'artist_id': [],
        'album_id': [],
        'path': [],
        'duration': [],
        'genres': [],
        'instruments': [],
        'moods_themes': []
    }

    with open(tsv_path, newline='') as tsv:
        reader = csv.reader(tsv, dialect='excel-tab')
        header = next(reader)

        for row in reader:
            processed['track_id'].append(int(row[0].replace('track_', '')))
            processed['artist_id'].append(int(row[1].replace('artist_', '')))
            processed['album_id'].append(int(row[2].replace('album_', '')))
            processed['path'].append(row[3].replace('path_', ''))
            processed['duration'].append(float(row[4].replace('duration_', '')))

            format_tags(processed, row)

    df = pd.DataFrame(processed)
    df.set_index('track_id', inplace=True)

    return df