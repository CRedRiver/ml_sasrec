import pandas as pd
import numpy as np

def preprocess(data_path, genre_path=None):
    df = pd.read_csv(data_path, sep = ',')

    # drop tags column
    df = df[["userId","movieId","timestamp"]]

    # sort by userid then for each userid sort by timestamp
    df = df.sort_values(by = ["userId","timestamp"])

    # drop duplicate pairs of [userId, movieId]
    df = df.drop_duplicates(subset = ["userId","movieId"], keep = 'first')
    
    # remap items so that they are in range 1 - N
    unique_items = df["movieId"].unique()
    item_map = {raw_id: new_id for new_id, raw_id in enumerate(unique_items, start=1)}
    df['movieId'] = df['movieId'].map(item_map)

    num_items = len(unique_items) + 1

    # get sequences of movies watched for each user
    user_sequences = df.groupby("userId")["movieId"].apply(list).to_dict()
    sequences = list(user_sequences.values())

    if genre_path:
        genre_df = pd.read_csv(genre_path, sep=',')
        unique_genres = genre_df["genres"].unique()
        genre_map = {genre: i for i, genre in enumerate(unique_genres, start=1)}

        movie_to_genre_id = {}
        for _, row in genre_df.iterrows():
            raw_movie_id = row['movieId']
            
            if raw_movie_id in item_map:
                remapped_id = item_map[raw_movie_id]
                movie_to_genre_id[remapped_id] = genre_map[row['genres']]
                
        num_genres = len(unique_genres)
        return sequences, num_items, item_map, movie_to_genre_id, num_genres
    else:
        return sequences, num_items, item_map

