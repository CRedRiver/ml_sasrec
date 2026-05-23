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
        
        # 1. Find every unique individual genre
        all_unique_genres = set()
        for genres_str in genre_df['genres']:
            all_unique_genres.update(genres_str.split('|'))
            
        # 2. Sort them so the index mapping is always consistent
        all_unique_genres = sorted(list(all_unique_genres))
        num_genres = len(all_unique_genres)
        
        # Create a mapping like {'Action': 0, 'Comedy': 1, 'Horror': 2...}
        genre_to_idx = {genre: i for i, genre in enumerate(all_unique_genres)}

        movie_to_multihot = {}
        
        for _, row in genre_df.iterrows():
            raw_movie_id = row['movieId']
            
            if raw_movie_id in item_map:
                remapped_id = item_map[raw_movie_id]
                
                # Start with an array of all 0s
                multi_hot_vector = [0.0] * num_genres
                
                # Flip the bit to 1.0 for every genre this movie belongs to
                for g in row['genres'].split('|'):
                    multi_hot_vector[genre_to_idx[g]] = 1.0
                    
                movie_to_multihot[remapped_id] = multi_hot_vector
                
        # Return the new multihot dictionary and the size of the multi-hot vector
        return sequences, num_items, item_map, movie_to_multihot, num_genres
    else:
        return sequences, num_items, item_map

