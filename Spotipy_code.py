import pandas as pd
import json
import time
import os
import spotipy
from spotipy.oauth2 import SpotifyOAuth
from dotenv import load_dotenv
import torch as tc

# load the environment variables    
load_dotenv()

# load the dataset
dataset = '/home/titwik/Projects/Spotify Project/song_dataset/data/mpd.slice.0-999.json'

# set the client id, client secret, redirect_uri and scope for project
client_id = os.getenv('client_id')
client_secret = os.getenv('client_secret')
redirect_uri = os.getenv('redirect_uri')
scope = "user-library-read"

# intialize the spotify client
sp = spotipy.Spotify(auth_manager=SpotifyOAuth(client_id=client_id,
                                               client_secret=client_secret,
                                               redirect_uri=redirect_uri,
                                               scope=scope))

# define a function to load unique songs from the dataset
def track_uris(dataset):           
     
    # Load the JSON file
    with open(dataset, 'r') as f:
        data = json.load(f)

    # Extract playlists
    playlists = data['playlists']

    # Use a list to maintain order and a set for quick duplicate checking
    track_uri_list = []
    track_uri_set = set()

    # Loop through the tracks in each playlist 
    for playlist in playlists:
        for track in playlist['tracks']:
            uri = track['track_uri']
            if uri not in track_uri_set:
                track_uri_list.append(uri)
                track_uri_set.add(uri)
    
    return track_uri_list   

# define a function to get a sample list of song uri's for testing
def sample_list(num_of_songs, dataset):
    sample_list = []
    uri = track_uris(dataset)
    for i in range(num_of_songs):
        sample_list.append(uri[i])
    return sample_list

# load the dataset 
# there are 130630 entries
def dataset(input_file = "dataset.csv", output_file = "audio_information.json"):

    # load the dataset
    df = pd.read_csv(input_file, low_memory=False)

    # add the information to a .json file
    try:
        with open(output_file, 'r') as file: 
            current_data = json.load(file)
            #print('Data exists')
            return current_data

    except FileNotFoundError:
        json_content = []

    errors = 0
    for i, row in df.iterrows():

        try:

            acousticness = float(row['acousticness'])
            danceability = float(row['danceability'])
            energy = float(row['energy'])
            key = float(row['key'])
            loudness = float(row['loudness'])
            mode = float(row['mode'])
            speechiness = float(row['speechiness'])
            instrumentalness = float(row['instrumentalness'])
            liveness = float(row['liveness'])
            valence = float(row['valence'])
            tempo = float(row['tempo'])
            

            track_info = {
                "track_name": row['track_name'],
                "track_id": row["track_id"],
                "artist_name": row["artist_name"],
                "danceability": danceability,
                "energy": energy,
                "key": key,
                "loudness": loudness,
                "mode": mode,
                "speechiness": speechiness,
                "acousticness": acousticness,
                "instrumentalness": instrumentalness,
                "liveness": liveness,
                "valence": valence,
                "tempo": tempo,
                "duration_ms": row['duration_ms'],
                "time_signature": row['time_signature']
            }
            json_content.append(track_info)
            
            # debugging statement
            #print(f"Song number {i} has been added.")
        
        except ValueError:
            print(f"Skipping song number {i} due to conversion error")
            errors+=1
            continue

    # Write JSON data to a file
    with open(output_file, 'w') as json_file:
        json.dump(json_content, json_file, indent=4)

    print(f"{130664 - errors} songs have been successfully written to {output_file}")
        
# get the audio features
def get_audio_features(num_tracks=130630, features_file = "audio_information.json"):
    
    features_json = dataset(output_file=features_file)
    
    # initialize tensor to store song data
    features_tensor = tc.zeros(num_tracks, 6)   # change the second dimension for the number of features

    # extract the song information from the .json file
    for index in range(num_tracks):
        features = features_json[index]
        features_tensor[index, 0] = features["acousticness"]
        features_tensor[index, 1] = features['energy']
        features_tensor[index, 2] = features['instrumentalness']
        features_tensor[index, 3] = features['danceability']
        features_tensor[index, 4] = features['loudness']
        features_tensor[index, 5] = features['valence']

        # if testing a sample size, use num_tracks to break the loop
        if index == (num_tracks-1):
            break

    return features_tensor

# use the API to get the genre of every artist in the dataset
def get_genre(artist_id):
    pass

if __name__ == "__main__":
    print(get_audio_features(2))
