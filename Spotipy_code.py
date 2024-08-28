import pandas as pd
import json
import time
import os
import spotipy
from spotipy.oauth2 import SpotifyOAuth
from dotenv import load_dotenv
import K_means_code
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


#----------------------------------------------------------------------------------------------------------------------------------

# load the dataset 
def song_data(input_file = "dataset.csv", output_file = "audio_information.json"):

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
    duplicates = set()
    for i, row in df.iterrows():

        # check for duplicate songs
        if row['track_id'] in duplicates:
            print(f'Skipping song number {i} as it is a duplicate')
            errors+=1
            continue
        else:
            duplicates.add(row['track_id'])

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

            if (acousticness > 1 or
                danceability > 1 or
                valence > 1 or
                energy > 1 or 
                instrumentalness > 1 or
                liveness > 1 or 
                speechiness > 1):

                print(f"Skipping song number {i} due to too large of a value in a field")
                errors+=1
                continue

            else:       

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
        
        except ValueError:
            print(f"Skipping song number {i} due to conversion error")
            errors+=1
            continue

    # Write JSON data to a file
    with open(output_file, 'w') as json_file:
        json.dump(json_content, json_file, indent=4)

    print(f"{130664 - errors-1} songs have been successfully written to {output_file}")

    with open(output_file, 'r') as file: 
        data = json.load(file)
        return data

# get the audio features
def get_audio_features(num_tracks=130291, features_file = "audio_information.json"):
    
    features_json = song_data(output_file=features_file)
    
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
    
    # get rid of rows with nan values
    nan_mask = ~tc.isnan(features_tensor).any(dim=1)
    features_tensor = features_tensor[nan_mask]

    return features_tensor

# use the API to get the genre of every artist in the dataset
def get_genre(num_tracks=130291, song_dataset = "audio_information.json"):

    try:
        with open(song_dataset, 'r') as file: 
            songs = json.load(file)

    except FileNotFoundError:
        print("Generating dataset...")
        songs = song_data()
        print('')

    track_ids = [songs[i]["track_id"] for i in range(num_tracks)]

    batch_size = 50
    artist_ids = []

    # Process track IDs in batches
    for i in range(0, len(track_ids), batch_size):
        batch = track_ids[i:i + batch_size]
        track_data = sp.tracks(batch)   
        time.sleep(5)

        # Extract artist IDs from each track in the batch
        for track in track_data['tracks']:
            for artist in track['artists']:
                artist_ids.append(artist['id'])
                break       

    genres = []

    for i in range(0, len(artist_ids), batch_size):
        batch = artist_ids[i:i+batch_size]
        artist_info = sp.artists(batch)
        time.sleep(5)
        for artist in artist_info['artists']:
            genres.append(artist['genres'])

    # edit the dataset to have genres
    for i in range(num_tracks):
        song = songs[i]     
        song['genre'] = genres[i]
        #print(f"Genre done for song {i}")

    with open(song_dataset, 'w') as json_file:
        json.dump(songs, json_file, indent=4)       
    
    print(f"Genres added to {num_tracks} songs.")

#-------------------------------------------------------------------------------------
def delete_bad_songs(num_tracks=130291, song_dataset = "audio_information.json"):

    with open(song_dataset, 'r') as file:   
        songs = json.load(file)
    
    # Create a new list with songs that have genres
    good_songs = [song for song in songs[:num_tracks] if len(song['genre']) > 0]
    
    # Calculate the number of deleted songs
    deleted = num_tracks - len(good_songs)
    
    # Save the filtered dataset
    with open(song_dataset, 'w') as json_file:
        json.dump(good_songs, json_file, indent=4)
    
    print(f"Deleted {deleted} songs.")


#def final_cleaning()

if __name__ == "__main__":
    #song_data()
    n = 20
    get_genre(n)    
    print('')
    delete_bad_songs(n)
    

    

    
