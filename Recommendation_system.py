import pandas as pd
import random
import json
import time
import os
import spotipy
from spotipy.oauth2 import SpotifyOAuth
from dotenv import load_dotenv
import K_means_code
import torch as tc
import Spotipy_code

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

def recommend_songs():

    song_name =  "Diving Bell" #input("What song do you like?\n")
    print('')
    artist_name = "Starset"# input("Who is the artist of the song you like?\n")
    print('')
    print('Processing. Please wait...')
        
    query = f"track:{song_name} artist:{artist_name}"
    results = sp.search(q=query, type='track', limit=1) 

    if results:
        track = results['tracks']['items'][0]
        track_id = track['id']
        track_uri = track['uri']
        audio_features = sp.audio_features(track_uri)
        audio_features = audio_features[0]
        track_info = {
                    "track_name": song_name,
                    "track_id": track_id,
                    "artist_name": artist_name,
                    "danceability": audio_features['danceability'],
                    "energy": audio_features["energy"],
                    "key": audio_features["key"],
                    "loudness": audio_features['loudness'],
                    "mode": audio_features['mode'],
                    "speechiness": audio_features["speechiness"],
                    "acousticness": audio_features["acousticness"],
                    "instrumentalness": audio_features['instrumentalness'],
                    "liveness": audio_features['liveness'],
                    "valence": audio_features['valence'],
                    "tempo": audio_features['tempo'],
                    "duration_ms": audio_features['duration_ms'],
                    "time_signature": audio_features['time_signature']
                }

        # get artist id and genre information
        track_data = sp.track(track_id)
        artist_ids = []

        for artist in track_data["artists"]:
            artist_ids.append(artist["id"])
            break

        artists_data = sp.artists(artist_ids)

        genres = []

        for artist in artists_data["artists"]:
            genres += artist["genres"]

        genres = set(genres)
        genres = list(genres)

        track_info['genre'] = genres
        song_data = Spotipy_code.song_data()
        song_data.append(track_info)
        with open("no_bad_songs.json", 'w') as json_file:
            json.dump(song_data, json_file, indent=4)    
            print("Song written to file\n")
            
            
        with open("no_bad_songs.json", 'r') as file:
            song_data = json.load(file)

        num_tracks = len(song_data)

        tensor_of_features = Spotipy_code.get_audio_features(num_tracks)

        _, tensor_of_features, labels = K_means_code.k_means(4, tensor_of_features)

        print('Almost there...\n')
        print('')

        # get points in the same cluster as input song
        #cluster_points = tensor_of_features[labels == labels[-1]]

        # keep track of indices of songs in the cluster
        indices_in_cluster = []
        for i, label in enumerate(labels):
            if label == labels[-1] and i != (len(song_data) - 1):
                indices_in_cluster.append(i)

        # get songs with similar genres 
        user_song = song_data[-1]
        #user_song_genres = user_song['genre']

        print(user_song)

        #print("Here are some songs you may like:\n")
        #for i in range(5):
            #random.seed(234)
            #recommendation_index = random.choice(indices_in_cluster)
            #recommendation = song_data[recommendation_index]
            
            #print(f"Song {i+1}: \n {tensor_of_features[recommendation_index]}")
            
            
            #name = recommendation['track_name']
            #artist = recommendation['artist_name']
            #print(f"{name} by {artist}")
            #indices_in_cluster.remove(recommendation_index)

        

#----------------------------------------------------------------------------------------------------------------------------------
        # delete the user-entered song from the dataset 
        del song_data[-1]
        with open("audio_information.json", 'w') as json_file:
            json.dump(song_data, json_file, indent=4)    
            #print("Song removed from file\n")    
        
    else:
        print('No song or artist found. Please try again')


if __name__ == "__main__":
    recommend_songs()
    